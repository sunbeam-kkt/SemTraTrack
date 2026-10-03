"""Full paper-objective integration with the official YOLOv13 Ultralytics fork.

This module does not duplicate YOLOv13 itself.  It subclasses the official
`DetectionModel`, taps the three feature levels feeding Detect (layers
23/27/31 of the official YOLOv13-S graph), and adds the paper's SPS, MSA-NWD,
TCR, MIFD and BNS-HM losses to the native detector criterion.

The integration follows the public YOLOv13/v8DetectionLoss API: native
TaskAlignedAssigner outputs are recomputed without changing the native loss so
that SPS foreground locations and scale-wise predicted/target boxes use the
same assignment as L_dec.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..config import SemTraTrackConfig
from ..sps import (
    SPSBranch,
    bns_hard_negative_loss,
    box_assignment_mask,
    foreground_background_contrastive_loss,
    instance_prototypes,
    instance_prototypes_flat,
    mifd_loss,
    q_sep,
    temporal_consistency_loss,
)

try:
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.tal import make_anchors
    _UPSTREAM_OK = True
    _UPSTREAM_ERROR = None
except Exception as exc:  # keep unit tests importable without the external fork
    DetectionModel = nn.Module  # type: ignore
    make_anchors = None
    _UPSTREAM_OK = False
    _UPSTREAM_ERROR = exc

EPS = 1e-8


def require_yolov13() -> None:
    if not _UPSTREAM_OK:
        raise ImportError(
            "The official iMoonLab YOLOv13 fork is required. Run `bash scripts/bootstrap_yolov13.sh` "
            "or `pip install -e third_party/yolov13` after checking out the pinned commit."
        ) from _UPSTREAM_ERROR


def norm_xywh_to_xyxy(bboxes: Tensor, batch_idx: Tensor, image_hw: Tuple[int, int]) -> Tensor:
    """Convert normalized xywh labels to pixel xyxy while preserving label order."""
    if bboxes.numel() == 0:
        return bboxes.new_empty((0, 4))
    h, w = image_hw
    x, y, bw, bh = bboxes.unbind(-1)
    return torch.stack(((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h), -1)


def _native_assignment(criterion, preds, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Reproduce YOLOv13 native TAL assignment for auxiliary-loss alignment.

    It mirrors the public v8DetectionLoss code used by the YOLOv13 repository
    but leaves the native criterion itself untouched.
    """
    feats = preds[1] if isinstance(preds, tuple) else preds
    if not isinstance(feats, (list, tuple)) or not feats:
        raise RuntimeError("Unsupported YOLOv13 prediction structure; expected Detect training feature list.")
    pred_distri, pred_scores = torch.cat(
        [xi.view(feats[0].shape[0], criterion.no, -1) for xi in feats], 2
    ).split((criterion.reg_max * 4, criterion.nc), 1)
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    dtype = pred_scores.dtype
    bs = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=pred_scores.device, dtype=dtype) * criterion.stride[0]
    anchor_points, stride_tensor = make_anchors(feats, criterion.stride, 0.5)
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = criterion.preprocess(targets.to(criterion.device), bs, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
    _, target_bboxes_px, target_scores, fg_mask, target_gt_idx = criterion.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )
    # Match native loss coordinate system after assignment.
    target_bboxes_grid = target_bboxes_px / stride_tensor
    return {
        "feats": feats,
        "pred_bboxes": pred_bboxes,
        "target_bboxes": target_bboxes_grid,
        "target_bboxes_px": target_bboxes_px,
        "target_scores": target_scores,
        "fg_mask": fg_mask,
        "target_gt_idx": target_gt_idx,
        "stride_tensor": stride_tensor,
        "imgsz": imgsz,
    }


class ScaleAttentionNWD(nn.Module):
    r"""MSA-NWD over native positive anchors at each detection scale.

    The manuscript defines :math:`\alpha_s` from the feature map of each input
    sample.  Consequently, scale attention is computed per image and only then
    averaged over the batch.  A previous implementation averaged feature
    logits over the whole batch first, which made every image share one set of
    scale weights and did not implement Eq. (6) faithfully.
    """
    def __init__(self, feature_dims=(128, 256, 512), hidden=32, eta=0.50):
        super().__init__()
        self.eta = eta
        self.maps = nn.ModuleList([
            nn.Sequential(nn.Linear(c, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)) for c in feature_dims
        ])

    @staticmethod
    def _moments(xyxy: Tensor) -> Tuple[Tensor, Tensor]:
        mu = (xyxy[..., :2] + xyxy[..., 2:]) * 0.5
        # The paper models scale as the Gaussian spread corresponding to box extent.
        sigma = (xyxy[..., 2:] - xyxy[..., :2]).clamp_min(0) * 0.5
        return mu, sigma

    def forward(self, visual_feats: Sequence[Tensor], assignment: Dict[str, Tensor]) -> Tensor:
        if len(visual_feats) != len(self.maps):
            raise ValueError("Expected three YOLOv13 detection-scale visual features.")
        fg = assignment["fg_mask"]
        pb, tb = assignment["pred_bboxes"], assignment["target_bboxes"]
        det_feats = assignment["feats"]
        lengths = [int(f.shape[-2] * f.shape[-1]) for f in det_feats]
        starts = [0]
        for n in lengths[:-1]: starts.append(starts[-1] + n)
        batch_size = int(fg.shape[0])
        logits, dists, valid = [], [], []
        for k, (vf, mapper, start, n) in enumerate(zip(visual_feats, self.maps, starts, lengths)):
            if vf.shape[1] != mapper[0].in_features:
                raise RuntimeError(
                    f"YOLOv13 feature-channel mismatch at scale {k}: got {vf.shape[1]}, "
                    f"expected {mapper[0].in_features}. Use the official scale=s graph."
                )
            if int(vf.shape[0]) != batch_size:
                raise RuntimeError("All MSA-NWD feature levels must share the assignment batch size.")
            # [B]: one attention logit per image and detection scale.
            logits.append(mapper(vf.mean(dim=(-2, -1))).squeeze(-1))
            m = fg[:, start:start+n]
            scale_distances, scale_valid = [], []
            for b in range(batch_size):
                mb = m[b]
                if mb.any():
                    p = pb[b, start:start+n][mb]
                    t = tb[b, start:start+n][mb]
                    mu_p, sig_p = self._moments(p)
                    mu_t, sig_t = self._moments(t)
                    distance = (
                        (mu_p - mu_t).pow(2).sum(-1)
                        + self.eta * (sig_p - sig_t).pow(2).sum(-1)
                    ).mean()
                    scale_distances.append(distance)
                    scale_valid.append(True)
                else:
                    # Retain the correct device/dtype without contributing to
                    # the loss or the softmax normalization.
                    scale_distances.append(vf[b].sum() * 0.0)
                    scale_valid.append(False)
            dists.append(torch.stack(scale_distances))
            valid.append(torch.tensor(scale_valid, device=vf.device, dtype=torch.bool))

        # [B,S], with attention normalized only across scales that contain a
        # native positive anchor for that image.
        logits_t = torch.stack(logits, dim=1)
        dists_t = torch.stack(dists, dim=1)
        valid_t = torch.stack(valid, dim=1)
        valid_images = valid_t.any(dim=1)
        if not valid_images.any():
            return visual_feats[0].new_tensor(0.0)
        logits_t = logits_t[valid_images]
        dists_t = dists_t[valid_images]
        valid_t = valid_t[valid_images]
        alpha = torch.softmax(logits_t.masked_fill(~valid_t, float("-inf")), dim=1)
        return (alpha * dists_t).sum(dim=1).mean()


if _UPSTREAM_OK:
    class PaperAlignedDetectionModel(DetectionModel):
        """YOLOv13-S with the complete manuscript training objective."""
        FEATURE_LAYERS = (23, 27, 31)
        FEATURE_DIMS = (128, 256, 512)  # official YOLOv13 scale=s channels

        def __init__(self, cfg="configs/yolov13s-UAV.yaml", nc=1, verbose=True, paper_cfg: Optional[SemTraTrackConfig]=None):
            self.paper_cfg = paper_cfg or SemTraTrackConfig()
            self.paper_cfg.validate()
            super().__init__(cfg=cfg, ch=3, nc=nc, verbose=verbose)
            if len(self.model) <= max(self.FEATURE_LAYERS):
                raise RuntimeError("Loaded detector is not the official YOLOv13-S graph required by SemTraTrack.")
            detect_from = getattr(self.model[-1], "f", None)
            if detect_from is not None and list(detect_from) != list(self.FEATURE_LAYERS):
                raise RuntimeError(
                    f"Detect inputs are {detect_from}, expected {self.FEATURE_LAYERS}. "
                    "Use configs/yolov13s-UAV.yaml or a checkpoint trained from that graph."
                )
            self.sps = SPSBranch(self.FEATURE_DIMS[0])
            self.msa_nwd = ScaleAttentionNWD(self.FEATURE_DIMS, eta=self.paper_cfg.eta_nwd)
            self._paper_feats: Dict[int, Tensor] = {}
            self._paper_handles = []
            for idx in self.FEATURE_LAYERS:
                self._paper_handles.append(self.model[idx].register_forward_hook(self._make_hook(idx)))
            self.last_aux: Dict[str, float] = {}

        def _make_hook(self, idx):
            def hook(_module, _inp, out):
                if not torch.is_tensor(out):
                    raise RuntimeError(f"Expected tensor at YOLOv13 layer {idx}, got {type(out)}")
                self._paper_feats[idx] = out
            return hook

        def _tcr(self, flat_protos: Tensor, batch: Dict[str, Tensor]) -> Tensor:
            if flat_protos.numel() == 0 or "track_ids" not in batch:
                return flat_protos.new_tensor(0.0)
            device = flat_protos.device
            bidx = batch["batch_idx"].long().to(device)
            tids = batch["track_ids"].long().to(device)
            vis = batch.get("visibility", torch.ones_like(batch["track_ids"], dtype=torch.float32)).to(device)
            pair_ids_img = batch["pair_id_per_image"].long().to(device)
            slots_img = batch["pair_slot_per_image"].long().to(device)
            losses = []
            for pair_id in pair_ids_img.unique().tolist():
                imgs = (pair_ids_img == int(pair_id)).nonzero(as_tuple=False).flatten()
                if len(imgs) != 2:
                    continue
                i0 = imgs[(slots_img[imgs] == 0).nonzero(as_tuple=False).flatten()[0]]
                i1 = imgs[(slots_img[imgs] == 1).nonzero(as_tuple=False).flatten()[0]]
                m0 = (bidx == i0) & (vis > 0)
                m1 = (bidx == i1) & (vis > 0)
                losses.append(temporal_consistency_loss(flat_protos[m0], tids[m0], flat_protos[m1], tids[m1]))
            return torch.stack(losses).mean() if losses else flat_protos.new_tensor(0.0)

        def loss(self, batch, preds=None):
            if getattr(self, "criterion", None) is None:
                self.criterion = self.init_criterion()
            if preds is None:
                self._paper_feats.clear()
                preds = self.predict(batch["img"])
            native_total, native_items = self.criterion(preds, batch)
            # Standard validation batches do not carry adjacent-frame identities.
            if not batch.get("paper_pair_mode", False):
                return native_total, native_items
            visual_feats = [self._paper_feats[i] for i in self.FEATURE_LAYERS]
            f = visual_feats[0]
            components = self.paper_cfg.enabled_detector_components()
            image_hw = (int(batch["img"].shape[-2]), int(batch["img"].shape[-1]))
            boxes_px = norm_xywh_to_xyxy(batch["bboxes"].to(f.device), batch["batch_idx"].to(f.device), image_hw)
            aux_bidx = batch["batch_idx"].to(f.device)
            assignment = _native_assignment(self.criterion, preds, batch)
            h3, w3 = f.shape[-2:]
            p3_n = h3 * w3
            fg = assignment["fg_mask"][:, :p3_n].reshape(f.shape[0], 1, h3, w3)
            zero = f.new_tensor(0.0)
            if "sps" in components:
                mask, m_f, m_b = self.sps(f)
                l_con = foreground_background_contrastive_loss(m_f, m_b, fg, self.paper_cfg.tau_con)
                protos_by_image = instance_prototypes(f, mask, boxes_px, aux_bidx, image_hw)
                l_mifd = (
                    mifd_loss(protos_by_image, self.paper_cfg.mifd_alpha, self.paper_cfg.tau_soft).to(f.device)
                    if "mifd" in components else zero
                )
                l_hard = (
                    bns_hard_negative_loss(
                        f, m_f, fg, self.paper_cfg.hard_negative_ratio, self.paper_cfg.hard_negative_margin
                    ) if "bns-hm" in components else zero
                )
                if "tcr" in components:
                    flat_protos = instance_prototypes_flat(f, mask, boxes_px, aux_bidx, image_hw)
                    l_tcr = self._tcr(flat_protos, batch)
                else:
                    l_tcr = zero
                # q_sep is monitoring only and uses GT-box support, not TAL positives.
                qmask = box_assignment_mask(boxes_px, aux_bidx, f.shape[0], (h3, w3), image_hw)
                q = q_sep(mask, qmask)
            else:
                l_con = l_tcr = l_mifd = l_hard = q = zero
            l_nwd = self.msa_nwd(visual_feats, assignment) if "msa-nwd" in components else zero
            aux = (
                self.paper_cfg.lambda_con * l_con
                + self.paper_cfg.lambda_nwd * l_nwd
                + self.paper_cfg.lambda_tcr * l_tcr
                + self.paper_cfg.lambda_mifd * l_mifd
                + self.paper_cfg.lambda_hard * l_hard
            )
            bs = int(batch["img"].shape[0])
            total = native_total + aux * bs
            self.last_aux = {
                "L_con": float(l_con.detach()), "L_nwd": float(l_nwd.detach()),
                "L_tcr": float(l_tcr.detach()), "L_mifd": float(l_mifd.detach()),
                "L_hard": float(l_hard.detach()), "q_sep": float(q.detach()),
                "L_aux_weighted": float(aux.detach()),
            }
            # Keep native three loss items so the official validator remains API compatible.
            return total, native_items
else:
    class PaperAlignedDetectionModel(nn.Module):  # pragma: no cover
        def __init__(self, *args, **kwargs):
            super().__init__(); require_yolov13()
