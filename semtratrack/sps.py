"""ISPS-SOD auxiliary components from the current SemTraTrack manuscript.

This module implements the paper equations independently from a particular
YOLOv13 release. It is called by ``semtratrack.training.model`` at the selected
high-resolution detector feature map.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import SemTraTrackConfig

EPS = 1e-8


def l2_normalize(x: Tensor, dim: int = -1, eps: float = EPS) -> Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class SPSBranch(nn.Module):
    """Position-wise normalization--MLP--sigmoid pseudo-segmentation branch.

    Input:  F [B,C,H,W]
    Output: A [B,1,H,W], M_F [B,C,H,W], M_B [B,C,H,W]
    """

    def __init__(self, channels: int, hidden_channels: Optional[int] = None):
        super().__init__()
        hidden = hidden_channels or max(16, channels // 4)
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feature: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if feature.ndim != 4:
            raise ValueError(f"SPS expects [B,C,H,W], got {tuple(feature.shape)}")
        # h(F) is applied position-wise over the channel vector.
        x = feature.permute(0, 2, 3, 1)  # B,H,W,C
        logits = self.mlp(self.norm(x)).permute(0, 3, 1, 2)  # B,1,H,W
        mask = torch.sigmoid(logits)
        m_f = mask * feature
        m_b = (1.0 - mask) * feature
        return mask, m_f, m_b


def box_assignment_mask(
    boxes_xyxy: Tensor,
    batch_indices: Tensor,
    batch_size: int,
    feature_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
) -> Tensor:
    """Rasterize box-level assignments to feature locations.

    This uses detection boxes only; it introduces no pixel-level segmentation
    annotation.  A detector-specific positive-assignment mask can be supplied
    instead if available.
    """
    h, w = feature_hw
    im_h, im_w = image_hw
    out = torch.zeros((batch_size, 1, h, w), dtype=torch.bool, device=boxes_xyxy.device)
    if boxes_xyxy.numel() == 0:
        return out
    sx, sy = w / float(im_w), h / float(im_h)
    for box, bi in zip(boxes_xyxy, batch_indices.long()):
        x1, y1, x2, y2 = box.tolist()
        fx1 = max(0, min(w - 1, int(torch.floor(torch.tensor(x1 * sx)).item())))
        fy1 = max(0, min(h - 1, int(torch.floor(torch.tensor(y1 * sy)).item())))
        fx2 = max(fx1 + 1, min(w, int(torch.ceil(torch.tensor(x2 * sx)).item())))
        fy2 = max(fy1 + 1, min(h, int(torch.ceil(torch.tensor(y2 * sy)).item())))
        out[int(bi), 0, fy1:fy2, fx1:fx2] = True
    return out


def q_sep(mask: Tensor, foreground_locations: Tensor) -> Tensor:
    """Box-conditioned activation-separation diagnostic from Eq. (4)."""
    fg = foreground_locations.bool()
    bg = ~fg
    vals = []
    for b in range(mask.shape[0]):
        a = mask[b, 0]
        f = fg[b, 0]
        g = bg[b, 0]
        fg_mean = a[f].mean() if f.any() else a.new_tensor(0.0)
        bg_mean = a[g].mean() if g.any() else a.new_tensor(0.0)
        vals.append(fg_mean - bg_mean)
    return torch.stack(vals).mean() if vals else mask.new_tensor(0.0)


def foreground_background_contrastive_loss(
    m_f: Tensor,
    m_b: Tensor,
    foreground_locations: Tensor,
    tau: float = 0.10,
    chunk_size: int = 256,
) -> Tensor:
    """Feature-level foreground/background contrastive objective.

    All background locations participate, but positives are processed in chunks
    to avoid allocating an enormous |Omega_F| x |Omega_B| matrix.
    """
    losses: List[Tensor] = []
    for b in range(m_f.shape[0]):
        fg_mask = foreground_locations[b, 0].bool()
        bg_mask = ~fg_mask
        fg = m_f[b].permute(1, 2, 0)[fg_mask]
        bg = m_b[b].permute(1, 2, 0)[bg_mask]
        if fg.numel() == 0 or bg.numel() == 0:
            continue
        proto = l2_normalize(fg.mean(dim=0, keepdim=True), dim=-1)
        fg_n = l2_normalize(fg, dim=-1)
        bg_n = l2_normalize(bg, dim=-1)
        foreground_terms: List[Tensor] = []
        for start in range(0, fg_n.shape[0], chunk_size):
            f = fg_n[start : start + chunk_size]
            pos = (f @ proto.T).squeeze(1) / tau
            neg = (f @ bg_n.T) / tau
            # -log exp(pos)/(exp(pos)+sum exp(neg))
            denom = torch.logsumexp(torch.cat([pos[:, None], neg], dim=1), dim=1)
            foreground_terms.append(denom - pos)
        # Exact 1/|Omega_F| normalization; chunking is memory-only.
        losses.append(torch.cat(foreground_terms).mean())
    return torch.stack(losses).mean() if losses else m_f.new_tensor(0.0)


def instance_prototypes(
    feature: Tensor,
    soft_mask: Tensor,
    boxes_xyxy: Tensor,
    batch_indices: Tensor,
    image_hw: Tuple[int, int],
) -> List[Tensor]:
    """Mask-weighted foreground prototypes for each image.

    Each GT box supplies an instance support M_i; the learned soft mask A weights
    locations inside that support.  No pixel-level GT mask is used.
    """
    bsz, c, h, w = feature.shape
    im_h, im_w = image_hw
    sx, sy = w / float(im_w), h / float(im_h)
    per_image: List[List[Tensor]] = [[] for _ in range(bsz)]
    for box, bi in zip(boxes_xyxy, batch_indices.long()):
        x1, y1, x2, y2 = box.tolist()
        fx1, fy1 = max(0, int(x1 * sx)), max(0, int(y1 * sy))
        fx2, fy2 = min(w, max(fx1 + 1, int(torch.ceil(torch.tensor(x2 * sx)).item()))), min(h, max(fy1 + 1, int(torch.ceil(torch.tensor(y2 * sy)).item())))
        if fx1 >= w or fy1 >= h or fx2 <= 0 or fy2 <= 0:
            continue
        f = feature[int(bi), :, fy1:fy2, fx1:fx2]
        a = soft_mask[int(bi), :, fy1:fy2, fx1:fx2]
        denom = a.sum() + EPS
        p = (f * a).sum(dim=(1, 2)) / denom
        per_image[int(bi)].append(l2_normalize(p, dim=0))
    return [torch.stack(items, dim=0) if items else feature.new_empty((0, c)) for items in per_image]


def mifd_loss(prototypes_by_image: Sequence[Tensor], alpha: float = 0.30, tau_soft: float = 0.10) -> Tensor:
    """Multi-instance feature decorrelation, Eq. (11)."""
    terms = []
    device = None
    for p in prototypes_by_image:
        device = p.device
        n = p.shape[0]
        if n < 2:
            continue
        p = l2_normalize(p, dim=1)
        sim = p @ p.T
        off = ~torch.eye(n, dtype=torch.bool, device=p.device)
        vals = sim[off]
        terms.append(F.softplus((vals - alpha) / tau_soft).mean())
    if terms:
        return torch.stack(terms).mean()
    return torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)


def temporal_consistency_loss(
    prototypes_t: Tensor,
    ids_t: Tensor,
    prototypes_next: Tensor,
    ids_next: Tensor,
) -> Tensor:
    """TCR over identities valid in both adjacent frames; missing IDs are skipped."""
    if prototypes_t.numel() == 0 or prototypes_next.numel() == 0:
        return prototypes_t.new_tensor(0.0)
    map_next = {int(i): k for k, i in enumerate(ids_next.tolist())}
    diffs = []
    p0 = l2_normalize(prototypes_t, dim=1)
    p1 = l2_normalize(prototypes_next, dim=1)
    for k, identity in enumerate(ids_t.tolist()):
        j = map_next.get(int(identity))
        if j is not None:
            diffs.append((p0[k] - p1[j]).pow(2).sum())
    return torch.stack(diffs).mean() if diffs else prototypes_t.new_tensor(0.0)


def bns_hard_negative_loss(
    feature: Tensor,
    m_f: Tensor,
    foreground_locations: Tensor,
    ratio: float = 0.02,
    margin: float = 0.30,
) -> Tensor:
    """Background Negative Sample Hard Mining (BNS-HM), Eqs. (12)-(14)."""
    terms = []
    for b in range(feature.shape[0]):
        fg_mask = foreground_locations[b, 0].bool()
        bg_mask = ~fg_mask
        fg = m_f[b].permute(1, 2, 0)[fg_mask]
        bg = feature[b].permute(1, 2, 0)[bg_mask]
        if fg.numel() == 0 or bg.numel() == 0:
            continue
        v = l2_normalize(fg.mean(dim=0), dim=0)
        bg = l2_normalize(bg, dim=1)
        similarities = bg @ v
        k = max(1, int(torch.ceil(similarities.new_tensor(ratio * similarities.numel())).item()))
        hard = similarities.topk(min(k, similarities.numel()), largest=True).values
        terms.append(F.relu(hard - margin).mean())
    return torch.stack(terms).mean() if terms else feature.new_tensor(0.0)


class MSANWD(nn.Module):
    """Multi-scale attention normalized Wasserstein distance.

    The caller supplies one (mu_p, sigma_p, mu_t, sigma_t, feature_s) tuple per
    scale.  feature_s is pooled and mapped to one attention logit g(f^s).
    """
    def __init__(self, feature_dims: Sequence[int], hidden: int = 32, eta_nwd: float = 0.50):
        super().__init__()
        self.eta_nwd = eta_nwd
        self.maps = nn.ModuleList([
            nn.Sequential(nn.Linear(c, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
            for c in feature_dims
        ])

    def forward(self, scales: Sequence[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]) -> Tensor:
        if len(scales) != len(self.maps):
            raise ValueError("Number of scale tuples must match feature_dims.")
        logits, distances = [], []
        for mapper, (mu_p, sigma_p, mu_t, sigma_t, feat) in zip(self.maps, scales):
            pooled = feat.mean(dim=(-2, -1))
            logits.append(mapper(pooled).squeeze(-1))
            pos = (mu_p - mu_t).pow(2).sum(dim=-1)
            size = (sigma_p - sigma_t).pow(2).sum(dim=-1)
            distance = pos + self.eta_nwd * size
            # The leading dimension is the image batch. Average any remaining
            # positive-anchor dimensions, retaining one distance per image.
            while distance.ndim > 1:
                distance = distance.mean(dim=-1)
            if distance.ndim == 0:
                distance = distance.expand(feat.shape[0])
            if distance.shape[0] != feat.shape[0]:
                raise ValueError("Each MSA-NWD scale must provide one distance set per image.")
            distances.append(distance)
        # Eq. (6): attention is normalized across scales independently for
        # every image, not once for the entire batch.
        alpha = torch.softmax(torch.stack(logits, dim=1), dim=1)
        distance_matrix = torch.stack(distances, dim=1)
        return (alpha * distance_matrix).sum(dim=1).mean()


@dataclass
class AuxiliaryLossOutput:
    total: Tensor
    con: Tensor
    nwd: Tensor
    tcr: Tensor
    mifd: Tensor
    hard: Tensor
    q_sep: Tensor


class ISPSAuxiliaryObjective(nn.Module):
    """Weighted auxiliary part of the complete ISPS-SOD objective.

    L_dec stays in the detector's native criterion.  This module returns
    lambda_con*L_con + ... + lambda_hard*L_hard and diagnostics.
    """
    def __init__(self, channels: int, cfg: Optional[SemTraTrackConfig] = None):
        super().__init__()
        self.cfg = cfg or SemTraTrackConfig()
        self.sps = SPSBranch(channels)

    def forward(
        self,
        feature: Tensor,
        foreground_locations: Tensor,
        boxes_xyxy: Tensor,
        batch_indices: Tensor,
        image_hw: Tuple[int, int],
        nwd_loss: Optional[Tensor] = None,
        tcr_loss: Optional[Tensor] = None,
    ) -> AuxiliaryLossOutput:
        mask, m_f, m_b = self.sps(feature)
        l_con = foreground_background_contrastive_loss(m_f, m_b, foreground_locations, self.cfg.tau_con)
        protos = instance_prototypes(feature, mask, boxes_xyxy, batch_indices, image_hw)
        l_mifd = mifd_loss(protos, self.cfg.mifd_alpha, self.cfg.tau_soft).to(feature.device)
        l_hard = bns_hard_negative_loss(feature, m_f, foreground_locations, self.cfg.hard_negative_ratio, self.cfg.hard_negative_margin)
        l_nwd = nwd_loss if nwd_loss is not None else feature.new_tensor(0.0)
        l_tcr = tcr_loss if tcr_loss is not None else feature.new_tensor(0.0)
        total = (
            self.cfg.lambda_con * l_con
            + self.cfg.lambda_nwd * l_nwd
            + self.cfg.lambda_tcr * l_tcr
            + self.cfg.lambda_mifd * l_mifd
            + self.cfg.lambda_hard * l_hard
        )
        return AuxiliaryLossOutput(total, l_con, l_nwd, l_tcr, l_mifd, l_hard, q_sep(mask, foreground_locations))


def instance_prototypes_flat(
    feature: Tensor,
    soft_mask: Tensor,
    boxes_xyxy: Tensor,
    batch_indices: Tensor,
    image_hw: Tuple[int, int],
) -> Tensor:
    """Return one normalized mask-weighted prototype per input box, preserving order."""
    _, c, h, w = feature.shape
    im_h, im_w = image_hw
    sx, sy = w / float(im_w), h / float(im_h)
    out: List[Tensor] = []
    for box, bi in zip(boxes_xyxy, batch_indices.long()):
        x1, y1, x2, y2 = [float(v) for v in box]
        fx1 = max(0, min(w - 1, int(math.floor(x1 * sx))))
        fy1 = max(0, min(h - 1, int(math.floor(y1 * sy))))
        fx2 = max(fx1 + 1, min(w, int(math.ceil(x2 * sx))))
        fy2 = max(fy1 + 1, min(h, int(math.ceil(y2 * sy))))
        f = feature[int(bi), :, fy1:fy2, fx1:fx2]
        a = soft_mask[int(bi), :, fy1:fy2, fx1:fx2]
        p = (f * a).sum(dim=(1, 2)) / (a.sum() + EPS)
        out.append(l2_normalize(p, dim=0))
    return torch.stack(out, 0) if out else feature.new_empty((0, c))
