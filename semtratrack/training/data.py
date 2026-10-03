"""Identity-aware adjacent-frame dataset for ISPS-SOD training.

The current paper applies TCR only to valid adjacent observations of the same
UAV identity.  Ordinary YOLO image datasets discard identity and adjacency, so
this module supplies adjacent frame pairs while preserving the standard YOLO
batch fields (`img`, `cls`, `bboxes`, `batch_idx`) plus `track_ids` and
`visibility` for TCR.

Input annotations use MOTChallenge-like rows:
    frame,id,x,y,w,h,conf,class,visibility
Coordinates are pixel xywh with x/y denoting top-left, as in the uploaded
Track-3 annotation example.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MOTObject:
    track_id: int
    xyxy: Tuple[float, float, float, float]
    visibility: float = 1.0


def _read_mot(path: Path) -> Dict[int, List[MOTObject]]:
    frames: Dict[int, List[MOTObject]] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        p = [x.strip() for x in raw.split(",")]
        if len(p) < 6:
            continue
        frame, tid = int(float(p[0])), int(float(p[1]))
        x, y, w, h = map(float, p[2:6])
        if w <= 0 or h <= 0:
            continue
        vis = float(p[8]) if len(p) > 8 else 1.0
        frames.setdefault(frame, []).append(MOTObject(tid, (x, y, x + w, y + h), vis))
    return frames


def _find_frame(video_dir: Path, frame: int, patterns: Sequence[str]) -> Optional[Path]:
    for pat in patterns:
        p = video_dir / pat.format(frame=frame)
        if p.exists():
            return p
    # Fallback for datasets whose original names are not zero padded.
    candidates = []
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidates.extend(video_dir.glob(f"*{frame}*{ext}"))
    return sorted(candidates)[0] if len(candidates) == 1 else None


def build_pair_manifest(
    frames_root: str,
    annotations_root: str,
    output: str,
    patterns: Sequence[str] = ("{frame:06d}.jpg", "{frame:05d}.jpg", "{frame:04d}.jpg", "{frame}.jpg", "{frame:06d}.png", "{frame}.png"),
    require_shared_identity: bool = False,
) -> int:
    """Create JSONL manifest of consecutive frame pairs from MOT annotations.

    All consecutive annotated frames are retained by default. TCR itself
    selects only identities visible in both frames; a pair with no shared
    identity still provides valid detector, SPS, MIFD, and BNS-HM supervision.
    ``require_shared_identity`` is an optional diagnostic/data-filtering mode,
    not the manuscript training default.
    """
    frames_root_p, ann_root_p = Path(frames_root), Path(annotations_root)
    rows = []
    for ann in sorted(ann_root_p.glob("*.txt")):
        video = ann.stem
        vdir_candidates = [frames_root_p / video, frames_root_p]
        vdir = next((p for p in vdir_candidates if p.exists()), None)
        if vdir is None:
            continue
        by_frame = _read_mot(ann)
        frame_ids = sorted(by_frame)
        frame_set = set(frame_ids)
        for f0 in frame_ids:
            f1 = f0 + 1
            if f1 not in frame_set:
                continue
            im0, im1 = _find_frame(vdir, f0, patterns), _find_frame(vdir, f1, patterns)
            if im0 is None or im1 is None:
                continue
            ids0 = {o.track_id for o in by_frame[f0] if o.visibility > 0}
            ids1 = {o.track_id for o in by_frame[f1] if o.visibility > 0}
            if require_shared_identity and not (ids0 & ids1):
                continue
            def enc(items):
                return [{"id": o.track_id, "xyxy": list(o.xyxy), "visibility": o.visibility} for o in items]
            rows.append({
                "video": video, "frame0": f0, "frame1": f1,
                "image0": str(im0.resolve()), "image1": str(im1.resolve()),
                "objects0": enc(by_frame[f0]), "objects1": enc(by_frame[f1]),
            })
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def _letterbox(im: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2.0, (size - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, (left, top)


def _transform_boxes_letterbox(boxes: np.ndarray, ratio: float, pad: Tuple[float, float], size: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    b = boxes.astype(np.float32).copy()
    b[:, [0, 2]] = b[:, [0, 2]] * ratio + pad[0]
    b[:, [1, 3]] = b[:, [1, 3]] * ratio + pad[1]
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, size - 1)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, size - 1)
    return b


def _xyxy_to_norm_xywh(boxes: np.ndarray, size: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5 / size
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5 / size
    out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / size
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / size
    return out



def _affine_image_boxes(im: np.ndarray, boxes: np.ndarray, size: int, params: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply synchronized scale/translation augmentation in the 640 canvas."""
    scale, tx, ty = params
    c = size * 0.5
    m = np.array([[scale, 0.0, (1.0-scale)*c + tx*size],
                  [0.0, scale, (1.0-scale)*c + ty*size]], dtype=np.float32)
    out = cv2.warpAffine(im, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=(114,114,114))
    if boxes.size == 0:
        return out, boxes.reshape(0,4)
    # Transform all four corners, then enclose them with an axis-aligned box.
    n = len(boxes)
    corners = np.stack([
        boxes[:, [0,1]], boxes[:, [2,1]], boxes[:, [2,3]], boxes[:, [0,3]]
    ], axis=1)
    ones = np.ones((n,4,1), dtype=np.float32)
    pts = np.concatenate([corners, ones], axis=-1) @ m.T
    b = np.concatenate([pts.min(1), pts.max(1)], axis=1)
    b[:, [0,2]] = np.clip(b[:, [0,2]], 0, size-1)
    b[:, [1,3]] = np.clip(b[:, [1,3]], 0, size-1)
    return out, b.astype(np.float32)

def _hsv(im: np.ndarray, gains: Tuple[float, float, float]) -> np.ndarray:
    gh, gs, gv = gains
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] * gh) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * gs, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * gv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


class PairedMOTDataset(Dataset):
    """Adjacent-frame training pairs with synchronized geometric preprocessing.

    A dataset item contains two frames. `collate_fn` flattens N pair samples to
    2N images, so `image_batch=16` corresponds to eight adjacent-frame pairs.
    """
    def __init__(
        self,
        manifest: str,
        imgsz: int = 640,
        augment: bool = True,
        hsv_h: float = 0.015,
        hsv_s: float = 0.70,
        hsv_v: float = 0.40,
        translate: float = 0.10,
        scale: float = 0.50,
    ):
        self.manifest = str(manifest)
        self.imgsz = int(imgsz)
        self.augment = augment
        self.hsv_h, self.hsv_s, self.hsv_v = hsv_h, hsv_s, hsv_v
        self.translate, self.scale = float(translate), float(scale)
        self.rows = [json.loads(x) for x in Path(manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
        if not self.rows:
            raise ValueError(f"No adjacent-frame pairs found in {manifest}")
        # Kept for Ultralytics plotting/autobatch helpers.
        self.labels = []
        for row in self.rows:
            for key in ("objects0", "objects1"):
                objs = row[key]
                self.labels.append({
                    "cls": np.zeros((len(objs), 1), dtype=np.float32),
                    "bboxes": np.zeros((len(objs), 4), dtype=np.float32),
                })

    def __len__(self) -> int:
        return len(self.rows)

    def _frame(self, path: str, objects: Sequence[dict], hsv_gains=None, affine_params=None) -> dict:
        im = cv2.imread(path)
        if im is None:
            raise FileNotFoundError(path)
        boxes = np.asarray([o["xyxy"] for o in objects], dtype=np.float32).reshape(-1, 4)
        tids = np.asarray([o["id"] for o in objects], dtype=np.int64)
        vis = np.asarray([o.get("visibility", 1.0) for o in objects], dtype=np.float32)
        im, ratio, pad = _letterbox(im, self.imgsz)
        boxes = _transform_boxes_letterbox(boxes, ratio, pad, self.imgsz)
        if self.augment and affine_params is not None:
            im, boxes = _affine_image_boxes(im, boxes, self.imgsz, affine_params)
        if self.augment and hsv_gains is not None:
            im = _hsv(im, hsv_gains)
        bboxes = _xyxy_to_norm_xywh(boxes, self.imgsz)
        # Drop degenerate boxes while keeping identity arrays aligned.
        keep = (bboxes[:, 2] > 1e-6) & (bboxes[:, 3] > 1e-6) if len(bboxes) else np.zeros(0, dtype=bool)
        bboxes, tids, vis = bboxes[keep], tids[keep], vis[keep]
        img = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))
        n = len(bboxes)
        return {
            "img": img,
            "cls": torch.zeros((n, 1), dtype=torch.float32),
            "bboxes": torch.as_tensor(bboxes, dtype=torch.float32),
            "track_ids": torch.as_tensor(tids, dtype=torch.long),
            "visibility": torch.as_tensor(vis, dtype=torch.float32),
            "im_file": path,
        }

    def __getitem__(self, index: int) -> dict:
        r = self.rows[index]
        if self.augment:
            gains = (
                1.0 + random.uniform(-self.hsv_h, self.hsv_h),
                1.0 + random.uniform(-self.hsv_s, self.hsv_s),
                1.0 + random.uniform(-self.hsv_v, self.hsv_v),
            )
            affine = (
                1.0 + random.uniform(-self.scale, self.scale),
                random.uniform(-self.translate, self.translate),
                random.uniform(-self.translate, self.translate),
            )
        else:
            gains = None
            affine = None
        # The same augmentation draw is used for the adjacent pair to avoid
        # manufacturing artificial appearance/motion jumps in TCR supervision.
        a = self._frame(r["image0"], r["objects0"], gains, affine)
        b = self._frame(r["image1"], r["objects1"], gains, affine)
        a["pair_id"], b["pair_id"] = index, index
        a["pair_slot"], b["pair_slot"] = 0, 1
        a["frame_index"], b["frame_index"] = int(r["frame0"]), int(r["frame1"])
        return {"frames": (a, b)}

    @staticmethod
    def collate_fn(batch: Sequence[dict]) -> dict:
        frames = [f for sample in batch for f in sample["frames"]]
        imgs = torch.stack([f["img"] for f in frames], 0)
        cls, boxes, bidx, tids, vis = [], [], [], [], []
        for i, f in enumerate(frames):
            n = len(f["bboxes"])
            if n:
                cls.append(f["cls"]); boxes.append(f["bboxes"])
                bidx.append(torch.full((n,), i, dtype=torch.long))
                tids.append(f["track_ids"]); vis.append(f["visibility"])
        cat = lambda xs, shape, dtype: torch.cat(xs, 0) if xs else torch.empty(shape, dtype=dtype)
        return {
            "img": imgs,
            "cls": cat(cls, (0, 1), torch.float32),
            "bboxes": cat(boxes, (0, 4), torch.float32),
            "batch_idx": cat(bidx, (0,), torch.long),
            "track_ids": cat(tids, (0,), torch.long),
            "visibility": cat(vis, (0,), torch.float32),
            "pair_id_per_image": torch.tensor([f["pair_id"] for f in frames], dtype=torch.long),
            "pair_slot_per_image": torch.tensor([f["pair_slot"] for f in frames], dtype=torch.long),
            "frame_index_per_image": torch.tensor([f["frame_index"] for f in frames], dtype=torch.long),
            "im_file": [f["im_file"] for f in frames],
            "paper_pair_mode": True,
        }
