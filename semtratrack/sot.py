"""Single-object tracking protocol described in the current manuscript.

Only the first-frame GT box is used. Later frames use the previous target
estimate to define a search region, generate class-agnostic detector candidates,
and choose the candidate with the highest SemTraTrack fused association score.
No later-frame GT correction or re-initialization is permitted.

The default search crop follows the published AQATrack-256 convention used by
the AQATrack-LINR comparison: a square crop whose side length is
``ceil(4 * sqrt(target_width * target_height))``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple
import math
import numpy as np

from .config import SemTraTrackConfig
from .types import Detection, Track, TrackObservation
from .lgatracker import LGATracker, coarse_state


DEFAULT_SEARCH_FACTOR = 4.0


def search_crop_from_box(
    frame: np.ndarray,
    xyxy: np.ndarray,
    factor: float = DEFAULT_SEARCH_FACTOR,
):
    """Return the square, zero-padded SOT search crop and its image origin.

    ``factor`` follows the standard area convention used by AQATrack: the crop
    side is ``ceil(factor * sqrt(box_width * box_height))``. Consequently, the
    default factor 4.0 produces a search area 16 times the target-box area.
    The returned origin may be negative when the crop crosses an image border.
    """
    if not math.isfinite(factor) or factor <= 1.0:
        raise ValueError("search_factor must be finite and greater than 1.0.")
    if frame.ndim not in (2, 3) or frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError("frame must be a non-empty grayscale or color image.")

    image_height, image_width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)
    if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("xyxy must contain one finite, positive-area box.")

    box_width, box_height = x2 - x1, y2 - y1
    crop_size = max(1, int(math.ceil(math.sqrt(box_width * box_height) * factor)))
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    crop_x1 = int(round(center_x - crop_size / 2.0))
    crop_y1 = int(round(center_y - crop_size / 2.0))
    crop_x2, crop_y2 = crop_x1 + crop_size, crop_y1 + crop_size

    # Allocate first, then copy the valid image intersection. This preserves a
    # square crop and constant-zero padding even when the box touches a border.
    crop_shape = (crop_size, crop_size) + frame.shape[2:]
    crop = np.zeros(crop_shape, dtype=frame.dtype)
    source_x1, source_y1 = max(0, crop_x1), max(0, crop_y1)
    source_x2, source_y2 = min(image_width, crop_x2), min(image_height, crop_y2)
    if source_x2 > source_x1 and source_y2 > source_y1:
        dest_x1, dest_y1 = source_x1 - crop_x1, source_y1 - crop_y1
        dest_x2 = dest_x1 + source_x2 - source_x1
        dest_y2 = dest_y1 + source_y2 - source_y1
        crop[dest_y1:dest_y2, dest_x1:dest_x2] = frame[
            source_y1:source_y2, source_x1:source_x2
        ]
    return crop, (crop_x1, crop_y1)


def offset_detections(
    detections: Sequence[Detection],
    offset: Tuple[int, int],
    frame_size: Optional[Tuple[int, int]] = None,
):
    """Map crop-local detections back to the image and optionally clip them."""
    ox, oy = offset
    out = []
    for d in detections:
        b = d.xyxy.copy()
        b[[0, 2]] += ox
        b[[1, 3]] += oy
        if frame_size is not None:
            width, height = frame_size
            b[[0, 2]] = np.clip(b[[0, 2]], 0, width)
            b[[1, 3]] = np.clip(b[[1, 3]], 0, height)
            if b[2] <= b[0] or b[3] <= b[1]:
                continue
        out.append(Detection(b, d.confidence, d.cls, d.state))
    return out


class SOTTracker(LGATracker):
    """One-active-track specialization of LGATracker."""
    def initialize(self, gt_xyxy: np.ndarray, confidence: float = 1.0):
        if self.frame_index != 0:
            raise RuntimeError("SOT initialization is allowed only in the first frame.")
        self.frame_index = 1
        d=Detection(np.asarray(gt_xyxy,dtype=np.float32),confidence,0)
        d.state=coarse_state(d.area,self.cfg)
        self._new_track(d)
        if self.frame_index % self.cfg.window_size == 0:
            self._complete_window()
        return next(iter(self.tracks.values()))

    def update_single(self, candidates: Sequence[Detection]) -> Optional[Track]:
        if not self.tracks:
            self.frame_index += 1
            if self.frame_index % self.cfg.window_size == 0:
                self._complete_window()
            return None
        self.frame_index += 1
        tr=next(iter(self.tracks.values()))
        dets=[self._prepare_detection(d) for d in candidates]
        if dets:
            scores=self._score_matrix(dets,[tr])[:,0]
            i=int(np.argmax(scores))
            if float(scores[i]) >= self.cfg.min_match_score:
                self._update_track(tr,dets[i])
            else:
                tr.lost += 1
        else:
            tr.lost += 1
        if tr.lost > self.cfg.max_lost:
            self.tracks.clear()
            tr=None
        if self.frame_index % self.cfg.window_size == 0:
            self._complete_window()
        return tr
