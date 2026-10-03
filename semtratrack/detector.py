"""Detector wrapper for the 640x640 AMP/autocast inference protocol."""
from __future__ import annotations

from typing import List, Optional
import contextlib
import numpy as np
import torch

from .config import SemTraTrackConfig
from .types import Detection


class UltralyticsDetector:
    def __init__(self, weights: str, device: str = "0", cfg: Optional[SemTraTrackConfig] = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Install a YOLOv13-compatible Ultralytics package before running the detector.") from exc
        self.cfg = cfg or SemTraTrackConfig()
        self.device_arg = device
        self.cuda = torch.cuda.is_available() and str(device).lower() not in {"cpu", "-1"}
        self.model = YOLO(weights)

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> List[Detection]:
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cfg.amp and self.cuda)
        with autocast:
            result = self.model.predict(
                source=frame,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.conf_threshold,
                device=self.device_arg,
                verbose=False,
                save=False,
            )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        conf = result.boxes.conf.detach().cpu().numpy()
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)
        return [Detection(b.astype(np.float32), float(c), int(k)) for b, c, k in zip(boxes, conf, cls)]
