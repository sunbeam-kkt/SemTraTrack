from __future__ import annotations

from typing import Optional, Tuple
import cv2

from .config import SemTraTrackConfig
from .context import build_context_encoder
from .detector import UltralyticsDetector
from .lgatracker import LGATracker


class SemTraTrackPipeline:
    """Two-stage SemTraTrack inference pipeline."""
    def __init__(
        self,
        detector_weights: str,
        assoc_checkpoint: Optional[str],
        frame_size: Tuple[int, int],
        device: str = "cuda",
        detector_device: str = "0",
        context_variant: Optional[str] = None,
        cfg: Optional[SemTraTrackConfig] = None,
    ):
        selected_variant = context_variant or (cfg.context_variant if cfg is not None else "full")
        self.cfg = cfg or SemTraTrackConfig(context_variant=selected_variant)
        if cfg is not None and context_variant is not None and context_variant != cfg.context_variant:
            raise ValueError("context_variant must match cfg.context_variant when both are supplied.")
        self.detector = UltralyticsDetector(detector_weights, detector_device, self.cfg)
        if self.cfg.context_variant == "geometry-only":
            self.tracker = LGATracker(frame_size, None, None, self.cfg, device)
            return
        if assoc_checkpoint is None:
            raise ValueError("--assoc-checkpoint is required unless context_variant is geometry-only.")
        encoder_name = "numeric-mlp" if self.cfg.context_variant == "numeric" else "structured-mlp"
        encoder = build_context_encoder(encoder_name, device=device)
        self.tracker = LGATracker.from_checkpoint(frame_size, encoder, assoc_checkpoint, self.cfg, device)

    def process(self, frame):
        detections = self.detector.detect(frame)
        return self.tracker.update(detections)
