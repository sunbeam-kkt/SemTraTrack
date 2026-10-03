from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """One detector output in original-frame pixel coordinates."""
    xyxy: np.ndarray
    confidence: float
    cls: int = 0
    state: Optional[int] = None

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = map(float, self.xyxy)
        return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    @property
    def width(self) -> float:
        return max(0.0, float(self.xyxy[2] - self.xyxy[0]))

    @property
    def height(self) -> float:
        return max(0.0, float(self.xyxy[3] - self.xyxy[1]))

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class TrackObservation:
    frame_index: int
    track_id: int
    center: np.ndarray
    xyxy: np.ndarray
    confidence: float
    state: int


@dataclass
class Track:
    track_id: int
    xyxy: np.ndarray
    confidence: float
    state: int
    lost: int = 0
    history: List[TrackObservation] = field(default_factory=list)

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = map(float, self.xyxy)
        return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    @property
    def area(self) -> float:
        return max(0.0, float(self.xyxy[2] - self.xyxy[0])) * max(0.0, float(self.xyxy[3] - self.xyxy[1]))

    def recent_centers(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.history:
            return None, None
        p1 = self.history[-1].center
        p2 = self.history[-2].center if len(self.history) >= 2 else None
        return p1, p2
