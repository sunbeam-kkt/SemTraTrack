"""Paper-faithful LGATracker implementation.

Implements Algorithm 1 in the current manuscript:
  1) per-frame pairwise cues,
  2) causal past-window context,
  3) context-aware score fusion,
  4) Hungarian assignment,
  5) online track update,
  6) completed-window prompt/context update.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math

import numpy as np
import torch
from torch import Tensor, nn
from scipy.optimize import linear_sum_assignment

from .config import SemTraTrackConfig
from .types import Detection, Track, TrackObservation

EPS = 1e-8


def coarse_state(area: float, cfg: SemTraTrackConfig) -> int:
    if area < cfg.small_area_threshold:
        return 0  # small
    if area > cfg.large_area_threshold:
        return 1  # large
    return 2  # medium


@dataclass
class WindowSummary:
    quantity: int
    state_hist: Tuple[int, int, int]  # small, large, medium counts over observations
    mean_x: float
    mean_y: float
    motion: str
    mean_confidence: float
    delta_x: float
    delta_y: float

    @classmethod
    def empty(cls) -> "WindowSummary":
        return cls(0, (0, 0, 0), 0.0, 0.0, "stationary", 0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "quantity": int(self.quantity),
            "state_hist": [int(v) for v in self.state_hist],
            "mean_x": float(self.mean_x),
            "mean_y": float(self.mean_y),
            "motion": self.motion,
            "mean_confidence": float(self.mean_confidence),
            "delta_x": float(self.delta_x),
            "delta_y": float(self.delta_y),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "WindowSummary":
        return cls(
            int(value["quantity"]),
            tuple(int(v) for v in value["state_hist"]),
            float(value["mean_x"]),
            float(value["mean_y"]),
            str(value["motion"]),
            float(value["mean_confidence"]),
            float(value.get("delta_x", 0.0)),
            float(value.get("delta_y", 0.0)),
        )

    def prompt(self, disabled_attributes: Sequence[str] = ()) -> str:
        disabled = set(disabled_attributes)
        small, large, medium = self.state_hist
        fields = []
        if "qs" not in disabled:
            fields.append(f"During this window, {self.quantity} UAV identities were observed.")
        if "ss" not in disabled:
            fields.append(f"States: medium-{medium}, small-{small}, large-{large}.")
        if "ap" not in disabled:
            fields.append(f"Mean normalized position: x={self.mean_x:.3f}, y={self.mean_y:.3f}.")
        if "md" not in disabled:
            fields.append(f"Dominant motion: {self.motion}.")
        if "ac" not in disabled:
            fields.append(f"Mean confidence: {self.mean_confidence:.2f}.")
        return " ".join(fields)

    def numeric_vector(self) -> np.ndarray:
        # A compact deterministic numeric baseline representation. Quantity is
        # log-compressed to avoid dominating the bounded statistics.
        motion_onehot = {
            "stationary": (1.0, 0.0, 0.0),
            "horizontal": (0.0, 1.0, 0.0),
            "vertical": (0.0, 0.0, 1.0),
        }[self.motion]
        total_states = max(1, sum(self.state_hist))
        state_frac = [v / total_states for v in self.state_hist]
        return np.asarray([
            math.log1p(self.quantity) / math.log(51.0),
            *state_frac,
            self.mean_x,
            self.mean_y,
            *motion_onehot,
            self.mean_confidence,
        ], dtype=np.float32)


def controlled_prompt(
    summary: WindowSummary,
    completed: Sequence[WindowSummary],
    variant: str,
    rng: np.random.Generator,
    disabled_attributes: Sequence[str] = (),
) -> str:
    """Apply one of the manuscript's controlled semantic-content variants.

    Controls draw only from windows completed before ``summary``. The first
    window therefore uses the neutral constant prompt when no causal pool is
    available. This rule avoids look-ahead information in every variant.
    """
    if variant in {"full", "numeric"}:
        return summary.prompt(disabled_attributes)
    if variant == "constant":
        return WindowSummary.empty().prompt(disabled_attributes)
    if not completed:
        return WindowSummary.empty().prompt(disabled_attributes)
    if variant == "shuffled-window":
        return completed[int(rng.integers(0, len(completed)))].prompt(disabled_attributes)
    if variant == "attribute-permuted":
        indices = rng.choice(len(completed), size=5, replace=len(completed) < 5)
        qs, ss, ap, md, ac = (completed[int(index)] for index in indices)
        mixed = WindowSummary(
            qs.quantity,
            ss.state_hist,
            ap.mean_x,
            ap.mean_y,
            md.motion,
            ac.mean_confidence,
            md.delta_x,
            md.delta_y,
        )
        return mixed.prompt(disabled_attributes)
    raise ValueError(f"Unknown context variant: {variant}")


class PairContextAdapter(nn.Module):
    """Learned W_g and W_e with the paper's cosine compatibility score."""
    def __init__(self, context_dim: int = 128, shared_dim: int = 128):
        super().__init__()
        self.w_g = nn.Linear(4, shared_dim, bias=False)
        self.w_e = nn.Linear(context_dim, shared_dim, bias=False)

    def compatibility(self, pair: Tensor, context: Tensor) -> Tensor:
        zg = self.w_g(pair)
        ze = self.w_e(context)
        zg = zg / (zg.norm(dim=-1, keepdim=True) + EPS)
        ze = ze / (ze.norm(dim=-1, keepdim=True) + EPS)
        return (1.0 + (zg * ze).sum(dim=-1)) * 0.5

    def association_loss(self, pair: Tensor, context: Tensor, labels: Tensor) -> Tensor:
        score = self.compatibility(pair, context).clamp(1e-6, 1.0 - 1e-6)
        return nn.functional.binary_cross_entropy(score, labels.float())


class LGATracker:
    def __init__(
        self,
        frame_size: Tuple[int, int],
        context_encoder: Optional[nn.Module],
        adapter: Optional[PairContextAdapter],
        cfg: Optional[SemTraTrackConfig] = None,
        device: str = "cuda",
    ):
        self.cfg = cfg or SemTraTrackConfig()
        self.cfg.validate()
        self.width, self.height = frame_size
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"frame_size must be positive, got {frame_size!r}.")
        self.diagonal = math.hypot(self.width, self.height)
        self.context_encoder = context_encoder.to(device).eval() if isinstance(context_encoder, nn.Module) else context_encoder
        self.adapter = adapter.to(device).eval() if adapter is not None else None
        self.device = device
        if self.cfg.context_variant != "geometry-only" and (context_encoder is None or adapter is None):
            raise ValueError("Context-enabled LGATracker requires both a context encoder and PairContextAdapter.")

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_index = 0

        self.window_observations: List[TrackObservation] = []
        self.context_history: deque[Tensor] = deque(maxlen=self.cfg.context_history)
        self.causal_context: Optional[Tensor] = None
        self.completed_summaries: List[WindowSummary] = []
        self.control_rng = np.random.default_rng(self.cfg.context_seed)

    @classmethod
    def from_checkpoint(
        cls,
        frame_size: Tuple[int, int],
        context_encoder: Optional[nn.Module],
        checkpoint: Optional[str],
        cfg: Optional[SemTraTrackConfig] = None,
        device: str = "cuda",
    ) -> "LGATracker":
        cfg = cfg or SemTraTrackConfig()
        if cfg.context_variant == "geometry-only":
            return cls(frame_size, None, None, cfg, device)
        if context_encoder is None or checkpoint is None:
            raise ValueError("A mapper and association checkpoint are required for context-enabled variants.")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        encoder_dim = int(getattr(context_encoder, "output_dim", cfg.context_dim))
        checkpoint_dim = int(state.get("context_dim", encoder_dim)) if isinstance(state, dict) else encoder_dim
        if checkpoint_dim != encoder_dim:
            raise ValueError(f"Association checkpoint expects context_dim={checkpoint_dim}, but encoder outputs {encoder_dim}.")
        if isinstance(state, dict) and state.get("variant") not in {None, cfg.context_variant}:
            raise ValueError(
                f"Association checkpoint was trained for variant={state['variant']!r}, "
                f"but runtime requested {cfg.context_variant!r}."
            )
        if isinstance(state, dict):
            runtime_encoder_name = (
                "numeric-mlp" if cfg.context_variant == "numeric" else "structured-mlp"
            )
            checkpoint_encoder_name = state.get("encoder_name")
            if checkpoint_encoder_name not in {None, runtime_encoder_name}:
                raise ValueError(
                    f"Checkpoint encoder_name={checkpoint_encoder_name!r}, but runtime "
                    f"requires {runtime_encoder_name!r}."
                )
            checkpoint_seed = int(state.get("context_seed", cfg.context_seed))
            stochastic_controls = {"shuffled-window", "attribute-permuted"}
            if cfg.context_variant in stochastic_controls and checkpoint_seed != cfg.context_seed:
                raise ValueError(
                    f"Checkpoint context_seed={checkpoint_seed}, but runtime requested "
                    f"{cfg.context_seed}. Prompt-control evaluation must use its training seed."
                )
            checkpoint_signature = state.get("association_signature")
            if checkpoint_signature is not None:
                runtime_signature = cfg.association_signature()
                mismatches = {
                    key: (checkpoint_signature.get(key), runtime_signature.get(key))
                    for key in runtime_signature
                    if checkpoint_signature.get(key) != runtime_signature.get(key)
                }
                if mismatches:
                    details = ", ".join(
                        f"{key}: checkpoint={old!r}, runtime={new!r}"
                        for key, (old, new) in sorted(mismatches.items())
                    )
                    raise ValueError(
                        "Association checkpoint/runtime protocol mismatch (" + details + ")."
                    )
            checkpoint_cues = tuple(state.get("disabled_cues", ()))
            checkpoint_attributes = tuple(state.get("disabled_attributes", ()))
            if checkpoint_cues != tuple(cfg.disabled_cues):
                raise ValueError(
                    f"Checkpoint disabled_cues={checkpoint_cues!r}, but runtime requested {cfg.disabled_cues!r}."
                )
            if checkpoint_attributes != tuple(cfg.disabled_attributes):
                raise ValueError(
                    "Checkpoint disabled_attributes="
                    f"{checkpoint_attributes!r}, but runtime requested {cfg.disabled_attributes!r}."
                )
        adapter = PairContextAdapter(encoder_dim, cfg.shared_dim)
        adapter.load_state_dict(state.get("adapter", state))
        if isinstance(state, dict) and state.get("context_encoder") is not None and isinstance(context_encoder, nn.Module):
            context_encoder.load_state_dict(state["context_encoder"], strict=True)
            context_encoder.to(device).eval()
        return cls(frame_size, context_encoder, adapter, cfg, device)

    def reset(self, frame_size: Optional[Tuple[int, int]] = None):
        if frame_size is not None:
            self.width, self.height = frame_size
            if self.width <= 0 or self.height <= 0:
                raise ValueError(f"frame_size must be positive, got {frame_size!r}.")
            self.diagonal = math.hypot(self.width, self.height)
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_index = 0
        self.window_observations.clear()
        self.context_history.clear()
        self.causal_context = None
        self.completed_summaries.clear()
        self.control_rng = np.random.default_rng(self.cfg.context_seed)

    def _prepare_detection(self, d: Detection) -> Detection:
        d.state = coarse_state(d.area, self.cfg)
        return d

    def _predicted_center(self, track: Track) -> np.ndarray:
        if len(track.history) >= 2:
            p_prev = track.history[-1].center
            p_prev2 = track.history[-2].center
            return p_prev + (p_prev - p_prev2)
        if track.history:
            return track.history[-1].center.copy()
        return track.center.copy()

    def pairwise_cues(self, det: Detection, track: Track) -> np.ndarray:
        p_i = det.center
        p_prev = track.history[-1].center if track.history else track.center
        p_hat = self._predicted_center(track)

        distance = float(np.linalg.norm(p_i - p_hat))
        s_pos = 1.0 / (1.0 + distance / max(self.diagonal, EPS))

        a_i, a_j = det.area, track.area
        s_size = min(a_i, a_j) / (max(a_i, a_j) + EPS) if max(a_i, a_j) > 0 else 0.0

        s_state = 1.0 if det.state == track.state else self.cfg.state_mismatch_score

        if len(track.history) >= 2:
            velocity = track.history[-1].center - track.history[-2].center
            candidate_disp = p_i - p_prev
            nv, nu = float(np.linalg.norm(velocity)), float(np.linalg.norm(candidate_disp))
            if nv > 1e-6 and nu > 1e-6:
                cos = float(np.dot(candidate_disp, velocity) / (nu * nv + EPS))
                cos = float(np.clip(cos, -1.0, 1.0))
                s_dir = 0.5 * (1.0 + cos)
            else:
                s_dir = self.cfg.neutral_direction_score
        else:
            s_dir = self.cfg.neutral_direction_score

        cues = np.asarray([s_pos, s_size, s_state, s_dir], dtype=np.float32)
        cue_index = {"pos": 0, "size": 1, "state": 2, "dir": 3}
        for cue in self.cfg.disabled_cues:
            cues[cue_index[cue]] = 0.0
        return cues

    def _geo_score(self, cues: np.ndarray) -> float:
        return float(np.dot(np.asarray(self.cfg.cue_weights, dtype=np.float32), cues))

    @torch.no_grad()
    def _context_score(self, cues: np.ndarray) -> float:
        if self.causal_context is None or self.adapter is None:
            raise RuntimeError("Context score requested before any completed window exists.")
        pair = torch.as_tensor(cues, dtype=torch.float32, device=self.device).unsqueeze(0)
        context = self.causal_context.to(self.device).unsqueeze(0)
        return float(self.adapter.compatibility(pair, context).item())

    def _score_matrix(self, detections: Sequence[Detection], tracks: Sequence[Track]) -> np.ndarray:
        scores = np.zeros((len(detections), len(tracks)), dtype=np.float32)
        has_context = self.causal_context is not None
        for i, d in enumerate(detections):
            for j, tr in enumerate(tracks):
                cues = self.pairwise_cues(d, tr)
                geo = self._geo_score(cues)
                if has_context:
                    ctx = self._context_score(cues)
                    scores[i, j] = (1.0 - self.cfg.beta) * geo + self.cfg.beta * ctx
                else:
                    scores[i, j] = geo
        return scores

    def _new_track(self, det: Detection) -> Track:
        tid = self.next_track_id
        self.next_track_id += 1
        tr = Track(tid, det.xyxy.copy(), float(det.confidence), int(det.state), lost=0)
        obs = TrackObservation(self.frame_index, tid, det.center.copy(), det.xyxy.copy(), float(det.confidence), int(det.state))
        tr.history.append(obs)
        self.tracks[tid] = tr
        self.window_observations.append(obs)
        return tr

    def _update_track(self, tr: Track, det: Detection):
        tr.xyxy = det.xyxy.copy()
        tr.confidence = float(det.confidence)
        tr.state = int(det.state)
        tr.lost = 0
        obs = TrackObservation(self.frame_index, tr.track_id, det.center.copy(), det.xyxy.copy(), float(det.confidence), int(det.state))
        tr.history.append(obs)
        self.window_observations.append(obs)

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        """Process one frame exactly in Algorithm 1 order."""
        self.frame_index += 1
        dets = [self._prepare_detection(d) for d in detections]

        if not self.tracks:
            for d in dets:
                self._new_track(d)
        else:
            active = list(self.tracks.values())
            matched_d, matched_t = set(), set()
            if dets and active:
                score = self._score_matrix(dets, active)
                rows, cols = linear_sum_assignment(1.0 - score)
                for i, j in zip(rows.tolist(), cols.tolist()):
                    if score[i, j] < self.cfg.min_match_score:
                        continue
                    self._update_track(active[j], dets[i])
                    matched_d.add(i)
                    matched_t.add(j)

            for i, d in enumerate(dets):
                if i not in matched_d:
                    self._new_track(d)

            remove = []
            for j, tr in enumerate(active):
                if j not in matched_t:
                    tr.lost += 1
                    if tr.lost > self.cfg.max_lost:
                        remove.append(tr.track_id)
            for tid in remove:
                self.tracks.pop(tid, None)

        # Causal update: only after current-frame association and track update.
        if self.frame_index % self.cfg.window_size == 0:
            self._complete_window()

        return list(self.tracks.values())

    def _complete_window(self):
        summary = summarize_window(
            self.window_observations, self.width, self.height, self.cfg.stationary_pixels
        )
        if self.cfg.context_variant == "geometry-only":
            self.window_observations = []
            return
        prompt = controlled_prompt(
            summary,
            self.completed_summaries,
            self.cfg.context_variant,
            self.control_rng,
            self.cfg.disabled_attributes,
        )
        numeric = summary.numeric_vector()
        if self.context_encoder is None:  # guarded by __init__; keeps type-checkers honest
            raise RuntimeError("Missing context encoder.")
        was_training = getattr(self.context_encoder, "training", False)
        if isinstance(self.context_encoder, nn.Module):
            self.context_encoder.eval()
        with torch.no_grad():
            embedding = self.context_encoder.encode(prompt, numeric=numeric).detach().float().to(self.device)
        if isinstance(self.context_encoder, nn.Module) and was_training:
            self.context_encoder.train()
        self.context_history.append(embedding)
        self.causal_context = torch.stack(list(self.context_history), dim=0).mean(dim=0)
        self.completed_summaries.append(summary)
        self.window_observations = []

    def export_rows(self) -> List[Tuple[int, int, float, float, float, float, float]]:
        """Return all stored observations as MOT rows (frame,id,x,y,w,h,conf)."""
        rows = []
        seen = set()
        for tr in self.tracks.values():
            for o in tr.history:
                key = (o.frame_index, o.track_id)
                if key in seen:
                    continue
                seen.add(key)
                x1, y1, x2, y2 = map(float, o.xyxy)
                rows.append((o.frame_index, o.track_id, x1, y1, x2 - x1, y2 - y1, o.confidence))
        return sorted(rows)


def summarize_window(
    observations: Sequence[TrackObservation],
    width: int,
    height: int,
    stationary_pixels: float = 50.0,
) -> WindowSummary:
    if not observations:
        return WindowSummary.empty()

    quantity = len({o.track_id for o in observations})
    state_hist = [0, 0, 0]
    for o in observations:
        state_hist[o.state] += 1

    mean_x = float(np.mean([o.center[0] / width for o in observations]))
    mean_y = float(np.mean([o.center[1] / height for o in observations]))
    mean_conf = float(np.mean([o.confidence for o in observations]))

    by_track: Dict[int, List[TrackObservation]] = defaultdict(list)
    for o in observations:
        by_track[o.track_id].append(o)
    valid = [sorted(v, key=lambda x: x.frame_index) for v in by_track.values() if len(v) >= 2]

    diagonal = math.hypot(width, height)
    if valid:
        dx = float(np.mean([abs(v[-1].center[0] - v[0].center[0]) / diagonal for v in valid]))
        dy = float(np.mean([abs(v[-1].center[1] - v[0].center[1]) / diagonal for v in valid]))
        threshold = stationary_pixels / diagonal
        if max(dx, dy) < threshold:
            motion = "stationary"
        else:
            motion = "horizontal" if dx >= dy else "vertical"
    else:
        dx = dy = 0.0
        motion = "stationary"

    return WindowSummary(quantity, tuple(state_hist), mean_x, mean_y, motion, mean_conf, dx, dy)
