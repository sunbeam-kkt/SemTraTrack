"""Paper-aligned configuration for SemTraTrack.

The defaults mirror the current manuscript. Keeping them in one dataclass makes
paper/code drift easy to audit.
"""
from dataclasses import dataclass, asdict
from typing import Any, Dict, FrozenSet, Tuple


@dataclass
class SemTraTrackConfig:
    # Detector / efficiency protocol
    imgsz: int = 640
    conf_threshold: float = 0.25
    amp: bool = True
    detector_ablation: str = "full"

    # SPS / detector objective
    lambda_con: float = 0.10
    lambda_nwd: float = 0.50
    lambda_tcr: float = 0.10
    lambda_mifd: float = 0.05
    lambda_hard: float = 0.10
    tau_con: float = 0.10
    mifd_alpha: float = 0.30
    tau_soft: float = 0.10
    hard_negative_ratio: float = 0.02
    hard_negative_margin: float = 0.30
    eta_nwd: float = 0.50

    # LGATracker
    window_size: int = 30
    context_history: int = 10
    max_lost: int = 5
    # Structured prompt mapper: E_tok (d_t=64) -> MLP hidden=128 -> d_e=128.
    token_embedding_dim: int = 64
    language_hidden_dim: int = 128
    context_dim: int = 128
    shared_dim: int = 128
    beta: float = 0.20
    cue_weights: Tuple[float, float, float, float] = (0.50, 0.20, 0.10, 0.20)
    context_variant: str = "full"
    context_seed: int = 0
    disabled_cues: Tuple[str, ...] = ()
    disabled_attributes: Tuple[str, ...] = ()

    # Coarse scale states from the manuscript
    small_area_threshold: float = 16.0 ** 2
    large_area_threshold: float = 96.0 ** 2
    state_mismatch_score: float = 0.05
    neutral_direction_score: float = 0.50
    stationary_pixels: float = 50.0

    # The paper does not introduce a separate matching threshold.  Setting 0.0
    # accepts all Hungarian pairs with a valid [0,1] score.  A non-zero value is
    # exposed only for diagnostic/practical use and should be reported if used.
    min_match_score: float = 0.0

    def validate(self) -> None:
        if self.imgsz not in {640, 1280}:
            raise ValueError("The manuscript reports only 640x640 and the explicit 1280x1280 resolution control.")
        if abs(sum(self.cue_weights) - 1.0) > 1e-6:
            raise ValueError("Association cue weights must sum to one.")
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError("beta must be in [0,1].")
        if not (0.0 <= self.conf_threshold <= 1.0):
            raise ValueError("conf_threshold must be in [0,1].")
        if not (0.0 <= self.min_match_score <= 1.0):
            raise ValueError("min_match_score must be in [0,1].")
        if not (0.0 < self.hard_negative_ratio <= 1.0):
            raise ValueError("hard_negative_ratio must be in (0,1].")
        if min(
            self.lambda_con,
            self.lambda_nwd,
            self.lambda_tcr,
            self.lambda_mifd,
            self.lambda_hard,
            self.tau_con,
            self.tau_soft,
            self.eta_nwd,
        ) < 0.0:
            raise ValueError("Loss weights, temperatures, and eta_nwd must be non-negative.")
        if self.tau_con == 0.0 or self.tau_soft == 0.0:
            raise ValueError("tau_con and tau_soft must be positive.")
        if self.window_size <= 0 or self.context_history <= 0 or self.max_lost < 0:
            raise ValueError("Invalid temporal configuration.")
        if self.small_area_threshold < 0 or self.large_area_threshold <= self.small_area_threshold:
            raise ValueError("Scale-state thresholds must satisfy 0 <= small < large.")
        if self.stationary_pixels < 0:
            raise ValueError("stationary_pixels must be non-negative.")
        if (self.token_embedding_dim, self.language_hidden_dim, self.context_dim, self.shared_dim) != (64, 128, 128, 128):
            raise ValueError("The manuscript uses d_t=64, hidden=128, d_e=128, and d_s=128.")
        valid_variants = {
            "full", "constant", "shuffled-window", "attribute-permuted",
            "numeric", "geometry-only",
        }
        if self.context_variant not in valid_variants:
            raise ValueError(f"Unknown context variant {self.context_variant!r}; expected one of {sorted(valid_variants)}.")
        valid_detector_ablations = {
            "full", "base", "w/o-tcr", "w/o-mifd", "w/o-stcr",
            "w/o-bns-hm", "w/o-msa-nwd", "w/o-sps",
        }
        if self.detector_ablation not in valid_detector_ablations:
            raise ValueError(
                f"Unknown detector ablation {self.detector_ablation!r}; "
                f"expected one of {sorted(valid_detector_ablations)}."
            )
        if not set(self.disabled_cues) <= {"pos", "size", "state", "dir"}:
            raise ValueError("disabled_cues accepts only pos, size, state, and dir.")
        if not set(self.disabled_attributes) <= {"qs", "ss", "ap", "md", "ac"}:
            raise ValueError("disabled_attributes accepts only qs, ss, ap, md, and ac.")
        if self.context_variant != "full" and (self.disabled_cues or self.disabled_attributes):
            raise ValueError("Cue/attribute ablations are defined only for the full structured-prompt variant.")

    def enabled_detector_components(self) -> FrozenSet[str]:
        """Return the exact switch row used by the manuscript's detector table."""
        rows = {
            "full": {"sps", "tcr", "mifd", "bns-hm", "msa-nwd"},
            "base": set(),
            "w/o-tcr": {"sps", "mifd", "bns-hm", "msa-nwd"},
            "w/o-mifd": {"sps", "tcr", "bns-hm", "msa-nwd"},
            "w/o-stcr": {"sps", "bns-hm", "msa-nwd"},
            "w/o-bns-hm": {"sps", "tcr", "mifd", "msa-nwd"},
            "w/o-msa-nwd": {"sps", "tcr", "mifd", "bns-hm"},
            # Dependency-aware row: TCR, MIFD, and BNS-HM require SPS.
            "w/o-sps": {"msa-nwd"},
        }
        return frozenset(rows[self.detector_ablation])

    def to_dict(self):
        return asdict(self)

    def association_signature(self) -> Dict[str, Any]:
        """Configuration values that define Stage-II supervision/inference.

        These values are persisted in the association corpus and checkpoint so
        a run cannot silently combine a mapper trained with one temporal or cue
        protocol and inference performed with another.
        """
        return {
            "window_size": int(self.window_size),
            "context_history": int(self.context_history),
            "max_lost": int(self.max_lost),
            "token_embedding_dim": int(self.token_embedding_dim),
            "language_hidden_dim": int(self.language_hidden_dim),
            "context_dim": int(self.context_dim),
            "shared_dim": int(self.shared_dim),
            "beta": float(self.beta),
            "cue_weights": [float(value) for value in self.cue_weights],
            "small_area_threshold": float(self.small_area_threshold),
            "large_area_threshold": float(self.large_area_threshold),
            "state_mismatch_score": float(self.state_mismatch_score),
            "neutral_direction_score": float(self.neutral_direction_score),
            "stationary_pixels": float(self.stationary_pixels),
        }
