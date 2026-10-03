"""Utilities for composing the current paper's stage-I objective."""
from dataclasses import dataclass
import torch
from torch import Tensor
from .sps import AuxiliaryLossOutput


@dataclass
class FullDetectorLoss:
    total: Tensor
    native: Tensor
    auxiliary: Tensor
    diagnostics: dict


def compose_detector_objective(native_detection_loss: Tensor, aux: AuxiliaryLossOutput) -> FullDetectorLoss:
    """L_ISPS-SOD = L_dec + the five weighted auxiliary terms."""
    total = native_detection_loss + aux.total
    return FullDetectorLoss(
        total=total,
        native=native_detection_loss,
        auxiliary=aux.total,
        diagnostics={
            "L_con": aux.con.detach(),
            "L_nwd": aux.nwd.detach(),
            "L_tcr": aux.tcr.detach(),
            "L_mifd": aux.mifd.detach(),
            "L_hard": aux.hard.detach(),
            "q_sep": aux.q_sep.detach(),
        },
    )
