"""Training integration for the paper-aligned ISPS-SOD detector."""
from .data import PairedMOTDataset, build_pair_manifest

__all__ = ["PairedMOTDataset", "build_pair_manifest"]
