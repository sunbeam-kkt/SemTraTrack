"""Task-specific context mappers used by SemTraTrack.

The proposed LGATracker does not use CLIP, MiniLM, or another pretrained
language model. It tokenizes the deterministic window prompt with a compact
fixed vocabulary, mean-pools trainable 64-D token embeddings, and applies a
two-layer GELU MLP (64 -> 128 -> 128), exactly as specified in the manuscript.
"""
from __future__ import annotations

import re
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn


class ContextEncoder(Protocol):
    output_dim: int

    def encode(self, prompt: str, numeric: Optional[np.ndarray] = None) -> Tensor: ...


class StructuredPromptTokenizer:
    """Deterministic tokenizer for the fixed SemTraTrack prompt template.

    Digits are tokenized individually, so arbitrary counts and decimal values
    remain inside a finite task-specific vocabulary. No external tokenizer or
    pretrained vocabulary is used.
    """

    PAD = "<pad>"
    UNK = "<unk>"
    WORDS: Tuple[str, ...] = (
        "during", "this", "window", "uav", "identities", "were", "observed",
        "states", "medium", "small", "large", "mean", "normalized", "position",
        "x", "y", "dominant", "motion", "stationary", "horizontal", "vertical",
        "confidence",
    )
    SYMBOLS: Tuple[str, ...] = (
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        ".", ",", ":", "-", "=",
    )
    TOKEN_PATTERN = re.compile(r"[a-z]+|\d|[.,:\-=]")

    def __init__(self) -> None:
        vocab = (self.PAD, self.UNK) + self.WORDS + self.SYMBOLS
        self.token_to_id = {token: index for index, token in enumerate(vocab)}
        self.id_to_token = vocab
        self.pad_id = self.token_to_id[self.PAD]
        self.unk_id = self.token_to_id[self.UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def tokenize(self, prompt: str) -> list[str]:
        return self.TOKEN_PATTERN.findall(prompt.lower())

    def encode(self, prompt: str) -> Tensor:
        tokens = self.tokenize(prompt)
        if not tokens:
            return torch.tensor([self.unk_id], dtype=torch.long)
        return torch.tensor(
            [self.token_to_id.get(token, self.unk_id) for token in tokens],
            dtype=torch.long,
        )

    def batch_encode(
        self,
        prompts: Sequence[str],
        device: torch.device | str,
    ) -> tuple[Tensor, Tensor]:
        if not prompts:
            raise ValueError("At least one prompt is required.")
        rows = [self.encode(prompt) for prompt in prompts]
        length = max(row.numel() for row in rows)
        ids = torch.full(
            (len(rows), length), self.pad_id, dtype=torch.long, device=device
        )
        valid = torch.zeros((len(rows), length), dtype=torch.bool, device=device)
        for index, row in enumerate(rows):
            n = row.numel()
            ids[index, :n] = row.to(device)
            valid[index, :n] = True
        return ids, valid


class StructuredPromptEncoder(nn.Module):
    """Trainable E_tok + mean pooling + two-layer GELU MLP."""

    output_dim = 128

    def __init__(
        self,
        token_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 128,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if (token_dim, hidden_dim, output_dim) != (64, 128, 128):
            raise ValueError("The manuscript fixes d_t=64, hidden=128, and d_e=128.")
        self.tokenizer = StructuredPromptTokenizer()
        self.token_embedding = nn.Embedding(
            self.tokenizer.vocab_size,
            token_dim,
            padding_idx=self.tokenizer.pad_id,
        )
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim
        self.device_name = device
        self.to(device)

    def forward(self, prompts: Sequence[str]) -> Tensor:
        device = self.token_embedding.weight.device
        ids, valid = self.tokenizer.batch_encode(prompts, device)
        embeddings = self.token_embedding(ids)
        weights = valid.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.mlp(pooled)

    def encode(self, prompt: str, numeric: Optional[np.ndarray] = None) -> Tensor:
        return self.forward([prompt])[0]


class NumericMLPEncoder(nn.Module):
    """NV + MLP control: the 10-D numeric window vector maps to 128-D."""

    output_dim = 128

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        output_dim: int = 128,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.device_name = device
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.to(device)

    def forward(self, numeric: Tensor) -> Tensor:
        return self.mlp(numeric)

    def encode(self, prompt: str, numeric: Optional[np.ndarray] = None) -> Tensor:
        if numeric is None:
            raise ValueError("NV + MLP requires the numeric window vector.")
        device = self.mlp[0].weight.device
        x = torch.as_tensor(numeric, dtype=torch.float32, device=device)
        return self.forward(x)


def build_context_encoder(name: str = "structured-mlp", device: str = "cuda") -> nn.Module:
    key = name.lower().replace("_", "-")
    if key in {"structured-mlp", "prompt-mlp", "lgatracker", "full"}:
        return StructuredPromptEncoder(device=device)
    if key in {"numeric-mlp", "nv-mlp", "numeric"}:
        return NumericMLPEncoder(device=device)
    raise ValueError(
        f"Unknown context encoder {name!r}. The manuscript implementation uses "
        "'structured-mlp'; 'numeric-mlp' is retained only for the NV + MLP control."
    )
