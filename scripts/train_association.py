#!/usr/bin/env python3
"""Train the LGATracker language mapper and W_g/W_e with pair supervision.

For the proposed model, all parameters in E_tok, the 64->128->128 GELU MLP,
W_g, and W_e are jointly optimized. The same entry point reproduces the four
context-enabled controls in the manuscript. Hungarian assignment is inference
only and is never differentiated through.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from semtratrack.config import SemTraTrackConfig
from semtratrack.context import NumericMLPEncoder, StructuredPromptEncoder
from semtratrack.lgatracker import PairContextAdapter, WindowSummary, controlled_prompt


VARIANTS = ("full", "constant", "shuffled-window", "attribute-permuted", "numeric")


def load_corpus(path: str) -> tuple[list[dict], dict[str, list[WindowSummary]], dict]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No association samples found in {path}")
    metadata_records = [record for record in records if record.get("record_type") == "metadata"]
    if len(metadata_records) > 1:
        raise ValueError(f"Corpus {path} contains multiple metadata records.")
    metadata = metadata_records[0] if metadata_records else {}
    rows = [record for record in records if record.get("record_type", "pair") == "pair"]
    if not rows:
        raise ValueError(f"No pair records found in {path}")
    required = {"pair_features", "history_indices", "history_summaries", "label"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"Corpus is missing {sorted(missing)}. Rebuild it with "
            "scripts/build_assoc_pairs_from_mot.py."
        )
    indexed: dict[str, dict[int, WindowSummary]] = {}
    for record in records:
        if record.get("record_type") != "summary":
            continue
        sequence = str(record.get("sequence", "default"))
        indexed.setdefault(sequence, {})[int(record["summary_index"])] = WindowSummary.from_dict(record["summary"])

    # Backward compatibility with the earlier, larger JSONL representation.
    if not indexed:
        for row in rows:
            if "summary_pool" not in row:
                continue
            sequence = str(row.get("sequence", "default"))
            pool = [WindowSummary.from_dict(item) for item in row["summary_pool"]]
            if len(pool) > len(indexed.get(sequence, {})):
                indexed[sequence] = {index: summary for index, summary in enumerate(pool)}

    pools: dict[str, list[WindowSummary]] = {}
    for sequence, values in indexed.items():
        if not values:
            continue
        expected = list(range(max(values) + 1))
        if sorted(values) != expected:
            raise ValueError(f"Non-contiguous summary indices for sequence {sequence!r}.")
        pools[sequence] = [values[index] for index in expected]
    return rows, pools, metadata


def validate_corpus_protocol(metadata: dict, cfg: SemTraTrackConfig) -> None:
    """Reject a corpus built with a different temporal/cue protocol."""
    signature = metadata.get("association_signature")
    if signature is None:
        print(
            "[WARN] Association corpus has no protocol metadata; rebuild it with "
            "scripts/build_assoc_pairs_from_mot.py for strict alignment checks."
        )
        return
    expected = cfg.association_signature()
    mismatches = {
        key: (signature.get(key), expected.get(key))
        for key in expected
        if signature.get(key) != expected.get(key)
    }
    if mismatches:
        details = ", ".join(
            f"{key}: corpus={old!r}, training={new!r}"
            for key, (old, new) in sorted(mismatches.items())
        )
        raise ValueError("Association corpus/training protocol mismatch (" + details + ").")


def prepare_variant(
    rows: list[dict],
    summary_pools: dict[str, list[WindowSummary]],
    variant: str,
    seed: int,
    disabled_attributes: tuple[str, ...] = (),
) -> None:
    """Precompute deterministic, causal prompt controls for every sequence."""
    if variant == "numeric":
        return
    controlled_by_sequence: dict[str, list[str]] = {}
    for sequence in sorted(summary_pools):
        rng = np.random.default_rng(seed)
        summaries = summary_pools[sequence]
        controlled = []
        for index, summary in enumerate(summaries):
            controlled.append(
                controlled_prompt(
                    summary, summaries[:index], variant, rng, disabled_attributes
                )
            )
        controlled_by_sequence[sequence] = controlled

    for row in rows:
        sequence = str(row.get("sequence", "default"))
        if sequence not in controlled_by_sequence:
            raise ValueError(f"Missing summary records for sequence {sequence!r}.")
        prompts = controlled_by_sequence[sequence]
        indices = [int(index) for index in row["history_indices"]]
        if any(index < 0 or index >= len(prompts) for index in indices):
            raise ValueError(f"Invalid history index in sequence {sequence!r}.")
        row["variant_prompts"] = [prompts[index] for index in indices]


def encode_history(
    encoder: torch.nn.Module,
    row: dict,
    variant: str,
    device: str,
) -> torch.Tensor:
    if variant == "numeric":
        vectors = [
            WindowSummary.from_dict(item).numeric_vector()
            for item in row["history_summaries"]
        ]
        x = torch.as_tensor(np.stack(vectors), dtype=torch.float32, device=device)
        return encoder(x).mean(dim=0)
    return encoder(row["variant_prompts"]).mean(dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="JSONL from build_assoc_pairs_from_mot.py")
    parser.add_argument("--out", default="weights/lgatracker.pt")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-cue", action="append", choices=["pos", "size", "state", "dir"], default=[])
    parser.add_argument("--disable-attribute", action="append", choices=["qs", "ss", "ap", "md", "ac"], default=[])
    args = parser.parse_args()
    if args.variant != "full" and (args.disable_cue or args.disable_attribute):
        parser.error("Cue/attribute ablations may be combined only with --variant full")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    disabled_cues = tuple(dict.fromkeys(args.disable_cue))
    disabled_attributes = tuple(dict.fromkeys(args.disable_attribute))
    cfg = SemTraTrackConfig(
        context_variant=args.variant,
        context_seed=args.seed,
        disabled_cues=disabled_cues,
        disabled_attributes=disabled_attributes,
    )
    cfg.validate()
    rows, summary_pools, corpus_metadata = load_corpus(args.corpus)
    validate_corpus_protocol(corpus_metadata, cfg)
    prepare_variant(rows, summary_pools, args.variant, args.seed, disabled_attributes)
    cue_index = {"pos": 0, "size": 1, "state": 2, "dir": 3}
    for row in rows:
        for cue in disabled_cues:
            row["pair_features"][cue_index[cue]] = 0.0
    if args.variant == "numeric":
        encoder = NumericMLPEncoder(
            input_dim=10,
            hidden_dim=cfg.language_hidden_dim,
            output_dim=cfg.context_dim,
            device=args.device,
        )
        encoder_name = "numeric-mlp"
    else:
        encoder = StructuredPromptEncoder(
            token_dim=cfg.token_embedding_dim,
            hidden_dim=cfg.language_hidden_dim,
            output_dim=cfg.context_dim,
            device=args.device,
        )
        encoder_name = "structured-mlp"
    adapter = PairContextAdapter(cfg.context_dim, cfg.shared_dim).to(args.device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(adapter.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(rows) / args.batch)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(total_steps, args.warmup_epochs * steps_per_epoch)

    def scheduled_lr(update: int) -> float:
        if warmup_steps and update < warmup_steps:
            return args.lr * float(update + 1) / float(warmup_steps)
        cosine_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (update - warmup_steps) / max(1, cosine_steps - 1))
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    update = 0
    for epoch in range(args.epochs):
        encoder.train()
        adapter.train()
        random.shuffle(rows)
        total = 0.0
        count = 0
        for start in range(0, len(rows), args.batch):
            lr = scheduled_lr(update)
            for group in optimizer.param_groups:
                group["lr"] = lr
            chunk = rows[start:start + args.batch]
            pair = torch.as_tensor(
                [row["pair_features"] for row in chunk],
                dtype=torch.float32,
                device=args.device,
            )
            labels = torch.as_tensor(
                [row["label"] for row in chunk],
                dtype=torch.float32,
                device=args.device,
            )
            context = torch.stack(
                [encode_history(encoder, row, args.variant, args.device) for row in chunk]
            )
            loss = adapter.association_loss(pair, context, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update += 1
            total += float(loss.item()) * len(chunk)
            count += len(chunk)
        print(f"epoch={epoch + 1:03d} assoc_loss={total / max(count, 1):.6f} lr={lr:.8f}")

    checkpoint = {
        "adapter": adapter.state_dict(),
        "context_encoder": encoder.state_dict(),
        "encoder_name": encoder_name,
        "variant": args.variant,
        "context_dim": cfg.context_dim,
        "shared_dim": cfg.shared_dim,
        "token_embedding_dim": cfg.token_embedding_dim,
        "language_hidden_dim": cfg.language_hidden_dim,
        "context_seed": cfg.context_seed,
        "disabled_cues": list(cfg.disabled_cues),
        "disabled_attributes": list(cfg.disabled_attributes),
        "association_signature": cfg.association_signature(),
        "corpus_metadata": corpus_metadata,
        "training_protocol": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "optimizer": "AdamW",
            "initial_lr": args.lr,
            "weight_decay": args.weight_decay,
            "schedule": "cosine",
            "warmup_epochs": args.warmup_epochs,
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
