#!/usr/bin/env python3
"""Train every semantic-control variant with the same three seeds.

The manuscript reports mean and standard deviation over three runs. This
launcher makes the seed set identical across variants and gives each run a
separate checkpoint. Evaluation remains a separate, dataset-specific step.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


VARIANTS = ("full", "constant", "shuffled-window", "attribute-permuted", "numeric")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-dir", default="weights/semantic_controls")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    trainer = Path(__file__).with_name("train_association.py")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for variant in args.variants:
            checkpoint = output / f"lgatracker_{variant}_seed{seed}.pt"
            command = [
                sys.executable,
                str(trainer),
                "--corpus", args.corpus,
                "--variant", variant,
                "--out", str(checkpoint),
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch", str(args.batch),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--warmup-epochs", str(args.warmup_epochs),
                "--device", args.device,
            ]
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
