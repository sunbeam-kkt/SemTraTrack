#!/usr/bin/env python3
"""Run the paper's LGATracker on every video and save MOT-format results.

Unlike the original repository, this script does NOT call Ultralytics
`model.track()` (BoT-SORT/ByteTrack), because that is not the method described
in the SemTraTrack paper.
"""
import argparse
import csv
import os
from pathlib import Path
import cv2
import torch

from semtratrack.config import SemTraTrackConfig
from semtratrack.context import build_context_encoder
from semtratrack.detector import UltralyticsDetector
from semtratrack.lgatracker import LGATracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="ISPS-SOD / YOLOv13 detector checkpoint")
    ap.add_argument("--assoc-checkpoint", default=None, help="jointly trained mapper/W_g/W_e checkpoint; not used by geometry-only")
    ap.add_argument("--input", default="./dataset/Videos")
    ap.add_argument("--output", default="./processed_results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--detector-device", default="0")
    ap.add_argument("--seed", type=int, default=0, help="fixed seed for prompt-control variants")
    ap.add_argument(
        "--context-variant",
        default="full",
        choices=["full", "constant", "shuffled-window", "attribute-permuted", "numeric", "geometry-only"],
        help="full is the proposed LGATracker; the remaining choices reproduce the context controls",
    )
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--disable-cue", action="append", choices=["pos", "size", "state", "dir"], default=[])
    ap.add_argument("--disable-attribute", action="append", choices=["qs", "ss", "ap", "md", "ac"], default=[])
    args = ap.parse_args()
    if args.context_variant != "geometry-only" and args.assoc_checkpoint is None:
        ap.error("--assoc-checkpoint is required for every context-enabled variant")

    cfg = SemTraTrackConfig(
        conf_threshold=args.conf,
        context_variant=args.context_variant,
        context_seed=args.seed,
        disabled_cues=tuple(dict.fromkeys(args.disable_cue)),
        disabled_attributes=tuple(dict.fromkeys(args.disable_attribute)),
    )
    cfg.validate()
    detector = UltralyticsDetector(args.weights, args.detector_device, cfg)
    if args.context_variant == "geometry-only":
        encoder = None
    else:
        encoder_name = "numeric-mlp" if args.context_variant == "numeric" else "structured-mlp"
        encoder = build_context_encoder(encoder_name, args.device)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted([p for p in Path(args.input).iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] cannot read {video}"); continue
        h, w = frame.shape[:2]
        tracker = LGATracker.from_checkpoint((w, h), encoder, args.assoc_checkpoint, cfg, args.device)
        path = out_dir / f"{video.stem}_MOT.txt"
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            frame_idx = 0
            while ok:
                frame_idx += 1
                dets = detector.detect(frame)
                active = tracker.update(dets)
                # Write only observations updated in this frame, not propagated lost tracks.
                for tr in active:
                    if not tr.history or tr.history[-1].frame_index != frame_idx:
                        continue
                    o = tr.history[-1]
                    x1, y1, x2, y2 = map(float, o.xyxy)
                    writer.writerow([frame_idx, tr.track_id, x1, y1, x2-x1, y2-y1, o.confidence, -1, -1, -1])
                ok, frame = cap.read()
        cap.release()
        print(f"saved: {path}")

if __name__ == "__main__":
    main()
