#!/usr/bin/env python3
"""Run SemTraTrack with the official Anti-UAV410 result convention.

Only the first-frame ground-truth box initializes the tracker. Later ``exist``
labels and boxes are never read by the tracking loop. A matched prediction is
written as ``[x, y, w, h]``; an unmatched/lost frame is written as ``[0]``,
which is the target-absence representation consumed by the official State
Accuracy evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np
import torch

from semtratrack.config import SemTraTrackConfig
from semtratrack.context import build_context_encoder
from semtratrack.detector import UltralyticsDetector
from semtratrack.lgatracker import LGATracker
from semtratrack.sot import (
    DEFAULT_SEARCH_FACTOR,
    SOTTracker,
    offset_detections,
    search_crop_from_box,
)


def frame_sort_key(path: Path):
    chunks = re.split(r"(\d+)", path.stem)
    return tuple(int(chunk) if chunk.isdigit() else chunk.lower() for chunk in chunks)


def xyxy_to_xywh(box: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def first_box_xyxy(label_path: Path) -> tuple[np.ndarray, int]:
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    boxes = labels.get("gt_rect")
    if not boxes or len(boxes[0]) != 4:
        raise ValueError(f"{label_path} has no valid first-frame gt_rect.")
    exists = labels.get("exist")
    if exists is not None and len(exists) != len(boxes):
        raise ValueError(f"{label_path} has inconsistent gt_rect/exist lengths.")
    x, y, width, height = map(float, boxes[0])
    if width <= 0 or height <= 0:
        raise ValueError(f"{label_path} has a non-positive first-frame box.")
    return np.asarray([x, y, x + width, y + height], dtype=np.float32), len(boxes)


def run_sequence(
    sequence_dir: Path,
    detector: UltralyticsDetector,
    assoc_checkpoint: str,
    search_factor: float,
    cfg: SemTraTrackConfig,
    device: str,
) -> list[list[float]]:
    frames = sorted(sequence_dir.glob("*.jpg"), key=frame_sort_key)
    if not frames:
        raise ValueError(f"No JPG frames found in {sequence_dir}")
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise ValueError(f"Cannot read {frames[0]}")
    height, width = first_frame.shape[:2]
    initial_box, labelled_frame_count = first_box_xyxy(sequence_dir / "IR_label.json")
    if len(frames) != labelled_frame_count:
        raise ValueError(
            f"{sequence_dir} contains {len(frames)} JPG frames but its label file "
            f"declares {labelled_frame_count}."
        )

    encoder = build_context_encoder("structured-mlp", device=device)
    loaded = LGATracker.from_checkpoint(
        (width, height), encoder, assoc_checkpoint, cfg, device
    )
    tracker = SOTTracker((width, height), encoder, loaded.adapter, cfg, device)
    track = tracker.initialize(initial_box)
    results: list[list[float]] = [xyxy_to_xywh(initial_box)]

    for frame_path in frames[1:]:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Cannot read {frame_path}")
        if track is None:
            # SemTraTrack does not use later ground truth to re-initialize.
            results.append([0])
            continue
        crop, offset = search_crop_from_box(frame, track.xyxy, search_factor)
        candidates = offset_detections(
            detector.detect(crop), offset, (frame.shape[1], frame.shape[0])
        )
        track = tracker.update_single(candidates)
        matched_now = (
            track is not None
            and bool(track.history)
            and track.history[-1].frame_index == tracker.frame_index
        )
        results.append(xyxy_to_xywh(track.xyxy) if matched_now else [0])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="Anti-UAV410 root containing train/val/test")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--weights", required=True, help="ISPS-SOD checkpoint")
    parser.add_argument("--assoc-checkpoint", required=True)
    parser.add_argument(
        "--search-factor",
        type=float,
        default=DEFAULT_SEARCH_FACTOR,
        help="square search-area factor (default: 4.0, AQATrack-256 convention)",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.search_factor <= 1.0:
        parser.error("--search-factor must be greater than 1.0")

    split_root = Path(args.dataset_root) / args.split
    if not split_root.is_dir():
        parser.error(f"Split directory does not exist: {split_root}")
    sequences = sorted(
        path.parent for path in split_root.glob("*/IR_label.json") if path.parent.is_dir()
    )
    if not sequences:
        parser.error(f"No Anti-UAV410 sequences found under {split_root}")

    cfg = SemTraTrackConfig(conf_threshold=args.conf)
    cfg.validate()
    detector = UltralyticsDetector(args.weights, args.detector_device, cfg)
    output = Path(args.output or f"results/AntiUAV410/{args.split}/SemTraTrack")
    output.mkdir(parents=True, exist_ok=True)

    for index, sequence in enumerate(sequences, start=1):
        destination = output / f"{sequence.name}.txt"
        if destination.exists() and not args.overwrite:
            print(f"[{index:03d}/{len(sequences):03d}] exists, skipping: {destination}")
            continue
        results = run_sequence(
            sequence,
            detector,
            args.assoc_checkpoint,
            args.search_factor,
            cfg,
            args.device,
        )
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps({"res": results}), encoding="utf-8")
        temporary.replace(destination)
        print(f"[{index:03d}/{len(sequences):03d}] saved: {destination}")


if __name__ == "__main__":
    main()
