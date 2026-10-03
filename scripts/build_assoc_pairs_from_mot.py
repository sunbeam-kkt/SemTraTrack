#!/usr/bin/env python3
"""Build causal Stage-II supervision from identity-labelled MOT detections.

The output is streaming JSONL rather than precomputed language features because the
manuscript jointly trains the task-specific token embedding, the two-layer MLP,
and W_g/W_e. Summary records are written once per completed window; pair
records store only the last K causal summaries, avoiding quadratic duplication
of the full sequence history.

Expected MOT rows (comma or whitespace separated):
    frame, identity, x, y, w, h, confidence, ...
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from semtratrack.config import SemTraTrackConfig
from semtratrack.lgatracker import (
    LGATracker,
    PairContextAdapter,
    WindowSummary,
    coarse_state,
    summarize_window,
)
from semtratrack.types import Detection, Track, TrackObservation


class _UnusedEncoder:
    output_dim = 128

    def encode(self, prompt, numeric=None):  # pragma: no cover - never called here
        raise RuntimeError("The corpus builder must not encode context.")


@dataclass(frozen=True)
class SequenceSpec:
    mot: Path
    width: int
    height: int
    name: str


def load_mot(path: str) -> dict[int, list[tuple[int, Detection]]]:
    frames: dict[int, list[tuple[int, Detection]]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = [part for part in re.split(r"[\s,]+", line.strip()) if part]
        if len(parts) < 6:
            continue
        frame, identity = int(float(parts[0])), int(float(parts[1]))
        x, y, width, height = map(float, parts[2:6])
        if frame < 1 or identity < 0 or width <= 0 or height <= 0:
            continue
        confidence = float(parts[6]) if len(parts) > 6 else 1.0
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Detection confidence must be in [0,1], got {confidence} "
                f"for identity {identity} in frame {frame} of {path}."
            )
        box = np.asarray([x, y, x + width, y + height], dtype=np.float32)
        frames[frame].append((identity, Detection(box, confidence)))
    for frame, observations in frames.items():
        identities = [identity for identity, _ in observations]
        if len(identities) != len(set(identities)):
            raise ValueError(f"Duplicate identity in frame {frame} of {path}.")
    return frames


def load_sequence_manifest(path: str) -> list[SequenceSpec]:
    """Load JSONL sequence metadata for a multi-sequence Stage-II corpus.

    Each row must contain ``mot``, ``width``, and ``height`` and may contain a
    stable ``sequence`` name. Relative MOT paths are resolved from the manifest
    directory, allowing the corpus recipe to be moved as one directory tree.
    """
    manifest = Path(path).resolve()
    specs: list[SequenceSpec] = []
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
            mot = Path(record["mot"])
            width = int(record["width"])
            height = int(record["height"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid sequence manifest row {line_number}: {raw}") from exc
        if not mot.is_absolute():
            mot = (manifest.parent / mot).resolve()
        if not mot.is_file():
            raise FileNotFoundError(f"MOT file in row {line_number} does not exist: {mot}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid frame size in row {line_number}: {width}x{height}")
        name = str(record.get("sequence") or mot.stem).strip()
        if not name:
            raise ValueError(f"Empty sequence name in row {line_number}.")
        specs.append(SequenceSpec(mot, width, height, name))
    if not specs:
        raise ValueError(f"No sequence rows found in {manifest}")
    _validate_sequence_names(specs)
    return specs


def resolve_mot_inputs(paths: list[str], width: int | None, height: int | None) -> list[SequenceSpec]:
    """Resolve one or more MOT files/directories sharing a frame size."""
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("--width and --height are required and must be positive with --mot.")
    files: list[Path] = []
    for value in paths:
        path = Path(value).resolve()
        if path.is_dir():
            files.extend(sorted(path.rglob("*.txt")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    files = sorted(dict.fromkeys(files))
    if not files:
        raise ValueError("No MOT text files were resolved from --mot.")
    specs = [SequenceSpec(path, width, height, path.stem) for path in files]
    _validate_sequence_names(specs)
    return specs


def _validate_sequence_names(specs: list[SequenceSpec]) -> None:
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Sequence names must be unique; use --sequence-manifest to disambiguate: "
            + ", ".join(duplicates)
        )


def build_sequence_records(
    spec: SequenceSpec,
    cfg: SemTraTrackConfig,
    neg_per_pos: int,
    rng: np.random.Generator,
    write_record,
) -> tuple[int, int, int]:
    """Write one sequence while keeping history and identities sequence-local."""
    helper = LGATracker(
        (spec.width, spec.height),
        _UnusedEncoder(),
        PairContextAdapter(cfg.context_dim, cfg.shared_dim),
        cfg,
        device="cpu",
    )
    frames = load_mot(str(spec.mot))
    if not frames:
        return 0, 0, 0

    active: dict[int, Track] = {}
    last_seen: dict[int, int] = {}
    window_observations: list[TrackObservation] = []
    history = deque(maxlen=cfg.context_history)
    completed: list[WindowSummary] = []
    pair_count = 0
    positives = 0

    for frame_index in range(1, max(frames) + 1):
        current = frames.get(frame_index, [])
        for identity in list(active):
            if frame_index - last_seen.get(identity, frame_index) > cfg.max_lost:
                active.pop(identity, None)
                last_seen.pop(identity, None)

        # Context is strictly causal: only completed previous windows are stored.
        if history and active:
            history_indices = list(range(len(completed) - len(history), len(completed)))
            history_records = [summary.to_dict() for summary in history]
            for identity, detection in current:
                detection.state = coarse_state(detection.area, cfg)
                candidates = [
                    (other_id, helper.pairwise_cues(detection, track))
                    for other_id, track in active.items()
                ]
                positives_for_detection = [candidate for candidate in candidates if candidate[0] == identity]
                negatives = [candidate for candidate in candidates if candidate[0] != identity]
                chosen = positives_for_detection[:1]
                if negatives:
                    count = min(len(negatives), neg_per_pos if positives_for_detection else 1)
                    indices = rng.choice(len(negatives), size=count, replace=False)
                    chosen.extend(negatives[int(index)] for index in np.atleast_1d(indices))
                for other_id, cues in chosen:
                    label = 1.0 if other_id == identity else 0.0
                    write_record(
                        {
                            "record_type": "pair",
                            "sequence": spec.name,
                            "frame_index": frame_index,
                            "window_index": (frame_index - 1) // cfg.window_size + 1,
                            "pair_features": cues.tolist(),
                            "history_indices": history_indices,
                            "history_summaries": history_records,
                            "label": label,
                        }
                    )
                    pair_count += 1
                    positives += int(label == 1.0)

        # GT identities update training-only tracks after pair construction.
        for identity, detection in current:
            detection.state = coarse_state(detection.area, cfg)
            if identity not in active:
                active[identity] = Track(
                    identity,
                    detection.xyxy.copy(),
                    detection.confidence,
                    detection.state,
                )
            track = active[identity]
            track.xyxy = detection.xyxy.copy()
            track.confidence = detection.confidence
            track.state = detection.state
            track.lost = 0
            observation = TrackObservation(
                frame_index,
                identity,
                detection.center.copy(),
                detection.xyxy.copy(),
                detection.confidence,
                detection.state,
            )
            track.history.append(observation)
            window_observations.append(observation)
            last_seen[identity] = frame_index

        if frame_index % cfg.window_size == 0:
            summary = summarize_window(
                window_observations,
                spec.width,
                spec.height,
                cfg.stationary_pixels,
            )
            summary_index = len(completed)
            completed.append(summary)
            history.append(summary)
            write_record(
                {
                    "record_type": "summary",
                    "sequence": spec.name,
                    "summary_index": summary_index,
                    "summary": summary.to_dict(),
                }
            )
            window_observations = []

    return pair_count, positives, len(completed)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--mot",
        action="append",
        help="identity-labelled MOT file or directory; repeat for multiple inputs",
    )
    source.add_argument(
        "--sequence-manifest",
        help="JSONL rows with mot/width/height and optional sequence name",
    )
    parser.add_argument("--width", type=int, help="shared original-frame width used with --mot")
    parser.add_argument("--height", type=int, help="shared original-frame height used with --mot")
    parser.add_argument("--out", required=True, help="output JSONL corpus")
    parser.add_argument("--neg-per-pos", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.neg_per_pos < 0:
        parser.error("--neg-per-pos must be non-negative")
    try:
        specs = (
            load_sequence_manifest(args.sequence_manifest)
            if args.sequence_manifest
            else resolve_mot_inputs(args.mot, args.width, args.height)
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    cfg = SemTraTrackConfig()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    pair_count = positives = summary_count = 0
    sequences_with_pairs = 0
    with temporary.open("w", encoding="utf-8") as stream:
        def write_record(record: dict) -> None:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_record(
            {
                "record_type": "metadata",
                "format_version": 2,
                "association_signature": cfg.association_signature(),
                "negative_pairs_per_positive": args.neg_per_pos,
                "seed": args.seed,
                "sequence_count": len(specs),
            }
        )
        for index, spec in enumerate(specs):
            counts = build_sequence_records(
                spec,
                cfg,
                args.neg_per_pos,
                np.random.default_rng(args.seed + index),
                write_record,
            )
            sequence_pairs, sequence_positives, sequence_summaries = counts
            pair_count += sequence_pairs
            positives += sequence_positives
            summary_count += sequence_summaries
            sequences_with_pairs += int(sequence_pairs > 0)
            print(
                f"sequence={spec.name} pairs={sequence_pairs} "
                f"positives={sequence_positives} summaries={sequence_summaries}"
            )

    if not pair_count:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("No pairs were produced; at least two windows and recurring identities are required.")
    os.replace(temporary, output)
    print(
        f"saved {pair_count} causal pairs from {sequences_with_pairs}/{len(specs)} sequences "
        f"-> {output}; positives={positives}; summaries={summary_count}"
    )


if __name__ == "__main__":
    main()
