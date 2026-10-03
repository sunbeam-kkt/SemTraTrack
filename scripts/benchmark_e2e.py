#!/usr/bin/env python3
"""Unified end-to-end timing boundary described in Sec. 4.4.

- batch size 1
- detector input 640x640
- AMP/autocast
- 100 warm-up frames
- 1,000 timed frames
- CUDA synchronize before/after each timed iteration
- includes preprocess + detector + pair cues + context fusion + Hungarian + update
- task-specific prompt-mapper cost is amortized because it runs once every 30 frames
- excludes file I/O, visualization, result serialization
"""
import argparse
import sys
from pathlib import Path

# Allow the documented ``python scripts/<name>.py`` invocation from a fresh
# source checkout without requiring SemTraTrack itself to be pre-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import time
import torch

from semtratrack.pipeline import SemTraTrackPipeline


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--assoc-checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--detector-device", default="0")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--frames", type=int, default=1000)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read video")
    h, w = frame.shape[:2]
    pipe = SemTraTrackPipeline(
        args.weights,
        args.assoc_checkpoint,
        (w, h),
        args.device,
        args.detector_device,
        context_variant="full",
    )

    # Decode requested frames before timing so disk/video I/O is excluded.
    frames = [frame]
    while len(frames) < args.warmup + args.frames:
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = cap.read()
            if not ok:
                break
        frames.append(f)
    cap.release()
    if len(frames) < args.warmup + args.frames:
        raise RuntimeError("Insufficient decodable frames for requested benchmark")

    for f in frames[:args.warmup]:
        pipe.process(f)
    times = []
    for f in frames[args.warmup:args.warmup+args.frames]:
        sync(); t0 = time.perf_counter()
        pipe.process(f)
        sync(); times.append((time.perf_counter() - t0) * 1000.0)

    latency = float(np.mean(times))
    print(f"mean_latency_ms={latency:.4f}")
    print(f"fps={1000.0/latency:.3f}")

if __name__ == "__main__":
    main()
