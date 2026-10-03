#!/usr/bin/env python3
"""Single-video paper-aligned SemTraTrack inference."""
import argparse
from pathlib import Path
import cv2
import torch

from semtratrack.config import SemTraTrackConfig
from semtratrack.context import build_context_encoder
from semtratrack.detector import UltralyticsDetector
from semtratrack.lgatracker import LGATracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--assoc-checkpoint", required=True)
    ap.add_argument("--out", default="processed_results/preview.mp4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--detector-device", default="0")
    args = ap.parse_args()

    cfg = SemTraTrackConfig()
    detector = UltralyticsDetector(args.weights, args.detector_device, cfg)
    encoder = build_context_encoder("structured-mlp", args.device)
    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    if not ok: raise RuntimeError("Cannot read input video")
    h, w = frame.shape[:2]
    tracker = LGATracker.from_checkpoint((w, h), encoder, args.assoc_checkpoint, cfg, args.device)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while ok:
        tracks = tracker.update(detector.detect(frame))
        for tr in tracks:
            if not tr.history or tr.history[-1].frame_index != tracker.frame_index: continue
            x1,y1,x2,y2 = map(int, tr.xyxy)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.putText(frame, str(tr.track_id), (x1, max(0,y1-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        writer.write(frame)
        ok, frame = cap.read()
    cap.release(); writer.release()
    print(f"saved: {args.out}")

if __name__ == "__main__": main()
