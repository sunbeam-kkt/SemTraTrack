#!/usr/bin/env python3
"""Measure batch-1 inference peak GPU memory for context variants.

Use the same detector checkpoint/video and a variant-specific association
checkpoint.  This reports allocated peak memory for the complete pipeline; it
does not hard-code the manuscript's measured values.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import torch
from semtratrack.pipeline import SemTraTrackPipeline

p=argparse.ArgumentParser(); p.add_argument('--video',required=True); p.add_argument('--weights',required=True); p.add_argument('--assoc-checkpoint',default=None); p.add_argument('--context-variant',choices=['full','constant','shuffled-window','attribute-permuted','numeric','geometry-only'],required=True); p.add_argument('--frames',type=int,default=300); p.add_argument('--device',default='cuda'); p.add_argument('--detector-device',default='0'); a=p.parse_args()
if a.context_variant != 'geometry-only' and a.assoc_checkpoint is None: p.error('--assoc-checkpoint is required for context-enabled variants')
if not torch.cuda.is_available(): raise SystemExit('CUDA is required for GPU-memory measurement.')
cap=cv2.VideoCapture(a.video); ok,frame=cap.read();
if not ok: raise RuntimeError('cannot read video')
h,w=frame.shape[:2]; pipe=SemTraTrackPipeline(a.weights,a.assoc_checkpoint,(w,h),a.device,a.detector_device,a.context_variant)
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); n=0
while ok and n<a.frames:
    pipe.process(frame); n+=1; ok,frame=cap.read()
cap.release(); torch.cuda.synchronize()
print(f'frames={n}')
print(f'peak_allocated_GB={torch.cuda.max_memory_allocated()/1024**3:.3f}')
print(f'peak_reserved_GB={torch.cuda.max_memory_reserved()/1024**3:.3f}')
