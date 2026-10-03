#!/usr/bin/env python3
"""Evaluate the manuscript's first-frame-initialized generic-SOT protocol.

The default factor and square-crop geometry follow the published AQATrack-256
configuration associated with the AQATrack-LINR comparison.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from semtratrack.config import SemTraTrackConfig
from semtratrack.context import build_context_encoder
from semtratrack.detector import UltralyticsDetector
from semtratrack.sot import (
    DEFAULT_SEARCH_FACTOR,
    SOTTracker,
    offset_detections,
    search_crop_from_box,
)


def parse_box(text):
    vals=[float(x) for x in text.split(',')]
    if len(vals)!=4: raise argparse.ArgumentTypeError('box must be x,y,w,h')
    x,y,w,h=vals; return np.array([x,y,x+w,y+h],np.float32)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',required=True)
    ap.add_argument('--first-box',required=True,type=parse_box,help='first-frame GT x,y,w,h; used once only')
    ap.add_argument(
        '--weights',
        required=True,
        help='Ultralytics-compatible, dataset-appropriate class-agnostic candidate localizer',
    )
    ap.add_argument('--assoc-checkpoint',required=True)
    ap.add_argument(
        '--search-factor',
        type=float,
        default=DEFAULT_SEARCH_FACTOR,
        help='square search-area factor (default: 4.0, AQATrack-256 convention)',
    )
    ap.add_argument('--out',default='sot_result.txt')
    ap.add_argument('--conf',type=float,default=0.25)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--detector-device',default='0')
    args=ap.parse_args()
    if args.search_factor <= 1.0:
        ap.error('--search-factor must be greater than 1.0')

    cfg=SemTraTrackConfig(conf_threshold=args.conf)
    cfg.validate()
    detector=UltralyticsDetector(args.weights,args.detector_device,cfg)
    encoder=build_context_encoder('structured-mlp',args.device)
    cap=cv2.VideoCapture(args.video); ok,frame=cap.read()
    if not ok: raise RuntimeError('Cannot read video')
    h,w=frame.shape[:2]
    from semtratrack.lgatracker import LGATracker
    tracker=LGATracker.from_checkpoint((w,h),encoder,args.assoc_checkpoint,cfg,args.device)
    # Reuse loaded adapter in SOT specialization.
    sot=SOTTracker((w,h),encoder,tracker.adapter,cfg,args.device)
    tr=sot.initialize(args.first_box)
    x1,y1,x2,y2=map(float,args.first_box)
    # Standard SOT result files contain exactly x,y,w,h, with no frame-index
    # column. The previous implementation accidentally wrote xyxy in the first
    # row and xywh in later rows, which invalidated benchmark evaluation.
    rows=[(x1,y1,x2-x1,y2-y1)]
    while True:
        ok,frame=cap.read()
        if not ok: break
        if tr is None:
            rows.append((0.0,0.0,0.0,0.0))
            continue
        crop,offset=search_crop_from_box(frame,tr.xyxy,args.search_factor)
        candidates=offset_detections(detector.detect(crop),offset,(w,h))
        tr=sot.update_single(candidates)
        if tr is not None and tr.history[-1].frame_index == sot.frame_index:
            x1,y1,x2,y2=map(float,tr.xyxy); rows.append((x1,y1,x2-x1,y2-y1))
        else:
            rows.append((0.0,0.0,0.0,0.0))
    cap.release()
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(args.out,np.asarray(rows),fmt='%.6f',delimiter=',')
    print(f'saved: {args.out}')

if __name__=='__main__': main()
