#!/usr/bin/env python3
"""Train the complete paper-aligned ISPS-SOD detector."""
import argparse
from semtratrack.config import SemTraTrackConfig
from semtratrack.training.model import require_yolov13
from semtratrack.training.trainer import PaperAlignedDetectionTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair-manifest", required=True, help="JSONL produced by scripts/build_pair_manifest.py")
    p.add_argument("--model", default="configs/yolov13s-UAV.yaml")
    p.add_argument("--weights", default=None, help="optional YOLOv13-S pretrained checkpoint")
    p.add_argument("--data", default="configs/MOT-UAV.yaml")
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/semtratrack")
    p.add_argument("--name", default="isps_sod")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--imgsz", type=int, choices=[640, 1280], default=640)
    p.add_argument(
        "--detector-ablation",
        choices=["full", "base", "w/o-tcr", "w/o-mifd", "w/o-stcr", "w/o-bns-hm", "w/o-msa-nwd", "w/o-sps"],
        default="full",
        help="exact component switch rows from the ISPS-SOD ablation table",
    )
    args = p.parse_args()
    require_yolov13()
    cfg = SemTraTrackConfig(imgsz=args.imgsz, detector_ablation=args.detector_ablation)
    overrides = dict(
        model=(args.weights if args.weights else args.model),
        data=args.data,
        epochs=50,
        batch=16,                 # image batch; trainer samples 8 frame pairs
        imgsz=args.imgsz,
        optimizer="AdamW",
        lr0=0.0005,
        weight_decay=0.01,
        cos_lr=True,
        warmup_epochs=5,
        amp=True,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=args.name,
    )
    trainer = PaperAlignedDetectionTrainer(pair_manifest=args.pair_manifest, paper_cfg=cfg, overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
