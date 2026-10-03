#!/usr/bin/env python3
"""One-batch smoke test for the full YOLOv13 + ISPS-SOD integration.

Run after `bash scripts/bootstrap_yolov13.sh`. It instantiates the exact UAV
model graph and executes a synthetic adjacent-frame batch through native loss +
all paper auxiliary losses. This is intended to catch upstream API/version
mismatches before launching a 50-epoch run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics.cfg import get_cfg

from semtratrack.config import SemTraTrackConfig
from semtratrack.training.model import PaperAlignedDetectionModel, require_yolov13

require_yolov13()
device='cuda' if torch.cuda.is_available() else 'cpu'
model=PaperAlignedDetectionModel('configs/yolov13s-UAV.yaml',nc=1,paper_cfg=SemTraTrackConfig()).to(device).train()
# DetectionTrainer normally attaches these native loss hyperparameters.  The
# standalone smoke test must do so explicitly before initializing the loss.
model.args=get_cfg()
# Two adjacent frames, two persistent identities (so TCR and MIFD are active).
batch={
 'img': torch.rand(2,3,640,640,device=device),
 'cls': torch.zeros(4,1,device=device),
 'bboxes': torch.tensor([[.35,.50,.04,.04],[.65,.55,.04,.04],[.36,.50,.04,.04],[.64,.55,.04,.04]],device=device),
 'batch_idx': torch.tensor([0,0,1,1],device=device),
 'track_ids': torch.tensor([5,8,5,8],device=device),
 'visibility': torch.ones(4,device=device),
 'pair_id_per_image': torch.tensor([0,0],device=device),
 'pair_slot_per_image': torch.tensor([0,1],device=device),
 'frame_index_per_image': torch.tensor([1,2],device=device),
 'paper_pair_mode': True,
}
loss,items=model.loss(batch)
assert torch.isfinite(loss), loss
print('full_loss=',float(loss.detach()))
print('native_items=',items.detach().cpu().tolist())
print('aux=',model.last_aux)
print('FULL DETECTOR SMOKE TEST PASSED')
