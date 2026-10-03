"""Ultralytics trainer bridge for complete SemTraTrack Stage-I training."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from ..config import SemTraTrackConfig
from .data import PairedMOTDataset
from .model import PaperAlignedDetectionModel, require_yolov13

try:
    from ultralytics.data import build_dataloader
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import LOGGER, RANK
    from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
    _OK = True
    _ERR = None
except Exception as exc:
    DetectionTrainer = object  # type: ignore
    _OK = False
    _ERR = exc


if _OK:
    class PaperAlignedDetectionTrainer(DetectionTrainer):
        """DetectionTrainer using adjacent-pair train data and standard YOLO val data.

        `args.batch=16` remains the paper's *image* batch size.  The training
        DataLoader samples eight adjacent pairs and the pair collate function
        flattens them to sixteen images before the model sees the batch.
        """
        def __init__(self, pair_manifest: str, paper_cfg: Optional[SemTraTrackConfig]=None, *args, **kwargs):
            self.pair_manifest = str(pair_manifest)
            self.paper_cfg = paper_cfg or SemTraTrackConfig()
            self.paper_cfg.validate()
            super().__init__(*args, **kwargs)
            self._install_aux_logger()

        def build_dataset(self, img_path, mode="train", batch=None):
            if mode == "train":
                return PairedMOTDataset(self.pair_manifest, imgsz=self.paper_cfg.imgsz, augment=True)
            return super().build_dataset(img_path, mode=mode, batch=batch)

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            if batch_size % 2:
                raise ValueError("Paper image batch size must be even because each sample is an adjacent frame pair.")
            with torch_distributed_zero_first(rank):
                dataset = self.build_dataset(dataset_path, mode="train", batch=batch_size)
            pair_batch = batch_size // 2
            workers = self.args.workers
            return build_dataloader(dataset, pair_batch, workers, shuffle=True, rank=rank)

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = PaperAlignedDetectionModel(
                cfg=cfg, nc=self.data["nc"], verbose=verbose and RANK == -1, paper_cfg=self.paper_cfg
            )
            if weights:
                model.load(weights)
            return model

        def plot_training_labels(self):
            # PairedMOTDataset labels are generated online after letterboxing;
            # the standard static-label plotting helper is not informative.
            return None

        def _install_aux_logger(self):
            """Log auxiliary terms and q_sep without changing validator loss API."""
            def epoch_start(trainer):
                trainer._paper_aux_accum = []

            def batch_end(trainer):
                try:
                    model = de_parallel(trainer.model)
                    if getattr(model, "last_aux", None):
                        trainer._paper_aux_accum.append(dict(model.last_aux))
                except Exception:
                    pass

            def epoch_end(trainer):
                rows = getattr(trainer, "_paper_aux_accum", [])
                if not rows:
                    return
                keys = list(rows[0])
                means = {k: sum(float(r.get(k, 0.0)) for r in rows) / len(rows) for k in keys}
                means["epoch"] = int(trainer.epoch) + 1
                out = Path(trainer.save_dir) / "paper_aux_metrics.csv"
                exists = out.exists()
                with out.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["epoch"] + keys)
                    if not exists: writer.writeheader()
                    writer.writerow(means)
                LOGGER.info("SemTraTrack auxiliary: " + ", ".join(f"{k}={means[k]:.4f}" for k in keys))

            try:
                self.add_callback("on_train_epoch_start", epoch_start)
                self.add_callback("on_train_batch_end", batch_end)
                self.add_callback("on_train_epoch_end", epoch_end)
            except Exception:
                LOGGER.warning("Could not register auxiliary CSV callbacks; training objective is unaffected.")
else:
    class PaperAlignedDetectionTrainer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_yolov13()
