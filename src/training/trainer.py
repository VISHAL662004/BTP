"""Training loop for the detection model (shared by all experiments)."""
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.data.patch_store import PatchStore, random_dihedral
from src.efficiency.parameters import count_parameters
from src.evaluation.evaluator import evaluate_store
from src.losses.detection_loss import DetectionLoss
from src.models.detector import build_detector
from src.utils.device import get_device
from src.utils.progress import Progress, text_bar
from src.utils.seed import set_seed


def make_logger(path: Path) -> logging.Logger:
    log = logging.getLogger(str(path))
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(path), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


class Trainer:
    def __init__(self, cfg: dict, out_dir, model=None, post_step=None):
        """`model`: use an existing (e.g. pruned) model instead of building one from cfg.
        `post_step`: callable run after every optimizer step (e.g. re-apply pruning masks)."""
        self.cfg, self.out = cfg, Path(out_dir)
        self.post_step = post_step
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / "best.pt").exists():
            raise FileExistsError(f"{self.out} already has results; never overwrite experiments (rules.md)")
        self.log = make_logger(self.out / "training.log")
        (self.out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        set_seed(cfg["experiment"]["seed"])
        self.device = get_device(cfg["device"]["preferred"])
        d = cfg["data"]
        self.train = PatchStore(d["cache_dir"], "train", d["channels"])
        self.val = PatchStore(d["cache_dir"], "val", d["channels"])
        self.model = (model if model is not None else build_detector(cfg)).to(self.device)
        l = cfg["loss"]
        self.loss = DetectionLoss(l["lambda_cls"], l["lambda_loc"], l["focal_alpha"], l["focal_gamma"])
        t = cfg["training"]
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=t["epochs"]) if t["scheduler"] == "cosine" else None
        ev = cfg["evaluation"]
        self.ann = pd.read_csv(ev["annotations"])
        self.exc = pd.read_csv(ev["excluded"])
        self.val_uids = pd.read_csv(Path(ev["splits_dir"]) / "val.csv").seriesuid.tolist()
        self.gen = torch.Generator().manual_seed(cfg["experiment"]["seed"])
        self.best = -1.0
        self._epoch = 0
        self.log.info(f"device={self.device} params={count_parameters(self.model)}")
        self.log.info(f"train={len(self.train)} (pos {int(self.train.y.sum())}) val={len(self.val)} (pos {int(self.val.y.sum())})")

    def _epoch_indices(self):
        y = self.train.y
        pos, neg = (y > 0.5).nonzero().squeeze(1), (y <= 0.5).nonzero().squeeze(1)
        idx = torch.cat([pos.repeat(self.cfg["data"]["pos_repeat"]), neg])
        return idx[torch.randperm(len(idx), generator=self.gen)]

    def train_epoch(self):
        self.model.train()
        bs, agg, n = self.cfg["training"]["batch_size"], {"loss": 0.0, "focal": 0.0, "loc": 0.0}, 0
        idx = self._epoch_indices()
        pbar = Progress(range(0, len(idx), bs), desc=f"  epoch {self._epoch}/{self.cfg['training']['epochs']} train",
                        leave=False, unit="batch", quiet_when_redirected=True)
        for i in pbar:
            b = idx[i:i + bs]
            x = self.train.x[b].float() / 255.0
            box = self.train.box[b]
            if self.cfg["data"]["augment"] == "dihedral":
                x, box = random_dihedral(x, box, self.gen)
            x, box, y = x.to(self.device), box.to(self.device), self.train.y[b].to(self.device)
            obj, braw = self.model(x)
            loss, parts = self.loss(obj, braw, y, box)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at batch {i // bs}: {parts}")
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.opt.step()
            if self.post_step is not None:
                self.post_step()
            for k in agg:
                agg[k] += parts[k] * len(b)
            n += len(b)
            pbar.set_postfix(loss=agg["loss"] / n)
        return {k: v / n for k, v in agg.items()}

    def checkpoint(self, name, epoch):
        torch.save({"model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                    "scheduler": self.sched.state_dict() if self.sched else None, "epoch": epoch,
                    "config": self.cfg, "best_val_cpm": self.best, "seed": self.cfg["experiment"]["seed"],
                    "pruning_state": None}, self.out / name)

    def fit(self):
        hist = []
        E = self.cfg["training"]["epochs"]
        epochs = Progress(range(1, E + 1), desc="epochs", unit="epoch", step_pct=5)
        for epoch in epochs:
            self._epoch = epoch
            t0 = time.time()
            tr = self.train_epoch()
            res, _ = evaluate_store(self.model, self.val, self.device, self.val_uids, self.ann, self.exc,
                                    self.cfg["evaluation"]["max_marks_per_scan"])
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, "val_cpm": res.cpm,
                   "val_sens@1": res.sensitivity_at[1], "val_max_sens": res.max_sensitivity,
                   "lr": self.opt.param_groups[0]["lr"], "seconds": time.time() - t0}
            hist.append(row)
            improved = res.cpm > self.best
            if improved:
                self.best = res.cpm
            self.checkpoint("last.pt", epoch)
            if improved:
                self.checkpoint("best.pt", epoch)
            epochs.set_postfix(loss=tr["loss"], val_cpm=res.cpm, best=self.best)
            self.log.info(f"epoch {epoch:02d}/{E} [{text_bar(epoch, E)}] loss {tr['loss']:.4f} (focal {tr['focal']:.4f} loc {tr['loc']:.4f}) "
                          f"val CPM {res.cpm:.4f} sens@1 {res.sensitivity_at[1]:.3f} {'*best*' if improved else ''} {row['seconds']:.0f}s")
            if self.sched:
                self.sched.step()
        pd.DataFrame(hist).to_csv(self.out / "history.csv", index=False)
        return hist
