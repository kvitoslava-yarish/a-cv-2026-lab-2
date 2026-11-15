import math
import time

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .data import IGNORE
from .head import SegHead
from .metrics import SegMetrics


class TrainSet(Dataset):
    def __init__(self, feats, masks, rows):
        self.feats, self.masks, self.rows = feats, masks, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return torch.from_numpy(self.feats[self.rows[i]].copy()), torch.from_numpy(self.masks[i].copy())


class EvalSet(Dataset):
    def __init__(self, feats, rows, masks):
        self.feats, self.rows, self.masks = feats, rows, masks

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return torch.from_numpy(self.feats[self.rows[i]].copy()), torch.from_numpy(self.masks[i])


def eval_collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def warmup_cosine(warmup, total):
    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return fn


class LitSeg(pl.LightningModule):
    def __init__(self, cfg, n_layers, dim, curve_names=(), prefix=""):
        super().__init__()
        t = cfg["train"]
        self.t, self.prefix, self.curve_names = t, prefix, list(curve_names)
        self.history = []
        self.num_classes = cfg["num_classes"]
        self.head = SegHead(n_layers, dim, t["hidden"], cfg["num_classes"], t["dropout"])

    def training_step(self, batch, _):
        x, y = batch
        logits = self.head(x.float(), size=y.shape[-2:])
        loss = F.cross_entropy(logits, y.long(), ignore_index=IGNORE)
        self.log(f"{self.prefix}train/loss", loss)
        self.log(f"{self.prefix}train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def on_validation_epoch_start(self):
        self.meters = [SegMetrics(self.num_classes, device=self.device) for _ in self.curve_names]

    def validation_step(self, batch, _, dataloader_idx=0):
        x, masks = batch
        logits = self.head(x.float())
        for lg, m in zip(logits, masks):
            probs = F.interpolate(lg[None].float(), size=m.shape, mode="bilinear", align_corners=False).softmax(1)[0]
            self.meters[dataloader_idx].update(probs, m.to(self.device))

    def on_validation_epoch_end(self):
        point = {"step": self.global_step}
        for name, meter in zip(self.curve_names, self.meters):
            res = meter.compute()
            point[name] = res["miou"]
            for k in ("miou", "ece", "loss", "pixel_acc"):
                self.log(f"{self.prefix}{name}/{k}", res[k], add_dataloader_idx=False)
        self.history.append(point)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.head.parameters(), lr=self.t["lr"], weight_decay=self.t["weight_decay"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(self.t["warmup_steps"], self.t["max_steps"]))
        return [opt], [{"scheduler": sched, "interval": "step"}]


class WeightHistograms(pl.Callback):
    def __init__(self, every):
        self.every = every

    def _log(self, trainer, module):
        if not isinstance(trainer.logger, WandbLogger):
            return
        hists = {
            f"{module.prefix}weights/{n}": wandb.Histogram(p.detach().float().cpu().numpy())
            for n, p in module.head.named_parameters()
        }
        trainer.logger.experiment.log({**hists, "trainer/global_step": trainer.global_step})

    def on_train_batch_end(self, trainer, module, *args):
        if trainer.global_step % self.every == 0:
            self._log(trainer, module)

    def on_train_end(self, trainer, module):
        self._log(trainer, module)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_info():
    if not torch.cuda.is_available():
        return "cpu", 0.0
    return torch.cuda.get_device_name(), torch.cuda.max_memory_allocated() / 2**20


def train_head(feats, masks, rows, cfg, seed, run=None, prefix="", curves=None):
    """Train a fresh head on feats[rows] with masks aligned to rows.
    curves: {name: (feats, rows, masks)} evaluated every `curve_every` steps; returns (head, seconds, history)."""
    t = cfg["train"]
    pl.seed_everything(seed, workers=True, verbose=False)
    steps, every = t["max_steps"], min(t["curve_every"], t["max_steps"])
    train_set = TrainSet(feats, masks, rows)
    sampler = RandomSampler(train_set, replacement=True, num_samples=steps * t["batch_size"])
    train_loader = DataLoader(train_set, batch_size=t["batch_size"], sampler=sampler)
    curves = curves or {}
    val_loaders = [
        DataLoader(EvalSet(*c), batch_size=t["eval_batch_size"], collate_fn=eval_collate) for c in curves.values()
    ]
    model = LitSeg(cfg, n_layers=feats.shape[1], dim=feats.shape[2], curve_names=curves, prefix=prefix)
    logger = WandbLogger(experiment=run) if run is not None else False
    trainer = pl.Trainer(
        max_steps=steps,
        precision=t["precision"] if torch.cuda.is_available() else "32-true",
        logger=logger,
        callbacks=[WeightHistograms(every)] if run is not None else [],
        val_check_interval=every if val_loaders else None,
        limit_val_batches=1.0 if val_loaders else 0,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator=device().type,
        devices=1,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.fit(model, train_loader, val_loaders or None)
    head = model.head.to(device()).eval()
    return head, time.time() - t0, model.history


@torch.no_grad()
def iter_logits(heads, feats, rows, batch_size=16, size=None):
    dev = device()
    for s in range(0, len(rows), batch_size):
        x = torch.from_numpy(np.stack([feats[r] for r in rows[s:s + batch_size]])).to(dev).float()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            yield [h(x, size=size).float() for h in heads]


@torch.no_grad()
def evaluate(head, feats, rows, masks, cfg):
    meter = SegMetrics(cfg["num_classes"], device=device())
    bs = cfg["train"]["eval_batch_size"]
    for s, (logits,) in zip(range(0, len(rows), bs), iter_logits([head], feats, rows, bs)):
        for lg, m in zip(logits, masks[s:s + bs]):
            m = torch.from_numpy(m).to(lg.device)
            probs = F.interpolate(lg[None], size=m.shape, mode="bilinear", align_corners=False).softmax(1)[0]
            meter.update(probs, m)
    return meter.compute()
