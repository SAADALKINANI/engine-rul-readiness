"""Training loop with early stopping, cosine LR schedule and MLflow tracking."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import rmse


RUL_SCALE = 125.0     # models regress RUL / 125 (unit-scale targets train far more stably)


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    min_epochs_frac: float = 0.5     # never early-stop before the LR has annealed
    warmup_epochs: int = 3
    seed: int = 0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(X[i:i + batch])) for i in range(0, len(X), batch)]
    return torch.cat(out).numpy() * RUL_SCALE


def fit(model: nn.Module, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray,
        cfg: TrainConfig, log=None) -> tuple[nn.Module, dict]:
    set_seed(cfg.seed)
    dl = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y / RUL_SCALE)),
                    batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total, warm = cfg.epochs * len(dl), cfg.warmup_epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total))))
    loss_fn = nn.HuberLoss(delta=10.0 / RUL_SCALE)      # robust to the noisy plateau region of the RUL target
    min_epochs = int(cfg.epochs * cfg.min_epochs_frac)
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(xb)
        val = rmse(yv, predict(model, Xv))
        hist.append({"epoch": ep, "train_loss": tot / len(X), "val_rmse": val})
        if log:
            log(ep, tot / len(X), val)
        if val < best - 1e-3:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience and ep >= min_epochs:
                break
    model.load_state_dict(best_state)
    return model, {"best_val_rmse": best, "epochs_run": len(hist), "train_seconds": time.time() - t0,
                   "history": hist}
