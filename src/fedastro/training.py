"""Shared local training. Every arm, and in Phase 3 every FL client, trains through
``train_local`` so that the arms differ only in what data a model sees and how
weights are combined, never in the optimisation code."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedastro.config import TrainConfig

EvalFn = Callable[[nn.Module], dict]


def derive_seed(*keys: int) -> int:
    """Deterministic 63-bit seed from a tuple of integers (run seed, arm id, silo idx...).

    The key count is prepended because SeedSequence ignores trailing zeros, which would
    otherwise make (a, b) and (a, b, 0) collide.
    """
    entropy = [len(keys), *keys]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0]) >> 1


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)


def class_weights(y: np.ndarray, n_classes: int, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def train_local(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    n_classes: int,
    seed: int,
    device: str = "cpu",
    epochs: int | None = None,
    eval_fn: EvalFn | None = None,
) -> list[dict]:
    """Train ``model`` in place for ``epochs`` (default ``cfg.epochs``) and return a
    per-epoch history. ``eval_fn``, if given, is called after each epoch and its
    metrics are added to the history; it is for reporting only and never used to pick
    a checkpoint (the final weights are always the last epoch's)."""
    model.to(device)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=cfg.batch_size, shuffle=True, generator=gen,
    )
    weight = class_weights(y, n_classes, cfg.class_weighting)
    loss_fn = nn.CrossEntropyLoss(weight=None if weight is None else weight.to(device))
    opt = make_optimizer(model, cfg)
    history = []
    for epoch in range(1, (epochs or cfg.epochs) + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        entry = {"epoch": epoch, "train_loss": total / n}
        if eval_fn is not None:
            entry.update(eval_fn(model))
        history.append(entry)
    return history


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: str = "cpu", batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
