"""Shared local training. Every arm, and in Phase 3 every FL client, trains through
``train_local`` so that the arms differ only in what data a model sees and how
weights are combined, never in the optimisation code."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedastro.config import TrainConfig

if TYPE_CHECKING:
    from fedastro.privacy.dp import DPPlan

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
    proximal_mu: float = 0.0,
    dp: DPPlan | None = None,
) -> list[dict]:
    """Train ``model`` in place for ``epochs`` (default ``cfg.epochs``) and return a
    per-epoch history. ``eval_fn``, if given, is called after each epoch and its
    metrics are added to the history; it is for reporting only and never used to pick
    a checkpoint (the final weights are always the last epoch's).

    ``proximal_mu > 0`` adds the FedProx term (mu / 2) * ||w - w_start||^2 to every
    batch loss, where w_start are the weights the model had on entry (the global
    model, for an FL client). With mu = 0 this is exactly plain local training.

    ``dp`` switches to DP-SGD (privacy/dp.py): Poisson-sampled batches, per-sample
    clipping and Gaussian noise. The model is trained in place either way."""
    if dp is not None and (proximal_mu > 0 or cfg.class_weighting != "none"):
        raise ValueError("DP-SGD supports neither the proximal term nor class weighting")
    model.to(device)
    # Seed the global RNG too (dropout draws from it), so a call is self-contained and
    # reproducible regardless of what ran earlier in this process or Ray actor.
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=cfg.batch_size, shuffle=True, generator=gen,
    )
    weight = class_weights(y, n_classes, cfg.class_weighting)
    loss_fn = nn.CrossEntropyLoss(weight=None if weight is None else weight.to(device))
    opt = make_optimizer(model, cfg)
    anchor = [p.detach().clone() for p in model.parameters()] if proximal_mu > 0 else None
    net = model
    if dp is not None:
        from fedastro.privacy.dp import make_private

        net, opt, loader = make_private(model, opt, loader, dp)
    try:
        return _epochs(net, model, loader, opt, loss_fn, anchor, proximal_mu,
                       epochs or cfg.epochs, device, eval_fn)
    finally:
        if dp is not None:
            net._close()  # remove Opacus hooks; ``model`` holds the trained weights


def _epochs(net, model, loader, opt, loss_fn, anchor, proximal_mu, n_epochs, device, eval_fn):
    history = []
    for epoch in range(1, n_epochs + 1):
        net.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            if len(yb) == 0:
                # Poisson sampling can draw an empty batch, with probability
                # (1 - 1/steps_per_epoch)^N, about e^-batch_size (< 1e-13 here).
                continue
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            if anchor is not None:
                prox = sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
                loss = loss + 0.5 * proximal_mu * prox
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        entry = {"epoch": epoch, "train_loss": total / n if n else None}
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
