"""Gradient inversion against a single client update (Inverting Gradients-style).

Threat model, stated explicitly because the attack's success depends on it:
- An honest-but-curious server observes ONE released update from ONE silo: the
  gradient of one local step on one batch (a FedSGD-style update). Multi-step FedAvg
  updates are harder to invert; no result here is claimed for them.
- The server knows the architecture and the global weights it sent, the batch size,
  the clipping norm, and the batch's labels (labels are the attacker's best case;
  for batch size 1 they are recoverable from the last-layer bias gradient anyway).
- The client computes its step in training mode, so dropout is active and its mask
  is unknown to the attacker, as in real local training.
- With DP on, the released gradient is exactly what DP-SGD releases for that step:
  per-sample clipping to C, Gaussian noise of std sigma*C, averaging over the batch
  (Opacus DPOptimizer). sigma is calibrated for the attacked batch size B: it is the
  noise a silo training with batch B for the run's epochs needs to meet the run's
  target epsilon. B = 1 is therefore not judged against a batch-32 noise level.

Attack: optimise dummy inputs so that their gradient (clipped per sample when DP is
on, as the attacker knows C) has maximal cosine similarity with the released gradient
(Geiping et al. 2020), plus a weak Gaussian prior on standardised inputs
(prior_weight * mean(x^2)) in place of the image-domain total-variation prior.
Defaults were chosen on WESAD seed 0 without DP (PROGRESS.md Phase 5).

Fairness: batches are drawn by seed, never picked; every trial is reported; the
model states include the untrained ``init`` global model and the ``final`` trained
one; batch sizes include the real training batch size.

Scores (reconstructions are matched to true records by the Hungarian algorithm):
- ``mse``: mean squared error in standardised units.
- ``rel_mse``: mse / mean(x_true^2), i.e. relative to guessing the population mean
  (the scaler mean is the zero vector). < 1 means the update leaked information.
- ``corr``: Pearson correlation between reconstruction and record, over features
  (feature vectors) or over time per channel (sequences), averaged.
- ``nn_identified``: fraction of reconstructions whose nearest record in the silo's
  whole training set is the true record (re-identification of the window).

Every cell also carries ``chance``: the same scores for random N(0, 1) guesses (the
attack's starting point, i.e. an attacker that learned nothing), matched and scored
identically. Matching alone lifts chance re-identification well above 1/N at larger
batch sizes, so attack scores are only meaningful relative to it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.func import functional_call, grad, vmap

from fedastro.config import AttackConfig, ExperimentConfig
from fedastro.privacy.dp import DPPlan, plan_dp
from fedastro.training import derive_seed

ATTACK_ID = 4  # derive_seed namespace; ARM_IDS uses 1-3

THREAT_MODEL = {
    "observer": "honest-but-curious server",
    "observed": "one single-step gradient update from one silo",
    "attacker_knows": ["architecture", "global weights", "batch size", "clipping norm", "batch labels"],
    "client_mode": "training (dropout active, mask unknown to attacker)",
    "not_claimed": "inversion of multi-step FedAvg updates",
}


def released_gradient(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor,
                      dp: DPPlan | None, seed: int) -> list[torch.Tensor]:
    """The update a client would release for one step on (xb, yb)."""
    torch.manual_seed(seed)
    model.train()
    if dp is None:
        model.zero_grad()
        nn.functional.cross_entropy(model(xb), yb).backward()
        g = [p.grad.detach().clone() for p in model.parameters()]
        model.zero_grad()
        return g
    from opacus import GradSampleModule
    from opacus.optimizers import DPOptimizer

    gsm = GradSampleModule(model)
    opt = DPOptimizer(torch.optim.SGD(model.parameters(), lr=0.0),
                      noise_multiplier=dp.noise_multiplier, max_grad_norm=dp.max_grad_norm,
                      expected_batch_size=len(yb))
    try:
        opt.zero_grad()
        nn.functional.cross_entropy(gsm(xb), yb).backward()
        opt.pre_step()  # clip per sample, add noise, average: what DP-SGD applies
        return [p.grad.detach().clone() for p in model.parameters()]
    finally:
        opt.zero_grad()
        gsm._close()


def invert(model: nn.Module, target: list[torch.Tensor], yb: torch.Tensor, input_shape: tuple[int, ...],
           cfg: AttackConfig, clip_norm: float | None, seed: int) -> torch.Tensor:
    """Reconstruct a batch from its released gradient."""
    model.eval()
    params = {k: v.detach() for k, v in model.named_parameters()}
    gen = torch.Generator().manual_seed(seed)
    dummy = torch.randn((len(yb), *input_shape), generator=gen).requires_grad_(True)
    opt = torch.optim.Adam([dummy], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, [int(cfg.iterations * f) for f in (0.375, 0.625, 0.875)], gamma=0.1)
    t_flat = torch.cat([t.reshape(-1) for t in target])
    t_norm = t_flat.norm()

    def loss_one(p, x, y):
        return nn.functional.cross_entropy(functional_call(model, p, (x.unsqueeze(0),)), y.unsqueeze(0))

    per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))
    for _ in range(cfg.iterations):
        opt.zero_grad()
        g = per_sample(params, dummy, yb)
        flat = torch.cat([g[k].reshape(len(yb), -1) for k in params], dim=1)
        if clip_norm is not None:
            factor = (clip_norm / (flat.norm(dim=1, keepdim=True) + 1e-6)).clamp(max=1.0)
            flat = flat * factor
        d = flat.mean(dim=0)
        loss = 1 - (d @ t_flat) / (d.norm() * t_norm + 1e-12)
        if cfg.prior_weight > 0:
            loss = loss + cfg.prior_weight * dummy.pow(2).mean()
        loss.backward()
        opt.step()
        sched.step()
    return dummy.detach()


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r over the last axis, averaged over any leading (channel) axes."""
    a = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a[None]
    b = b.reshape(-1, b.shape[-1]) if b.ndim > 1 else b[None]
    rs = []
    for u, v in zip(a, b):
        su, sv = u.std(), v.std()
        rs.append(0.0 if su == 0 or sv == 0 else float(np.mean((u - u.mean()) * (v - v.mean())) / (su * sv)))
    return float(np.mean(rs))


def score(recon: np.ndarray, true: np.ndarray, true_idx: np.ndarray, pool: np.ndarray) -> dict[str, float]:
    n = len(true)
    r2, t2, p2 = recon.reshape(n, -1), true.reshape(n, -1), pool.reshape(len(pool), -1)
    cost = ((r2[:, None, :] - t2[None, :, :]) ** 2).mean(axis=2)
    rows, cols = linear_sum_assignment(cost)
    mse = float(cost[rows, cols].mean())
    nn_idx = ((r2[:, None, :] - p2[None, :, :]) ** 2).mean(axis=2).argmin(axis=1)
    return {
        "mse": mse,
        "rel_mse": mse / float((t2 ** 2).mean()),
        "corr": float(np.mean([_corr(recon[i], true[j]) for i, j in zip(rows, cols)])),
        "nn_identified": float(np.mean([nn_idx[i] == true_idx[j] for i, j in zip(rows, cols)])),
    }


SCORES = ("mse", "rel_mse", "corr", "nn_identified")


def chance_scores(silo_X: np.ndarray, idx: np.ndarray, draws: int, seed: int) -> dict[str, float]:
    """Mean score of ``draws`` random N(0, 1) guesses for the batch ``silo_X[idx]``."""
    rng = np.random.default_rng(seed)
    shape = (len(idx), *silo_X.shape[1:])
    runs = [score(rng.standard_normal(shape).astype(np.float32), silo_X[idx], idx, silo_X) for _ in range(draws)]
    return {k: float(np.mean([r[k] for r in runs])) for k in SCORES}


def _mean_std(vals: list[float]) -> dict[str, float | None]:
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}


def run_attack(cfg: ExperimentConfig, seed: int, silo_X: np.ndarray, silo_y: np.ndarray,
               global_weights: dict[str, list[np.ndarray]], model: nn.Module) -> dict[str, Any]:
    """Attack every (model state, batch size) cell for ``cfg.privacy.attack.trials``
    seeded batches. ``silo_X`` is already standardised with the federated scaler.
    ``model`` is a freshly built instance of the run's architecture."""
    # Thread count changes float reduction order, so it is restored afterwards: a later
    # run in the same process must not depend on whether an attack ran before it.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # Matching a noise-dominated DP gradient drives values into the float32 denormal
    # range, which is ~1000x slower on x86 CPUs. Flushing them to zero changes only
    # values below ~1e-38. torch has no getter; False is its default.
    torch.set_flush_denormal(True)
    try:
        return _run_attack(cfg, seed, silo_X, silo_y, global_weights, model)
    finally:
        torch.set_flush_denormal(False)
        torch.set_num_threads(threads)


def _run_attack(cfg: ExperimentConfig, seed: int, silo_X: np.ndarray, silo_y: np.ndarray,
                global_weights: dict[str, list[np.ndarray]], model: nn.Module) -> dict[str, Any]:
    from fedastro.fl.client import set_weights

    acfg = cfg.privacy.attack
    epochs = cfg.fl.rounds * cfg.fl.local_epochs
    X, y = torch.from_numpy(silo_X), torch.from_numpy(silo_y)
    cells = []
    for si, state in enumerate(acfg.model_states):
        set_weights(model, global_weights[state])
        for b in acfg.batch_sizes:
            if b > len(X):
                raise ValueError(f"attack batch size {b} exceeds silo size {len(X)}")
            dp = plan_dp(cfg.privacy, len(X), b, epochs) if cfg.privacy.enabled else None
            trials = []
            for t in range(acfg.trials):
                rng = np.random.default_rng(derive_seed(seed, ATTACK_ID, b, t))
                idx = np.sort(rng.choice(len(X), size=b, replace=False))
                target = released_gradient(model, X[idx], y[idx], dp, derive_seed(seed, ATTACK_ID, b, t, si, 1))
                recon = invert(model, target, y[idx], tuple(X.shape[1:]), acfg,
                               None if dp is None else dp.max_grad_norm, derive_seed(seed, ATTACK_ID, b, t, si, 2))
                chance = chance_scores(silo_X, idx, acfg.chance_draws, derive_seed(seed, ATTACK_ID, b, t, si, 3))
                trials.append({"indices": idx.tolist(), **score(recon.numpy(), silo_X[idx], idx, silo_X),
                               "chance": chance})
            cells.append({
                "state": state, "batch_size": b,
                "noise_multiplier": None if dp is None else dp.noise_multiplier,
                **{k: _mean_std([tr[k] for tr in trials]) for k in SCORES},
                "chance": {k: float(np.mean([tr["chance"][k] for tr in trials])) for k in SCORES},
                "trials": trials,
            })
    headline = next(c for c in cells if c["state"] == acfg.model_states[-1]
                    and c["batch_size"] == min(acfg.batch_sizes))
    return {
        "mse": headline["mse"]["mean"],
        "headline": {"state": headline["state"], "batch_size": headline["batch_size"],
                     "rule": "last model state, smallest batch size (worst case for the client)"},
        "threat_model": THREAT_MODEL,
        "n_silo_records": len(X),
        "settings": {"iterations": acfg.iterations, "lr": acfg.lr, "prior_weight": acfg.prior_weight,
                     "trials": acfg.trials, "chance_draws": acfg.chance_draws,
                     "loss": "cosine gradient distance"},
        "cells": cells,
    }
