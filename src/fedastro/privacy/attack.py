"""Gradient inversion against a single client update (Inverting Gradients-style).

Threat model, stated explicitly because the attack's success depends on it:
- An honest-but-curious server observes ONE released update from ONE silo: the
  gradient of one local step on one batch (a FedSGD-style update). Multi-step FedAvg
  updates are harder to invert; no result here is claimed for them.
- The server knows the architecture and the global weights it sent, the batch size
  and the clipping norm. Labels are attacked two ways (``label_modes``): ``known``
  (the attacker's best case) and ``inferred`` from the last-layer bias gradient
  (``infer_labels``), with the label-recovery accuracy reported.
- Every silo is attacked (``silos``; empty = all): small silos carry more DP noise
  but have fewer records to hide among.
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
    "attacker_knows": ["architecture", "global weights", "batch size", "clipping norm",
                       "batch labels (label mode 'known' only; 'inferred' recovers them from the update)"],
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


def infer_labels(model: nn.Module, target: list[torch.Tensor], batch_size: int, n_classes: int,
                 input_shape: tuple[int, ...], seed: int, n_probe: int = 256) -> torch.Tensor:
    """Recover the batch's label counts from the released gradient, without knowing them.

    For softmax cross-entropy with mean reduction, the last layer's bias gradient is
    g_c = mean_i(p_ic - y_ic), so the count of class c is n_c = B * (mean_i p_ic - g_c).
    The batch's mean prediction is unknown; it is estimated by the model's mean
    prediction on random N(0, 1) inputs (standardised space), as in Geng et al. 2021.
    Counts are rounded to integers summing to B by largest remainder. Under DP the
    attacker cannot undo clipping or noise and uses the same formula anyway.
    The returned labels are a multiset; order is irrelevant because dummies are free.
    """
    bias_grad = target[-1]
    if bias_grad.shape != (n_classes,):
        raise ValueError("label inference needs the final layer's bias as the last parameter")
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    with torch.no_grad():
        p_bar = torch.softmax(model(torch.randn((n_probe, *input_shape), generator=gen)), dim=1).mean(0)
    raw = (batch_size * (p_bar - bias_grad)).clamp(min=0).double().numpy()
    if raw.sum() == 0:
        raw = np.ones(n_classes)
    raw = raw * batch_size / raw.sum()
    counts = np.floor(raw).astype(int)
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[: batch_size - counts.sum()]] += 1
    return torch.from_numpy(np.repeat(np.arange(n_classes), counts))


def label_accuracy(inferred: torch.Tensor, true: torch.Tensor, n_classes: int) -> float:
    """Fraction of the batch's labels recovered (multiset overlap / B)."""
    a = np.bincount(inferred.numpy(), minlength=n_classes)
    b = np.bincount(true.numpy(), minlength=n_classes)
    return float(np.minimum(a, b).sum() / len(true))


def run_attack(cfg: ExperimentConfig, seed: int, silos: dict[str, tuple[np.ndarray, np.ndarray]],
               global_weights: dict[str, list[np.ndarray]], model: nn.Module) -> dict[str, Any]:
    """Attack every (silo, model state, batch size, label mode) cell for
    ``cfg.privacy.attack.trials`` seeded batches. ``silos`` maps silo name (topology
    order) to (X standardised with the federated scaler, y). ``model`` is a freshly
    built instance of the run's architecture."""
    # Thread count changes float reduction order, so it is restored afterwards: a later
    # run in the same process must not depend on whether an attack ran before it.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # Matching a noise-dominated DP gradient drives values into the float32 denormal
    # range, which is ~1000x slower on x86 CPUs. Flushing them to zero changes only
    # values below ~1e-38. torch has no getter; False is its default.
    torch.set_flush_denormal(True)
    acfg = cfg.privacy.attack
    try:
        # Seeded by the silo's topology position, so a silo's batches do not depend on
        # which other silos are selected for attack.
        order = [s.name for s in cfg.topology.silos]
        out = {name: _attack_silo(cfg, seed, order.index(name), X, y, global_weights, model)
               for name, (X, y) in silos.items()}
    finally:
        torch.set_flush_denormal(False)
        torch.set_num_threads(threads)
    # Headline: the worst case for a client -- last model state, smallest batch, known
    # labels, and the silo where the attack did best (lowest mse).
    def head(name):
        return next(c for c in out[name]["cells"] if c["state"] == acfg.model_states[-1]
                    and c["batch_size"] == min(acfg.batch_sizes) and c["labels"] == acfg.label_modes[0])
    worst = min(out, key=lambda n: head(n)["mse"]["mean"])
    return {
        "mse": head(worst)["mse"]["mean"],
        "headline": {"silo": worst, "state": acfg.model_states[-1], "batch_size": min(acfg.batch_sizes),
                     "labels": acfg.label_modes[0],
                     "rule": "last model state, smallest batch, first label mode, silo with the "
                             "lowest mse (worst case for a client)"},
        "threat_model": THREAT_MODEL,
        "settings": {"iterations": acfg.iterations, "lr": acfg.lr, "prior_weight": acfg.prior_weight,
                     "trials": acfg.trials, "chance_draws": acfg.chance_draws,
                     "label_modes": list(acfg.label_modes), "loss": "cosine gradient distance"},
        "silos": out,
    }


def _attack_silo(cfg: ExperimentConfig, seed: int, silo_idx: int, silo_X: np.ndarray, silo_y: np.ndarray,
                 global_weights: dict[str, list[np.ndarray]], model: nn.Module) -> dict[str, Any]:
    from fedastro.fl.client import set_weights

    acfg = cfg.privacy.attack
    epochs = cfg.fl.rounds * cfg.fl.local_epochs
    n_classes = list(model.parameters())[-1].shape[0]  # final layer bias
    X, y = torch.from_numpy(silo_X), torch.from_numpy(silo_y)
    shape = tuple(X.shape[1:])
    cells = []
    for si, state in enumerate(acfg.model_states):
        set_weights(model, global_weights[state])
        for b in acfg.batch_sizes:
            if b > len(X):
                raise ValueError(f"attack batch size {b} exceeds silo size {len(X)}")
            dp = plan_dp(cfg.privacy, len(X), b, epochs) if cfg.privacy.enabled else None
            clip = None if dp is None else dp.max_grad_norm
            trials: dict[str, list[dict[str, Any]]] = {mode: [] for mode in acfg.label_modes}
            for t in range(acfg.trials):
                key = (seed, ATTACK_ID, silo_idx, b, t)
                rng = np.random.default_rng(derive_seed(*key))
                idx = np.sort(rng.choice(len(X), size=b, replace=False))
                # One released update per trial, attacked under every label mode (paired).
                target = released_gradient(model, X[idx], y[idx], dp, derive_seed(*key, si, 1))
                chance = chance_scores(silo_X, idx, acfg.chance_draws, derive_seed(*key, si, 3))
                for mode in acfg.label_modes:
                    extra: dict[str, Any] = {}
                    labels = y[idx]
                    if mode == "inferred":
                        labels = infer_labels(model, target, b, n_classes, shape, derive_seed(*key, si, 4))
                        extra["label_accuracy"] = label_accuracy(labels, y[idx], n_classes)
                    recon = invert(model, target, labels, shape, acfg, clip, derive_seed(*key, si, 2))
                    trials[mode].append({"indices": idx.tolist(),
                                         **score(recon.numpy(), silo_X[idx], idx, silo_X),
                                         **extra, "chance": chance})
            for mode, trs in trials.items():
                metrics = SCORES + (("label_accuracy",) if mode == "inferred" else ())
                cells.append({
                    "state": state, "batch_size": b, "labels": mode,
                    "noise_multiplier": None if dp is None else dp.noise_multiplier,
                    **{k: _mean_std([tr[k] for tr in trs]) for k in metrics},
                    "chance": {k: float(np.mean([tr["chance"][k] for tr in trs])) for k in SCORES},
                    "trials": trs,
                })
    return {"n_records": len(X), "cells": cells}
