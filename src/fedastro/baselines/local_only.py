"""Local-only arm (lower bound): each silo trains alone on its own data, with its own
scaler, and is evaluated on the shared global holdout. Every silo uses the same
TrainConfig (same epochs), so small silos take fewer optimisation steps; that is part
of what being a small silo means."""

from __future__ import annotations

from typing import Any

import numpy as np

from fedastro.baselines.common import ARM_IDS, dp_plan, holdout_evaluator, init_model
from fedastro.config import ExperimentConfig
from fedastro.data.partition import Federation
from fedastro.data.preprocess import Standardizer
from fedastro.metrics.classification import METRICS
from fedastro.training import derive_seed, train_local


def run_local_only(fed: Federation, cfg: ExperimentConfig, seed: int, device: str) -> dict[str, Any]:
    n_classes = len(fed.holdout.class_names)
    per_silo: dict[str, Any] = {}
    for idx, (name, split) in enumerate(fed.silos.items()):
        data = split.data
        scaler = Standardizer.fit(data.X)
        silo_seed = derive_seed(seed, ARM_IDS["local_only"], idx)
        model = init_model(cfg, data.X.shape[1:], n_classes, silo_seed)
        dp = dp_plan(cfg, len(data), cfg.train.epochs)
        history = train_local(
            model, scaler.transform(data.X), data.y, cfg.train, n_classes, silo_seed, device,
            eval_fn=holdout_evaluator(fed.holdout, scaler, device), dp=dp,
        )
        final = {k: v for k, v in history[-1].items() if k not in ("epoch", "train_loss")}
        per_silo[name] = {"final": final, "history": history, "n_train": len(data)}
        if dp is not None:
            per_silo[name]["privacy"] = dp.to_dict()
    return {"per_silo": per_silo, **summarise_silos(per_silo)}


def summarise_silos(per_silo: dict[str, Any]) -> dict[str, Any]:
    """Unweighted mean/std/min/max across silos for each metric."""
    out: dict[str, dict[str, float | None]] = {"mean": {}, "std_across_silos": {}, "min": {}, "max": {}}
    for m in METRICS:
        vals = [s["final"][m] for s in per_silo.values() if s["final"][m] is not None]
        if not vals:
            for key in out:
                out[key][m] = None
            continue
        out["mean"][m] = float(np.mean(vals))
        out["std_across_silos"][m] = float(np.std(vals))
        out["min"][m] = float(np.min(vals))
        out["max"][m] = float(np.max(vals))
    return out
