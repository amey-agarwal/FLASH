"""Held-out-group evaluation (Phase 12): every arm's final model scored on each whole
held-out group, and on the held-out individuals of the training groups, separately.

Scoring uses each arm's final model on the held-out set only, after training, so it
cannot change what any arm learns. Runs only when the federation has held-out groups.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fedastro.baselines.common import init_model
from fedastro.baselines.ensemble import ensemble_proba, evaluate_subset
from fedastro.config import ExperimentConfig
from fedastro.data.partition import Federation
from fedastro.metrics.classification import METRICS
from fedastro.training import predict_proba


def _local_mean(per_silo: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": per_silo[0]["n"]}
    for m in METRICS:
        vals = [p[m] for p in per_silo if p[m] is not None]
        out[m] = float(np.mean(vals)) if vals else None
    return out


def arm_probabilities(cfg: ExperimentConfig, fed: Federation, models: dict[str, Any], capture: dict[str, Any],
                      device: str) -> dict[str, Any]:
    """Held-out class probabilities per arm. local_only maps silo -> probabilities."""
    X = fed.holdout.X
    out: dict[str, Any] = {}
    if "centralized" in models:
        model, scaler = models["centralized"]
        out["centralized"] = predict_proba(model, scaler.transform(X), device)
    if "local" in models:
        if "local_only" in cfg.arms:
            out["local_only"] = {name: predict_proba(m, s.transform(X), device) for name, m, s in models["local"]}
        if "ensemble_of_locals" in cfg.arms:
            out["ensemble_of_locals"] = ensemble_proba(models["local"], X, device)
    if "global_weights" in capture:
        from fedastro.fl.client import set_weights

        model = init_model(cfg, X.shape[1:], len(fed.holdout.class_names), seed=0).to(device)  # weights replaced
        set_weights(model, capture["global_weights"]["final"])
        out["federated"] = predict_proba(model, capture["scaler"].transform(X), device)
    return out


def holdout_group_report(fed: Federation, probas: dict[str, Any]) -> dict[str, Any]:
    """{"per_holdout_group": {group: {arm: metrics}}, "held_out_individuals": {arm: metrics} | None}.
    local_only is the unweighted mean over silos (each silo's own model), as in the arm."""
    holdout = fed.holdout
    groups = fed.groups["holdout_groups"]
    masks = {g: holdout.in_groups([g]) for g in groups}
    individuals = ~holdout.in_groups(groups)

    def score(mask: np.ndarray) -> dict[str, Any]:
        row = {}
        for arm, p in probas.items():
            if arm == "local_only":
                row[arm] = _local_mean([evaluate_subset(holdout, mask, q) for q in p.values()])
            else:
                row[arm] = evaluate_subset(holdout, mask, p)
        return row

    return {"per_holdout_group": {g: score(m) for g, m in masks.items()},
            "held_out_individuals": score(individuals) if individuals.any() else None}
