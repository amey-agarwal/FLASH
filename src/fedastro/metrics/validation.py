"""Validation scores of every trained model (CLAUDE.md Phase H8, rule 9).

Hierarchical topologies give each node a frozen validation split (data/partition.py). Every
model a run trains is scored on each node's own validation set:
- ``centralized``: the one centralized model;
- ``local``: each node's own local-only model (pooled = each node scored with its own model);
- ``flat_federated``: the flat federated arm's final global model;
- ``regional`` / ``global``: the hierarchical arm's models (a node scored with its region's).

This section is the **only** input to hyperparameter selection (tuning.py). It never reads a
test set. Validation is never used for anything that is reported as a result.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from fedastro.data.partition import HierFederation
from fedastro.metrics.classification import evaluate

SELECTION_METRIC = "macro_f1"
# Which validation model selects each arm's hyperparameters (PREREG section 5).
ARM_MODEL = {"centralized": "centralized", "local_only": "local", "federated": "flat_federated",
             "hierarchical": "global"}


def _scores(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    m = evaluate(y, proba)
    m.pop("recall_per_class")
    return {"n_val": int(len(y)), **m}


def validation_section(fed: HierFederation, region_of: dict[str, str],
                       predictors: dict[str, Callable[[str, np.ndarray], np.ndarray]]) -> dict[str, Any]:
    """``predictors[model](node, X)`` gives that model's class probabilities for raw inputs X
    at ``node`` (the model applies its own scaler)."""
    nodes = list(fed.nodes)
    ys = {n: fed.nodes[n].val.y for n in nodes}
    models: dict[str, Any] = {}
    for name, predict in predictors.items():
        proba = {n: predict(n, fed.nodes[n].val.X) for n in nodes}

        def pooled(group: list[str]) -> dict[str, Any]:
            return _scores(np.concatenate([ys[n] for n in group]), np.concatenate([proba[n] for n in group]))

        regions: dict[str, list[str]] = {}
        for n in nodes:
            regions.setdefault(region_of[n], []).append(n)
        models[name] = {"pooled": pooled(nodes), "per_region": {r: pooled(g) for r, g in regions.items()},
                        "per_node": {n: _scores(ys[n], proba[n]) for n in nodes}}
    return {"models": models, "selection_metric": SELECTION_METRIC, "arm_model": ARM_MODEL,
            "scope": "each node's own validation split (never test data); input to hyperparameter selection only"}


__all__ = ["validation_section", "ARM_MODEL", "SELECTION_METRIC"]
