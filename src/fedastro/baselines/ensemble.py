"""ensemble_of_locals arm (Phase 12): the stronger no-federation baseline.

Each silo trains alone, exactly as in local_only (same seeds, same scalers, so the very
same models), and the held-out prediction is the unweighted mean of the local models'
predicted class probabilities. No weights are exchanged: only predictions are combined,
at test time. If federation does not beat this, its gain over local_only came from
combining models, not from training together.

Unweighted, like the local_only mean: each silo is one stakeholder. Under DP every local
model keeps its own record-level budget; averaging their outputs is post-processing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fedastro.baselines.local_only import run_local_only
from fedastro.config import ExperimentConfig
from fedastro.data.base import DatasetBundle
from fedastro.data.partition import Federation
from fedastro.metrics.classification import evaluate
from fedastro.training import predict_proba

WEIGHTING = "unweighted mean of the local models' predicted probabilities"


def ensemble_proba(local_models: list, X: np.ndarray, device: str) -> np.ndarray:
    """Mean predicted probabilities; each local model sees X through its own scaler."""
    return np.mean([predict_proba(model, scaler.transform(X), device) for _, model, scaler in local_models], axis=0)


def run_ensemble_of_locals(fed: Federation, cfg: ExperimentConfig, seed: int, device: str,
                           local_models: list | None = None,
                           local_arm: dict[str, Any] | None = None) -> tuple[dict[str, Any], list]:
    """Returns (arm, local models). ``local_models``/``local_arm`` from a local_only run
    of the same experiment are reused; otherwise the local models are trained here."""
    if local_models is None:
        local_models = []
        local_arm = run_local_only(fed, cfg, seed, device, models=local_models)
    arm: dict[str, Any] = {
        "final": evaluate(fed.holdout.y, ensemble_proba(local_models, fed.holdout.X, device)),
        "n_models": len(local_models),
        "weighting": WEIGHTING,
    }
    if local_arm is not None and cfg.privacy.enabled:
        arm["privacy"] = {name: s["privacy"] for name, s in local_arm["per_silo"].items()}
    return arm, local_models


def evaluate_subset(bundle: DatasetBundle, mask: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    """Metrics on the masked part of a held-out bundle, from already computed probabilities."""
    return {"n": int(mask.sum()), **evaluate(bundle.y[mask], proba[mask])}
