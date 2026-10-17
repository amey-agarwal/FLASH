"""Centralized arm (upper bound): all silos' training data pooled, trained normally.

The scaler is fit on the pooled training data only; the holdout is never seen in
training or preprocessing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fedastro.baselines.common import ARM_IDS, concat, dp_plan, holdout_evaluator, init_model
from fedastro.config import ExperimentConfig
from fedastro.data.partition import Federation
from fedastro.data.preprocess import Standardizer
from fedastro.models.model import count_parameters
from fedastro.training import derive_seed, train_local


def run_centralized(fed: Federation, cfg: ExperimentConfig, seed: int, device: str) -> dict[str, Any]:
    X, y = concat([s.data for s in fed.silos.values()])
    n_classes = len(fed.holdout.class_names)
    scaler = Standardizer.fit(X)
    arm_seed = derive_seed(seed, ARM_IDS["centralized"])
    model = init_model(cfg, X.shape[1:], n_classes, arm_seed)
    dp = dp_plan(cfg, len(y), cfg.train.epochs)
    history = train_local(
        model, scaler.transform(X), y, cfg.train, n_classes, arm_seed, device,
        eval_fn=holdout_evaluator(fed.holdout, scaler, device), dp=dp,
    )
    final = {k: v for k, v in history[-1].items() if k not in ("epoch", "train_loss")}
    out = {"final": final, "history": history, "n_train": len(y), "n_params": count_parameters(model)}
    if dp is not None:
        from fedastro.privacy.dp import records_per_subject

        subjects = np.concatenate([s.data.subject_id for s in fed.silos.values()])
        out["privacy"] = dp.to_dict(records_per_subject(subjects))
    return out
