"""Pieces shared by the baseline arms (and reused by the federated arm in Phase 3)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fedastro.config import ExperimentConfig
from fedastro.data.base import DatasetBundle
from fedastro.data.preprocess import Standardizer
from fedastro.metrics.classification import evaluate
from fedastro.models.model import build_model
from fedastro.training import predict_proba

# Stable integer ids for deriving per-arm seeds; never reorder.
ARM_IDS = {"centralized": 1, "local_only": 2, "federated": 3}


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    return requested


def init_model(cfg: ExperimentConfig, input_shape: tuple[int, ...], n_classes: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return build_model(cfg.model, input_shape, n_classes)


def holdout_evaluator(holdout: DatasetBundle, scaler: Standardizer, device: str):
    X = scaler.transform(holdout.X)
    y = holdout.y

    def _eval(model: nn.Module) -> dict:
        return evaluate(y, predict_proba(model, X, device))

    return _eval


def dp_plan(cfg: ExperimentConfig, n_train: int, epochs: int):
    """DP-SGD plan for one trained model, or None when privacy is off. ``epochs`` is
    the total the budget must cover (a federated client: rounds x local epochs)."""
    if not cfg.privacy.enabled:
        return None
    from fedastro.privacy.dp import plan_dp

    return plan_dp(cfg.privacy, n_train, cfg.train.batch_size, epochs)


def concat(bundles: list[DatasetBundle]) -> tuple[np.ndarray, np.ndarray]:
    return (np.concatenate([b.X for b in bundles]), np.concatenate([b.y for b in bundles]))
