"""Global seeding for Python, NumPy and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> np.random.Generator:
    """Seed every RNG this project uses and return a NumPy Generator for local use.

    With ``deterministic=True`` PyTorch is told to use deterministic kernels and to
    raise if an op has no deterministic implementation. ``CUBLAS_WORKSPACE_CONFIG`` is
    set for CUDA determinism; it only takes effect if set before the first cuBLAS call.
    ``PYTHONHASHSEED`` cannot be changed after interpreter start, so nothing here relies
    on ``hash()`` ordering.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)
