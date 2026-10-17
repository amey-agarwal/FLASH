"""Server-side aggregation strategies.

Aggregation is done here rather than with Flower's built-in strategies so that it is
deterministic: updates are always combined in silo order, in float64, regardless of the
order in which replies arrive from the simulation engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedastro.config import FLConfig


@dataclass(frozen=True)
class ClientUpdate:
    silo_index: int
    silo: str
    arrays: list[np.ndarray]
    num_examples: int
    train_loss: float | None


class FedAvg:
    """McMahan et al. 2017: example-weighted mean of client models."""

    name = "fedavg"

    def aggregate(self, updates: list[ClientUpdate]) -> list[np.ndarray]:
        if not updates:
            raise ValueError("no client updates to aggregate")
        ordered = sorted(updates, key=lambda u: u.silo_index)
        total = sum(u.num_examples for u in ordered)
        out = []
        for i, ref in enumerate(ordered[0].arrays):
            acc = np.zeros(ref.shape, dtype=np.float64)
            for u in ordered:
                acc += u.arrays[i].astype(np.float64) * (u.num_examples / total)
            out.append(acc.astype(ref.dtype))
        return out


class FedProx(FedAvg):
    """Li et al. 2020. Server aggregation is identical to FedAvg; the difference is the
    client-side proximal term (mu / 2) * ||w - w_global||^2, applied in
    ``train_local`` via ``client_proximal_mu``."""

    name = "fedprox"

    def __init__(self, mu: float):
        if mu <= 0:
            raise ValueError("FedProx mu must be > 0")
        self.mu = mu


def build_strategy(cfg: FLConfig) -> FedAvg:
    if cfg.strategy == "fedavg":
        return FedAvg()
    if cfg.strategy == "fedprox":
        return FedProx(cfg.proximal_mu)
    raise ValueError(f"unknown strategy {cfg.strategy!r}")


def client_proximal_mu(cfg: FLConfig) -> float:
    """Proximal coefficient each client applies during local training (0 = none)."""
    return cfg.proximal_mu if cfg.strategy == "fedprox" else 0.0
