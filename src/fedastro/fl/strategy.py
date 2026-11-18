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
    epsilon: float | None = None


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


@dataclass(frozen=True)
class BufferedUpdate:
    update: ClientUpdate
    start: list[np.ndarray]  # the global weights the client trained from
    staleness: int  # server versions applied since the client was sent ``start``


class FedBuff:
    """Nguyen et al. 2022, buffered asynchronous aggregation. Each update contributes
    its delta (trained - start) scaled by staleness weight 1/sqrt(1 + staleness) and
    by its share of the buffer's examples; the server applies
    w <- w + server_lr * sum_i (n_i / sum n) * s(tau_i) * delta_i.
    The staleness factor is not renormalised, so stale buffers move the model less.
    Deterministic: float64, buffer order (= simulated arrival order)."""

    name = "fedbuff"

    def __init__(self, server_lr: float):
        if server_lr <= 0:
            raise ValueError("server_lr must be > 0")
        self.server_lr = server_lr

    @staticmethod
    def staleness_weight(staleness: int) -> float:
        return 1.0 / float(np.sqrt(1.0 + staleness))

    def aggregate(self, current: list[np.ndarray], buffer: list[BufferedUpdate]) -> list[np.ndarray]:
        if not buffer:
            raise ValueError("empty buffer")
        total = sum(b.update.num_examples for b in buffer)
        out = []
        for j, ref in enumerate(current):
            acc = ref.astype(np.float64)
            for b in buffer:
                w = self.server_lr * (b.update.num_examples / total) * self.staleness_weight(b.staleness)
                acc = acc + w * (b.update.arrays[j].astype(np.float64) - b.start[j].astype(np.float64))
            out.append(acc.astype(ref.dtype))
        return out


class ServerOpt:
    """Server optimizer on the round's aggregate (Phase H8; Reddi et al. 2021, "Adaptive
    Federated Optimization"). The pseudo-gradient is ``delta = aggregate - current``:
    - ``momentum`` (FedAvgM): v <- beta1 * v + delta; w <- current + lr * v.
      With beta1 = 0 and lr = 1 this is plain FedAvg.
    - ``adam`` (FedAdam): m <- beta1 * m + (1 - beta1) * delta;
      v <- beta2 * v + (1 - beta2) * delta^2; w <- current + lr * m / (sqrt(v) + tau).
      No bias correction, as in Reddi et al.
    State starts at zero and is kept across rounds by whoever holds the object (one per
    aggregator). Float64 throughout, cast back to each array's dtype."""

    def __init__(self, kind: str, lr: float, beta1: float, beta2: float, tau: float):
        if kind not in ("momentum", "adam"):
            raise ValueError(f"unknown server optimizer {kind!r}")
        self.kind, self.lr, self.beta1, self.beta2, self.tau = kind, lr, beta1, beta2, tau
        self.m: list[np.ndarray] | None = None
        self.v: list[np.ndarray] | None = None

    def step(self, current: list[np.ndarray], aggregate: list[np.ndarray]) -> list[np.ndarray]:
        deltas = [a.astype(np.float64) - c.astype(np.float64) for c, a in zip(current, aggregate)]
        if self.m is None:
            self.m = [np.zeros_like(d) for d in deltas]
            self.v = [np.zeros_like(d) for d in deltas]
        out = []
        for j, (c, d) in enumerate(zip(current, deltas)):
            if self.kind == "momentum":
                self.m[j] = self.beta1 * self.m[j] + d
                step = self.m[j]
            else:
                self.m[j] = self.beta1 * self.m[j] + (1 - self.beta1) * d
                self.v[j] = self.beta2 * self.v[j] + (1 - self.beta2) * d * d
                step = self.m[j] / (np.sqrt(self.v[j]) + self.tau)
            out.append((c.astype(np.float64) + self.lr * step).astype(c.dtype))
        return out


def build_server_opt(cfg: FLConfig) -> ServerOpt | None:
    """A fresh optimizer (zero state) for one aggregator, or None for plain aggregation."""
    if cfg.server_opt == "none":
        return None
    return ServerOpt(cfg.server_opt, cfg.server_opt_lr, cfg.server_beta1, cfg.server_beta2, cfg.server_tau)


def build_strategy(cfg: FLConfig) -> FedAvg:
    if cfg.strategy == "fedavg":
        return FedAvg()
    if cfg.strategy == "fedprox":
        return FedProx(cfg.proximal_mu)
    if cfg.strategy == "fedbuff":
        raise ValueError("fedbuff is asynchronous; the server uses FedBuff directly")
    raise ValueError(f"unknown strategy {cfg.strategy!r}")


def client_proximal_mu(cfg: FLConfig) -> float:
    """Proximal coefficient each client applies during local training (0 = none)."""
    return cfg.proximal_mu if cfg.strategy == "fedprox" else 0.0
