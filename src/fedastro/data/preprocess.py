"""Standardisation fit on training data only.

Statistics are kept as sufficient statistics (count, sum, sum of squares) so that
several parties' statistics can be combined exactly. Combining all silos' statistics
equals fitting on the pooled data, which is what the centralized arm does.
Features are standardised per column; sequences per channel (over windows and time).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-8


def _reduce_axes(X: np.ndarray) -> tuple[int, ...]:
    return (0,) if X.ndim == 2 else (0, 2)


@dataclass(frozen=True)
class Stats:
    count: float
    total: np.ndarray
    total_sq: np.ndarray

    @staticmethod
    def of(X: np.ndarray) -> "Stats":
        axes = _reduce_axes(X)
        X64 = X.astype(np.float64)
        count = float(np.prod([X.shape[a] for a in axes]))
        return Stats(count, X64.sum(axis=axes), (X64**2).sum(axis=axes))

    def __add__(self, other: "Stats") -> "Stats":
        return Stats(self.count + other.count, self.total + other.total,
                     self.total_sq + other.total_sq)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def from_stats(stats: Stats) -> "Standardizer":
        mean = stats.total / stats.count
        var = np.maximum(stats.total_sq / stats.count - mean**2, 0.0)
        return Standardizer(mean, np.sqrt(var))

    @staticmethod
    def fit(X: np.ndarray) -> "Standardizer":
        return Standardizer.from_stats(Stats.of(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        shape = (1, -1) if X.ndim == 2 else (1, -1, 1)
        out = (X - self.mean.reshape(shape)) / (self.std.reshape(shape) + EPS)
        return out.astype(np.float32)
