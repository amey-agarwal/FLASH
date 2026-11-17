"""Calibration (CLAUDE.md Phase H6): top-label expected calibration error (ECE) and
reliability bins.

For each example, confidence = the largest predicted class probability, correct = whether
that class is the true one. Confidences are put in ``n_bins`` equal-width bins over [0, 1]
(the last bin includes 1.0); ECE = sum over bins of (bin size / n) x |accuracy - mean
confidence|. 10 bins (Phase H6 decision 2): test sets here are small, so more bins would
leave many nearly empty. Bin counts are always reported next to ECE, since ECE on a small
test set is noisy.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def reliability_bins(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        out.append({"lo": b / n_bins, "hi": (b + 1) / n_bins, "n": n,
                    "confidence": float(conf[m].mean()) if n else None,
                    "accuracy": float(correct[m].mean()) if n else None})
    return out


def ece(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float | None:
    if len(y) == 0:
        return None
    bins = reliability_bins(y, proba, n_bins)
    return float(sum(b["n"] * abs(b["accuracy"] - b["confidence"]) for b in bins if b["n"]) / len(y))


def calibration_section(models: dict[str, tuple[np.ndarray, np.ndarray]], n_bins: int, scope: str) -> dict[str, Any]:
    """``result.calibration``: models maps a model name to (y, probabilities)."""
    return {"enabled": True, "kind": "top-label", "n_bins": n_bins, "scope": scope,
            "ece": {m: ece(y, p, n_bins) for m, (y, p) in models.items()},
            "reliability": {m: reliability_bins(y, p, n_bins) for m, (y, p) in models.items()}}
