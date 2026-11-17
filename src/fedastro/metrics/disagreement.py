"""Agreement between a node's local, regional and global models (hypothesis H3).

For every test example of a node, the three models' predicted classes either all agree
or not. Reported per node, per region (pooled over the region's nodes) and overall:
- ``rate``: fraction of examples where the three models do not all agree.
- ``pairwise``: disagreement rate of each pair of models.
- ``acc_agree`` / ``acc_disagree``: accuracy of each model on the agreeing and on the
  disagreeing examples (None when a subset is empty).

The headline (``result.disagreement.acc_agree`` / ``acc_disagree``) is the **regional**
model's accuracy on the pooled agreeing / disagreeing examples, fixed before any result
(Phase H3 decision 6).

Caveat (CLAUDE.md section 2): the three models share training data, so their agreement is
correlated. Disagreement is a flag for clinician review, never an independent second
opinion, and agreement is never independent confirmation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

MODELS = ("local", "regional", "global")
HEADLINE = "regional"


def _report(y: np.ndarray, preds: dict[str, np.ndarray]) -> dict[str, Any]:
    agree = (preds["local"] == preds["regional"]) & (preds["regional"] == preds["global"])
    n_dis = int((~agree).sum())

    def acc(mask: np.ndarray) -> dict[str, float | None]:
        return {m: (float(np.mean(preds[m][mask] == y[mask])) if mask.any() else None) for m in MODELS}

    return {
        "n": int(len(y)), "n_disagree": n_dis,
        "rate": float(n_dis / len(y)) if len(y) else None,
        "pairwise": {f"{a}-{b}": float(np.mean(preds[a] != preds[b])) for a, b in combinations(MODELS, 2)},
        "acc_agree": acc(agree), "acc_disagree": acc(~agree),
    }


def disagreement_report(ys: dict[str, np.ndarray], preds: dict[str, dict[str, np.ndarray]],
                        region_of: dict[str, str]) -> dict[str, Any]:
    """``ys[node]`` true labels and ``preds[node][model]`` predicted classes for each node's
    test set; ``region_of[node]`` its region. Returns the result's ``disagreement`` section."""
    def pooled(nodes: list[str]) -> dict[str, Any]:
        y = np.concatenate([ys[n] for n in nodes])
        return _report(y, {m: np.concatenate([preds[n][m] for n in nodes]) for m in MODELS})

    regions: dict[str, list[str]] = {}
    for node, region in region_of.items():
        regions.setdefault(region, []).append(node)
    overall = pooled(list(ys))
    return {
        "per_node": {n: _report(ys[n], preds[n]) for n in ys},
        "per_region": {r: pooled(nodes) for r, nodes in regions.items()},
        "global": overall,
        "headline_model": HEADLINE,
        "acc_agree": overall["acc_agree"][HEADLINE],
        "acc_disagree": overall["acc_disagree"][HEADLINE],
        "note": "The three models share training data, so agreement is correlated: disagreement is a "
                "flag for review, not an independent second opinion.",
    }
