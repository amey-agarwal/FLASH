"""Paired statistics for "federated beats X" claims (CLAUDE2.md rule 12).

A pair is one run, i.e. one (seed, holdout draw): both arms trained on the same silos and
scored on the same held-out set. For each comparator and metric the difference
federated - comparator is taken per run, and summarised by its mean and a 95%
hierarchical percentile bootstrap CI (``aggregate.bootstrap_ci``: resample seeds, then
draws within each seed). The primary comparison is the pre-registered one
(``evaluation.primary``); every other comparison is exploratory, and the number of
comparisons is reported. No p-values.
"""

from __future__ import annotations

from typing import Any

from fedastro.config import COMPARATORS
from fedastro.metrics.classification import METRICS


def arm_metric(result: dict[str, Any], arm: str, metric: str) -> float | None:
    """The arm's final held-out metric; local_only is its unweighted mean over silos."""
    a = result["arms"].get(arm) or {}
    value = (a.get("mean") or {}).get(metric) if arm == "local_only" else (a.get("final") or {}).get(metric)
    return value


def holdout_draw(result: dict[str, Any]) -> int:
    return (result.get("evaluation") or {}).get("holdout_draw", 0)


def paired_comparisons(results: list[dict[str, Any]], primary: str = "") -> dict[str, Any] | None:
    """The ``stats`` block, or None when the federated arm or every comparator is missing."""
    from fedastro.metrics.aggregate import bootstrap_ci

    if not all(arm_metric(r, "federated", "accuracy") is not None for r in results):
        return None
    comparisons = []
    for b in COMPARATORS:
        for metric in METRICS:
            diffs: dict[int, list[float]] = {}
            for r in sorted(results, key=lambda r: (r["seed"], holdout_draw(r))):
                fa, fb = arm_metric(r, "federated", metric), arm_metric(r, b, metric)
                if fa is not None and fb is not None:
                    diffs.setdefault(r["seed"], []).append(fa - fb)
            pairs = [d for ds in diffs.values() for d in ds]
            if not pairs:
                continue
            comparisons.append({
                "a": "federated", "b": b, "metric": metric,
                "mean_diff": float(sum(pairs) / len(pairs)),
                "ci95": bootstrap_ci(list(diffs.values())) if len(pairs) > 1 else None,
                "n_pairs": len(pairs), "n_seeds": len(diffs),
                "n_a_better": sum(d > 0 for d in pairs),
                "role": "primary" if primary == f"federated-{b}:{metric}" else "exploratory",
            })
    if not comparisons:
        return None
    return {"primary": primary or None, "n_comparisons": len(comparisons),
            "ci_method": "hierarchical percentile bootstrap of the mean paired difference "
                         "(resample seeds, then draws within a seed), 2000 draws, fixed rng",
            "comparisons": comparisons}
