"""The centralized vs federated gap table (CLAUDE.md Phase H8).

Inputs: per model family, the evaluation runs of each arm at its selected configuration
(tuning.py), one run per evaluation seed. Each arm can have its own configuration (its own
learning rate), so the arms come from separate runs, paired by seed.

Per family and metric:
- ``centralized - federated``: centralized minus the flat federated arm's final global model;
- ``centralized - hierarchical``: centralized minus the hierarchical arm's global model, and
  ``centralized - hierarchical_regional``: minus each node scored with its region's model.
All on the same test data (every region's test data pooled; data/partition and
fl/hierarchy.pooled_test_set). Mean and std over seeds of the paired difference, and the
hierarchical percentile bootstrap 95% CI of metrics/aggregate.bootstrap_ci (one value per seed).
A positive gap = centralized is better. A negative gap is reported as it is.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fedastro.metrics.aggregate import bootstrap_ci

GAP_METRICS = ("accuracy", "macro_f1", "auc")
# (label, arm whose run holds it, key in result.arms[arm])
COMPARED = (("federated", "federated", "final"), ("hierarchical", "hierarchical", "final"),
            ("hierarchical_regional", "hierarchical", "regional_final"))


def _score(result: dict[str, Any], arm: str, key: str, metric: str) -> float | None:
    return ((result["arms"].get(arm) or {}).get(key) or {}).get(metric)


def _by_seed(results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out = {}
    for r in results:
        if r["seed"] in out:
            raise ValueError(f"two runs for seed {r['seed']} ({r['run_id']}, {out[r['seed']]['run_id']})")
        out[r["seed"]] = r
    return out


def gap_rows(family: str, runs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """``runs[arm]``: that arm's evaluation results (arm in centralized / federated /
    hierarchical). Seeds must match the centralized runs'."""
    cen = _by_seed(runs["centralized"])
    rows = []
    for label, arm, key in COMPARED:
        if arm not in runs:
            continue
        other = _by_seed(runs[arm])
        if set(other) != set(cen):
            raise ValueError(f"{family}: {arm} seeds {sorted(other)} != centralized seeds {sorted(cen)}")
        for metric in GAP_METRICS:
            pairs = [(cen[s]["arms"]["centralized"]["final"].get(metric), _score(other[s], arm, key, metric))
                     for s in sorted(cen)]
            diffs = [a - b for a, b in pairs if a is not None and b is not None]
            if not diffs:
                continue
            rows.append({"family": family, "comparison": f"centralized - {label}", "metric": metric,
                         "centralized": float(np.mean([a for a, _ in pairs if a is not None])),
                         "other": float(np.mean([b for _, b in pairs if b is not None])),
                         "gap_mean": float(np.mean(diffs)),
                         "gap_std": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
                         "ci95": bootstrap_ci([[d] for d in diffs]) if len(diffs) > 1 else None,
                         "n_seeds": len(diffs)})
    return rows


def format_gap(rows: list[dict[str, Any]]) -> str:
    lines = [f"{'family':<8} {'comparison':<38} {'metric':<9} {'centr.':>7} {'other':>7} {'gap':>8} "
             f"{'std':>7}  95% CI          n"]
    for r in rows:
        ci = "      -       " if r["ci95"] is None else f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
        std = "   -   " if r["gap_std"] is None else f"{r['gap_std']:7.3f}"
        lines.append(f"{r['family']:<8} {r['comparison']:<38} {r['metric']:<9} {r['centralized']:7.3f} "
                     f"{r['other']:7.3f} {r['gap_mean']:+8.3f} {std}  {ci}  {r['n_seeds']}")
    lines.append("gap = centralized minus the other arm (positive = centralized better); paired by seed; "
                 "CI = hierarchical percentile bootstrap over seeds")
    return "\n".join(lines)


__all__ = ["gap_rows", "format_gap", "GAP_METRICS"]
