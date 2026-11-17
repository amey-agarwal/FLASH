"""``result.model_report`` (CLAUDE.md Phase H6): one row per model of a run, for the model-
family comparison (hypothesis H4). Each run has one architecture (decision 5); the comparison
across logreg / MLP / 1D-CNN is a sweep (``variants:`` in configs/sweeps/model_families_*.yaml).

Row fields: accuracy, macro_f1, auc, ece, worst_node_accuracy (and the node), plus the run's
n_params and payload_bytes (one model's weights, float32, as sent per update).
- Hierarchical runs: every model scored on each node's own test set, pooled
  (``result.hierarchy.global``); worst node = the lowest per-node accuracy.
- Flat runs: each arm on the shared held-out set; worst node is defined only for local_only
  (the lowest per-silo accuracy), null otherwise (one shared test set, no node split).
"""

from __future__ import annotations

from typing import Any


def _row(m: dict[str, Any], ece: float | None) -> dict[str, Any]:
    return {"accuracy": m.get("accuracy"), "macro_f1": m.get("macro_f1"), "auc": m.get("auc"), "ece": ece}


def hier_report(arch: str, n_params: int, payload: int, metrics: dict[str, Any],
                calibration: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for model, m in metrics["global"].items():
        if model == "n_test":
            continue
        per_node = {n: v[model]["accuracy"] for n, v in metrics["per_node"].items()}
        worst = min(per_node, key=per_node.get)
        rows[model] = {**_row(m, calibration["ece"].get(model)), "worst_node_accuracy": per_node[worst],
                       "worst_node": worst}
    return {"arch": arch, "n_params": n_params, "payload_bytes": payload, "rows": rows,
            "scope": "each node's own test set, pooled; worst node = lowest per-node accuracy"}


def flat_report(arch: str, n_params: int, payload: int, arms: dict[str, Any], cfg_arms: tuple[str, ...],
                calibration: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for arm in cfg_arms:
        a = arms[arm]
        if arm == "local_only":
            per = {s: v["final"]["accuracy"] for s, v in a["per_silo"].items()}
            worst = min(per, key=per.get)
            rows[arm] = {**_row(a["mean"], calibration["ece"].get(arm)), "worst_node_accuracy": per[worst],
                         "worst_node": worst}
        else:
            rows[arm] = {**_row(a["final"], calibration["ece"].get(arm)), "worst_node_accuracy": None,
                         "worst_node": None}
    return {"arch": arch, "n_params": n_params, "payload_bytes": payload, "rows": rows,
            "scope": "shared held-out set; worst node defined for local_only only (lowest per-silo accuracy)"}
