"""Per-silo fairness: how well one model serves each silo's case mix.

There is no subject-disjoint per-silo test data (silos hold 1-5 subjects, and
overlapping windows of the same subject would leak). So each silo's accuracy is
estimated on the shared, subject-disjoint holdout, re-weighted to that silo's own
training label distribution:

    acc_silo = sum_k  p_silo(k) * recall_k(holdout)

This captures label-mix (prior) heterogeneity across silos exactly; it does NOT capture
feature heterogeneity between subjects. Spread = max - min across silos; lower is fairer.
For local_only, each silo is scored with its own model; for centralized and federated,
every silo is scored with the single shared model.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def label_mix_accuracy(recall_per_class: list[float], label_counts: dict[str, int],
                       class_names: list[str]) -> float:
    counts = np.array([label_counts[c] for c in class_names], dtype=np.float64)
    if counts.sum() == 0:
        raise ValueError("silo has no training windows")
    return float(np.dot(counts / counts.sum(), np.asarray(recall_per_class, dtype=np.float64)))


def summarise(per_silo: dict[str, float]) -> dict[str, Any]:
    vals = np.array(list(per_silo.values()))
    return {
        "per_silo": per_silo,
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "spread": float(vals.max() - vals.min()),
    }


def fairness_report(result: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    """Fairness section for every arm that ran, computed from final metrics."""
    silos = {s["name"]: s["label_counts"] for s in result["silos"]}
    arms = result["arms"]
    out: dict[str, Any] = {
        "metric": "holdout recall re-weighted to each silo's training label mix",
    }
    for arm in ("centralized", "federated"):
        recall = arms[arm]["final"].get("recall_per_class")
        if recall is not None:
            out[arm] = summarise({n: label_mix_accuracy(recall, lc, class_names) for n, lc in silos.items()})
    if arms["local_only"]["per_silo"]:
        out["local_only"] = summarise({
            n: label_mix_accuracy(arms["local_only"]["per_silo"][n]["final"]["recall_per_class"], lc, class_names)
            for n, lc in silos.items()
        })
    return out
