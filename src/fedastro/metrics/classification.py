"""Classification metrics. Accuracy alone misleads under WESAD class imbalance, so
macro-F1, balanced accuracy and macro one-vs-rest AUC are always reported alongside."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "auc")


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float | None]:
    """``proba`` has shape (n, n_classes). AUC is the macro mean of one-vs-rest AUCs over
    classes that have both positives and negatives in ``y_true`` (None if none do)."""
    n_classes = proba.shape[1]
    y_pred = proba.argmax(axis=1)
    labels = list(range(n_classes))
    aucs = []
    for k in labels:
        pos = y_true == k
        if pos.any() and (~pos).any():
            aucs.append(roc_auc_score(pos, proba[:, k]))
    return {
        "accuracy": float(np.mean(y_pred == y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "auc": float(np.mean(aucs)) if aucs else None,
    }
