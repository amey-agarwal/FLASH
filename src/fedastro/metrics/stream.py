"""Test-then-train metrics (CLAUDE.md Phase H7): each streamed example is scored once, on
arrival, before any model has trained on it (fl/stream.py). Scope: every node's training
split, streamed; never the fixed test sets, which keep the fixed-split results.

Per model (local, regional, global, and deployed_* with the update gate):
- ``rounds``: accuracy over the examples arriving in each round, every node pooled;
- ``per_node`` / ``per_region`` / ``global``: accuracy, balanced accuracy, macro-F1, AUC over
  every example the model scored, pooled over the stream;
- ``n_scored``: how many examples it scored. A model a node had not received yet scores
  nothing, so models can be compared on different example sets; ``disagreement`` compares
  the three on the examples all three scored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fedastro.metrics.classification import evaluate
from fedastro.metrics.disagreement import MODELS, disagreement_report


def _models(records: dict[str, list[dict[str, Any]]]) -> list[str]:
    for recs in records.values():
        if recs:
            return list(recs[0]["proba"])
    return list(MODELS)


def _pool(recs: list[dict[str, Any]], model: str) -> tuple[np.ndarray, np.ndarray] | None:
    parts = [(r["y"], r["proba"][model]) for r in recs if r["proba"].get(model) is not None]
    if not parts:
        return None
    return np.concatenate([y for y, _ in parts]), np.concatenate([p for _, p in parts])


def _scores(recs: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_arrived": int(sum(len(r["y"]) for r in recs)), "n_scored": {}}
    for m in models:
        pooled = _pool(recs, m)
        out["n_scored"][m] = 0 if pooled is None else int(len(pooled[0]))
        if pooled is None:
            out[m] = None
            continue
        metrics = evaluate(*pooled)
        metrics.pop("recall_per_class")
        out[m] = metrics
    return out


def stream_report(records: dict[str, list[dict[str, Any]]], region_of: dict[str, str],
                           n_rounds: int) -> dict[str, Any]:
    """``records[node]``: one entry per round with arrivals: round, y, proba[model] (None if the
    node did not hold that model yet). Returns ``result.stream.test_then_train``."""
    models = _models(records)
    rounds = []
    for r in range(2, n_rounds + 1):
        recs = [x for n in records for x in records[n] if x["round"] == r]
        entry: dict[str, Any] = {"round": r, "n_arrived": int(sum(len(x["y"]) for x in recs)),
                                 "accuracy": {}, "n_scored": {}}
        for m in models:
            pooled = _pool(recs, m)
            entry["n_scored"][m] = 0 if pooled is None else int(len(pooled[0]))
            entry["accuracy"][m] = None if pooled is None else float(np.mean(pooled[1].argmax(axis=1) == pooled[0]))
        rounds.append(entry)
    regions: dict[str, list[str]] = {}
    for n, reg in region_of.items():
        regions.setdefault(reg, []).append(n)
    per_node = {n: {"region": region_of[n], **_scores(records[n], models)} for n in records}
    per_region = {reg: _scores([x for n in nodes for x in records[n]], models) for reg, nodes in regions.items()}
    overall = _scores([x for n in records for x in records[n]], models)
    # H3 on the stream: the examples each of local, regional and global scored
    ys, preds = {}, {}
    for n, recs in records.items():
        full = [x for x in recs if all(x["proba"].get(m) is not None for m in MODELS)]
        ys[n] = np.concatenate([x["y"] for x in full]) if full else np.zeros(0, dtype=np.int64)
        preds[n] = {m: (np.concatenate([x["proba"][m].argmax(axis=1) for x in full]) if full
                        else np.zeros(0, dtype=np.int64)) for m in MODELS}
    keep = [n for n in ys if len(ys[n])]
    dis = (disagreement_report({n: ys[n] for n in keep}, {n: preds[n] for n in keep},
                               {n: region_of[n] for n in keep}) if keep else None)
    if dis is not None:
        dis["n_compared"] = int(sum(len(ys[n]) for n in keep))
        dis["scope"] = "streamed examples scored by all three of local, regional and global"
    return {"models": models, "rounds": rounds, "per_node": per_node, "per_region": per_region,
            "global": overall, "disagreement": dis,
            "scope": "each node's training split, streamed; every example scored once, on arrival, before "
                     "any model trained on it"}


__all__ = ["stream_report"]
