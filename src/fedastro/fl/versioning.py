"""On-the-fly model updates with a validation gate (CLAUDE.md Phase H5).

Every node of the hierarchical arm holds two **deployed** model versions, one per track:
- ``regional``: the latest version of its region's model the node accepted;
- ``global``: the latest version of the global model the node accepted.
Version 0 of both tracks is the node's own local-only model (Phase H5 decision 2): a federated
version must be no worse than the model the node already has.

When a new version reaches a node (decision 4: a new regional model goes to the nodes that
took part in that round; a new global model to the nodes of every region that received the
merge and were in contact that round), the node scores it on its **own validation set** and
deploys it only if

    val_after >= val_before - tolerance,

with ``val_before`` the deployed version's score. Otherwise the version is rejected, kept
out (the deployed version stays) and logged. Metric: macro-F1. Tolerance (PREREG.md
section 4): ``k`` x the bootstrap standard error of the deployed version's validation
macro-F1 (``n_bootstrap`` seeded resamples of the validation set), recomputed whenever a
new version is deployed. Only validation data is used (CLAUDE.md rule 9).

The gate controls deployment only (decision 1): training, aggregation and every H3/H4 result
are unchanged with it on. ``updates.inject`` adds a SCRIPTED fault for testing the gate:
the copy of a version delivered to one region's nodes in one round is degraded (decision 5);
the aggregators' models are not.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.metrics import f1_score

from fedastro.config import UPDATE_TRACKS, UpdatesConfig
from fedastro.training import derive_seed

VERSIONING_ID = 8  # derive_seed namespace (ARM_IDS 1-3, attack 4, COMMS 5, HIER_LINK 6, SECAGG 7)


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over every class (absent classes score 0, as in metrics/classification.py)."""
    with warnings.catch_warnings():  # tiny resamples often miss a class; that is expected here
        warnings.simplefilter("ignore", UserWarning)
        return float(f1_score(y, pred, labels=list(range(n_classes)), average="macro", zero_division=0))


def bootstrap_se(y: np.ndarray, pred: np.ndarray, n_classes: int, n_boot: int, seed: int) -> float:
    """Standard error of macro-F1 over ``n_boot`` seeded resamples (with replacement)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    scores = [macro_f1(y[ix], pred[ix], n_classes) for ix in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.std(scores, ddof=1))


def degrade(weights: list[np.ndarray], kind: str, scale: float, seed: int) -> list[np.ndarray]:
    """The SCRIPTED fault of ``updates.inject``: a degraded copy (the input is not changed)."""
    if kind == "zero":
        return [np.zeros_like(w) for w in weights]
    rng = np.random.default_rng(seed)
    out = []
    for w in weights:
        sd = float(np.std(w)) if w.size > 1 else float(abs(w).max(initial=0.0))
        out.append((w + scale * (sd or 1.0) * rng.standard_normal(w.shape)).astype(w.dtype))
    return out


@dataclass
class Deployed:
    """A node's deployed version on one track."""

    version: int  # 0 = the node's local-only model
    score: float  # validation macro-F1
    tolerance: float
    weights: list[np.ndarray] | None  # None = the local-only model


class UpdateGate:
    """Holds every node's deployed versions and applies the gate to incoming versions.

    ``predict(weights, node)`` returns the predicted classes of a model on the node's
    validation set (the caller scales the inputs). ``y_val[node]`` and ``local_pred[node]``
    are the node's validation labels and its local-only model's predictions on them."""

    def __init__(self, cfg: UpdatesConfig, seed: int, n_classes: int, nodes: list[str],
                 y_val: dict[str, np.ndarray], local_pred: dict[str, np.ndarray],
                 predict: Callable[[list[np.ndarray], str], np.ndarray], regions: list[str]):
        empty = [n for n in nodes if len(y_val[n]) == 0]
        if empty:
            raise ValueError(f"the update gate needs a validation set at every node; empty at {empty}")
        self.cfg, self.seed, self.n_classes = cfg, seed, n_classes
        self.nodes, self.regions = list(nodes), list(regions)
        self.y_val, self.predict = y_val, predict
        self.events: list[dict[str, Any]] = []
        self.deployed: dict[str, dict[str, Deployed]] = {}
        for n in self.nodes:
            score = macro_f1(y_val[n], local_pred[n], n_classes)
            self.deployed[n] = {t: Deployed(0, score, self._tolerance(n, t, 0, local_pred[n]), None)
                                for t in UPDATE_TRACKS}

    def _tolerance(self, node: str, track: str, version: int, pred: np.ndarray) -> float:
        seed = derive_seed(self.seed, VERSIONING_ID, self.nodes.index(node), UPDATE_TRACKS.index(track), version)
        return self.cfg.k * bootstrap_se(self.y_val[node], pred, self.n_classes, self.cfg.n_bootstrap, seed)

    def offer(self, rnd: int, track: str, region: str, version: int, weights: list[np.ndarray],
              recipients: list[str]) -> list[dict[str, Any]]:
        """A new ``track`` version (number ``version``) reaches ``recipients`` (nodes of
        ``region``) in round ``rnd``. Returns the round's accept/reject events."""
        inj = next((i for i in self.cfg.inject
                    if i.round == rnd and i.region == region and i.track == track), None)
        if inj is not None:
            weights = degrade(weights, inj.kind, inj.scale,
                              derive_seed(self.seed, VERSIONING_ID, 1_000 + self.regions.index(region), rnd,
                                          UPDATE_TRACKS.index(track)))
        out = []
        for node in recipients:
            cur = self.deployed[node][track]
            pred = self.predict(weights, node)
            after = macro_f1(self.y_val[node], pred, self.n_classes)
            accepted = after >= cur.score - cur.tolerance
            ev = {"round": rnd, "node": node, "region": region, "model": track, "version": version,
                  "version_before": cur.version, "accepted": bool(accepted), "val_before": cur.score,
                  "val_after": after, "tolerance_used": cur.tolerance, "injected": inj is not None}
            if accepted:
                self.deployed[node][track] = Deployed(version, after, self._tolerance(node, track, version, pred),
                                                      weights)
            out.append(ev)
        self.events.extend(out)
        return out

    def section(self) -> dict[str, Any]:
        """``result.updates``."""
        c = self.cfg
        rejected = [e for e in self.events if not e["accepted"]]
        return {
            "enabled": True,
            "tolerance": {"rule": c.tolerance_rule, "k": c.k, "n_bootstrap": c.n_bootstrap, "metric": c.metric,
                          "baseline_version": "0 = the node's local-only model",
                          "definition": "accept iff val_after >= val_before - k x bootstrap SE of the deployed "
                                        "version's validation macro-F1"},
            "events": self.events,
            "n_offered": len(self.events), "n_rejected": len(rejected),
            "n_injected": sum(e["injected"] for e in self.events),
            "deployed": {n: {t: d.version for t, d in tracks.items()} for n, tracks in self.deployed.items()},
            "inject": [{"round": i.round, "region": i.region, "track": i.track, "kind": i.kind, "scale": i.scale}
                       for i in c.inject],
            "note": ("The gate decides which version each node deploys; training and aggregation are "
                     "unchanged. Scores are on each node's own validation set only. Injected versions "
                     "are SCRIPTED faults for testing the gate."),
        }
