"""Test-then-train run of the hierarchy (CLAUDE.md Phase H7, the ``hierarchical_stream`` arm).

The same regional / global schedule as the hierarchical arm (fl/hierarchy.run_rounds), on
the time-ordered streams of data/stream.py. This object is the ``stream`` hook of
``run_rounds``. Per round r:

1. ``begin_round(r)``: every example arriving at a node in round r is predicted by
   - ``local``: the node's own local model, trained through round r - 1;
   - ``regional``: the last regional model **delivered to that node** (a node in loss of signal
     or in a fallback region keeps an older one);
   - ``global``: the last global model delivered to that node;
   - with the update gate on, ``deployed_regional`` / ``deployed_global``: the versions the node
     deployed (version 0 = its local model after the warm-up round).
   A model the node has not received yet (no regional before its first successful regional
   round, no global before the first merge reaches it) predicts nothing; those examples are
   counted but left out of that model's metrics. Then each local model trains for
   ``fl.local_epochs`` on the node's labels usable in round r.
2. The nodes train the regional models on the same labels (fl/client.py with ``streams``).
3. Each new regional / global version is ``deliver``-ed to the nodes that receive it (the same
   recipients as the update gate's, Phase H5 decision 4).

Every example is scored before any model trains on it. ``audit()`` checks this from the actual
logs: the round each example was scored, the rounds the local models trained on it, and the
schedule the clients train from.

Local stream models use their own scaler (fit on the node's warm-up block), the federated ones
the federated scaler (warm-up statistics only, fl/client.py). Nothing here uses validation or
test data except the gate (validation, as in Phase H5).
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from fedastro.baselines.common import init_model
from fedastro.config import ExperimentConfig
from fedastro.data.base import DatasetBundle
from fedastro.data.preprocess import Standardizer
from fedastro.data.stream import STREAM_ID, NodeStream
from fedastro.training import derive_seed, predict_proba, train_local

TRACKS = ("regional", "global")


class StreamRun:
    """``fed_proba(weights, X_scaled)`` gives a federated model's class probabilities;
    ``fed_scaler`` standardises inputs for it."""

    def __init__(self, cfg: ExperimentConfig, seed: int, device: str, train: dict[str, DatasetBundle],
                 streams: dict[str, NodeStream], fed_scaler: Standardizer,
                 fed_proba: Callable[[list[np.ndarray], np.ndarray], np.ndarray]):
        self.cfg, self.seed, self.device = cfg, seed, device
        self.names = list(train)
        self.train, self.streams = train, streams
        self.fed_scaler, self.fed_proba = fed_scaler, fed_proba
        first = train[self.names[0]]
        self.input_shape, self.n_classes = first.X.shape[1:], len(first.class_names)
        self.held: dict[str, dict[str, list[np.ndarray] | None]] = {n: {t: None for t in TRACKS} for n in self.names}
        self.gate = None  # set by the caller when updates.enabled (after local_v0 exists)
        self.records: dict[str, list[dict[str, Any]]] = {n: [] for n in self.names}
        self.trained_log: dict[str, list[tuple[int, np.ndarray]]] = {n: [] for n in self.names}
        self.local: dict[str, tuple[Any, Standardizer]] = {}
        for i, n in enumerate(self.names):
            warm = train[n].subset(streams[n].arrival == 0)
            scaler = Standardizer.fit(warm.X)
            model = init_model(cfg, self.input_shape, self.n_classes, derive_seed(seed, STREAM_ID, i)).to(device)
            self.local[n] = (model, scaler)
        self._train_local(1)  # round 1: the warm-up block (nothing arrives in round 1)
        self.local_v0 = {n: (copy.deepcopy(m), sc) for n, (m, sc) in self.local.items()}
        self.last_round = 1

    # -- hooks called by run_rounds ------------------------------------------------------

    def n_train(self, rnd: int) -> dict[str, int]:
        return {n: int(self.streams[n].train_mask(rnd).sum()) for n in self.names}

    def begin_round(self, rnd: int) -> None:
        if rnd == 1:
            return  # warm-up already trained in __init__
        if rnd != self.last_round + 1:
            raise RuntimeError(f"stream rounds out of order: {rnd} after {self.last_round}")
        for n in self.names:
            idx = self.streams[n].arriving(rnd)
            if len(idx):
                self.records[n].append({"round": rnd, "idx": idx, "y": self.train[n].y[idx],
                                        "proba": self._predict(n, self.train[n].X[idx])})
        self._train_local(rnd)
        self.last_round = rnd

    def deliver(self, track: str, weights: list[np.ndarray], recipients: list[str]) -> None:
        for n in recipients:
            self.held[n][track] = weights

    # -- internals -----------------------------------------------------------------------

    def _predict(self, node: str, X: np.ndarray) -> dict[str, np.ndarray | None]:
        model, scaler = self.local[node]
        out: dict[str, np.ndarray | None] = {"local": predict_proba(model, scaler.transform(X), self.device)}
        Xf = self.fed_scaler.transform(X)
        for t in TRACKS:
            w = self.held[node][t]
            out[t] = None if w is None else self.fed_proba(w, Xf)
        if self.gate is not None:
            for t in TRACKS:
                d = self.gate.deployed[node][t]
                if d.weights is None:
                    m0, sc0 = self.local_v0[node]
                    out[f"deployed_{t}"] = predict_proba(m0, sc0.transform(X), self.device)
                else:
                    out[f"deployed_{t}"] = self.fed_proba(d.weights, Xf)
        return out

    def _train_local(self, rnd: int) -> None:
        for i, n in enumerate(self.names):
            mask = self.streams[n].train_mask(rnd)
            idx = np.flatnonzero(mask)
            model, scaler = self.local[n]
            data = self.train[n]
            train_local(model, scaler.transform(data.X[idx]), data.y[idx], self.cfg.train, self.n_classes,
                        seed=derive_seed(self.seed, STREAM_ID, i, rnd), device=self.device,
                        epochs=self.cfg.fl.local_epochs)
            self.trained_log[n].append((rnd, idx))

    # -- outputs -------------------------------------------------------------------------

    def local_v0_classes(self, node: str, X: np.ndarray) -> np.ndarray:
        m, sc = self.local_v0[node]
        return predict_proba(m, sc.transform(X), self.device).argmax(axis=1)

    def final_local_models(self) -> dict[str, tuple[Any, Standardizer]]:
        return dict(self.local)

    def audit(self) -> dict[str, Any]:
        """Checks, from the logs, that no example was scored after a model trained on it.
        Scoring in round r happens before any training in round r, so an example trained on
        in its own arrival round (delay 0) is fine; one trained on earlier is a violation."""
        n_scored = n_bad_local = n_bad_sched = 0
        n_unlabelled_trained = 0
        for n in self.names:
            st = self.streams[n]
            first_local = np.full(len(st.arrival), np.iinfo(np.int64).max, dtype=np.int64)
            for rnd, idx in self.trained_log[n]:
                first_local[idx] = np.minimum(first_local[idx], rnd)
                n_unlabelled_trained += int((st.usable_from[idx] > rnd).sum())
            for rec in self.records[n]:
                idx, r = rec["idx"], rec["round"]
                n_scored += len(idx)
                n_bad_local += int((first_local[idx] < r).sum())
                n_bad_sched += int((st.usable_from[idx] < r).sum())
        return {"n_scored": n_scored, "n_scored_after_local_training": n_bad_local,
                "n_scored_after_client_training": n_bad_sched, "n_trained_without_usable_label": n_unlabelled_trained,
                "ok": n_bad_local == 0 and n_bad_sched == 0 and n_unlabelled_trained == 0,
                "note": "scoring in round r precedes all training in round r; client training follows the "
                        "same schedule (data/stream.py), checked through usable_from"}


__all__ = ["StreamRun"]
