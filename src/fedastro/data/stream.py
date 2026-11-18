"""Time-ordered streams for test-then-train evaluation (CLAUDE.md Phase H7).

Each node's **training split** becomes a stream; its validation split (update gate, tuning)
and its test split (fixed-split results) are untouched. Per node:

1. **Arrival order.** ``time``: index order, which is time order for loaders that store each
   subject's windows in time order (WESAD, asserted in tests). ``patient_shuffle``: patients
   in a seeded random order, each patient's windows kept together in index order; this is a
   SIMULATED arrival order (CinC and synthetic data have no common clock across patients).
   ``auto`` = ``time`` for one_subject_per_node topologies, ``patient_shuffle`` for by_source.
2. **Warm-up block.** The first ``max(1, round(warmup_frac * n))`` examples, fully labelled
   and usable for training from round 1: data a node holds before the mission starts.
3. **Chunks.** The rest arrives in ``fl.rounds - 1`` equal chunks (``np.array_split``) in
   rounds 2..fl.rounds. An example arriving in round r is predicted at the start of round r,
   by the models as they stand after round r - 1, before any training in round r.
4. **Labels.** Only a fraction gets labels. Cumulative rounding: after chunk k, exactly
   ``floor(label_fraction * arrived + 0.5)`` of the arrived examples are labelled, so a
   fraction too small for a single chunk still yields labels over the stream. Which examples
   in a chunk get labelled is seeded and random. A label is usable for training from round
   ``arrival + label_delay_rounds``. With delay 0 that is the arrival round, after the
   example was predicted. Unlabelled examples are never trained on.

Everything is a pure function of (node data, config, run seed, node index), so the server
and the simulated clients compute the same schedule without exchanging index lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fedastro.config import ExperimentConfig, StreamConfig
from fedastro.data.base import DatasetBundle

STREAM_ID = 9  # derive_seed / rng namespace (ARM_IDS 1-3, attack 4, COMMS 5, HIER_LINK 6, SECAGG 7, VERSIONING 8)
NEVER = np.iinfo(np.int64).max  # usable_from of an unlabelled example


@dataclass(frozen=True)
class NodeStream:
    """One node's schedule. Arrays are indexed like the node's training bundle."""

    order: np.ndarray  # training-bundle indices in arrival order
    arrival: np.ndarray  # round each example arrives (0 = warm-up block)
    usable_from: np.ndarray  # first round its label can be trained on (NEVER = no label)
    n_rounds: int

    def train_mask(self, rnd: int) -> np.ndarray:
        """Examples whose labels may be trained on in round ``rnd``."""
        return self.usable_from <= rnd

    def arriving(self, rnd: int) -> np.ndarray:
        """Indices arriving (and predicted) in round ``rnd``, in arrival order. Round 0 is the
        warm-up block, which is never predicted."""
        return self.order[self.arrival[self.order] == rnd]

    @property
    def n_warmup(self) -> int:
        return int((self.arrival == 0).sum())

    def summary(self) -> dict[str, Any]:
        stream = self.arrival > 0
        labelled = self.usable_from < NEVER
        return {"n_train_split": int(len(self.arrival)), "n_warmup": self.n_warmup,
                "n_stream": int(stream.sum()), "n_stream_labelled": int((stream & labelled).sum()),
                "n_trainable_final_round": int(self.train_mask(self.n_rounds).sum()),
                "chunk_sizes": [int((self.arrival == r).sum()) for r in range(2, self.n_rounds + 1)]}


def resolve_order(cfg: ExperimentConfig) -> str:
    s = cfg.stream
    if s.order != "auto":
        return s.order
    return "time" if cfg.hierarchy is not None and cfg.hierarchy.partition == "one_subject_per_node" \
        else "patient_shuffle"


def arrival_order(data: DatasetBundle, order: str, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(len(data))
    if order == "time":
        return idx
    patients = rng.permutation(np.unique(data.subject_id))
    return np.concatenate([idx[data.subject_id == p] for p in patients])


def make_stream(data: DatasetBundle, s: StreamConfig, order: str, n_rounds: int, seed: int,
                node_index: int) -> NodeStream:
    n = len(data)
    if n < 2:
        raise ValueError(f"node {node_index}: a stream needs at least 2 training examples, got {n}")
    if n_rounds < 2:
        raise ValueError("a stream needs at least 2 rounds")
    rng = np.random.default_rng([seed, STREAM_ID, node_index])
    seq = arrival_order(data, order, rng)
    n_w = min(n - 1, max(1, int(round(s.warmup_frac * n))))
    arrival = np.zeros(n, dtype=np.int64)
    usable = np.full(n, NEVER, dtype=np.int64)
    usable[seq[:n_w]] = 1  # warm-up: labelled, usable from round 1
    arrived, labelled = 0, 0
    for k, chunk in enumerate(np.array_split(seq[n_w:], n_rounds - 1), start=2):
        arrival[chunk] = k
        arrived += len(chunk)
        target = int(np.floor(s.label_fraction * arrived + 0.5))
        take = min(target - labelled, len(chunk))
        if take > 0:
            usable[rng.choice(chunk, size=take, replace=False)] = k + s.label_delay_rounds
        labelled += max(take, 0)
    return NodeStream(order=seq, arrival=arrival, usable_from=usable, n_rounds=n_rounds)


def make_streams(cfg: ExperimentConfig, train: dict[str, DatasetBundle], seed: int) -> dict[str, NodeStream]:
    """Every node's schedule, in topology node order (node index = position)."""
    order = resolve_order(cfg)
    return {name: make_stream(train[name], cfg.stream, order, cfg.fl.rounds, seed, i)
            for i, name in enumerate(train)}


__all__ = ["NodeStream", "make_stream", "make_streams", "resolve_order", "NEVER", "STREAM_ID"]
