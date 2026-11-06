"""DatasetBundle interface and provenance record."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

REAL = "REAL"
SYNTHETIC = "SYNTHETIC"
# What the data stands for (CLAUDE2.md rule 10); figure captions must state it.
DOMAINS = ("terrestrial_proxy", "terrestrial_analog", "spaceflight_rodent", "synthetic")


@dataclass(frozen=True)
class Provenance:
    """Where a dataset came from. Written verbatim into every result file.

    ``kind`` is ``SYNTHETIC`` for generated data so it can never be mistaken for real
    measurements downstream. ``domain`` is filled in from the dataset registry on load
    (``fedastro.data.DatasetSpec.load``); it is omitted from the record when empty, which
    only happens for a bundle built outside the registry.
    """

    dataset: str
    version: str
    checksum: str  # identity of the raw source (archive hash, or array hash if generated)
    source: str
    kind: str
    processed_checksum: str = ""  # hash of the arrays actually used, after preprocessing
    domain: str = ""

    def __post_init__(self) -> None:
        if self.kind not in (REAL, SYNTHETIC):
            raise ValueError(f"provenance kind must be {REAL} or {SYNTHETIC}")
        if self.domain and self.domain not in DOMAINS:
            raise ValueError(f"provenance domain must be one of {DOMAINS}, got {self.domain!r}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d["domain"]:
            del d["domain"]
        return d


@dataclass(frozen=True)
class DatasetBundle:
    """Windowed samples with per-sample subject ids.

    X: float32 array, shape (n_samples, n_features) or (n_samples, n_channels, length).
    y: int64 class labels, shape (n_samples,).
    subject_id: int64 subject identifiers, shape (n_samples,).
    site_id: optional int64 site identifiers, shape (n_samples,). None if the dataset
        has no notion of site (e.g. WESAD, a single-lab study).
    """

    X: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray
    class_names: tuple[str, ...]
    provenance: Provenance
    site_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.y)
        if len(self.X) != n or len(self.subject_id) != n:
            raise ValueError("X, y and subject_id must have the same length")
        if self.site_id is not None and len(self.site_id) != n:
            raise ValueError("site_id must have the same length as y")
        if n and (self.y.min() < 0 or self.y.max() >= len(self.class_names)):
            raise ValueError("labels must lie in [0, len(class_names))")

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask_or_idx: np.ndarray) -> "DatasetBundle":
        return DatasetBundle(
            X=self.X[mask_or_idx],
            y=self.y[mask_or_idx],
            subject_id=self.subject_id[mask_or_idx],
            class_names=self.class_names,
            provenance=self.provenance,
            site_id=None if self.site_id is None else self.site_id[mask_or_idx],
        )

    @property
    def subjects(self) -> list[int]:
        return sorted(int(s) for s in np.unique(self.subject_id))

    def label_counts(self) -> dict[str, int]:
        counts = np.bincount(self.y, minlength=len(self.class_names))
        return {name: int(c) for name, c in zip(self.class_names, counts)}


@dataclass(frozen=True)
class GroupedDatasetBundle(DatasetBundle):
    """A DatasetBundle whose samples also belong to natural groups (a country or a
    cohort), alongside ``subject_id``.

    group_id: int64 group index per sample, shape (n_samples,), indexing ``group_names``.
    group_names: name of each group, e.g. ("USA", "JPN").

    Every subject belongs to exactly one group, so splitting by group never splits a
    person (CLAUDE2.md rule 11).
    """

    group_id: np.ndarray = None  # type: ignore[assignment]  # required; base fields have defaults
    group_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.group_id is None or len(self.group_id) != len(self.y):
            raise ValueError("group_id must have the same length as y")
        if len(set(self.group_names)) != len(self.group_names):
            raise ValueError("group_names must be distinct")
        if not len(self):
            return
        if self.group_id.min() < 0 or self.group_id.max() >= len(self.group_names):
            raise ValueError("group_id must index group_names")
        pairs = np.unique(np.stack([self.subject_id, self.group_id]), axis=1)
        if pairs.shape[1] != len(np.unique(self.subject_id)):
            raise ValueError("a subject belongs to more than one group")

    def subset(self, mask_or_idx: np.ndarray) -> "GroupedDatasetBundle":
        base = DatasetBundle.subset(self, mask_or_idx)
        return GroupedDatasetBundle(**{f: getattr(base, f) for f in base.__dataclass_fields__},
                                    group_id=self.group_id[mask_or_idx], group_names=self.group_names)

    @property
    def groups(self) -> list[str]:
        """Names of the groups present, in group-index order."""
        return [self.group_names[int(g)] for g in np.unique(self.group_id)]

    def in_groups(self, names: tuple[str, ...] | list[str]) -> np.ndarray:
        """Boolean mask of samples whose group is in ``names``."""
        unknown = set(names) - set(self.group_names)
        if unknown:
            raise ValueError(f"unknown groups {sorted(unknown)}; have {self.group_names}")
        return np.isin(self.group_id, [self.group_names.index(n) for n in names])


def array_checksum(*arrays: np.ndarray) -> str:
    """SHA-256 over the raw bytes, dtype and shape of each array."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return "sha256:" + h.hexdigest()
