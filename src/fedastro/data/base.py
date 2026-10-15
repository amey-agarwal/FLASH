"""DatasetBundle interface and provenance record."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

REAL = "REAL"
SYNTHETIC = "SYNTHETIC"


@dataclass(frozen=True)
class Provenance:
    """Where a dataset came from. Written verbatim into every result file.

    ``kind`` is ``SYNTHETIC`` for generated data so it can never be mistaken for real
    measurements downstream.
    """

    dataset: str
    version: str
    checksum: str
    source: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in (REAL, SYNTHETIC):
            raise ValueError(f"provenance kind must be {REAL} or {SYNTHETIC}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetBundle:
    """Windowed samples with per-sample subject ids.

    X: float32 array, shape (n_samples, n_features) or (n_samples, n_channels, length).
    y: int64 class labels, shape (n_samples,).
    subject_id: int64 subject identifiers, shape (n_samples,).
    """

    X: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray
    class_names: tuple[str, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        n = len(self.y)
        if len(self.X) != n or len(self.subject_id) != n:
            raise ValueError("X, y and subject_id must have the same length")
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
        )

    def label_counts(self) -> dict[str, int]:
        counts = np.bincount(self.y, minlength=len(self.class_names))
        return {name: int(c) for name, c in zip(self.class_names, counts)}


def array_checksum(*arrays: np.ndarray) -> str:
    """SHA-256 over the raw bytes, dtype and shape of each array."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return "sha256:" + h.hexdigest()
