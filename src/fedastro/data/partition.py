"""Subject-disjoint partitioning of a DatasetBundle across silos.

Phase 0 provides only ``by_subject``. Phase 1 adds ``by_site``, ``dirichlet(alpha)``
and the global subject-disjoint held-out test set.
"""

from __future__ import annotations

import numpy as np

from fedastro.data.base import DatasetBundle


def by_subject(
    bundle: DatasetBundle,
    silo_shares: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, DatasetBundle]:
    """Assign whole subjects to silos in proportion to ``silo_shares``.

    Every silo gets at least one subject; no subject appears in two silos.
    """
    subjects = np.unique(bundle.subject_id)
    names = list(silo_shares)
    if len(subjects) < len(names):
        raise ValueError(f"need >= {len(names)} subjects, have {len(subjects)}")
    shuffled = rng.permutation(subjects)
    counts = _allocate(len(subjects), np.array([silo_shares[n] for n in names], dtype=float))
    out: dict[str, DatasetBundle] = {}
    start = 0
    for name, c in zip(names, counts):
        chosen = shuffled[start : start + c]
        start += c
        out[name] = bundle.subset(np.isin(bundle.subject_id, chosen))
    return out


def _allocate(n: int, shares: np.ndarray) -> np.ndarray:
    """Largest-remainder allocation of n items, with a floor of 1 per bucket."""
    k = len(shares)
    w = shares / shares.sum()
    raw = w * (n - k)
    counts = np.floor(raw).astype(int) + 1
    remainder = n - counts.sum()
    order = np.argsort(-(raw - np.floor(raw)), kind="stable")
    counts[order[:remainder]] += 1
    return counts
