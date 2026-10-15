"""Subject-disjoint partitioning of a DatasetBundle into a held-out set and silos.

Order of operations, enforced by ``make_federation``:
1. ``holdout_split`` removes whole subjects into the global test set. No silo ever
   sees them.
2. The remaining subjects are partitioned across silos by one of:
   - ``by_subject``: whole subjects, allocated in proportion to silo data shares.
   - ``by_site``: whole sites, allocated the same way (needs ``site_id``).
   - ``dirichlet``: ``by_subject``, then per-silo class-skew subsampling with
     proportions drawn from Dirichlet(alpha). This keeps subjects disjoint at the cost
     of dropping windows; dropped counts are reported per silo.

Invariant for every method: no subject appears in two silos or in a silo and the
held-out set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedastro.data.base import DatasetBundle


@dataclass(frozen=True)
class SiloSplit:
    data: DatasetBundle
    n_dropped: int = 0  # windows discarded by dirichlet subsampling
    target_proportions: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Federation:
    holdout: DatasetBundle
    silos: dict[str, SiloSplit]


def holdout_split(
    bundle: DatasetBundle, n_subjects: int, rng: np.random.Generator
) -> tuple[DatasetBundle, DatasetBundle]:
    """Return (train_pool, holdout) with ``n_subjects`` whole subjects in the holdout."""
    subjects = np.array(bundle.subjects)
    if not 1 <= n_subjects < len(subjects):
        raise ValueError(f"holdout needs 1..{len(subjects) - 1} subjects, got {n_subjects}")
    held = rng.choice(subjects, size=n_subjects, replace=False)
    mask = np.isin(bundle.subject_id, held)
    return bundle.subset(~mask), bundle.subset(mask)


def by_subject(
    bundle: DatasetBundle, silo_shares: dict[str, float], rng: np.random.Generator
) -> dict[str, DatasetBundle]:
    """Assign whole subjects to silos in proportion to ``silo_shares`` (>= 1 each)."""
    return _assign_groups(bundle, bundle.subject_id, silo_shares, rng, "subjects")


def by_site(
    bundle: DatasetBundle, silo_shares: dict[str, float], rng: np.random.Generator
) -> dict[str, DatasetBundle]:
    """Assign whole sites to silos. Subjects must not span sites."""
    if bundle.site_id is None:
        raise ValueError(f"dataset {bundle.provenance.dataset} has no site ids; use by_subject")
    for s in bundle.subjects:
        if len(np.unique(bundle.site_id[bundle.subject_id == s])) > 1:
            raise ValueError(f"subject {s} spans multiple sites; by_site would leak it")
    return _assign_groups(bundle, bundle.site_id, silo_shares, rng, "sites")


def dirichlet(
    bundle: DatasetBundle,
    silo_shares: dict[str, float],
    alpha: float,
    rng: np.random.Generator,
) -> dict[str, SiloSplit]:
    """Subject-disjoint silos with Dirichlet(alpha) label skew imposed by subsampling.

    For each silo, draw target class proportions p ~ Dir(alpha * 1). Keep the largest
    subset of that silo's windows whose class proportions match p (rounded down), chosen
    uniformly at random within each class. Small alpha -> near single-class silos.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    n_classes = len(bundle.class_names)
    out: dict[str, SiloSplit] = {}
    for name, part in by_subject(bundle, silo_shares, rng).items():
        p = rng.dirichlet(np.full(n_classes, alpha))
        counts = np.bincount(part.y, minlength=n_classes)
        present = p > 0
        scale = np.min(counts[present] / p[present])
        keep_n = np.floor(scale * p + 1e-9).astype(int)
        keep_idx = []
        for k in range(n_classes):
            idx = np.flatnonzero(part.y == k)
            keep_idx.append(rng.choice(idx, size=min(keep_n[k], len(idx)), replace=False))
        keep = np.sort(np.concatenate(keep_idx))
        if len(keep) == 0:  # pathological draw: keep the silo's majority class
            k = int(np.argmax(counts))
            keep = np.flatnonzero(part.y == k)
        out[name] = SiloSplit(
            data=part.subset(keep),
            n_dropped=len(part) - len(keep),
            target_proportions=tuple(round(float(x), 6) for x in p),
        )
    return out


def make_federation(
    bundle: DatasetBundle,
    silo_shares: dict[str, float],
    method: str,
    holdout_subjects: int,
    rng: np.random.Generator,
    alpha: float = 0.0,
) -> Federation:
    pool, holdout = holdout_split(bundle, holdout_subjects, rng)
    if method == "by_subject":
        silos = {k: SiloSplit(v) for k, v in by_subject(pool, silo_shares, rng).items()}
    elif method == "by_site":
        silos = {k: SiloSplit(v) for k, v in by_site(pool, silo_shares, rng).items()}
    elif method == "dirichlet":
        silos = dirichlet(pool, silo_shares, alpha, rng)
    else:
        raise ValueError(f"unknown partition method {method!r}")
    fed = Federation(holdout=holdout, silos=silos)
    assert_disjoint(fed)
    return fed


def assert_disjoint(fed: Federation) -> None:
    """Raise if any subject is shared between two silos or a silo and the holdout."""
    seen: dict[int, str] = {s: "holdout" for s in fed.holdout.subjects}
    for name, split in fed.silos.items():
        for s in split.data.subjects:
            if s in seen:
                raise AssertionError(f"subject {s} in both {seen[s]} and {name}")
            seen[s] = name


def _assign_groups(
    bundle: DatasetBundle,
    group_id: np.ndarray,
    silo_shares: dict[str, float],
    rng: np.random.Generator,
    what: str,
) -> dict[str, DatasetBundle]:
    groups = np.unique(group_id)
    names = list(silo_shares)
    if len(groups) < len(names):
        raise ValueError(f"need >= {len(names)} {what} for {len(names)} silos, have {len(groups)}")
    shuffled = rng.permutation(groups)
    counts = _allocate(len(groups), np.array([silo_shares[n] for n in names], dtype=float))
    out: dict[str, DatasetBundle] = {}
    start = 0
    for name, c in zip(names, counts):
        out[name] = bundle.subset(np.isin(group_id, shuffled[start : start + c]))
        start += c
    return out


def _allocate(n: int, shares: np.ndarray) -> np.ndarray:
    """Largest-remainder (Hamilton) allocation of n items by share, then a floor of 1.

    Any bucket left at 0 takes one item from the currently largest bucket.
    """
    if n < len(shares):
        raise ValueError("fewer items than buckets")
    raw = shares / shares.sum() * n
    counts = np.floor(raw).astype(int)
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[: n - counts.sum()]] += 1
    while (counts == 0).any():
        counts[int(np.argmax(counts))] -= 1
        counts[int(np.flatnonzero(counts == 0)[0])] += 1
    return counts
