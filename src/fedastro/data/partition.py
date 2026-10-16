"""Subject-disjoint partitioning of a DatasetBundle into a held-out set and silos.

Order of operations, enforced by ``make_federation``:
1. ``holdout_split`` removes whole subjects into the global test set. No silo ever
   sees them.
2. The remaining subjects are partitioned across silos by one of:
   - ``by_subject``: whole subjects, allocated in proportion to silo data shares.
   - ``by_site``: whole sites, allocated the same way (needs ``site_id``).
   - ``dirichlet``: ``by_subject``, then per-silo class-skew subsampling with
     proportions q ~ Dirichlet(alpha * p), where p is the training pool's class
     distribution (Hsu et al. 2019). alpha -> infinity recovers the natural mix p;
     alpha -> 0 gives single-class silos. Subjects stay disjoint at the cost of
     dropping windows; dropped counts are reported per silo.
   - ``dirichlet_quantity_control``: identical subjects per silo and identical window
     counts per silo as ``dirichlet`` (same alpha, same rng), but windows are kept
     uniformly at random, so each silo retains its natural label mix. Comparing the
     two isolates the effect of label skew from the effect of losing data.

The held-out set is drawn from its own fixed seed (``holdout_seed``), so it is the same
for every run seed of an experiment.

Invariant for every method: no subject appears in two silos or in a silo and the
held-out set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedastro.data.base import DatasetBundle

# Separates the holdout RNG stream from run-seed streams even when the integers coincide.
_HOLDOUT_STREAM = 0x484F4C44


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
    quantity_control: bool = False,
) -> dict[str, SiloSplit]:
    """Subject-disjoint silos with Dirichlet(alpha) label skew imposed by subsampling.

    For each silo, draw target class proportions q ~ Dir(alpha * p), with p the class
    distribution of the whole training pool (Hsu et al. 2019, "Measuring the Effects of
    Non-Identical Data Distribution"). Keep the largest subset of that silo's windows
    whose class proportions match q (rounded down), chosen uniformly at random within
    each class. Large alpha -> q ~ p (natural mix); small alpha -> near single-class.

    (A symmetric Dir(alpha * 1) prior would instead pull every silo towards uniform
    class proportions as alpha grows, i.e. a label-prior shift relative to the natural
    mix even at "low skew". That was the Phase 4 first-pass bug; see PROGRESS.md.)

    With ``quantity_control=True`` the same number of windows is kept per silo, but
    chosen uniformly at random regardless of class. RNG draws are ordered so that the
    subject assignment, the p draws and therefore the per-silo counts are identical in
    both modes for the same input rng state.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    n_classes = len(bundle.class_names)
    prior = np.bincount(bundle.y, minlength=n_classes) / len(bundle)
    if (prior == 0).any():
        raise ValueError("every class must occur in the training pool for dirichlet partitioning")
    parts = by_subject(bundle, silo_shares, rng)
    props = {name: rng.dirichlet(alpha * prior) for name in parts}
    out: dict[str, SiloSplit] = {}
    for name, part in parts.items():
        p = props[name]
        keep_n = _dirichlet_keep_counts(np.bincount(part.y, minlength=n_classes), p)
        if quantity_control:
            keep = np.sort(rng.choice(len(part), size=int(keep_n.sum()), replace=False))
        else:
            keep = np.sort(np.concatenate([
                rng.choice(np.flatnonzero(part.y == k), size=keep_n[k], replace=False)
                for k in range(n_classes)
            ]))
        out[name] = SiloSplit(
            data=part.subset(keep),
            n_dropped=len(part) - len(keep),
            target_proportions=None if quantity_control else tuple(round(float(x), 6) for x in p),
        )
    return out


def _dirichlet_keep_counts(counts: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-class counts of the largest subset of ``counts`` with proportions ~= p.

    Pathological draws that would keep nothing keep the silo's whole majority class.
    """
    present = p > 0
    scale = np.min(counts[present] / p[present])
    keep_n = np.minimum(np.floor(scale * p + 1e-9).astype(int), counts)
    if keep_n.sum() == 0:
        keep_n = np.zeros_like(counts)
        k = int(np.argmax(counts))
        keep_n[k] = counts[k]
    return keep_n


def make_federation(
    bundle: DatasetBundle,
    silo_shares: dict[str, float],
    method: str,
    holdout_subjects: int,
    rng: np.random.Generator,
    alpha: float = 0.0,
    holdout_seed: int = 0,
) -> Federation:
    """Hold out test subjects (fixed ``holdout_seed``), then partition the rest (``rng``)."""
    holdout_rng = np.random.default_rng([holdout_seed, _HOLDOUT_STREAM])
    pool, holdout = holdout_split(bundle, holdout_subjects, holdout_rng)
    if method == "by_subject":
        silos = {k: SiloSplit(v) for k, v in by_subject(pool, silo_shares, rng).items()}
    elif method == "by_site":
        silos = {k: SiloSplit(v) for k, v in by_site(pool, silo_shares, rng).items()}
    elif method == "dirichlet":
        silos = dirichlet(pool, silo_shares, alpha, rng)
    elif method == "dirichlet_quantity_control":
        silos = dirichlet(pool, silo_shares, alpha, rng, quantity_control=True)
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
