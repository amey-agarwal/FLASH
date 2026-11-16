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
   - ``by_group``: natural groups (countries, cohorts) of a GroupedDatasetBundle. One
     silo per group, or the groups listed per silo in ``group_map``. Deterministic: no
     rng, and data shares are not used (natural silos keep their natural sizes).

The held-out set is drawn from its own fixed seed (``holdout_seed``), so it is the same
for every run seed of an experiment.

``holdout_groups`` (grouped data only) removes whole groups before anything else. They
join the held-out set and contribute nothing to training: no samples, so no scaler
statistics either (every scaler, local or federated, is fit on silo data only).

Invariant for every method: no subject appears in two silos or in a silo and the
held-out set; no held-out group appears in a silo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fedastro.data.base import DatasetBundle, GroupedDatasetBundle

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
    # Grouped data only: {"silo_to_groups", "holdout_groups", "unused_groups"}; else None.
    groups: dict[str, Any] | None = None


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


def by_group(
    bundle: DatasetBundle, silo_names: list[str], group_map: dict[str, tuple[str, ...]] | None = None
) -> dict[str, DatasetBundle]:
    """Natural silos: each silo holds whole groups, never re-mixed (CLAUDE2.md rule 11).
    Without ``group_map`` every silo name must be a group name. Groups not listed are
    left out (the caller reports them)."""
    if not isinstance(bundle, GroupedDatasetBundle):
        raise ValueError(f"dataset {bundle.provenance.dataset} has no groups; by_group needs grouped data")
    mapping = _group_mapping(silo_names, group_map)
    if set(mapping) != set(silo_names):
        raise ValueError(f"group_map silos {sorted(mapping)} != topology silos {sorted(silo_names)}")
    out = {}
    for name in silo_names:
        part = bundle.subset(bundle.in_groups(mapping[name]))
        if not len(part):
            raise ValueError(f"silo {name}: groups {list(mapping[name])} have no training samples")
        out[name] = part
    return out


def _group_mapping(silo_names: list[str], group_map: dict[str, tuple[str, ...]] | None) -> dict[str, tuple[str, ...]]:
    return group_map or {name: (name,) for name in silo_names}


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
    group_map: dict[str, tuple[str, ...]] | None = None,
    holdout_groups: tuple[str, ...] = (),
    holdout_draw: int = 0,
) -> Federation:
    """Hold out whole groups (grouped data), then test subjects (fixed ``holdout_seed``),
    then partition the rest (``rng``). Without groups this is exactly the earlier path.

    Under by_group, groups that are in no silo and not held out are unused: they are
    removed before held-out subjects are drawn, so they appear neither in training nor
    in the test set (e.g. a cohort excluded by CLAUDE2.md rule 14).

    ``holdout_draw`` (Phase 12) selects one of several fixed held-out subject sets. Draw 0
    is exactly the earlier stream; draw d > 0 has its own stream, still independent of
    the run seed."""
    holdout_key = [holdout_seed, _HOLDOUT_STREAM] + ([holdout_draw] if holdout_draw else [])
    holdout_rng = np.random.default_rng(holdout_key)
    grouped = isinstance(bundle, GroupedDatasetBundle)
    if holdout_groups and not grouped:
        raise ValueError(f"dataset {bundle.provenance.dataset} has no groups; holdout_groups needs grouped data")
    if grouped:
        held_group = bundle.in_groups(holdout_groups)
        keep = ~held_group
        if method == "by_group":
            mapped = [g for gs in _group_mapping(list(silo_shares), group_map).values() for g in gs]
            keep &= bundle.in_groups(mapped)
        pool = bundle.subset(keep)
        held_subjects: list[int] = []
        if holdout_subjects:
            pool, held = holdout_split(pool, holdout_subjects, holdout_rng)
            held_subjects = held.subjects
        holdout = bundle.subset(held_group | np.isin(bundle.subject_id, held_subjects))
    else:
        pool, holdout = holdout_split(bundle, holdout_subjects, holdout_rng)
    if method == "by_group":
        silos = {k: SiloSplit(v) for k, v in by_group(pool, list(silo_shares), group_map).items()}
    elif method == "by_subject":
        silos = {k: SiloSplit(v) for k, v in by_subject(pool, silo_shares, rng).items()}
    elif method == "by_site":
        silos = {k: SiloSplit(v) for k, v in by_site(pool, silo_shares, rng).items()}
    elif method == "dirichlet":
        silos = dirichlet(pool, silo_shares, alpha, rng)
    elif method == "dirichlet_quantity_control":
        silos = dirichlet(pool, silo_shares, alpha, rng, quantity_control=True)
    else:
        raise ValueError(f"unknown partition method {method!r}")
    groups = None
    if grouped:
        silo_groups = {name: split.data.groups for name, split in silos.items()}
        used = {g for gs in silo_groups.values() for g in gs} | set(holdout_groups)
        groups = {"silo_to_groups": silo_groups, "holdout_groups": list(holdout_groups),
                  "unused_groups": [g for g in bundle.group_names if g not in used]}
    fed = Federation(holdout=holdout, silos=silos, groups=groups)
    assert_disjoint(fed)
    return fed


def assert_disjoint(fed: Federation) -> None:
    """Raise if any subject is shared between two silos or a silo and the holdout, or a
    held-out group has samples in a silo."""
    seen: dict[int, str] = {s: "holdout" for s in fed.holdout.subjects}
    for name, split in fed.silos.items():
        for s in split.data.subjects:
            if s in seen:
                raise AssertionError(f"subject {s} in both {seen[s]} and {name}")
            seen[s] = name
    held = set(fed.groups["holdout_groups"]) if fed.groups else set()
    for name, split in fed.silos.items():
        leaked = held & set(getattr(split.data, "groups", []))
        if leaked:
            raise AssertionError(f"held-out groups {sorted(leaked)} have samples in silo {name}")


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
