"""Phase H1: per-node train / validation / test partitioning for hierarchical topologies.
The headline invariant: no patient in two nodes or two splits."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from fedastro.config import load_config
from fedastro.data import wesad
from fedastro.data.partition import (NodeSplit, _allocate_free, assert_hier_disjoint,
                                     make_hier_federation)
from fedastro.pipeline import build_hier_federation, load_dataset

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cinc():
    cfg = load_config(REPO / "configs" / "experiments" / "smoke_hier.yaml")
    return cfg, *build_hier_federation(cfg, 0)[:2]


@pytest.fixture(scope="module")
def wes():
    cfg = load_config(REPO / "configs" / "experiments" / "smoke_hier_wesad12.yaml")
    return cfg, *build_hier_federation(cfg, 0)[:2]


def _patients(split: NodeSplit) -> dict[str, set[int]]:
    return {"train": set(split.train.subjects), "val": set(split.val.subjects),
            "test": set(split.test.subjects) if split.test is not None else set()}


def test_no_patient_in_two_nodes_or_two_splits(cinc):
    _, bundle, fed = cinc
    assert len(fed.nodes) == 15
    seen: set[int] = set()
    for split in fed.nodes.values():
        p = _patients(split)
        assert not (p["train"] & p["val"] or p["train"] & p["test"] or p["val"] & p["test"])
        assert all(p.values())  # every split has at least one patient
        mine = p["train"] | p["val"] | p["test"]
        assert not mine & seen
        seen |= mine
    assert seen == set(bundle.subjects)  # every source is used, so every patient is placed
    assert fed.holdout is None and fed.unused_sources == ()


def test_nodes_hold_only_their_sources_and_roscosmos_is_leo_only(cinc):
    cfg, bundle, fed = cinc
    group_of = {int(s): bundle.group_names[int(g)] for s, g in zip(bundle.subject_id, bundle.group_id)}
    for node in cfg.hierarchy.nodes:
        split = fed.nodes[node.name]
        ids = set().union(*_patients(split).values())
        assert {group_of[s] for s in ids} <= set(node.sources)
        assert split.region == cfg.hierarchy.region_of(node.name)
    holders = [n.name for n in cfg.hierarchy.nodes if "incart" in n.sources]
    assert holders == ["leo_roscosmos"]
    incart = {s for s, g in group_of.items() if g == "incart"}
    assert set().union(*_patients(fed.nodes["leo_roscosmos"]).values()) == incart


def test_split_fractions_are_about_70_10_20_of_patients(cinc):
    cfg, _, fed = cinc
    for name, split in fed.nodes.items():
        p = {k: len(v) for k, v in _patients(split).items()}
        n = sum(p.values())
        assert n >= cfg.hierarchy.min_patients_per_node
        assert p["test"] == max(1, round(0.2 * n)) and p["val"] == max(1, round(0.1 * n)), name


def test_by_source_partition_is_frozen_across_run_seeds_and_follows_holdout_seed(cinc):
    cfg, _, fed = cinc
    _, other = build_hier_federation(cfg, 1)[:2]
    # Synthetic data is regenerated per seed, but subject ids are stable: same assignment.
    for name in fed.nodes:
        assert _patients(fed.nodes[name]) == _patients(other.nodes[name])
    moved = dataclasses.replace(cfg, partition=dataclasses.replace(cfg.partition, holdout_seed=1))
    _, third = build_hier_federation(moved, 0)[:2]
    assert any(_patients(fed.nodes[n]) != _patients(third.nodes[n]) for n in fed.nodes)


def test_quantity_skew_comes_from_the_dirichlet_shares(cinc):
    cfg, _, fed = cinc
    sizes = {n: sum(len(v) for v in _patients(s).values()) for n, s in fed.nodes.items()}
    nasa = [sizes[n] for n in ("leo_nasa", "moon_nasa", "mars_nasa")]
    assert sum(nasa) == 30 and len(set(nasa)) > 1  # one source, unequal region shares
    near_equal = dataclasses.replace(cfg, hierarchy=dataclasses.replace(cfg.hierarchy, quantity_alpha=1e6))
    _, even = build_hier_federation(near_equal, 0)[:2]
    assert [sum(len(v) for v in _patients(even.nodes[n]).values())
            for n in ("leo_nasa", "moon_nasa", "mars_nasa")] == [10, 10, 10]


def test_a_source_too_small_for_its_nodes_is_an_error(cinc):
    cfg, bundle, _ = cinc
    keep = ~bundle.in_groups(["g12ec"]) | np.isin(bundle.subject_id, bundle.subset(bundle.in_groups(["g12ec"])).subjects[:8])
    with pytest.raises(ValueError, match="fewer than 3 .* x 3 nodes"):
        make_hier_federation(bundle.subset(keep), cfg.hierarchy, np.random.default_rng(0))


def test_by_source_needs_grouped_data_and_known_sources(cinc, wes):
    cfg, bundle, _ = cinc
    with pytest.raises(ValueError, match="needs grouped data"):
        make_hier_federation(wes[1], cfg.hierarchy, np.random.default_rng(0))
    renamed = dataclasses.replace(bundle, group_names=tuple(f"g{i}" for i in range(8)))
    with pytest.raises(ValueError, match="are not groups of the dataset"):
        make_hier_federation(renamed, cfg.hierarchy, np.random.default_rng(0))


def test_allocate_free_sums_and_allows_empty_buckets():
    assert _allocate_free(10, np.array([0.5, 0.3, 0.2])).tolist() == [5, 3, 2]
    assert _allocate_free(1, np.array([0.9, 0.05, 0.05])).tolist() == [1, 0, 0]
    assert _allocate_free(0, np.array([0.5, 0.5])).tolist() == [0, 0]


# -- one subject per node (WESAD-style) ----------------------------------------------


def test_one_subject_per_node_with_shared_holdout(wes):
    cfg, bundle, fed = wes
    assert len(fed.nodes) == 12 and len(fed.holdout.subjects) == 3
    subjects = [split.train.subjects for split in fed.nodes.values()]
    assert all(len(s) == 1 for s in subjects)
    assert len({s[0] for s in subjects}) == 12
    assert not {s[0] for s in subjects} & set(fed.holdout.subjects)
    for name, split in fed.nodes.items():
        assert split.test is None and fed.test_of(name) is fed.holdout
        assert split.val.subjects == split.train.subjects


def test_validation_is_the_time_ordered_tail_of_each_class_with_a_gap(wes):
    cfg, bundle, fed = wes
    for split in fed.nodes.values():
        s = split.train.subjects[0]
        mask = bundle.subject_id == s
        X, y = bundle.X[mask], bundle.y[mask]
        n_gap = 0
        for k in np.unique(y):
            cls = np.flatnonzero(y == k)
            n_val = round(0.2 * len(cls))
            np.testing.assert_array_equal(split.val.X[split.val.y == k], X[cls[len(cls) - n_val:]])
            np.testing.assert_array_equal(split.train.X[split.train.y == k], X[cls[: len(cls) - n_val - 1]])
            n_gap += 1
        assert split.n_gap_dropped == n_gap
        assert set(np.unique(split.val.y)) == set(np.unique(y))  # the validation set keeps every class


def test_tail_split_keeps_a_class_too_small_for_a_tail_in_training():
    from fedastro.data.partition import tail_split
    from fedastro.data.base import DatasetBundle, Provenance
    prov = Provenance(dataset="SYNTHETIC", version="t", checksum="c", source="t", kind="SYNTHETIC")
    y = np.array([0] * 10 + [1] * 2 + [2] * 10)
    b = DatasetBundle(X=np.arange(22, dtype=np.float32)[:, None], y=y, subject_id=np.zeros(22, dtype=np.int64),
                      class_names=("a", "b", "c"), provenance=prov)
    train, val, dropped = tail_split(b, 0, 0.2, 1)
    assert val.tolist() == [8, 9, 20, 21] and dropped == 2
    assert train.tolist() == list(range(7)) + [10, 11] + list(range(12, 19))


@pytest.mark.skipif(not wesad.archive_path().exists(), reason="real WESAD archive not downloaded")
def test_real_wesad_validation_sets_hold_every_condition():
    """The Phase H3 fix: with a whole-session tail, no WESAD validation set had baseline."""
    cfg = load_config(REPO / "configs" / "experiments" / "smoke_hier_wesad12.yaml")
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, name="wesad"))
    _, fed, _ = build_hier_federation(cfg, 0)
    for name, split in fed.nodes.items():
        assert all(v > 0 for v in split.val.label_counts().values()), name
        assert all(v > 0 for v in split.train.label_counts().values()), name


def test_holdout_is_the_flat_holdout_and_node_assignment_follows_the_run_seed(wes):
    cfg, _, fed = wes
    from fedastro.data.partition import _HOLDOUT_STREAM, holdout_split
    bundle = load_dataset(cfg, 0)
    _, flat_holdout = holdout_split(bundle, 3, np.random.default_rng([0, _HOLDOUT_STREAM]))
    assert fed.holdout.subjects == flat_holdout.subjects
    _, other = build_hier_federation(cfg, 1)[:2]
    assert other.holdout.subjects == fed.holdout.subjects  # test set never depends on the run seed
    assert [s.train.subjects for s in other.nodes.values()] != [s.train.subjects for s in fed.nodes.values()]
    _, again = build_hier_federation(cfg, 0)[:2]
    assert [s.train.subjects for s in again.nodes.values()] == [s.train.subjects for s in fed.nodes.values()]


def test_wrong_subject_count_is_an_error(wes):
    cfg, bundle, _ = wes
    with pytest.raises(ValueError, match="12 nodes"):
        make_hier_federation(bundle, cfg.hierarchy, np.random.default_rng(0), holdout_subjects=2)


def test_disjointness_assertion_catches_a_leak(cinc, wes):
    _, _, fed = cinc
    a, b = fed.nodes["leo_nasa"], fed.nodes["moon_nasa"]
    leaked = dataclasses.replace(fed, nodes={**fed.nodes, "moon_nasa": dataclasses.replace(b, val=a.val)})
    with pytest.raises(AssertionError, match="in both"):
        assert_hier_disjoint(leaked)
    within = dataclasses.replace(fed, nodes={**fed.nodes, "leo_nasa": dataclasses.replace(a, test=a.train)})
    with pytest.raises(AssertionError, match="in both train and test"):
        assert_hier_disjoint(within)
    _, _, wfed = wes
    first = next(iter(wfed.nodes))
    held = dataclasses.replace(wfed, holdout=wfed.nodes[first].train)
    with pytest.raises(AssertionError, match="in both holdout"):
        assert_hier_disjoint(held)


def test_wesad_windows_are_stored_in_time_order():
    """The tail split relies on it: label_windows emits start times in increasing order,
    and the loader keeps that order within a subject."""
    labels = np.concatenate([np.full(700 * 200, 1), np.full(700 * 50, 0), np.full(700 * 300, 2),
                             np.full(700 * 150, 3)])
    t, _ = wesad.label_windows(labels, 60.0, 30.0)
    assert len(t) > 10 and np.all(np.diff(t) > 0)
