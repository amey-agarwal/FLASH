import numpy as np
import pytest

from fedastro.config import SyntheticParams
from fedastro.data.base import SYNTHETIC
from fedastro.data.partition import _allocate, by_site, by_subject, dirichlet, make_federation
from fedastro.data.synthetic import load_synthetic

SHARES = {"a": 0.4, "b": 0.2, "c": 0.2, "d": 0.12, "e": 0.08}


def test_synthetic_is_labelled_and_deterministic():
    p = SyntheticParams()
    a, b = load_synthetic(p, 0), load_synthetic(p, 0)
    assert a.provenance.kind == SYNTHETIC
    assert a.provenance.dataset == "SYNTHETIC"
    assert a.provenance.checksum == b.provenance.checksum
    assert a.provenance.checksum != load_synthetic(p, 1).provenance.checksum
    assert a.X.shape == (p.n_subjects * p.windows_per_subject, p.n_features)
    assert a.X.dtype == np.float32


def test_by_subject_disjoint_and_complete():
    bundle = load_synthetic(SyntheticParams(n_subjects=20), 0)
    parts = by_subject(bundle, SHARES, np.random.default_rng(0))
    subject_sets = [set(p.subject_id.tolist()) for p in parts.values()]
    for i in range(len(subject_sets)):
        for j in range(i + 1, len(subject_sets)):
            assert not subject_sets[i] & subject_sets[j], "a subject appears in two silos"
    assert set().union(*subject_sets) == set(bundle.subject_id.tolist())
    assert sum(len(p) for p in parts.values()) == len(bundle)
    assert all(len(s) >= 1 for s in subject_sets)
    sizes = [len(s) for s in subject_sets]
    assert sizes == sorted(sizes, reverse=True), "allocation should follow the share ordering"


def test_by_subject_minimum_one_each():
    bundle = load_synthetic(SyntheticParams(n_subjects=5), 0)
    parts = by_subject(bundle, SHARES, np.random.default_rng(0))
    assert [len(set(p.subject_id.tolist())) for p in parts.values()] == [1] * 5


def test_allocation_respects_share_ordering_for_wesad_sized_pool():
    shares = np.array([0.40, 0.20, 0.20, 0.12, 0.08])
    counts = _allocate(12, shares)
    assert counts.sum() == 12 and counts.min() >= 1
    assert counts[0] == counts.max() and counts[-1] == counts.min()


@pytest.mark.parametrize("method,alpha", [("by_subject", 0.0), ("dirichlet", 0.3), ("by_site", 0.0)])
def test_federation_holdout_is_subject_disjoint(method, alpha):
    bundle = load_synthetic(SyntheticParams(n_subjects=24, n_sites=8), 0)
    fed = make_federation(bundle, SHARES, method, 3, np.random.default_rng(1), alpha=alpha)
    held = set(fed.holdout.subjects)
    assert len(held) == 3
    seen = set(held)
    for split in fed.silos.values():
        s = set(split.data.subjects)
        assert not s & seen, "subject shared between holdout/silos"
        seen |= s
        # no training sample comes from a holdout subject
        assert not np.isin(split.data.subject_id, list(held)).any()


def test_dirichlet_small_alpha_is_more_skewed_than_large():
    bundle = load_synthetic(SyntheticParams(n_subjects=40, windows_per_subject=60), 0)

    def mean_max_share(alpha):
        vals = []
        for seed in range(10):
            for split in dirichlet(bundle, SHARES, alpha, np.random.default_rng(seed)).values():
                c = np.bincount(split.data.y, minlength=3)
                vals.append(c.max() / c.sum())
        return np.mean(vals)

    assert mean_max_share(0.1) > mean_max_share(100.0) + 0.2


def test_dirichlet_reports_dropped_windows():
    bundle = load_synthetic(SyntheticParams(n_subjects=20), 0)
    parts = dirichlet(bundle, SHARES, 0.5, np.random.default_rng(0))
    assert sum(len(p.data) + p.n_dropped for p in parts.values()) == len(bundle)
    for p in parts.values():
        assert len(p.target_proportions) == 3


def test_by_site_requires_site_ids():
    bundle = load_synthetic(SyntheticParams(n_subjects=20), 0)
    with pytest.raises(ValueError, match="no site ids"):
        by_site(bundle, SHARES, np.random.default_rng(0))


def test_by_site_keeps_sites_whole():
    bundle = load_synthetic(SyntheticParams(n_subjects=20, n_sites=10), 0)
    parts = by_site(bundle, SHARES, np.random.default_rng(0))
    site_sets = [set(p.site_id.tolist()) for p in parts.values()]
    assert sum(len(s) for s in site_sets) == len(set().union(*site_sets)) == 10


def test_holdout_fixed_across_run_seeds_and_set_by_holdout_seed():
    bundle = load_synthetic(SyntheticParams(n_subjects=24), 0)
    held = [
        fed.holdout.subjects
        for fed in (make_federation(bundle, SHARES, "by_subject", 3, np.random.default_rng(s))
                    for s in range(5))
    ]
    assert all(h == held[0] for h in held)
    other = make_federation(bundle, SHARES, "by_subject", 3, np.random.default_rng(0), holdout_seed=7)
    assert other.holdout.subjects != held[0]


def test_quantity_control_matches_dirichlet_subjects_and_counts():
    bundle = load_synthetic(SyntheticParams(n_subjects=30, windows_per_subject=40), 0)
    for seed in range(5):
        skew = make_federation(bundle, SHARES, "dirichlet", 3, np.random.default_rng(seed), alpha=0.3)
        ctrl = make_federation(bundle, SHARES, "dirichlet_quantity_control", 3,
                               np.random.default_rng(seed), alpha=0.3)
        assert skew.holdout.subjects == ctrl.holdout.subjects
        for name in SHARES:
            assert skew.silos[name].data.subjects == ctrl.silos[name].data.subjects
            assert len(skew.silos[name].data) == len(ctrl.silos[name].data)
            assert skew.silos[name].n_dropped == ctrl.silos[name].n_dropped
            assert ctrl.silos[name].target_proportions is None


def test_quantity_control_is_less_skewed():
    bundle = load_synthetic(SyntheticParams(n_subjects=40, windows_per_subject=60), 0)

    def mean_max_share(method):
        vals = []
        for seed in range(10):
            fed = make_federation(bundle, SHARES, method, 3, np.random.default_rng(seed), alpha=0.2)
            for split in fed.silos.values():
                c = np.bincount(split.data.y, minlength=3)
                vals.append(c.max() / c.sum())
        return np.mean(vals)

    assert mean_max_share("dirichlet") > mean_max_share("dirichlet_quantity_control") + 0.2
