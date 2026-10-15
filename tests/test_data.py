import numpy as np

from fedastro.config import SyntheticParams
from fedastro.data.base import SYNTHETIC
from fedastro.data.partition import by_subject
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
