"""Phase H2: the SIMULATED location shift (data/shift.py)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from fedastro.config import ConfigError, ShiftConfig, load_config
from fedastro.data.base import DatasetBundle, Provenance
from fedastro.data.partition import HierFederation, NodeSplit
from fedastro.data.shift import PROVENANCE, apply_shift, format_shift_table, make_region_shifts, strengths
from fedastro.metrics.schema import empty_result, validate_result
from fedastro.pipeline import build_hier_federation

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "configs" / "experiments"


def _cfg(name="smoke_hier_shift", **shift):
    cfg = load_config(EXP / f"{name}.yaml")
    return dataclasses.replace(cfg, shift=dataclasses.replace(cfg.shift, **shift)) if shift else cfg


def _same(a: DatasetBundle, b: DatasetBundle) -> bool:
    return a.X.tobytes() == b.X.tobytes() and a.y.tobytes() == b.y.tobytes() \
        and a.subject_id.tobytes() == b.subject_id.tobytes()


@pytest.fixture(scope="module")
def plain():
    return build_hier_federation(load_config(EXP / "smoke_hier.yaml"), 0)[1]


@pytest.fixture(scope="module")
def shifted():
    return build_hier_federation(_cfg(), 0)[1:]


@pytest.mark.parametrize("kind", ["covariate", "concept", "both"])
def test_magnitude_zero_is_bit_identical_to_no_shift(plain, kind):
    _, fed, section = build_hier_federation(_cfg(magnitude=0.0, kind=kind), 0)
    assert section["enabled"] and section["magnitude"] == 0.0 and section["provenance"] == "SIMULATED"
    for name, split in plain.nodes.items():
        for part in ("train", "val", "test"):
            assert _same(getattr(split, part), getattr(fed.nodes[name], part)), (name, part)
    for r in section["regions"].values():
        assert r["distance_before"] == r["distance_after"]


def test_magnitude_zero_is_bit_identical_with_a_shared_holdout():
    _, plain, _ = build_hier_federation(load_config(EXP / "smoke_hier_wesad12.yaml"), 0)
    _, fed, _ = build_hier_federation(_cfg("smoke_hier_shift_wesad12", magnitude=0.0), 0)
    for name, split in plain.nodes.items():
        assert _same(split.train, fed.nodes[name].train) and _same(split.val, fed.nodes[name].val)
        assert _same(plain.test_of(name), fed.test_of(name))


def test_reference_region_is_untouched_and_others_change(plain, shifted):
    fed, section = shifted
    assert section["reference_region"] == "leo"
    for name, split in plain.nodes.items():
        same = _same(split.train, fed.nodes[name].train)
        assert same == (split.region == "leo"), name
        # labels, patients and sizes never change
        assert np.array_equal(split.test.y, fed.nodes[name].test.y)
        assert np.array_equal(split.val.subject_id, fed.nodes[name].val.subject_id)


def test_shifted_regions_are_measurably_further_from_the_reference(shifted):
    _, section = shifted
    assert section["regions"]["leo"]["distance_after"]["smd_mean"] == 0.0
    for region in ("moon", "mars"):
        before, after = (section["regions"][region][k] for k in ("distance_before", "distance_after"))
        for key in ("smd_mean", "wasserstein_mean", "smd_class_mean"):
            assert after[key] > before[key] + 0.1, (region, key)


def test_distance_grows_with_magnitude():
    d = [build_hier_federation(_cfg(magnitude=m), 0)[2]["regions"]["mars"]["distance_after"]["wasserstein_mean"]
         for m in (0.0, 0.5, 1.0, 2.0)]
    assert d == sorted(d) and d[3] > d[0] + 0.5


def test_only_selected_features_move_and_the_transform_is_the_same_in_every_split(plain, shifted):
    fed, section = shifted
    selected = section["features"]
    assert len(selected) == 4 and selected == sorted(selected)  # half of 8 features
    others = [i for i in range(8) if i not in selected]
    cfg = _cfg()
    pooled = np.concatenate([s.train.X for s in plain.nodes.values()]).astype(np.float64)
    shifts, _ = make_region_shifts(cfg.shift, cfg.hierarchy, pooled.mean(axis=0), pooled.std(axis=0), 3)
    for name, split in plain.nodes.items():
        rs = shifts[split.region]
        for part in ("train", "val", "test"):
            a, b = getattr(split, part), getattr(fed.nodes[name], part)
            assert np.array_equal(a.X[:, others], b.X[:, others])
            want = rs.mean + (a.X.astype(np.float64) - rs.mean) * rs.gain + rs.offset + rs.class_offset[a.y]
            np.testing.assert_allclose(b.X, want.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_kinds_are_separate_mechanisms():
    cfg = _cfg()
    mean, std = np.zeros(8), np.ones(8)
    cov, _ = make_region_shifts(dataclasses.replace(cfg.shift, kind="covariate"), cfg.hierarchy, mean, std, 3)
    con, _ = make_region_shifts(dataclasses.replace(cfg.shift, kind="concept"), cfg.hierarchy, mean, std, 3)
    both, sel = make_region_shifts(cfg.shift, cfg.hierarchy, mean, std, 3)
    m = cov["mars"]
    assert not m.class_offset.any() and m.offset.any() and (m.gain != 1).any()
    c = con["mars"]
    assert c.class_offset.any() and not c.offset.any() and (c.gain == 1).all()
    # "both" is exactly the two parts together
    assert np.array_equal(both["mars"].offset, m.offset) and np.array_equal(both["mars"].class_offset, c.class_offset)
    # concept shift has no marginal component: it sums to zero over classes
    np.testing.assert_allclose(c.class_offset.sum(axis=0), 0.0, atol=1e-12)
    # magnitude 1 = an offset of 1 training std (RMS over the selected features)
    assert np.sqrt((m.offset[sel] ** 2).mean()) == pytest.approx(1.0, abs=1e-6)
    assert np.sqrt((c.class_offset[:, sel] ** 2).mean(axis=1)) == pytest.approx(np.sqrt(2 / 3) * np.ones(3), rel=0.5)
    assert not np.allclose(both["moon"].offset, both["mars"].offset)  # regions differ in direction


def test_shift_does_not_depend_on_the_run_seed_but_on_the_shift_seed():
    cfg = _cfg()
    mean, std = np.zeros(8), np.ones(8)
    a, sel_a = make_region_shifts(cfg.shift, cfg.hierarchy, mean, std, 3)
    b, sel_b = make_region_shifts(dataclasses.replace(cfg.shift, seed=1), cfg.hierarchy, mean, std, 3)
    assert not np.array_equal(a["mars"].offset, b["mars"].offset)
    s0 = build_hier_federation(cfg, 0)[2]
    s1 = build_hier_federation(cfg, 1)[2]
    assert s0["features"] == s1["features"] == [int(i) for i in sel_a]


def test_strength_and_magnitude_changes_keep_other_regions_directions():
    cfg = _cfg()
    mean, std = np.zeros(8), np.ones(8)
    base, _ = make_region_shifts(cfg.shift, cfg.hierarchy, mean, std, 3)
    moon_off = dataclasses.replace(cfg.shift, region_strength={"moon": 0.0, "leo": 0.5})
    alt, _ = make_region_shifts(moon_off, cfg.hierarchy, mean, std, 3)
    assert strengths(moon_off, cfg.hierarchy) == {"leo": 0.5, "moon": 0.0, "mars": 1.0}
    assert np.array_equal(alt["mars"].offset, base["mars"].offset)
    assert alt["moon"].amount == 0 and alt["leo"].amount == 0.5
    double, _ = make_region_shifts(dataclasses.replace(cfg.shift, magnitude=2.0), cfg.hierarchy, mean, std, 3)
    np.testing.assert_allclose(double["mars"].offset, 2 * base["mars"].offset)
    explicit, sel = make_region_shifts(dataclasses.replace(cfg.shift, features=(7, 1)), cfg.hierarchy, mean, std, 3)
    assert sel.tolist() == [1, 7] and not explicit["mars"].offset[[0, 2, 3, 4, 5, 6]].any()


def test_shared_holdout_is_shifted_per_region():
    cfg = _cfg("smoke_hier_shift_wesad12")
    _, fed, section = build_hier_federation(cfg, 0)
    leo, moon, mars = (fed.test_of(f"{r}_nasa") for r in ("leo", "moon", "mars"))
    assert _same(leo, fed.holdout)
    assert not _same(moon, fed.holdout) and not _same(mars, moon)
    assert np.array_equal(moon.y, fed.holdout.y)
    assert fed.test_of("moon_esa") is fed.test_of("moon_nasa")  # one test set per region


def test_sequences_shift_per_channel():
    """(n, channels, length) input: offsets are per channel, constant over time."""
    cfg = _cfg()
    rng = np.random.default_rng(0)
    prov = Provenance(dataset="SYNTHETIC", version="t", checksum="c", source="test", kind="SYNTHETIC")

    def bundle(subject):
        return DatasetBundle(X=rng.normal(size=(20, 4, 16)).astype(np.float32), y=rng.integers(0, 3, 20),
                             subject_id=np.full(20, subject), class_names=("a", "b", "c"), provenance=prov)

    nodes = {n.name: NodeSplit(region=cfg.hierarchy.region_of(n.name), train=bundle(3 * i), val=bundle(3 * i + 1),
                               test=bundle(3 * i + 2)) for i, n in enumerate(cfg.hierarchy.nodes)}
    fed, section = apply_shift(HierFederation(nodes=nodes, holdout=None), cfg.hierarchy, cfg.shift)
    assert len(section["features"]) == 2  # half of 4 channels
    before, after = nodes["mars_nasa"].train, fed.nodes["mars_nasa"].train
    assert after.X.shape == before.X.shape and after.X.dtype == np.float32
    untouched = [c for c in range(4) if c not in section["features"]]
    assert np.array_equal(before.X[:, untouched], after.X[:, untouched])
    assert not np.array_equal(before.X[:, section["features"]], after.X[:, section["features"]])
    assert section["regions"]["mars"]["distance_after"]["smd_mean"] > section["regions"]["mars"]["distance_before"]["smd_mean"]


def test_section_fits_the_result_schema_and_is_always_labelled_simulated(shifted):
    _, section = shifted
    r = empty_result()
    r["provenance"] = [{"dataset": "SYNTHETIC", "version": "v", "checksum": "c"}]
    r["shift"] = section
    validate_result(r)
    assert PROVENANCE == section["provenance"] == "SIMULATED" and "not measured" in section["note"]
    assert "SIMULATED" in format_shift_table(section)
    off = build_hier_federation(load_config(EXP / "smoke_hier.yaml"), 0)[2]
    assert off == {"enabled": False, "kind": None, "magnitude": 0.0, "provenance": "SIMULATED"}


def test_config_rules(tmp_path):
    flat = load_config(EXP / "smoke.yaml")
    assert "shift" not in flat.experiment_dict()  # locked run ids unchanged
    assert "shift" in _cfg().experiment_dict()
    with pytest.raises(ConfigError, match="needs a hierarchy"):
        flat.with_overrides(shift=ShiftConfig(enabled=True, magnitude=1.0))
    cfg = _cfg()
    for bad, match in ((dict(kind="drift"), "shift.kind"), (dict(magnitude=-1.0), "magnitude"),
                       (dict(feature_fraction=0.0), "feature_fraction"), (dict(features=(1, 1)), "distinct"),
                       (dict(region_strength={"venus": 1.0}), "unknown regions"),
                       (dict(region_strength={"mars": -1.0}), ">= 0")):
        with pytest.raises(ConfigError, match=match):
            cfg.with_overrides(shift=dataclasses.replace(cfg.shift, **bad))
    with pytest.raises(ValueError, match="out of range"):
        build_hier_federation(_cfg(features=(99,)), 0)
    # the control (magnitude 0) is its own run, distinct from the unshifted config
    assert _cfg(magnitude=0.0).config_hash() != _cfg().config_hash()
