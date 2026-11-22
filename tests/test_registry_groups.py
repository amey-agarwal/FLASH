"""Phase 11: dataset registry, provenance domain, grouped data, by_group, holdout_groups."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from fedastro import config as config_mod
from fedastro.cli import main
from fedastro.config import ConfigError, SyntheticParams, load_config
from fedastro.data import REGISTRY, dataset_names, domain_of, get_dataset
from fedastro.data.base import DOMAINS, GroupedDatasetBundle, Provenance
from fedastro.data.partition import by_group, make_federation
from fedastro.data.preprocess import Standardizer
from fedastro.data.synthetic import load_synthetic
from fedastro.metrics.schema import SchemaError, validate_result
from fedastro.pipeline import build_federation

REPO = Path(__file__).resolve().parents[1]
GROUPED = REPO / "configs" / "experiments" / "smoke_grouped.yaml"
SHARES = {"nasa_earth": 0.4, "esa_analog": 0.2, "iss_commercial": 0.2, "lunar_habitat": 0.12, "mars_transit": 0.08}


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _grouped(n_subjects=24, n_groups=8, seed=0) -> GroupedDatasetBundle:
    return load_synthetic(SyntheticParams(n_subjects=n_subjects, n_groups=n_groups), seed)


# -- registry ------------------------------------------------------------------------
def test_registry_drives_config_validation():
    assert config_mod.DATASETS == dataset_names() == ("synthetic", "wesad", "osd435", "cinc2021")
    assert all(spec.domain in DOMAINS for spec in REGISTRY.values())
    with pytest.raises(KeyError):
        get_dataset("mimic")  # not a registered dataset


def test_registry_path_gives_identical_data_and_byte_identical_smoke(tmp_path, smoke_config):
    cfg = load_config(smoke_config)
    via_registry = get_dataset("synthetic").load(cfg.data, 0)
    direct = load_synthetic(cfg.data.synthetic, 0)
    for f in ("X", "y", "subject_id"):
        assert np.array_equal(getattr(via_registry, f), getattr(direct, f))
    # The registry adds the domain and changes nothing else.
    assert via_registry.provenance == dataclasses.replace(direct.provenance, domain="synthetic")
    # `fedastro run` loads through the registry; its result equals the smoke lock, which
    # differs from the Phase 10 version only by the domain line (approved 2026-11-05).
    golden = _module("golden", REPO / "scripts" / "golden.py")
    assert golden.smoke_result_bytes(tmp_path) == golden.SMOKE_RESULT.read_bytes()


def test_run_report_legacy_domains_match_registry():
    report = _module("run_report", REPO / "scripts" / "run_report.py")
    assert report.LEGACY_DOMAINS == {spec.label: spec.domain for spec in REGISTRY.values()}


# -- provenance domain ---------------------------------------------------------------
def test_registry_stamps_the_domain_and_old_artifacts_still_resolve(smoke_config):
    cfg = load_config(smoke_config)
    assert get_dataset("synthetic").load(cfg.data, 0).provenance.to_dict()["domain"] == "synthetic"
    p = load_synthetic(SyntheticParams(), 0).provenance  # built outside the registry
    assert "domain" not in p.to_dict()
    # Artifacts written before Phase 11 have no domain field; readers fall back to the registry.
    assert domain_of(p.to_dict()) == "synthetic"
    assert domain_of({"dataset": "WESAD"}) == "terrestrial_proxy"
    assert domain_of({"dataset": "OTHER", "domain": "terrestrial_analog"}) == "terrestrial_analog"
    assert domain_of({"dataset": "OTHER"}) is None


def test_registry_rejects_a_loader_with_a_contradicting_domain(monkeypatch, smoke_config):
    import fedastro.data.synthetic as synthetic

    original = synthetic.load

    def wrong(data, seed):
        b = original(data, seed)
        return dataclasses.replace(b, provenance=dataclasses.replace(b.provenance, domain="terrestrial_analog"))

    monkeypatch.setattr(synthetic, "load", wrong)
    with pytest.raises(ValueError, match="registry"):
        get_dataset("synthetic").load(load_config(smoke_config).data, 0)


def test_domain_is_validated():
    p = dataclasses.replace(load_synthetic(SyntheticParams(), 0).provenance, domain="terrestrial_analog")
    assert p.to_dict()["domain"] == "terrestrial_analog"
    with pytest.raises(ValueError):
        Provenance("d", "v", "c", "s", "REAL", domain="spaceflight_human")
    result = json.loads((REPO / "tests" / "golden" / "smoke_result.json").read_text())
    validate_result(result)
    result["provenance"][0]["domain"] = "spaceflight_human"
    with pytest.raises(SchemaError):
        validate_result(result)


# -- grouped bundles -----------------------------------------------------------------
def test_grouped_synthetic_bundle():
    b = _grouped()
    assert isinstance(b, GroupedDatasetBundle)
    assert b.groups == [f"group_{g}" for g in range(8)]
    sub = b.subset(b.in_groups(["group_1"]))
    assert isinstance(sub, GroupedDatasetBundle) and sub.groups == ["group_1"]
    assert sub.subjects == [3, 4, 5]  # contiguous blocks of 24 // 8 = 3 subjects
    with pytest.raises(ValueError):
        b.in_groups(["group_99"])


def test_ungrouped_synthetic_unchanged_by_the_groups_option():
    a = load_synthetic(SyntheticParams(), 0)
    assert type(a).__name__ == "DatasetBundle"
    b = _grouped(n_subjects=20, n_groups=4)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.subject_id, b.subject_id)  # same draws first


def test_a_subject_in_two_groups_is_rejected():
    b = _grouped()
    gid = b.group_id.copy()
    gid[0] = 7  # subject 0 now spans group_0 and group_7
    with pytest.raises(ValueError, match="more than one group"):
        dataclasses.replace(b, group_id=gid)


# -- by_group and holdout_groups -----------------------------------------------------
def test_by_group_one_silo_per_group_and_mapped():
    b = _grouped(n_groups=5)
    names = [f"group_{g}" for g in range(5)]
    parts = by_group(b, names)
    assert {n: p.groups for n, p in parts.items()} == {n: [n] for n in names}
    mapped = by_group(b, ["a", "b"], {"a": ("group_0", "group_1"), "b": ("group_2",)})
    assert mapped["a"].groups == ["group_0", "group_1"] and mapped["b"].groups == ["group_2"]
    with pytest.raises(ValueError):
        by_group(load_synthetic(SyntheticParams(), 0), names)  # not grouped
    with pytest.raises(ValueError):
        by_group(b, ["a"], {"a": ("group_42",)})


def test_holdout_groups_contribute_no_samples():
    b = _grouped()
    group_map = {"nasa_earth": ("group_0", "group_1"), "esa_analog": ("group_2",), "iss_commercial": ("group_3",),
                 "lunar_habitat": ("group_4",), "mars_transit": ("group_5",)}
    fed = make_federation(b, SHARES, "by_group", 2, np.random.default_rng(0),
                          group_map=group_map, holdout_groups=("group_6",))
    held_subjects = set(b.subset(b.in_groups(["group_6"])).subjects)
    unused_subjects = set(b.subset(b.in_groups(["group_7"])).subjects)
    for split in fed.silos.values():
        assert "group_6" not in split.data.groups and "group_7" not in split.data.groups
        assert not held_subjects & set(split.data.subjects)
    assert held_subjects <= set(fed.holdout.subjects)
    assert not unused_subjects & set(fed.holdout.subjects)  # unused groups are nowhere
    assert fed.groups == {"silo_to_groups": {k: list(v) for k, v in group_map.items()},
                          "holdout_groups": ["group_6"], "unused_groups": ["group_7"]}
    n_train = sum(len(s.data) for s in fed.silos.values())
    assert n_train + len(fed.holdout) + len(unused_subjects) * 30 == len(b)
    assert len(fed.holdout.subjects) == 3 + 2  # the whole group + 2 held-out individuals


def test_holdout_groups_need_grouped_data():
    with pytest.raises(ValueError, match="no groups"):
        make_federation(load_synthetic(SyntheticParams(), 0), SHARES, "by_subject", 3,
                        np.random.default_rng(0), holdout_groups=("group_0",))


def _sentinel_loader(original, value=1e6):
    """Wrap a loader so the held-out group's features are an impossible value."""
    def load(data, seed):
        b = original(data, seed)
        X = b.X.copy()
        X[b.in_groups(["group_6"])] = value
        return dataclasses.replace(b, X=X)
    return load


def test_holdout_groups_contribute_no_scaler_statistics(monkeypatch):
    """The held-out group's features are 1e6. If any of its samples reached a silo, the
    centralized, local-only or federated scaler would show it. The federated scaler is
    taken from the real Flower stats exchange (one round)."""
    import fedastro.data.synthetic as synthetic
    from fedastro.fl.server import run_federated

    monkeypatch.setattr(synthetic, "load", _sentinel_loader(synthetic.load))
    cfg = load_config(GROUPED, {"arms": ["federated"], "fl.rounds": 1})
    _, fed = build_federation(cfg, 0)
    assert np.abs(fed.holdout.X).max() == 1e6  # the sentinel is really in the data
    silo_X = np.concatenate([s.data.X for s in fed.silos.values()])
    assert np.abs(silo_X).max() < 1e3  # centralized and local-only scalers fit on silo data only
    capture = {}
    run_federated(fed, cfg, 0, "cpu", None, capture)
    expected = Standardizer.fit(silo_X)
    np.testing.assert_allclose(capture["scaler"].mean, expected.mean, rtol=1e-6)
    np.testing.assert_allclose(capture["scaler"].std, expected.std, rtol=1e-6)
    assert np.abs(capture["scaler"].mean).max() < 1e3


# -- config --------------------------------------------------------------------------
def test_new_fields_stay_out_of_the_run_id_when_unused(smoke_config):
    cfg = load_config(smoke_config)
    d = cfg.experiment_dict()
    assert "n_groups" not in d["data"]["synthetic"]
    assert "group_map" not in d["partition"] and "holdout_groups" not in d["partition"]
    grouped = load_config(GROUPED).experiment_dict()
    assert grouped["data"]["synthetic"]["n_groups"] == 8 and grouped["partition"]["holdout_groups"] == ["group_6"]


@pytest.mark.parametrize("overrides,match", [
    ({"partition.group_map": {"jaxa": ["group_7"]}}, "one entry per topology silo"),  # merged: 6 silos
    ({"partition.holdout_groups": ["group_0"]}, "cannot also train"),
    ({"partition.method": "by_subject", "partition.holdout_groups": []}, "only meaningful for method 'by_group'"),
    ({"data.synthetic.n_groups": 0}, "need grouped data"),
    ({"partition.holdout_groups": [], "partition.holdout_subjects": 0}, "holdout_subjects must be >= 1"),
])
def test_grouped_config_validation(overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_config(GROUPED, overrides)


def test_holdout_subjects_zero_is_allowed_with_holdout_groups():
    cfg = load_config(GROUPED, {"partition.holdout_subjects": 0})
    _, fed = build_federation(cfg, 0)
    assert fed.holdout.groups == ["group_6"]


def test_grouped_smoke_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    assert main(["run", str(GROUPED), "--seed", "0", "--output-dir", str(tmp_path)]) == 0
    (path,) = tmp_path.glob("*/result.json")
    r = json.loads(path.read_text())
    validate_result(r)
    assert r["groups"]["holdout_groups"] == ["group_6"] and r["groups"]["unused_groups"] == ["group_7"]
    assert [s["groups"] for s in r["silos"]][0] == ["group_0", "group_1"]
    assert "group_6" in r["holdout"]["groups"] and "group_7" not in r["holdout"]["groups"]
    assert r["provenance"][0]["domain"] == "synthetic"
    assert "Held-out groups" in r["notes"] and "Unused groups" in r["notes"]
    assert r["arms"]["federated"]["final"]["accuracy"] is not None
    # Phase 12: every arm, the federated one included, is scored on the held-out group
    assert r["evaluation"]["per_holdout_group"]["group_6"]["federated"]["n"] == 3 * 30
