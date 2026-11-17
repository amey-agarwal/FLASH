"""Phase H6: model comparison (logreg / MLP / 1D-CNN), calibration and the model report."""

import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from fedastro.config import CalibrationConfig, ConfigError, ModelConfig, load_config
from fedastro.data.synthetic import load_synthetic
from fedastro.grid import load_grid
from fedastro.metrics.calibration import ece, reliability_bins
from fedastro.metrics.schema import validate_result
from fedastro.models.model import LogReg, build_model, count_parameters
from fedastro.pipeline import run_experiment

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "configs" / "experiments"
SWEEP = REPO / "configs" / "sweeps" / "model_families_smoke.yaml"


# -- models ------------------------------------------------------------------------------


def test_logreg_is_one_linear_layer():
    m = build_model(ModelConfig(arch="logreg"), (8,), 3)
    assert isinstance(m, LogReg) and count_parameters(m) == 8 * 3 + 3
    with pytest.raises(ValueError, match="1-D input"):
        build_model(ModelConfig(arch="logreg"), (4, 16), 3)


def test_input_rules():
    cfg = load_config(EXP / "smoke.yaml")
    seq = dataclasses.replace(cfg.data, synthetic=dataclasses.replace(cfg.data.synthetic, representation="sequence"))
    for arch in ("logreg", "mlp"):
        with pytest.raises(ConfigError, match="needs feature input"):
            dataclasses.replace(cfg, model=ModelConfig(arch=arch), data=seq).validate()
    with pytest.raises(ConfigError, match="needs sequence input"):
        dataclasses.replace(cfg, model=ModelConfig(arch="cnn1d")).validate()
    dataclasses.replace(cfg, model=ModelConfig(arch="cnn1d"), data=seq).validate()
    with pytest.raises(ConfigError, match="calibration settings need calibration.enabled"):
        dataclasses.replace(cfg, calibration=CalibrationConfig(n_bins=15)).validate()


def test_defaults_stay_out_of_the_run_id():
    cfg = load_config(EXP / "smoke.yaml")
    d = cfg.experiment_dict()
    assert "calibration" not in d and "representation" not in d["data"]["synthetic"]
    on = dataclasses.replace(cfg, calibration=CalibrationConfig(enabled=True))
    assert on.config_hash() != cfg.config_hash() and on.experiment_dict()["calibration"]["n_bins"] == 10


# -- synthetic sequences ----------------------------------------------------------------


def test_synthetic_sequences_are_seeded_and_leave_features_unchanged():
    p = load_config(EXP / "smoke.yaml").data.synthetic
    feats = load_synthetic(p, 0)
    sq = dataclasses.replace(p, representation="sequence", seq_length=32)
    a, b = load_synthetic(sq, 0), load_synthetic(sq, 0)
    assert a.X.shape == (len(feats), p.n_features, 32) and a.X.dtype == np.float32
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, feats.y)
    assert a.provenance.kind == "SYNTHETIC" and a.provenance.checksum != feats.provenance.checksum
    assert np.array_equal(load_synthetic(p, 0).X, feats.X)  # the feature path is untouched
    r = np.corrcoef(a.X.mean(axis=2).ravel(), feats.X.ravel())[0, 1]
    assert r > 0.8  # a channel's mean level tracks its feature


# -- calibration ------------------------------------------------------------------------


def test_ece_known_cases():
    y = np.array([0, 1, 0, 1])
    assert ece(y, np.eye(2)[y]) == 0.0  # confident and right
    p = np.array([[0.8, 0.2]] * 4)  # 80% confident in class 0, right half the time
    assert ece(y, p) == pytest.approx(0.3)
    bins = reliability_bins(y, p)
    assert len(bins) == 10 and sum(b["n"] for b in bins) == 4 and bins[8]["n"] == 4
    assert bins[8]["accuracy"] == 0.5 and bins[8]["confidence"] == pytest.approx(0.8)
    assert ece(np.array([], dtype=int), np.zeros((0, 2))) is None


def test_ece_top_bin_includes_confidence_one():
    y = np.array([0, 0])
    bins = reliability_bins(y, np.array([[1.0, 0.0], [1.0, 0.0]]), 10)
    assert bins[-1]["n"] == 2


# -- the sweep --------------------------------------------------------------------------


def test_family_sweep_pairs_architecture_and_representation():
    _, variants = load_grid(SWEEP)
    got = [(v.config.model.arch, v.config.data.synthetic.representation) for v in variants]
    assert got == [("logreg", "features"), ("mlp", "features"), ("cnn1d", "sequence")]


# -- through every arm and the hierarchy ------------------------------------------------


@pytest.mark.parametrize("index", [0, 1, 2], ids=["logreg", "mlp", "cnn1d"])
def test_every_family_runs_through_every_arm_and_the_hierarchy(index):
    v = load_grid(SWEEP)[1][index]
    t0 = time.perf_counter()
    r = run_experiment(v.config, 0)
    elapsed = time.perf_counter() - t0
    validate_result(r)
    for arm in ("centralized", "local_only", "federated", "ensemble_of_locals", "hierarchical"):
        assert r["arms"][arm], arm
    mr = r["model_report"]
    assert mr["arch"] == v.config.model.arch and mr["n_params"] > 0 and mr["payload_bytes"] == 4 * mr["n_params"]
    assert {"local", "regional", "global", "flat_federated", "centralized"} <= set(mr["rows"])
    for row in mr["rows"].values():
        assert 0 <= row["ece"] <= 1 and 0 <= row["worst_node_accuracy"] <= row["accuracy"] + 1e-12
    cal = r["calibration"]
    assert cal["n_bins"] == 10 and set(cal["ece"]) == set(mr["rows"])
    assert r["arms"]["hierarchical"]["n_params"] == mr["n_params"]
    print(f"{v.config.model.arch}: {elapsed:.0f} s")


def test_flat_calibration_changes_no_arm_result(tmp_path):
    """smoke.yaml with calibration on: the arms equal the golden smoke result's, and the
    report and calibration are filled."""
    golden = json.loads((REPO / "tests" / "golden" / "smoke_result.json").read_text())
    cfg = load_config(EXP / "smoke.yaml")
    cfg = dataclasses.replace(cfg, calibration=CalibrationConfig(enabled=True))
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        r = run_experiment(cfg, 0)
    finally:
        del os.environ["SOURCE_DATE_EPOCH"]
    validate_result(r)
    assert json.dumps(r["arms"], sort_keys=True) == json.dumps(golden["arms"], sort_keys=True)
    mr = r["model_report"]
    assert set(mr["rows"]) == {"centralized", "local_only"} and mr["rows"]["centralized"]["worst_node_accuracy"] is None
    lo = mr["rows"]["local_only"]
    assert lo["worst_node_accuracy"] == min(v["final"]["accuracy"] for v in r["arms"]["local_only"]["per_silo"].values())
    assert set(r["calibration"]["ece_local_only_per_silo"]) == set(r["arms"]["local_only"]["per_silo"])
