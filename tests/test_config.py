import dataclasses
from pathlib import Path

import pytest
import yaml

from fedastro.config import ConfigError, load_config


def _write_tree(tmp: Path, repo: Path, experiment: dict) -> Path:
    (tmp / "silos").mkdir()
    (tmp / "experiments").mkdir()
    (tmp / "base.yaml").write_text((repo / "configs/base.yaml").read_text())
    (tmp / "silos/five_agency.yaml").write_text((repo / "configs/silos/five_agency.yaml").read_text())
    exp = tmp / "experiments/x.yaml"
    exp.write_text(yaml.safe_dump(experiment))
    return exp


def test_smoke_config_loads(smoke_config):
    cfg = load_config(smoke_config)
    assert cfg.name == "smoke"
    assert cfg.data.is_synthetic
    assert [s.name for s in cfg.topology.silos] == [
        "nasa_earth", "esa_analog", "iss_commercial", "lunar_habitat", "mars_transit",
    ]
    mars = cfg.topology.silos[-1]
    assert set(mars.latency_presets_s) == {"near", "mean", "far"}
    assert cfg.seeds == (0, 1, 2, 3, 4)


def test_unknown_key_rejected(tmp_path, repo):
    exp = _write_tree(tmp_path, repo, {"name": "x", "learning_rat": 0.1})
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(exp)


def test_nested_unknown_key_rejected(tmp_path, repo):
    exp = _write_tree(tmp_path, repo, {"name": "x", "data": {"synthetic": {"n_subject": 3}}})
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(exp)


@pytest.mark.parametrize("bad", [
    {"name": "x", "device": "tpu"},
    {"name": "x", "seed": -1},
    {"name": "x", "seed": "zero"},
    {"name": "x", "data": {"name": "imaginary"}},
    {"name": "has space"},
    {"name": "x", "data": {"synthetic": {"n_subjects": 7}}},  # < silos + holdout
    {"name": "x", "partition": {"method": "dirichlet"}},  # alpha missing
    {"name": "x", "partition": {"method": "by_subject", "alpha": 0.5}},  # alpha unused
    {"name": "x", "partition": {"holdout_subjects": 0}},
    {"name": "x", "data": {"wesad": {"devices": ["ankle"]}}},
    {"name": "x", "data": {"wesad": {"representation": "spectrogram"}}},
    {"name": "x", "data": {"wesad": {"sequence_hz": 8.0}}},
])
def test_invalid_values_rejected(tmp_path, repo, bad):
    exp = _write_tree(tmp_path, repo, bad)
    with pytest.raises(ConfigError):
        load_config(exp)


def test_hash_ignores_execution_keys(smoke_config):
    cfg = load_config(smoke_config)
    other = dataclasses.replace(cfg, seed=7, seeds=(7,), output_dir="/elsewhere")
    assert cfg.config_hash() == other.config_hash()
    changed = dataclasses.replace(cfg, device="cuda")
    assert cfg.config_hash() != changed.config_hash()


def test_recorded_config_omits_unused_dataset_params(smoke_config):
    d = load_config(smoke_config).experiment_dict()
    assert "synthetic" in d["data"] and "wesad" not in d["data"]


@pytest.mark.parametrize("bad", [
    {"name": "x", "model": {"arch": "cnn1d"}},  # synthetic features cannot feed a CNN
    {"name": "x", "data": {"name": "wesad", "wesad": {"representation": "sequence"}}},  # mlp + sequence
    {"name": "x", "model": {"arch": "transformer"}},
    {"name": "x", "model": {"hidden": [64, 64, 64, 64]}},
    {"name": "x", "model": {"kernel_size": 4}},
    {"name": "x", "train": {"optimizer": "lbfgs"}},
    {"name": "x", "train": {"epochs": 0}},
    {"name": "x", "arms": ["centralized", "centralized"]},
])
def test_invalid_model_train_arms_rejected(tmp_path, repo, bad):
    exp = _write_tree(tmp_path, repo, bad)
    with pytest.raises(ConfigError):
        load_config(exp)


def test_cnn_with_sequence_accepted(tmp_path, repo):
    exp = _write_tree(tmp_path, repo, {"name": "x", "model": {"arch": "cnn1d"},
                                       "data": {"name": "wesad", "wesad": {"representation": "sequence"}}})
    assert load_config(exp).model.arch == "cnn1d"


def test_resources_excluded_from_run_identity(smoke_config):
    from fedastro.config import ResourcesConfig
    cfg = load_config(smoke_config)
    gpu = dataclasses.replace(cfg, resources=ResourcesConfig(client_num_cpus=4, client_num_gpus=0.5))
    assert cfg.config_hash() == gpu.config_hash()
    assert "resources" not in cfg.experiment_dict()


@pytest.mark.parametrize("bad", [
    {"name": "x", "device": "cuda", "arms": ["federated"]},  # needs client_num_gpus > 0
    {"name": "x", "fl": {"rounds": 0}},
    {"name": "x", "fl": {"strategy": "scaffold"}},
    {"name": "x", "fl": {"relative_target": 1.5}},
    {"name": "x", "resources": {"client_num_cpus": 0}},
])
def test_invalid_fl_rejected(tmp_path, repo, bad):
    exp = _write_tree(tmp_path, repo, bad)
    with pytest.raises(ConfigError):
        load_config(exp)
