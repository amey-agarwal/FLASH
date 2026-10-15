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
    {"name": "x", "data": {"synthetic": {"n_subjects": 2}}},  # fewer subjects than silos
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
