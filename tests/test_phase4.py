import dataclasses
import json

import numpy as np
import pytest
import torch
import yaml

from fedastro.cli import main
from fedastro.config import ConfigError, FLConfig, ModelConfig, TrainConfig, load_config
from fedastro.fl.strategy import FedAvg, FedProx, build_strategy, client_proximal_mu
from fedastro.grid import load_grid
from fedastro.metrics.classification import evaluate
from fedastro.metrics.fairness import fairness_report, label_mix_accuracy
from fedastro.models.model import build_model
from fedastro.training import train_local


def _toy(n=96, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n).astype(np.int64)
    X = (np.eye(3)[y] * 2 + rng.normal(size=(n, 3))).astype(np.float32)
    return X, y


def _drift(mu):
    X, y = _toy()
    torch.manual_seed(0)
    model = build_model(ModelConfig(), (3,), 3)
    start = [p.detach().clone() for p in model.parameters()]
    train_local(model, X, y, TrainConfig(epochs=5), 3, seed=1, proximal_mu=mu)
    return sum(((p.detach() - s) ** 2).sum().item() for p, s in zip(model.parameters(), start))


def test_proximal_term_limits_drift_from_global_weights():
    assert _drift(10.0) < 0.5 * _drift(0.0)


def test_mu_zero_is_plain_training():
    X, y = _toy()

    def weights(**kw):
        torch.manual_seed(0)
        m = build_model(ModelConfig(), (3,), 3)
        train_local(m, X, y, TrainConfig(epochs=2), 3, seed=1, **kw)
        return torch.cat([p.detach().flatten() for p in m.parameters()])

    assert torch.equal(weights(), weights(proximal_mu=0.0))


def test_strategy_selection_and_client_mu():
    assert type(build_strategy(FLConfig())) is FedAvg
    prox = build_strategy(FLConfig(strategy="fedprox", proximal_mu=0.1))
    assert isinstance(prox, FedProx) and prox.mu == 0.1
    assert client_proximal_mu(FLConfig(strategy="fedavg", proximal_mu=0.1)) == 0.0
    assert client_proximal_mu(FLConfig(strategy="fedprox", proximal_mu=0.1)) == 0.1


def test_mu_only_recorded_for_fedprox(smoke_config):
    cfg = load_config(smoke_config)
    assert "proximal_mu" not in cfg.experiment_dict()["fl"]
    changed_mu = dataclasses.replace(cfg, fl=dataclasses.replace(cfg.fl, proximal_mu=0.5))
    assert cfg.config_hash() == changed_mu.config_hash(), "unused mu must not change FedAvg run ids"
    prox = load_config(smoke_config, {"fl.strategy": "fedprox"})
    assert prox.experiment_dict()["fl"]["proximal_mu"] == 0.01
    with pytest.raises(ConfigError):
        load_config(smoke_config, {"fl.strategy": "fedprox", "fl.proximal_mu": 0.0})


def test_overrides_are_validated(smoke_config):
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(smoke_config, {"partition.alpah": 0.1})


def test_recall_per_class_reported():
    y = np.array([0, 0, 1, 2])
    proba = np.eye(3)[[0, 1, 1, 2]]
    assert evaluate(y, proba)["recall_per_class"] == [0.5, 1.0, 1.0]


def test_label_mix_accuracy_and_report():
    names = ["a", "b", "c"]
    assert label_mix_accuracy([1.0, 0.0, 0.5], {"a": 2, "b": 0, "c": 2}, names) == pytest.approx(0.75)
    result = {
        "silos": [{"name": "s1", "label_counts": {"a": 1, "b": 0, "c": 0}},
                  {"name": "s2", "label_counts": {"a": 0, "b": 1, "c": 0}}],
        "arms": {
            "centralized": {"final": {"recall_per_class": [0.9, 0.5, 0.7]}},
            "federated": {"final": {}},
            "local_only": {"per_silo": {"s1": {"final": {"recall_per_class": [1.0, 0.0, 0.0]}},
                                        "s2": {"final": {"recall_per_class": [0.0, 1.0, 0.0]}}}},
        },
    }
    rep = fairness_report(result, names)
    assert rep["centralized"]["spread"] == pytest.approx(0.4)
    assert rep["local_only"]["spread"] == 0.0 and rep["local_only"]["mean"] == 1.0
    assert "federated" not in rep


def _sweep(tmp_path, repo, axes, **extra):
    spec = {"name": "t", "base": str(repo / "configs/experiments/smoke.yaml"), "seeds": [0], "axes": axes, **extra}
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def test_grid_expands_cartesian_product(tmp_path, repo):
    path = _sweep(tmp_path, repo, {"partition.method": ["dirichlet", "dirichlet_quantity_control"],
                                   "partition.alpha": [0.1, 1.0]})
    name, variants = load_grid(path)
    assert name == "t" and len(variants) == 4
    assert len({v.name for v in variants}) == 4
    v = variants[1]
    assert v.config.partition.method == "dirichlet" and v.config.partition.alpha == 1.0
    assert v.config.seeds == (0,)
    assert len({v.config.config_hash() for v in variants}) == 4


def test_grid_rejects_bad_specs(tmp_path, repo):
    with pytest.raises(ConfigError):
        load_grid(_sweep(tmp_path, repo, {"partition.alpha": []}))
    with pytest.raises(ConfigError):
        load_grid(_sweep(tmp_path, repo, {"train.epochs": [1]}, extra_key=1))


def test_grid_index_and_resume(tmp_path, repo, monkeypatch, capsys):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    path = _sweep(tmp_path, repo, {"train.epochs": [1, 2]})
    out = tmp_path / "runs"
    assert main(["grid", str(path), "--index", "1", "--output-dir", str(out)]) == 0
    results = list(out.glob("*/result.json"))
    assert len(results) == 1 and json.loads(results[0].read_text())["config"]["train"]["epochs"] == 2
    stamp = results[0].stat().st_mtime_ns
    assert main(["grid", str(path), "--index", "1", "--resume", "--output-dir", str(out)]) == 0
    assert results[0].stat().st_mtime_ns == stamp, "--resume must not rerun an existing run"
    assert main(["grid", str(path), "--index", "7", "--output-dir", str(out)]) == 2


def test_grid_report_collects_variant_summaries(tmp_path, repo, monkeypatch, capsys):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    path = _sweep(tmp_path, repo, {"train.epochs": [1, 2]})
    out = tmp_path / "runs"
    assert main(["grid", str(path), "--index", "0", "--output-dir", str(out)]) == 0
    capsys.readouterr()
    assert main(["grid", str(path), "--report", "--output-dir", str(out)]) == 0
    text = capsys.readouterr().out
    assert "t-epochs1" in text and "missing" in text  # variant 1 never ran
    rows = json.loads((out / "t-grid.json").read_text())["rows"]
    assert [r["variant"] for r in rows] == ["t-epochs1"]
    assert rows[0]["local_only_accuracy"]["n"] == 1


def test_dirichlet_large_alpha_recovers_natural_mix():
    """With Dir(alpha * prior), huge alpha keeps each silo near the pool's own class mix
    (a symmetric prior would wrongly pull silos towards uniform proportions)."""
    from fedastro.config import SyntheticParams
    from fedastro.data.partition import dirichlet
    from fedastro.data.synthetic import load_synthetic
    bundle = load_synthetic(SyntheticParams(n_subjects=30, windows_per_subject=40), 0)
    keep = np.isin(bundle.y, [0]) | (np.arange(len(bundle)) % 3 == 0)  # skew the pool: class 0 dominant
    pool = bundle.subset(keep)
    prior = np.bincount(pool.y, minlength=3) / len(pool)
    assert prior[0] > 0.5
    parts = dirichlet(pool, {"a": 0.5, "b": 0.5}, 1e6, np.random.default_rng(0))
    for split in parts.values():
        np.testing.assert_allclose(split.target_proportions, prior, atol=1e-3)
        mix = np.bincount(split.data.y, minlength=3) / len(split.data)
        np.testing.assert_allclose(mix, prior, atol=0.05)


def test_grid_continues_after_a_failed_run(tmp_path, repo, monkeypatch, capsys):
    import fedastro.cli as cli
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    real = cli.run_and_write
    calls = []

    def flaky(cfg, seed):
        calls.append(cfg.name)
        if cfg.train.epochs == 1:
            raise RuntimeError("simulated stall")
        return real(cfg, seed)

    monkeypatch.setattr(cli, "run_and_write", flaky)
    path = _sweep(tmp_path, repo, {"train.epochs": [1, 2]})
    assert main(["grid", str(path), "--output-dir", str(tmp_path / "runs")]) == 1
    assert calls == ["t-epochs1", "t-epochs2"], "second variant must still run"
    err = capsys.readouterr().err
    assert "FAILED t-epochs1 seed 0" in err and "--resume" in err
    assert len(list((tmp_path / "runs").glob("t-epochs2-*-summary.json"))) == 1
