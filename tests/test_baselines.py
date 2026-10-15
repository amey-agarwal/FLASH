import dataclasses
import json

import numpy as np
import pytest

from fedastro.baselines.centralized import run_centralized
from fedastro.baselines.local_only import run_local_only
from fedastro.cli import main
from fedastro.config import ConfigError, TrainConfig, load_config
from fedastro.metrics.aggregate import summarise
from fedastro.metrics.classification import METRICS
from fedastro.pipeline import build_federation


@pytest.fixture
def small(smoke_config):
    cfg = load_config(smoke_config)
    cfg = dataclasses.replace(cfg, train=TrainConfig(epochs=2))
    _, fed = build_federation(cfg, 0)
    return cfg, fed


def test_centralized_pools_every_silo_and_reports_all_metrics(small):
    cfg, fed = small
    out = run_centralized(fed, cfg, 0, "cpu")
    assert out["n_train"] == sum(len(s.data) for s in fed.silos.values())
    assert set(METRICS) <= set(out["final"])
    assert [h["epoch"] for h in out["history"]] == [1, 2]


def test_local_only_one_model_per_silo(small):
    cfg, fed = small
    out = run_local_only(fed, cfg, 0, "cpu")
    assert list(out["per_silo"]) == list(fed.silos)
    for name, s in out["per_silo"].items():
        assert s["n_train"] == len(fed.silos[name].data)
    accs = [s["final"]["accuracy"] for s in out["per_silo"].values()]
    assert out["mean"]["accuracy"] == pytest.approx(np.mean(accs))
    assert out["min"]["accuracy"] == min(accs) and out["max"]["accuracy"] == max(accs)


def test_arm_results_independent_of_which_arms_run(small):
    cfg, fed = small
    a = run_local_only(fed, cfg, 0, "cpu")
    run_centralized(fed, cfg, 0, "cpu")
    b = run_local_only(fed, cfg, 0, "cpu")
    assert a == b



def test_summary_mean_std_and_mixing_guard(tmp_path, smoke_config, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    assert main(["sweep", str(smoke_config), "--seeds", "0", "1", "2", "--output-dir", str(tmp_path)]) == 0
    (summary_path,) = tmp_path.glob("*-summary.json")
    summary = json.loads(summary_path.read_text())
    results = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*/result.json"))]
    accs = [r["arms"]["centralized"]["final"]["accuracy"] for r in results]
    c = summary["arms"]["centralized"]["accuracy"]
    assert c["n"] == 3 and c["mean"] == pytest.approx(np.mean(accs))
    assert c["std"] == pytest.approx(np.std(accs, ddof=1))
    assert set(summary["arms"]["local_only"]["per_silo"]) == {s["name"] for s in results[0]["silos"]}
    other = json.loads(json.dumps(results[0]))
    other["run_id"] = "different-seed9-0123456789"
    other["seed"] = 9
    with pytest.raises(ValueError, match="mix experiments"):
        summarise(results + [other])
