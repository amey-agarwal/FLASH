import itertools
import json

import numpy as np
import pytest

from fedastro.cli import main
from fedastro.fl.strategy import ClientUpdate, FedAvg
from fedastro.metrics.aggregate import _h1_tally
from fedastro.metrics.comms import payload_bytes, rounds_to_target
from fedastro.metrics.schema import validate_result


def _upd(i, value, n):
    return ClientUpdate(i, f"s{i}", [np.full((2, 2), value, np.float32), np.full(3, value, np.float32)], n, 0.0)


def test_fedavg_is_example_weighted_mean():
    out = FedAvg().aggregate([_upd(0, 1.0, 10), _upd(1, 4.0, 30)])
    np.testing.assert_allclose(out[0], 3.25)
    assert out[0].dtype == np.float32 and out[1].shape == (3,)


def test_fedavg_bitwise_independent_of_arrival_order():
    rng = np.random.default_rng(0)
    ups = [ClientUpdate(i, f"s{i}", [rng.normal(size=50).astype(np.float32)], int(n), 0.0)
           for i, n in enumerate([346, 139, 137, 139, 70])]
    ref = FedAvg().aggregate(ups)[0]
    for perm in itertools.permutations(ups):
        assert np.array_equal(FedAvg().aggregate(list(perm))[0], ref)


def test_rounds_to_target_and_payload():
    rounds = [{"round": i, "bytes": 100, "global": {"accuracy": a}} for i, a in enumerate([0.5, 0.7, 0.9], 1)]
    assert rounds_to_target(rounds, 0.7) == {"threshold": 0.7, "metric": "accuracy", "round": 2, "cumulative_bytes": 200,
                                               "sim_wallclock_s": None}
    assert rounds_to_target(rounds, 0.95)["round"] is None
    assert payload_bytes([np.zeros(10, np.float32), np.zeros((2, 3), np.float32)]) == 64


def test_h1_tally_counts_honestly():
    def r(fed, lo, cen):
        return {"arms": {"federated": {"final": {"accuracy": fed}}, "local_only": {"mean": {"accuracy": lo}},
                         "centralized": {"final": {"accuracy": cen}}}}
    t = _h1_tally([r(0.7, 0.6, 0.8), r(0.5, 0.6, 0.8), r(0.9, 0.6, 0.8)])
    assert t["federated_beats_local_only_mean"] == 2
    assert t["federated_between_local_only_and_centralized"] == 1


@pytest.fixture(scope="module")
def fl_runs(tmp_path_factory, repo=None):
    """Two identical federated smoke runs through the real Flower/Ray simulation."""
    import os
    from pathlib import Path
    cfg = Path(__file__).resolve().parents[1] / "configs/experiments/smoke_fl.yaml"
    root = tmp_path_factory.mktemp("fl")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        for sub in ("a", "b"):
            assert main(["run", str(cfg), "--seed", "0", "--output-dir", str(root / sub)]) == 0
    finally:
        del os.environ["SOURCE_DATE_EPOCH"]
    return [next((root / sub).glob("*/result.json")) for sub in ("a", "b")]


def test_federated_smoke_result_contract(fl_runs):
    result = json.loads(fl_runs[0].read_text())
    validate_result(result)
    fed = result["arms"]["federated"]
    silos = [s["name"] for s in result["silos"]]
    assert [r["round"] for r in fed["rounds"]] == [1, 2, 3]
    for r in fed["rounds"]:
        assert r["participants"] == silos
        assert r["bytes"] == fed["payload_bytes"] * len(silos) * 2
        assert r["sim_wallclock_s"] is None
        assert set(r["per_silo"]) == set(silos)
        for name, ps in r["per_silo"].items():
            assert {"update_on_holdout", "global_on_local", "train_loss", "n_train"} <= set(ps)
    assert fed["final"] == fed["rounds"][-1]["global"]
    assert fed["payload_bytes"] == fed["n_params"] * 4
    assert fed["total_bytes"] == fed["setup_bytes"] + sum(r["bytes"] for r in fed["rounds"])
    assert "relative_to_centralized" in fed["rounds_to_target"]


def test_federated_run_is_byte_identical(fl_runs):
    assert fl_runs[0].read_bytes() == fl_runs[1].read_bytes()


def test_federated_scaler_matches_pooled_fit():
    """The setup exchange (sum of per-silo sufficient statistics) equals a pooled fit."""
    from fedastro.config import SyntheticParams
    from fedastro.data.partition import make_federation
    from fedastro.data.preprocess import Standardizer, Stats
    from fedastro.data.synthetic import load_synthetic
    bundle = load_synthetic(SyntheticParams(n_subjects=20), 0)
    fed = make_federation(bundle, {"a": 0.5, "b": 0.3, "c": 0.2}, "by_subject", 3, np.random.default_rng(0))
    stats = [Stats.of(s.data.X) for s in fed.silos.values()]
    combined = stats[0] + stats[1] + stats[2]
    pooled = Standardizer.fit(np.concatenate([s.data.X for s in fed.silos.values()]))
    np.testing.assert_allclose(Standardizer.from_stats(combined).mean, pooled.mean, rtol=1e-10)
