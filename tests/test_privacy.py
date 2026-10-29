"""Phase 5: DP-SGD accounting/wiring and the gradient-inversion attack."""

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedastro.cli import main
from fedastro.config import AttackConfig, ConfigError, PrivacyConfig, load_config
from fedastro.metrics.aggregate import load_results, summarise
from fedastro.metrics.schema import validate_result
from fedastro.models.model import CNN1D, MLP
from fedastro.privacy.attack import invert, released_gradient, score
from fedastro.privacy.dp import DPPlan, plan_dp
from fedastro.training import train_local

REPO = Path(__file__).resolve().parents[1]
ON = PrivacyConfig(enabled=True, target_epsilon=2.0)


# --- accounting -------------------------------------------------------------------

def test_delta_strictly_below_one_over_n_and_epsilon_within_budget():
    plan = plan_dp(ON, n_train=345, batch_size=32, epochs=30)
    assert plan.delta < 1 / 345
    assert plan.steps_per_epoch == 11 and plan.sample_rate == pytest.approx(1 / 11)
    spent = plan.epsilon_after(30)
    assert 1.95 <= spent <= 2.0  # get_noise_multiplier's 0.01 tolerance, never over budget
    eps = [plan.epsilon_after(k) for k in range(0, 31, 5)]
    assert eps[0] == 0.0 and all(a < b for a, b in zip(eps, eps[1:]))


def test_subject_level_group_privacy_bound():
    from fedastro.privacy.dp import records_per_subject, subject_level

    assert records_per_subject(np.array([4, 4, 9, 4, 9])) == 3
    one = subject_level(1.0, 1e-4, 1)
    assert one["epsilon"] == 1.0 and one["delta"] == pytest.approx(1e-4) and not one["vacuous"]
    small = subject_level(0.5, 1e-4, 3)
    assert small["epsilon"] == 1.5 and small["delta"] == pytest.approx(3 * np.exp(1.0) * 1e-4)
    wesad = subject_level(1.0, 0.1 / 345, 70)  # a WESAD subject: ~70 windows
    assert wesad["epsilon"] == 70.0 and wesad["vacuous"] and wesad["delta"] == 1.0


def test_smaller_epsilon_needs_more_noise():
    sig = [plan_dp(dataclasses.replace(ON, target_epsilon=e), 345, 32, 30).noise_multiplier
           for e in (0.5, 1, 2, 3, 5, 10, 13)]
    assert all(a > b for a, b in zip(sig, sig[1:]))


def test_split_calibration_matches_make_private_with_epsilon():
    """Our plan (get_noise_multiplier + make_private, closed-form epsilon) equals
    Opacus's make_private_with_epsilon + get_epsilon for one training call."""
    from opacus import PrivacyEngine

    n, b, epochs = 96, 32, 4
    plan = plan_dp(ON, n, b, epochs)
    torch.manual_seed(0)
    model = MLP(5, (8,), 3, 0.0)
    loader = DataLoader(TensorDataset(torch.randn(n, 5), torch.randint(0, 3, (n,))), batch_size=b)
    engine = PrivacyEngine(accountant="rdp")
    m, opt, dl = engine.make_private_with_epsilon(
        module=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), data_loader=loader,
        target_epsilon=ON.target_epsilon, target_delta=plan.delta, epochs=epochs, max_grad_norm=1.0)
    assert opt.noise_multiplier == pytest.approx(plan.noise_multiplier)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            nn.functional.cross_entropy(m(xb), yb).backward()
            opt.step()
    assert engine.get_epsilon(plan.delta) == pytest.approx(plan.epsilon_after(epochs), rel=1e-9)


def test_train_local_dp_is_deterministic_and_leaves_a_plain_module():
    from fedastro.config import TrainConfig

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(100, 6)).astype(np.float32), rng.integers(0, 3, 100)
    cfg = TrainConfig(epochs=2)
    plan = plan_dp(ON, 100, cfg.batch_size, 2)

    def run(dp):
        torch.manual_seed(0)
        m = MLP(6, (8,), 3, 0.2)
        train_local(m, X, y, cfg, 3, seed=7, dp=dp)
        return m

    a, b, plain = run(plan), run(plan), run(None)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert not all(torch.equal(p, q) for p, q in zip(a.parameters(), plain.parameters()))
    assert list(a.state_dict()) == list(plain.state_dict())
    assert all(not mod._forward_hooks and not mod._backward_hooks for mod in a.modules())


def test_dp_rejects_proximal_term():
    from fedastro.config import TrainConfig

    X, y = np.zeros((40, 3), np.float32), np.zeros(40, np.int64)
    with pytest.raises(ValueError):
        train_local(MLP(3, (4,), 2, 0.0), X, y, TrainConfig(epochs=1), 2, 0, proximal_mu=0.1,
                    dp=plan_dp(ON, 40, 32, 1))


# --- config ---------------------------------------------------------------------

SMOKE = REPO / "configs/experiments/smoke_fl.yaml"


@pytest.mark.parametrize("overrides, message", [
    ({"privacy.enabled": True}, "target_epsilon must be > 0"),
    ({"privacy.target_epsilon": 1.0}, "only meaningful when privacy.enabled"),
    ({"privacy.enabled": True, "privacy.target_epsilon": 1.0, "privacy.delta_factor": 1.0}, "delta < 1/N_train"),
    ({"privacy.enabled": True, "privacy.target_epsilon": 1.0, "fl.strategy": "fedprox"}, "fedprox"),
    ({"privacy.enabled": True, "privacy.target_epsilon": 1.0, "train.class_weighting": "balanced"}, "class_weighting"),
    ({"privacy.attack.enabled": True, "arms": ["centralized"]}, "needs the federated arm"),
    ({"privacy.attack.enabled": True, "privacy.attack.silos": ["pluto"]}, "not silos of the topology"),
    ({"privacy.attack.label_modes": ["guessed"]}, "label_modes"),
    ({"privacy.attack.silos": ["nasa_earth", "nasa_earth"]}, "must not repeat"),
    ({"privacy.enabled": "yes"}, "expected true/false"),
])
def test_privacy_config_validation(overrides, message):
    with pytest.raises(ConfigError, match=message):
        load_config(SMOKE, overrides)


def test_privacy_absent_from_experiment_identity_when_unused():
    """Pre-Phase-5 configs must keep their run ids."""
    off = load_config(SMOKE)
    assert "privacy" not in off.experiment_dict()
    on = load_config(SMOKE, {"privacy.enabled": True, "privacy.target_epsilon": 1.0})
    assert "attack" not in on.experiment_dict()["privacy"]
    assert on.config_hash() != off.config_hash()


# --- attack ---------------------------------------------------------------------

def _batch(n=4, d=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randint(0, 3, (n,), generator=g)


def test_released_gradient_without_dp_is_the_plain_gradient():
    torch.manual_seed(0)
    m = MLP(10, (16,), 3, 0.0)
    X, y = _batch()
    g = released_gradient(m, X, y, None, seed=0)
    m.zero_grad()
    nn.functional.cross_entropy(m(X), y).backward()
    assert all(torch.allclose(a, p.grad) for a, p in zip(g, m.parameters()))


def test_released_gradient_with_dp_clips_per_sample_then_averages():
    """sigma = 0 isolates clipping: result = mean of per-sample grads scaled to norm <= C."""
    torch.manual_seed(0)
    m = MLP(10, (16,), 3, 0.0)
    X, y = _batch()
    C = 0.05
    plan = DPPlan(n_train=100, batch_size=4, epochs=1, target_epsilon=1.0, delta=1e-3,
                  max_grad_norm=C, noise_multiplier=0.0)
    g = released_gradient(m, X, y, plan, seed=0)
    expected = [torch.zeros_like(p) for p in m.parameters()]
    for i in range(len(y)):
        m.zero_grad()
        nn.functional.cross_entropy(m(X[i:i + 1]), y[i:i + 1]).backward()
        gi = [p.grad.clone() for p in m.parameters()]
        norm = torch.sqrt(sum((t ** 2).sum() for t in gi))
        f = min(1.0, C / (norm.item() + 1e-6))
        expected = [e + f * t / len(y) for e, t in zip(expected, gi)]
    assert all(torch.allclose(a, e, atol=1e-6) for a, e in zip(g, expected))
    assert all(not mod._forward_hooks for mod in m.modules())


def test_score_is_permutation_invariant_and_perfect_on_exact_reconstruction():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(20, 6)).astype(np.float32)
    idx = np.array([3, 7, 11])
    s = score(pool[idx][::-1].copy(), pool[idx], idx, pool)
    assert s["mse"] == pytest.approx(0.0) and s["corr"] == pytest.approx(1.0) and s["nn_identified"] == 1.0
    big = rng.normal(size=(200, 50)).astype(np.float32)
    bidx = np.arange(8)
    noise = score(rng.normal(size=(8, 50)).astype(np.float32), big[bidx], bidx, big)
    assert noise["rel_mse"] > 1.0  # an uninformed guess is worse than the mean (~2)


def test_attack_recovers_single_record_from_untrained_mlp():
    """Sanity check of the attack itself (no DP, batch 1, untrained net). This is the
    attacker's best case and is NOT the headline result; see the sweep for that."""
    torch.manual_seed(0)
    m = MLP(10, (16,), 3, 0.0)
    pool, labels = _batch(n=30)
    idx = np.array([5])
    g = released_gradient(m, pool[idx], labels[idx], None, seed=0)
    rec = invert(m, g, labels[idx], (10,), AttackConfig(iterations=300), None, seed=0)
    s = score(rec.numpy(), pool[idx].numpy(), idx, pool.numpy())
    assert s["rel_mse"] < 0.1 and s["nn_identified"] == 1.0


def test_attack_runs_on_sequences():
    torch.manual_seed(0)
    m = CNN1D(3, (4,), 3, 3, 0.0)
    X = torch.randn(2, 3, 16)
    y = torch.tensor([0, 1])
    g = released_gradient(m, X, y, None, seed=0)
    rec = invert(m, g, y, (3, 16), AttackConfig(iterations=3), clip_norm=1.0, seed=0)
    assert rec.shape == X.shape
    s = score(rec.numpy(), X.numpy(), np.array([0, 1]), X.numpy())
    assert -1.0 <= s["corr"] <= 1.0


# --- end to end (real Flower/Ray simulation) ------------------------------------

@pytest.fixture(scope="module")
def dp_runs(tmp_path_factory):
    cfg = REPO / "configs/experiments/smoke_dp.yaml"
    root = tmp_path_factory.mktemp("dp")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        for sub in ("a", "b"):
            assert main(["run", str(cfg), "--seed", "0", "--output-dir", str(root / sub)]) == 0
    finally:
        del os.environ["SOURCE_DATE_EPOCH"]
    return [next((root / sub).glob("*/result.json")) for sub in ("a", "b")]


def test_dp_run_result_contract(dp_runs):
    r = json.loads(dp_runs[0].read_text())
    validate_result(r)
    p = r["privacy"]
    assert p["enabled"] and p["target_epsilon"] == 3.0
    assert 0 < p["epsilon"] <= 3.0
    acc = p["accounting"]
    assert set(acc) == {"centralized", "local_only", "federated"}
    assert set(acc["federated"]) == {s["name"] for s in r["silos"]}
    for m in [acc["centralized"], *acc["local_only"].values(), *acc["federated"].values()]:
        assert m["delta"] < 1 / m["n_train"] and m["epsilon"] <= 3.0
    for silo in acc["federated"]:
        eps = [rd["per_silo"][silo]["epsilon"] for rd in r["arms"]["federated"]["rounds"]]
        assert all(a < b for a, b in zip(eps, eps[1:]))
        assert eps[-1] == pytest.approx(acc["federated"][silo]["epsilon"])
    a = p["attack"]
    assert set(a["silos"]) == {x["name"] for x in r["silos"]}  # every silo
    assert a["headline"]["silo"] in a["silos"] and a["mse"] is not None
    head = [c for c in a["silos"][a["headline"]["silo"]]["cells"]
            if (c["state"], c["batch_size"], c["labels"]) == ("final", 1, "known")]
    assert a["mse"] == head[0]["mse"]["mean"] == min(
        c["mse"]["mean"] for e in a["silos"].values() for c in e["cells"]
        if (c["state"], c["batch_size"], c["labels"]) == ("final", 1, "known"))
    for name, entry in a["silos"].items():
        cells = entry["cells"]
        assert [(c["state"], c["batch_size"], c["labels"]) for c in cells] == [
            (st, b, lm) for st in ("init", "final") for b in (1, 4) for lm in ("known", "inferred")]
        for c in cells:
            assert len(c["trials"]) == 2 and c["noise_multiplier"] > 0
            assert set(c["chance"]) == {"mse", "rel_mse", "corr", "nn_identified"}
            assert ("label_accuracy" in c) == (c["labels"] == "inferred")
            ex = c["example"]
            assert ex["record_index"] == min(c["trials"][0]["indices"])
            assert len(ex["true"]) == len(ex["reconstruction"]) == 8  # synthetic n_features
        # one released update per trial, attacked under both label modes on the same records
        assert cells[0]["trials"][0]["indices"] == cells[1]["trials"][0]["indices"]
        # batches depend on the seed only, so both model states attack the same records
        assert cells[0]["trials"][0]["indices"] == cells[4]["trials"][0]["indices"]
    sl = p["subject_level"]
    assert sl["epsilon_max"] > p["epsilon"]
    assert all("subject_level" in m for m in acc["federated"].values())


def test_dp_run_is_byte_identical(dp_runs):
    assert dp_runs[0].read_bytes() == dp_runs[1].read_bytes()


def test_summary_reports_privacy(dp_runs):
    s = summarise(load_results(dp_runs[:1]))
    assert s["privacy"]["target_epsilon"] == 3.0
    cells = s["privacy"]["attack"]["cells"]
    assert len(cells) == 5 * 8
    c = cells["nasa_earth/final-B4-inferred"]
    assert c["n_trials"] == 2 and "label_accuracy" in c["ci95"]
    lo, hi = c["ci95"]["rel_mse"]
    assert lo <= c["rel_mse"]["mean"] <= hi


# --- robustness additions ----------------------------------------------------------

def test_infer_labels_recovers_counts_without_dp():
    from fedastro.privacy.attack import infer_labels, label_accuracy

    torch.manual_seed(0)
    m = MLP(10, (16,), 3, 0.0)
    for n, seed in ((1, 0), (1, 1), (4, 2), (8, 3)):
        X, y = _batch(n=n, seed=seed)
        g = released_gradient(m, X, y, None, seed=0)
        inferred = infer_labels(m, g, n, 3, (10,), seed=0)
        assert len(inferred) == n
        assert label_accuracy(inferred, y, 3) >= (1.0 if n == 1 else 0.75)


def test_attack_batches_do_not_depend_on_silo_selection():
    from fedastro.privacy.attack import run_attack

    cfg = load_config(SMOKE, {"privacy.attack.enabled": True, "privacy.attack.batch_sizes": [2],
                              "privacy.attack.trials": 1, "privacy.attack.iterations": 2,
                              "privacy.attack.model_states": ["init"], "privacy.attack.chance_draws": 1,
                              "privacy.attack.label_modes": ["known"]})
    torch.manual_seed(0)
    m = MLP(8, (8,), 3, 0.0)
    w = [p.detach().numpy().copy() for p in m.state_dict().values()]
    rng = np.random.default_rng(0)
    data = {s.name: (rng.normal(size=(20, 8)).astype(np.float32), rng.integers(0, 3, 20))
            for s in cfg.topology.silos}
    last = cfg.topology.silos[-1].name
    both = run_attack(cfg, 0, data, {"init": w}, m)
    alone = run_attack(cfg, 0, {last: data[last]}, {"init": w}, m)
    idx = lambda r: r["silos"][last]["cells"][0]["trials"][0]["indices"]
    assert idx(both) == idx(alone)


def test_label_accuracy_is_multiset_overlap():
    from fedastro.privacy.attack import label_accuracy

    assert label_accuracy(torch.tensor([0, 0, 1, 2]), torch.tensor([2, 1, 0, 0]), 3) == 1.0
    assert label_accuracy(torch.tensor([0, 0, 0, 0]), torch.tensor([0, 1, 1, 2]), 3) == 0.25


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    from fedastro.metrics.aggregate import bootstrap_ci

    groups = [[0.1, 0.2, 0.3], [0.2, 0.25], [0.15, 0.4, 0.3]]
    a, b = bootstrap_ci(groups), bootstrap_ci(groups)
    assert a == b
    pooled = np.mean([v for g in groups for v in g])
    assert a[0] <= pooled <= a[1]
    assert bootstrap_ci([[0.5, 0.5], [0.5]]) == [0.5, 0.5]
    assert bootstrap_ci([]) is None


def test_grid_common_applies_to_base_variant_too():
    from fedastro.grid import load_grid

    _, variants = load_grid(REPO / "configs/sweeps/phase5_eps_cluster.yaml")
    assert variants[0].name == "p5epsc-base" and not variants[0].config.privacy.enabled
    assert all(v.config.privacy.attack.trials == 10 for v in variants)
    assert all(v.config.privacy.enabled for v in variants[1:])
