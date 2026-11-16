"""Phase 12: ensemble_of_locals, holdout redraws, paired statistics, fairness, per-group evaluation."""

import json
from pathlib import Path

import numpy as np
import pytest

from fedastro import config as config_mod
from fedastro.baselines.ensemble import ensemble_proba
from fedastro.baselines.local_only import run_local_only
from fedastro.cli import main
from fedastro.config import ConfigError, SyntheticParams, load_config
from fedastro.data.partition import make_federation
from fedastro.data.synthetic import load_synthetic
from fedastro.manifest import make_run_id, run_keys
from fedastro.metrics.aggregate import summarise
from fedastro.metrics.classification import METRICS
from fedastro.metrics.stats import paired_comparisons
from fedastro.pipeline import build_federation, run_arms, run_experiment
from fedastro.training import predict_proba

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "configs" / "experiments" / "smoke_eval.yaml"
GROUPED = REPO / "configs" / "experiments" / "smoke_grouped.yaml"
SHARES = {"nasa_earth": 0.4, "esa_analog": 0.2, "iss_commercial": 0.2, "lunar_habitat": 0.12, "mars_transit": 0.08}
NO_FL = ["centralized", "local_only", "ensemble_of_locals"]  # fast: no Ray


def test_constants_agree():
    assert config_mod.STAT_METRICS == METRICS


# -- config and run ids --------------------------------------------------------------
def test_evaluation_block_is_out_of_the_run_id_at_defaults(smoke_config):
    assert "evaluation" not in load_config(smoke_config).experiment_dict()
    d = load_config(EVAL).experiment_dict()
    assert d["evaluation"] == {"holdout_draws": 2, "primary": "federated-local_only:macro_f1"}


@pytest.mark.parametrize("overrides,match", [
    ({"evaluation.primary": "local_only-federated:accuracy"}, "evaluation.primary must be"),
    ({"evaluation.primary": "federated-local_only:f1"}, "evaluation.primary must be"),
    ({"evaluation.holdout_draws": 0}, "holdout_draws must be >= 1"),
    ({"arms": ["federated", "centralized"]}, "needs arms"),  # primary names local_only
])
def test_evaluation_validation(overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_config(EVAL, overrides)


def test_group_redraws_are_deferred():
    with pytest.raises(ConfigError, match="re-draws held-out subjects only"):
        load_config(GROUPED, {"evaluation.holdout_draws": 2})


def test_run_ids_with_draws(smoke_config):
    cfg = load_config(EVAL)
    h = cfg.config_hash()[:10]
    assert make_run_id(cfg, 1, 1) == f"smoke_eval-seed1-draw1-{h}"
    assert run_keys(cfg, (0, 1)) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    plain = load_config(smoke_config)
    assert make_run_id(plain, 0) == f"smoke-seed0-{plain.config_hash()[:10]}"
    with pytest.raises(ValueError):
        make_run_id(plain, 0, 1)


# -- holdout redraws -----------------------------------------------------------------
def test_draw_zero_is_the_existing_holdout_and_draws_differ():
    b = load_synthetic(SyntheticParams(), 0)
    fed = lambda draw, seed=0: make_federation(b, SHARES, "by_subject", 3, np.random.default_rng(seed), holdout_draw=draw)
    assert fed(0).holdout.subjects == make_federation(b, SHARES, "by_subject", 3, np.random.default_rng(0)).holdout.subjects
    sets = [tuple(fed(d).holdout.subjects) for d in range(3)]
    assert len(set(sets)) == 3
    assert fed(2, seed=7).holdout.subjects == list(sets[2])  # a draw does not depend on the run seed


# -- ensemble ------------------------------------------------------------------------
def test_ensemble_is_the_mean_of_the_local_probabilities(smoke_config):
    cfg = load_config(smoke_config)
    _, fed = build_federation(cfg, 0)
    models = []
    run_local_only(fed, cfg, 0, "cpu", models=models)
    manual = np.mean([predict_proba(m, s.transform(fed.holdout.X)) for _, m, s in models], axis=0)
    np.testing.assert_allclose(ensemble_proba(models, fed.holdout.X, "cpu"), manual)
    assert len(models) == 5


def test_adding_the_ensemble_does_not_change_the_other_arms(smoke_config):
    cfg = load_config(smoke_config, {"arms": ["centralized", "local_only"]})
    with_ens = load_config(smoke_config, {"arms": NO_FL})
    a, b = json.loads(json.dumps(run_experiment(cfg, 0)["arms"])), run_experiment(with_ens, 0)["arms"]
    b = json.loads(json.dumps(b))
    assert a["centralized"] == b["centralized"] and a["local_only"] == b["local_only"]
    # without local_only the ensemble trains the very same local models
    alone = run_experiment(load_config(smoke_config, {"arms": ["ensemble_of_locals"]}), 0)["arms"]["ensemble_of_locals"]
    assert alone["final"] == b["ensemble_of_locals"]["final"]


# -- paired statistics ---------------------------------------------------------------
def _fake(seed, draw, fed, lo, ens=None, cen=None):
    m = lambda v: {k: v for k in METRICS}
    arms = {"federated": {"final": m(fed)}, "local_only": {"mean": m(lo)}}
    if ens is not None:
        arms["ensemble_of_locals"] = {"final": m(ens)}
    if cen is not None:
        arms["centralized"] = {"final": m(cen)}
    return {"seed": seed, "evaluation": {"holdout_draw": draw}, "arms": arms}


def test_paired_comparisons():
    rs = [_fake(s, d, 0.7 + 0.01 * s, 0.6, ens=0.72) for s in range(3) for d in range(2)]
    st = paired_comparisons(rs, "federated-local_only:accuracy")
    by = {(c["b"], c["metric"]): c for c in st["comparisons"]}
    assert st["n_comparisons"] == 8  # 2 comparators present x 4 metrics; centralized absent
    c = by[("local_only", "accuracy")]
    assert c["role"] == "primary" and c["n_pairs"] == 6 and c["n_seeds"] == 3 and c["n_a_better"] == 6
    assert c["mean_diff"] == pytest.approx(0.11)
    assert c["ci95"][0] <= 0.11 <= c["ci95"][1]
    e = by[("ensemble_of_locals", "macro_f1")]
    assert e["role"] == "exploratory" and e["mean_diff"] == pytest.approx(-0.01)
    assert paired_comparisons([_fake(0, 0, 0.7, 0.6)])["comparisons"][0]["ci95"] is None  # one pair, no CI


def test_summary_counts_runs_and_rejects_duplicates(tmp_path):
    assert main(["sweep", str(EVAL), "--seeds", "0", "--output-dir", str(tmp_path)]) == 0
    paths = sorted(tmp_path.glob("*/result.json"))
    assert [p.parent.name.split("-")[2] for p in paths] == ["draw0", "draw1"]
    results = [json.loads(p.read_text()) for p in paths]
    s = summarise(results)
    assert s["n_runs"] == 2 and s["holdout_draws"] == [0, 1] and s["n_seeds"] == 1
    assert "ensemble_of_locals" in s["arms"] and s["stats"]["primary"] == "federated-local_only:macro_f1"
    f = s["fairness"]["federated"]
    assert f["gap"] == f["spread"] and set(f["worst_silo"]["silo_counts"]) <= set(SHARES)
    assert {r["evaluation"]["holdout_draw"] for r in results} == {0, 1}
    assert results[0]["holdout"]["subjects"] != results[1]["holdout"]["subjects"]
    with pytest.raises(ValueError, match="duplicate"):
        summarise(results + results[:1])
    assert main(["run", str(EVAL), "--draw", "5", "--output-dir", str(tmp_path)]) == 2


# -- per-held-out-group evaluation ---------------------------------------------------
def test_per_holdout_group_metrics():
    cfg = load_config(GROUPED, {"arms": NO_FL})
    r = run_experiment(cfg, 0)
    ev = r["evaluation"]
    g6 = ev["per_holdout_group"]["group_6"]
    assert set(g6) == {"centralized", "local_only", "ensemble_of_locals"}
    assert g6["centralized"]["n"] == 3 * 30  # the whole held-out group
    assert ev["held_out_individuals"]["centralized"]["n"] == 2 * 30  # 2 held-out individuals
    assert r["arms"]["ensemble_of_locals"]["per_holdout_group"]["group_6"] == g6["ensemble_of_locals"]
    # per-group numbers recombine to the arm's overall accuracy
    n_total = g6["centralized"]["n"] + ev["held_out_individuals"]["centralized"]["n"]
    acc = (g6["centralized"]["accuracy"] * 90 + ev["held_out_individuals"]["centralized"]["accuracy"] * 60) / n_total
    assert acc == pytest.approx(r["arms"]["centralized"]["final"]["accuracy"])


def test_no_evaluation_block_without_draws_or_groups(smoke_config):
    assert "evaluation" not in run_experiment(load_config(smoke_config), 0)
