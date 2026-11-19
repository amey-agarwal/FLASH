"""Phase H9: PREREG section 2 decision rules, generated H9 sweeps, stream summary across seeds.

Every result dict below is FABRICATED test input (not a run, not a number to report)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from fedastro.grid import load_grid
from fedastro.h9_sweeps import SHIFT_SETTINGS, emit_h9_sweeps
from fedastro.metrics.aggregate import _summarise_stream
from fedastro.metrics.hypotheses import (INCONCLUSIVE, NOT_SUPPORTED, SUPPORTED, by_setting,
                                         evaluate_hypotheses, verdict)

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "configs" / "experiments" / "wesad12_hier.yaml"
REGIONS = {"leo": ["l1", "l2"], "moon": ["m1", "m2"], "mars": ["r1", "r2"]}


def _shift(mag):
    return {"enabled": True, "kind": "both", "magnitude": mag}


def _silos():
    return [{"name": n, "region": r, "n_test": 10} for r, ns in REGIONS.items() for n in ns]


def _local(seed, mag, acc_by_region, mean):
    per = {n: {"final": {"accuracy": acc_by_region[r], "macro_f1": acc_by_region[r]}}
           for r, ns in REGIONS.items() for n in ns}
    return {"seed": seed, "run_id": f"lo{seed}{mag}", "shift": _shift(mag), "silos": _silos(),
            "arms": {"local_only": {"per_silo": per, "mean": {"accuracy": mean, "macro_f1": mean}}}}


def _fed(seed, mag, acc):
    return {"seed": seed, "run_id": f"fe{seed}{mag}", "shift": _shift(mag),
            "arms": {"federated": {"final": {"accuracy": acc, "macro_f1": acc}}}}


def _hier(seed, mag, regional, glob, agree=0.9, disagree=0.5):
    pr = {r: {"n_test": 10, "regional": {"accuracy": regional[r], "macro_f1": regional[r]},
              "global": {"accuracy": glob[r], "macro_f1": glob[r]}} for r in REGIONS}
    return {"seed": seed, "run_id": f"hi{seed}{mag}", "shift": _shift(mag), "hierarchy": {"per_region": pr},
            "disagreement": {"acc_agree": agree + 0.01 * seed, "acc_disagree": disagree, "headline_model": "regional"}}


def _family(fed_minus_local=0.1, reg_minus_local=0.1, reg_minus_glob=0.1, ctrl=-0.02, seeds=range(5)):
    out = {"local_only": [], "federated": [], "hierarchical": []}
    for s in seeds:
        j = 0.002 * s  # seed jitter
        lo = {r: 0.6 + j for r in REGIONS}
        out["local_only"] += [_local(s, 0.0, lo, 0.6 + j), _local(s, 1.0, lo, 0.6 + j)]
        out["federated"] += [_fed(s, 0.0, 0.6 + j + fed_minus_local), _fed(s, 1.0, 0.5)]
        reg = {r: 0.6 + j + reg_minus_local for r in REGIONS}
        out["hierarchical"].append(_hier(s, 1.0, reg, {r: reg[r] - reg_minus_glob - j for r in REGIONS}))
        out["hierarchical"].append(_hier(s, 0.0, reg, {r: reg[r] - ctrl for r in REGIONS}))
    return out


def test_verdict_rule():
    assert verdict([0.01, 0.2]) == SUPPORTED
    assert verdict([-0.2, -0.01]) == NOT_SUPPORTED
    assert verdict([-0.01, 0.2]) == INCONCLUSIVE and verdict(None) == INCONCLUSIVE


def test_h1_h2_h3_supported():
    rep = evaluate_hypotheses({"mlp": _family()}, ("both", 1.0), ["moon", "mars"])
    f = rep["families"]["mlp"]
    assert f["H1"]["verdict"] == SUPPORTED and f["H1"]["rows"][0]["diff_mean"] == pytest.approx(0.1)
    assert f["H2"]["verdict"] == SUPPORTED and f["H2"]["control"]["verdict"] == SUPPORTED
    assert f["H3"]["verdict"] == SUPPORTED and f["H3"]["model"] == "regional"
    assert rep["H4"]["H1"]["verdict"] == "holds in every family"


def test_losses_are_reported_as_not_supported():
    rep = evaluate_hypotheses({"mlp": _family(fed_minus_local=-0.1, reg_minus_glob=-0.1)}, ("both", 1.0),
                              ["moon", "mars"])
    f = rep["families"]["mlp"]
    assert f["H1"]["verdict"] == NOT_SUPPORTED
    assert f["H2"]["verdict"] == NOT_SUPPORTED  # (b) lost, (a) won: either not supported -> not supported
    assert rep["H4"]["H1"]["verdict"] == "does not hold"


def test_h2_control_fails_only_when_regional_is_clearly_better():
    f = evaluate_hypotheses({"m": _family(ctrl=0.05)}, ("both", 1.0), ["moon", "mars"])["families"]["m"]
    assert f["H2"]["control"]["verdict"] == NOT_SUPPORTED


def test_h2_uses_the_shifted_regions_only():
    fam = _family(reg_minus_local=0.0)
    for r in fam["hierarchical"]:  # regional wins on leo only
        if r["shift"]["magnitude"] == 1.0:
            r["hierarchy"]["per_region"]["leo"]["regional"]["accuracy"] = 0.99
    f = evaluate_hypotheses({"m": fam}, ("both", 1.0), ["moon", "mars"])["families"]["m"]
    assert f["H2"]["rows"][0]["diff_mean"] == pytest.approx(0.0, abs=1e-12)


def test_mixed_families_and_bad_inputs():
    rep = evaluate_hypotheses({"a": _family(), "b": _family(fed_minus_local=0.0)}, ("both", 1.0), ["moon", "mars"])
    assert rep["H4"]["H1"]["per_family"] == {"a": SUPPORTED, "b": INCONCLUSIVE}
    assert rep["H4"]["H1"]["verdict"] == "mixed / inconclusive"
    fam = _family()
    with pytest.raises(ValueError, match="two runs for seed"):
        by_setting(fam["federated"] + fam["federated"][:1])
    fam["federated"] = [r for r in fam["federated"] if r["seed"] != 4]
    with pytest.raises(ValueError, match="seeds"):
        evaluate_hypotheses({"m": fam}, ("both", 1.0), ["moon", "mars"])


def _eval_sweeps(d: Path, families=("mlp", "cnn1d"), arms=("centralized", "local_only", "federated", "hierarchical")):
    run = {"centralized": ["centralized"], "local_only": ["local_only"], "federated": ["federated"],
           "hierarchical": ["local_only", "hierarchical"]}
    extra = {"mlp": {}, "cnn1d": {"data.wesad.representation": "sequence", "data.wesad.sequence_hz": 4.0}}
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for fam in families:
        for arm in arms:
            v = {"model.arch": fam, **extra[fam], "train.lr": 0.003, "arms": run[arm]}
            if arm in ("federated", "hierarchical"):
                v.update({"fl.local_epochs": 2, "fl.server_opt": "momentum"})
            p = d / f"eval_{fam}_{arm}.yaml"
            p.write_text(yaml.safe_dump({"name": p.stem, "base": os.path.relpath(BASE, d), "seeds": [0, 1, 2, 3, 4],
                                         "variants": [v]}))
            out.append(p)
    return out


def test_h9_sweeps_carry_the_h8_winner_and_the_prereg_settings(tmp_path):
    paths = emit_h9_sweeps(_eval_sweeps(tmp_path / "gen"), tmp_path / "sweeps", "wesad12", tmp_path / "hyp")
    assert len(paths) == 2 * 6 + 1
    _, vs = load_grid(tmp_path / "sweeps" / "h9_wesad12_mlp_hierarchical.yaml")
    assert [(v.config.shift.kind, v.config.shift.magnitude) for v in vs] == list(SHIFT_SETTINGS)
    assert all(v.config.train.lr == 0.003 and v.config.fl.local_epochs == 2 and v.config.fl.server_opt == "momentum"
               and v.config.secagg.enabled and v.config.seeds == (0, 1, 2, 3, 4) and not v.config.comms.enabled
               for v in vs)
    _, vs = load_grid(tmp_path / "sweeps" / "h9_wesad12_cnn1d_local_only.yaml")
    assert all(v.config.arms == ("local_only",) and v.config.data.wesad.representation == "sequence" for v in vs)
    _, (full,) = load_grid(tmp_path / "sweeps" / "h9_wesad12_mlp_full.yaml")
    c = full.config
    assert c.comms.enabled and c.secagg.enabled and c.updates.enabled and "hierarchical_stream" in c.arms
    assert (c.shift.kind, c.shift.magnitude) == ("both", 1.0)
    _, vs = load_grid(tmp_path / "sweeps" / "h9_wesad12_mlp_stream.yaml")
    assert {(v.config.stream.label_fraction, v.config.stream.label_delay_rounds) for v in vs} == \
        {(0.1, 0), (0.1, 1), (1.0, 0), (1.0, 1)}
    spec = yaml.safe_load((tmp_path / "hyp" / "h9_wesad12.yaml").read_text())
    assert spec["shifted_regions"] == ["moon", "mars"]
    for arms in spec["families"].values():
        assert all((tmp_path / "hyp" / p).resolve().is_file() for p in arms.values())


def test_h9_sweeps_need_every_arm_once(tmp_path):
    with pytest.raises(ValueError, match="need"):
        emit_h9_sweeps(_eval_sweeps(tmp_path / "g", arms=("local_only", "federated", "hierarchical")),
                       tmp_path / "s", "wesad12", tmp_path / "h")


def test_stream_summary_across_seeds():
    def res(seed, acc):
        tt = {"models": ["local", "regional", "global"], "scope": "s",
              "global": {"n_arrived": 100, "n_scored": {"local": 100, "regional": 90, "global": 0},
                         "local": {"accuracy": acc, "balanced_accuracy": acc, "macro_f1": acc, "auc": acc},
                         "regional": {"accuracy": acc + 0.1, "balanced_accuracy": acc, "macro_f1": acc, "auc": acc},
                         "global": None},
              "disagreement": {"acc_agree": 0.9, "acc_disagree": 0.4}}
        return {"seed": seed, "stream": {"enabled": True, "label_fraction": 0.1, "label_delay_rounds": 1,
                                         "test_then_train": tt}}
    s = _summarise_stream([res(0, 0.5), res(1, 0.7)])
    assert s["models"]["local"]["accuracy"]["mean"] == pytest.approx(0.6)
    assert s["models"]["regional"]["n_scored"]["mean"] == 90
    assert s["models"]["global"]["n_runs_scored"] == 0 and s["models"]["global"]["accuracy"]["mean"] is None
    assert s["disagreement"]["acc_disagree"]["mean"] == pytest.approx(0.4)
