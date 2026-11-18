"""Phase H8: validation section, selection on validation only, generated sweeps, gap table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from fedastro.config import load_config
from fedastro.grid import load_grid
from fedastro.manifest import make_run_id
from fedastro.metrics.gap import gap_rows
from fedastro.metrics.schema import validate_result
from fedastro.metrics.validation import validation_section
from fedastro.pipeline import build_hier_federation, run_experiment
from fedastro.tuning import emit_sweep, select

REPO = Path(__file__).resolve().parents[1]
SWEEPS = REPO / "configs" / "sweeps"
STAGE_A = SWEEPS / "gapA_wesad12_mlp.yaml"


def _write(out: Path, cfg, seed: int, val: dict, test_acc: float = 0.5) -> None:
    run = out / make_run_id(cfg, seed)
    run.mkdir(parents=True)
    result = {"run_id": run.name, "seed": seed,
              "validation": {"models": {m: {"pooled": {"macro_f1": v}} for m, v in val.items()}},
              "arms": {"federated": {"final": {"accuracy": test_acc}}, "hierarchical": {"final": {"accuracy": test_acc}}},
              "hierarchy": {"global": {"global": {"accuracy": test_acc}}}}
    (run / "result.json").write_text(json.dumps(result))


def _fake_stage(tmp_path, score, test=lambda v: 0.5):
    _, variants = load_grid(STAGE_A)
    for v in variants:
        for seed in v.config.seeds:
            s = score(v)
            _write(tmp_path, v.config, seed, {"flat_federated": s + 0.001 * seed % 3, "global": s}, test(v))
    return variants


def _lr_e(v):
    return v.config.train.lr, v.config.fl.local_epochs


def test_cheapest_within_tolerance_wins(tmp_path):
    # best = (lr 1e-3, E 5) at 0.800; (lr 1e-3, E 1) at 0.797 is within 0.005 and cheaper
    table = {(1e-3, 5): 0.800, (1e-3, 1): 0.797}
    variants = _fake_stage(tmp_path, lambda v: table.get(_lr_e(v), 0.5))
    sel = select(variants, tmp_path, "hierarchical")
    assert _lr_e(variants[sel["winner"]["index"]]) == (1e-3, 1)
    assert sel["best_mean"] == pytest.approx(0.800) and len(sel["candidates"]) == 12


def test_outside_tolerance_the_best_wins(tmp_path):
    table = {(1e-3, 5): 0.800, (1e-3, 1): 0.790}
    variants = _fake_stage(tmp_path, lambda v: table.get(_lr_e(v), 0.5))
    assert _lr_e(variants[select(variants, tmp_path, "hierarchical")["winner"]["index"]]) == (1e-3, 5)


def test_selection_is_blind_to_test_scores(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    score = lambda v: 0.5 + 0.01 * v.index  # noqa: E731
    va = _fake_stage(a, score, test=lambda v: 0.5)
    vb = _fake_stage(b, score, test=lambda v: float(np.random.default_rng(v.index).random()))
    for arm in ("federated", "hierarchical"):
        sa, sb = select(va, a, arm), select(vb, b, arm)
        assert sa["winner"]["name"] == sb["winner"]["name"]
        assert [c["scores"] for c in sa["candidates"]] == [c["scores"] for c in sb["candidates"]]


def test_a_missing_tuning_run_is_an_error(tmp_path):
    variants = _fake_stage(tmp_path, lambda v: 0.5)
    v = variants[3]
    (tmp_path / make_run_id(v.config, v.config.seeds[1]) / "result.json").unlink()
    with pytest.raises(FileNotFoundError, match="every seed"):
        select(variants, tmp_path, "federated")


def test_emitted_stage_b_and_evaluation_sweeps_load(tmp_path):
    variants = _fake_stage(tmp_path, lambda v: 0.9 if _lr_e(v) == (3e-3, 2) else 0.5)
    sel = select(variants, tmp_path, "federated")
    template = yaml.safe_load((SWEEPS / "templates/gapB.yaml").read_text())
    out = tmp_path / "gen" / "gapB_mlp_federated.yaml"
    emit_sweep(STAGE_A, sel, out, "gapB_mlp_federated", template)
    name, vs = load_grid(out)
    assert len(vs) == 16 and all(v.config.arms == ("federated",) for v in vs)
    assert all(v.config.train.lr == 3e-3 and v.config.fl.local_epochs == 2 for v in vs)
    assert {v.config.fl.server_opt for v in vs} == {"none", "momentum", "adam"}
    assert all(v.config.seeds == (100, 101, 102) for v in vs)
    sel_h = select(variants, tmp_path, "hierarchical")
    ev = tmp_path / "gen" / "eval_h.yaml"
    emit_sweep(STAGE_A, sel_h, ev, "eval_h", seeds=[0, 1, 2, 3, 4])
    _, (v,) = load_grid(ev)
    assert v.config.seeds == (0, 1, 2, 3, 4) and v.config.arms == ("local_only", "hierarchical")
    assert v.config.model.arch == "mlp"


def test_emitted_sweeps_for_flat_arms_drop_hierarchy_only_settings(tmp_path):
    smoke = SWEEPS / "gap_tuning_smoke.yaml"
    _, variants = load_grid(smoke)
    for v in variants:
        for seed in v.config.seeds:
            _write(tmp_path, v.config, seed, {"centralized": 0.5 + 0.1 * v.index, "flat_federated": 0.5})
    for arm in ("centralized", "federated"):
        out = tmp_path / f"eval_{arm}.yaml"
        emit_sweep(smoke, select(variants, tmp_path, arm), out, f"eval_{arm}", seeds=[0, 1])
        _, (v,) = load_grid(out)
        assert "hierarchical" not in v.config.arms and v.config.fl.rounds == 4


def test_validation_section_reads_validation_sets_only():
    cfg = load_config(REPO / "configs" / "experiments" / "smoke_hier_wesad12.yaml")
    _, hfed, _ = build_hier_federation(cfg, 0)
    seen = []

    def predictor(node, X):
        seen.append((node, X.shape))
        return np.tile([0.7, 0.2, 0.1], (len(X), 1))

    region_of = {n.name: r.name for r in cfg.hierarchy.regions for n in r.nodes}
    sec = validation_section(hfed, region_of, {"m": predictor})
    assert sorted(seen) == sorted((n, s.val.X.shape) for n, s in hfed.nodes.items())
    n_val = sum(len(s.val) for s in hfed.nodes.values())
    assert sec["models"]["m"]["pooled"]["n_val"] == n_val
    y = np.concatenate([s.val.y for s in hfed.nodes.values()])
    assert sec["models"]["m"]["pooled"]["accuracy"] == pytest.approx(np.mean(y == 0))


def _res(seed, arm, key, acc):
    return {"seed": seed, "run_id": f"{arm}-{seed}", "arms": {arm: {key: {"accuracy": acc, "macro_f1": acc}}}}


def test_gap_rows():
    cen = [_res(s, "centralized", "final", 0.8 + 0.01 * s) for s in range(3)]
    fed = [_res(s, "federated", "final", 0.7) for s in range(3)]
    hier = [{"seed": s, "run_id": f"h{s}", "arms": {"hierarchical": {"final": {"accuracy": 0.9},
                                                                       "regional_final": {"accuracy": 0.6}}}}
            for s in range(3)]
    rows = {(r["comparison"], r["metric"]): r for r in gap_rows("mlp", {"centralized": cen, "federated": fed,
                                                                         "hierarchical": hier})}
    assert rows[("centralized - federated", "accuracy")]["gap_mean"] == pytest.approx(0.11)
    assert rows[("centralized - hierarchical", "accuracy")]["gap_mean"] == pytest.approx(-0.09)  # reported as is
    assert rows[("centralized - hierarchical_regional", "accuracy")]["gap_mean"] == pytest.approx(0.21)
    lo, hi = rows[("centralized - federated", "accuracy")]["ci95"]
    assert lo <= 0.11 <= hi
    with pytest.raises(ValueError, match="seeds"):
        gap_rows("mlp", {"centralized": cen, "federated": fed[:2]})


@pytest.mark.slow
def test_a_hierarchical_run_reports_validation_scores_for_every_model():
    _, variants = load_grid(SWEEPS / "gap_tuning_smoke.yaml")
    r = run_experiment(variants[0].config, 100)
    validate_result(r)
    models = r["validation"]["models"]
    assert set(models) == {"centralized", "local", "flat_federated", "regional", "global"}
    n_val = sum(s["n_val"] for s in r["silos"])
    assert all(m["pooled"]["n_val"] == n_val for m in models.values())
