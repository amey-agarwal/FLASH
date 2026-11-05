"""Golden Tier A (CLAUDE2.md section 3): the reproducibility lock, checked on every test
run. Hash computation and one smoke run only; Tier B (``make golden``) is the slow part.

If one of these fails, a golden finding's identity or the smoke result moved. Stop and
report it; never regenerate a golden file without the user's approval in PROGRESS.md."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("golden", REPO / "scripts" / "golden.py")
golden = importlib.util.module_from_spec(_spec)
sys.modules["golden"] = golden  # dataclasses resolve the defining module through sys.modules
_spec.loader.exec_module(golden)


def test_run_ids_match_golden():
    locked = json.loads(golden.RUN_IDS.read_text())
    problems, _ = golden.compare_run_ids(locked, golden.collect_run_ids())
    assert not problems, "golden run ids changed:\n" + "\n".join(problems)


def test_run_id_lock_covers_g1_hash():
    locked = json.loads(golden.RUN_IDS.read_text())
    g1 = locked["experiments"]["configs/experiments/wesad_by_subject.yaml"]
    assert g1["config_hash"].startswith("ce31e25dda")  # the laptop-vs-cluster hash (PROGRESS.md)


def test_smoke_result_is_byte_identical_to_golden(tmp_path):
    assert golden.smoke_result_bytes(tmp_path) == golden.SMOKE_RESULT.read_bytes()


def test_metrics_golden_covers_the_pinned_subset():
    m = json.loads(golden.METRICS.read_text())
    assert m["tolerance"] == golden.TOLERANCE == 1e-3
    pinned = golden.pinned_run_ids()
    assert set(pinned) == {"G1", "G2", "G3", "G4", "G4b", "G5"}
    assert len(pinned["G1"]) == 5
    if m["status"] == "PENDING":
        assert m["needs_clean_runs"] == pinned
    else:
        assert m["status"] == "FROZEN"
        assert {k: sorted(v["runs"]) for k, v in m["items"].items()} == {k: sorted(v) for k, v in pinned.items()}


def test_compare_run_ids_flags_changes_and_allows_additions():
    entry = {"name": "x", "config_hash": "a" * 64, "run_ids": ["x-seed0-aaaaaaaaaa"]}
    locked = {"experiments": {"e.yaml": entry, "gone.yaml": entry}, "sweeps": {"s.yaml": {"v": entry}}}
    moved = {**entry, "config_hash": "b" * 64, "run_ids": ["x-seed0-bbbbbbbbbb"]}
    current = {"experiments": {"e.yaml": moved, "new.yaml": entry},
               "sweeps": {"s.yaml": {"v": entry, "v2": entry}}}
    problems, additions = golden.compare_run_ids(locked, current)
    assert len(problems) == 2  # e.yaml changed, gone.yaml missing
    assert set(additions) == {"new.yaml", "s.yaml [v2]"}


def test_compare_metrics_tolerance_and_structure():
    want = {"a.accuracy": 0.7000, "a.auc": None, "d.mars": 0, "d.nasa": 42}
    assert golden.compare_metrics(want, {**want, "a.accuracy": 0.7009}) == []
    assert len(golden.compare_metrics(want, {**want, "a.accuracy": 0.7011})) == 1
    assert len(golden.compare_metrics(want, {**want, "d.mars": 1})) == 1  # integers beyond tolerance
    assert len(golden.compare_metrics(want, {**want, "a.auc": 0.5})) == 1  # None -> number
    extra = golden.compare_metrics(want, {**want, "a.f1": 0.6})
    assert extra == ["a.f1: new field"]


def test_g4b_compares_every_attack_cell():
    item = next(i for i in golden.PINNED if i.id == "G4b")
    assert item.variant == "p5eps-base" and item.attack_cells
    cell = {"state": "final", "batch_size": 1, "labels": "known",
            "mse": {"mean": 0.08, "std": 0.04}, "nn_identified": {"mean": 0.93, "std": 0.15}}
    result = {"arms": {}, "privacy": {"attack": {
        "mse": 0.08, "silos": {"nasa_earth": {"cells": [cell, {**cell, "batch_size": 32}]}}}}}
    m = golden.extract_metrics(result, item)
    assert m["privacy.attack.cells.nasa_earth.final.B1.known.nn_identified"] == 0.93
    assert "privacy.attack.cells.nasa_earth.final.B32.known.mse" in m
    assert m["privacy.attack.mse"] == 0.08


def test_freeze_rejects_dirty_mixed_or_synthetic_runs():
    def result(sha, kind="REAL"):
        return {"git_sha": sha, "provenance": [{"kind": kind}]}

    ok = {"G1": {"r0": result("abc"), "r1": result("abc")}}
    assert golden.source_problems(ok) == ([], {"abc"})
    assert golden.source_problems({"G1": {"r0": result("abc-dirty")}})[0]
    assert golden.source_problems({"G1": {"r0": result("abc"), "r1": result("def")}})[0]
    assert golden.source_problems({"G1": {"r0": result("abc", "SYNTHETIC")}})[0]


def test_mask_git_sha_replaces_exactly_one_field():
    data = b'{\n  "git_sha": "9bd7753-dirty",\n  "seed": 0\n}\n'
    assert golden.mask_git_sha(data) == b'{\n  "git_sha": "GOLDEN-MASKED",\n  "seed": 0\n}\n'
    with pytest.raises(ValueError):
        golden.mask_git_sha(b'{"seed": 0}')
