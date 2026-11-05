#!/usr/bin/env python3
"""Golden reproducibility lock (CLAUDE2.md section 3).

Tier A (fast; part of ``make test`` through tests/test_golden.py):
- every existing config and sweep variant still has the run id recorded in
  ``tests/golden/run_ids.json`` (hash computation only, no training);
- the smoke result is byte-identical to ``tests/golden/smoke_result.json``, with
  ``git_sha`` masked: it names the commit by design and changes with every commit.

Tier B (slow; ``make golden``, or ``scripts/slurm/golden.sbatch`` on the cluster):
- ``run`` re-runs the pinned subset (PINNED) through the normal CLI;
- ``check`` compares its final metrics with ``tests/golden/metrics.json`` at TOLERANCE;
- ``freeze`` writes ``tests/golden/metrics.json`` from clean-commit runs of the subset.

A golden file is never rewritten to make a check pass without the user's approval
recorded in PROGRESS.md: overwriting an existing golden file needs ``--force``.

Usage:
    python scripts/golden.py ids [--write [--force]]
    python scripts/golden.py smoke [--write [--force]]
    python scripts/golden.py pending [--force]
    python scripts/golden.py run --output-dir runs/_golden [--item G1 ...] [--resume]
    python scripts/golden.py check --runs-dir runs/_golden
    python scripts/golden.py freeze --runs-dir runs/golden-<sha> [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fedastro.config import ExperimentConfig, load_config
from fedastro.grid import load_grid
from fedastro.manifest import dumps, make_run_id

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden"
RUN_IDS = GOLDEN / "run_ids.json"
SMOKE_RESULT = GOLDEN / "smoke_result.json"
METRICS = GOLDEN / "metrics.json"
SMOKE_CONFIG = "configs/experiments/smoke.yaml"
GIT_SHA_MASK = "GOLDEN-MASKED"
TOLERANCE = 1e-3  # absolute, the laptop-vs-cluster tolerance (PROGRESS.md, cluster shakedown)

# Final metrics compared for every pinned run; items add their finding-specific fields.
BASE_FIELDS = ("arms.centralized.final", "arms.local_only.mean", "arms.federated.final")
# Per-cell attack scores (mean over trials) compared when an item sets attack_cells.
ATTACK_CELL_SCORES = ("mse", "nn_identified")


@dataclass(frozen=True)
class Pinned:
    id: str
    finding: str
    config: str  # repo-relative path
    variant: str | None  # sweep variant name; None for an experiment config
    extra: tuple[str, ...] = ()
    attack_cells: bool = False  # also compare every (silo, state, batch size, labels) cell


# One entry per golden finding (CLAUDE2.md section 3): G1 at 5 seeds, one variant each of
# G2-G5, plus G4b (user decision 2026-11-05), so both halves of G4 are covered. Choices are
# recorded in PROGRESS.md, Phase 10.
PINNED = (
    Pinned("G1", "three-arm H1 on WESAD by-subject", "configs/experiments/wesad_by_subject.yaml", None),
    Pinned("G2", "federated performance vs Dirichlet alpha (strongest skew)",
           "configs/sweeps/phase4_alpha.yaml", "p4alpha-a0p1-dirichlet-fedavg"),
    Pinned("G3", "FedProx mu sweep (the only mu with a measurable effect)",
           "configs/sweeps/phase4_mu.yaml", "p4mu-mu1p0"),
    Pinned("G4", "DP utility collapse at small epsilon; attack under DP",
           "configs/sweeps/phase5_eps.yaml", "p5eps-eps1p0",
           ("privacy.epsilon", "privacy.attack.mse")),
    Pinned("G5", "FedBuff speed from excluding far silos (Mars far, loss of signal on)",
           "configs/sweeps/phase6_presets.yaml", "p6pre-marsfar-fedbuff",
           ("arms.federated.sim_wallclock_s_total", "arms.federated.dispatches")),
    Pinned("G4b", "attack success depends on batch size without DP (B=1 vs B=32)",
           "configs/sweeps/phase5_eps.yaml", "p5eps-base", ("privacy.attack.mse",), attack_cells=True),
)


# -- Tier A: run ids -----------------------------------------------------------------
def _rel(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def _entry(cfg: ExperimentConfig) -> dict[str, Any]:
    seeds = sorted({cfg.seed, *cfg.seeds})
    return {"name": cfg.name, "config_hash": cfg.config_hash(),
            "run_ids": [make_run_id(cfg, s) for s in seeds]}


def collect_run_ids() -> dict[str, Any]:
    """Run ids of every experiment config and every sweep variant in configs/."""
    experiments = {_rel(p): _entry(load_config(p))
                   for p in sorted((REPO / "configs" / "experiments").glob("*.yaml"))}
    sweeps = {}
    for p in sorted((REPO / "configs" / "sweeps").glob("*.yaml")):
        _, variants = load_grid(p)
        sweeps[_rel(p)] = {v.name: {"index": v.index, **_entry(v.config)} for v in variants}
    return {"experiments": experiments, "sweeps": sweeps}


def compare_run_ids(golden: dict[str, Any], current: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(problems, additions). A golden config or variant that is missing or whose entry
    changed is a problem; configs added since the lock are allowed and listed."""
    problems, additions = [], []
    for path, want in golden["experiments"].items():
        got = current["experiments"].get(path)
        if got is None:
            problems.append(f"{path}: config missing")
        elif got != want:
            problems.append(f"{path}: {want['run_ids'][0]} -> {got['run_ids'][0]}")
    additions += [p for p in current["experiments"] if p not in golden["experiments"]]
    for path, variants in golden["sweeps"].items():
        got_variants = current["sweeps"].get(path)
        if got_variants is None:
            problems.append(f"{path}: sweep missing")
            continue
        for name, want in variants.items():
            got = got_variants.get(name)
            if got is None:
                problems.append(f"{path}: variant {name} missing")
            elif got != want:
                problems.append(f"{path} [{name}]: {want} -> {got}")
        additions += [f"{path} [{n}]" for n in got_variants if n not in variants]
    additions += [p for p in current["sweeps"] if p not in golden["sweeps"]]
    return problems, additions


# -- Tier A: smoke result ------------------------------------------------------------
def mask_git_sha(data: bytes) -> bytes:
    masked, n = re.subn(rb'"git_sha": "[^"]*"', f'"git_sha": "{GIT_SHA_MASK}"'.encode(), data)
    if n != 1:
        raise ValueError(f"expected exactly one git_sha field, found {n}")
    return masked


def smoke_result_bytes(out_dir: Path) -> bytes:
    """``fedastro run configs/experiments/smoke.yaml --seed 0`` with SOURCE_DATE_EPOCH=0,
    exactly as ``make repro-check`` runs it; returns result.json with git_sha masked."""
    from fedastro.cli import main

    previous = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        if main(["run", str(REPO / SMOKE_CONFIG), "--seed", "0", "--output-dir", str(out_dir)]) != 0:
            raise RuntimeError("smoke run failed")
    finally:
        if previous is None:
            del os.environ["SOURCE_DATE_EPOCH"]
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous
    (path,) = out_dir.glob("*/result.json")
    return mask_git_sha(path.read_bytes())


# -- Tier B --------------------------------------------------------------------------
def pinned_config(item: Pinned) -> tuple[ExperimentConfig, int | None]:
    """(config, sweep variant index or None)."""
    if item.variant is None:
        return load_config(REPO / item.config), None
    _, variants = load_grid(REPO / item.config)
    for v in variants:
        if v.name == item.variant:
            return v.config, v.index
    raise KeyError(f"{item.id}: variant {item.variant} not in {item.config}")


def pinned_run_ids() -> dict[str, list[str]]:
    out = {}
    for item in PINNED:
        cfg, _ = pinned_config(item)
        out[item.id] = [make_run_id(cfg, s) for s in cfg.seeds]
    return out


def run_pinned(items: list[Pinned], out_dir: Path, resume: bool) -> None:
    """Re-run pinned items through the normal CLI: ``fedastro sweep`` for an experiment,
    ``fedastro grid --index i`` for a sweep variant."""
    from fedastro.cli import main

    for item in items:
        _, index = pinned_config(item)
        if index is None:
            argv = ["sweep", str(REPO / item.config), "--output-dir", str(out_dir)]
        else:
            argv = ["grid", str(REPO / item.config), "--index", str(index), "--output-dir", str(out_dir)]
            argv += ["--resume"] if resume else []
        print(f"[{item.id}] fedastro {' '.join(argv)}", flush=True)
        if main(argv) != 0:
            raise RuntimeError(f"{item.id}: run failed")


def _get(d: Any, dotted: str) -> Any:
    for key in dotted.split("."):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k in sorted(value):
            _flatten(f"{prefix}.{k}", value[k], out)
    else:
        out[prefix] = value


def extract_metrics(result: dict[str, Any], item: Pinned) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in BASE_FIELDS + item.extra:
        _flatten(field, _get(result, field), out)
    if item.attack_cells:
        silos = _get(result, "privacy.attack.silos") or {}
        for silo in sorted(silos):
            for c in silos[silo]["cells"]:
                key = f"privacy.attack.cells.{silo}.{c['state']}.B{c['batch_size']}.{c['labels']}"
                for s in ATTACK_CELL_SCORES:
                    out[f"{key}.{s}"] = c[s]["mean"]
    return out


def load_pinned_results(runs_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """{item id: {run id: result}} for every pinned run; raises if any is missing."""
    found, missing = {}, []
    for item_id, run_ids in pinned_run_ids().items():
        found[item_id] = {}
        for run_id in run_ids:
            path = runs_dir / run_id / "result.json"
            if path.is_file():
                found[item_id][run_id] = json.loads(path.read_text())
            else:
                missing.append(run_id)
    if missing:
        raise FileNotFoundError(f"{len(missing)} pinned run(s) missing in {runs_dir}: {missing}")
    return found


def source_problems(results: dict[str, dict[str, dict[str, Any]]]) -> tuple[list[str], set[str]]:
    """Reasons the runs cannot be frozen (dirty or unknown commit, mixed commits,
    synthetic data), and the set of commits seen."""
    problems, shas = [], set()
    for item_id, runs in results.items():
        for run_id, r in runs.items():
            sha = r["git_sha"]
            shas.add(sha)
            if sha.endswith("-dirty") or sha == "unknown":
                problems.append(f"{run_id}: not a clean commit ({sha})")
            if any(p.get("kind") != "REAL" for p in r["provenance"]):
                problems.append(f"{run_id}: provenance is not REAL")
    if len(shas) > 1:
        problems.append(f"runs come from {len(shas)} commits: {sorted(shas)}")
    return problems, shas


def compare_metrics(want: dict[str, Any], got: dict[str, Any], tol: float = TOLERANCE) -> list[str]:
    """Differences between golden and current metrics: changed key sets, a number that
    moved by more than ``tol``, or any other value that differs."""
    diffs = [f"{k}: missing now" for k in sorted(set(want) - set(got))]
    diffs += [f"{k}: new field" for k in sorted(set(got) - set(want))]
    for k in sorted(set(want) & set(got)):
        a, b = want[k], got[k]
        numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b))
        if numeric:
            if abs(a - b) > tol:
                diffs.append(f"{k}: {a} -> {b} (|diff| {abs(a - b):.3g} > {tol})")
        elif a != b:
            diffs.append(f"{k}: {a!r} -> {b!r}")
    return diffs


def pending_metrics() -> dict[str, Any]:
    return {
        "status": "PENDING",
        "tolerance": TOLERANCE,
        "reason": ("Only -dirty runs exist (PROGRESS.md, Phase 10). Run scripts/slurm/golden.sbatch "
                   "on a clean commit, then: python scripts/golden.py freeze --runs-dir <dir>."),
        "items": {item.id: {"finding": item.finding, "config": item.config, "variant": item.variant}
                  for item in PINNED},
        "needs_clean_runs": pinned_run_ids(),
    }


def frozen_metrics(results: dict[str, dict[str, dict[str, Any]]], sha: str, runs_dir: Path) -> dict[str, Any]:
    items = {}
    for item in PINNED:
        runs = {run_id: {"seed": r["seed"], "metrics": extract_metrics(r, item)}
                for run_id, r in results[item.id].items()}
        items[item.id] = {"finding": item.finding, "config": item.config, "variant": item.variant,
                          "fields": list(BASE_FIELDS + item.extra), "runs": runs}
    return {
        "status": "FROZEN",
        "tolerance": TOLERANCE,
        "source": {"git_sha": sha, "runs_dir": runs_dir.name,
                   "frozen_utc": dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
        "items": items,
    }


def check_runs(runs_dir: Path) -> int:
    golden = json.loads(METRICS.read_text())
    if golden["status"] != "FROZEN":
        print(f"tests/golden/metrics.json is {golden['status']}: {golden['reason']}", file=sys.stderr)
        return 2
    results = load_pinned_results(runs_dir)
    _, shas = source_problems(results)
    print(f"golden from {golden['source']['git_sha']}; checking runs from {sorted(shas)}")
    failed = 0
    for item in PINNED:
        for run_id, want in golden["items"][item.id]["runs"].items():
            diffs = compare_metrics(want["metrics"], extract_metrics(results[item.id][run_id], item),
                                    golden["tolerance"])
            status = "OK" if not diffs else "MOVED"
            print(f"[{item.id}] {run_id}: {status}")
            for d in diffs:
                print(f"    {d}")
            failed += bool(diffs)
    print(f"{failed} run(s) moved" if failed else "Tier B PASSED")
    return 1 if failed else 0


# -- CLI -----------------------------------------------------------------------------
def _write(path: Path, data: bytes, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"{_rel(path)} exists. Overwriting a golden file needs --force and the "
                         "user's approval recorded in PROGRESS.md.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"wrote {_rel(path)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("ids", "smoke"):
        sp = sub.add_parser(name)
        sp.add_argument("--write", action="store_true", help="write the golden file instead of checking")
        sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("pending", help="write a PENDING metrics.json listing the clean runs needed")
    sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("run")
    sp.add_argument("--output-dir", required=True, type=Path)
    sp.add_argument("--item", nargs="+", choices=[i.id for i in PINNED])
    sp.add_argument("--resume", action="store_true", help="skip sweep-variant runs already done")
    for name in ("check", "freeze"):
        sp = sub.add_parser(name)
        sp.add_argument("--runs-dir", required=True, type=Path)
        if name == "freeze":
            sp.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    if args.command == "ids":
        current = collect_run_ids()
        if args.write:
            _write(RUN_IDS, dumps(current).encode(), args.force)
            return 0
        problems, additions = compare_run_ids(json.loads(RUN_IDS.read_text()), current)
        for a in additions:
            print(f"new since the lock (allowed): {a}")
        for pr in problems:
            print(f"CHANGED: {pr}")
        print("run ids match the lock" if not problems else f"{len(problems)} golden run id(s) changed")
        return 1 if problems else 0
    if args.command == "smoke":
        with tempfile.TemporaryDirectory() as tmp:
            data = smoke_result_bytes(Path(tmp))
        if args.write:
            _write(SMOKE_RESULT, data, args.force)
            return 0
        same = data == SMOKE_RESULT.read_bytes()
        print("smoke result byte-identical to the lock" if same else "smoke result DIFFERS from the lock")
        return 0 if same else 1
    if args.command == "pending":
        _write(METRICS, dumps(pending_metrics()).encode(), args.force)
        return 0
    if args.command == "run":
        items = [i for i in PINNED if args.item is None or i.id in args.item]
        run_pinned(items, args.output_dir, args.resume)
        return 0
    try:
        if args.command == "check":
            return check_runs(args.runs_dir)
        results = load_pinned_results(args.runs_dir)  # freeze
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    problems, shas = source_problems(results)
    if problems:
        for pr in problems:
            print(f"cannot freeze: {pr}", file=sys.stderr)
        return 1
    if METRICS.exists() and json.loads(METRICS.read_text())["status"] == "PENDING":
        args.force = True  # replacing the PENDING placeholder is the intended first freeze
    _write(METRICS, dumps(frozen_metrics(results, shas.pop(), args.runs_dir)).encode(), args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
