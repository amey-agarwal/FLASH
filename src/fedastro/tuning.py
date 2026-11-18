"""Hyperparameter selection on validation scores only (CLAUDE.md Phase H8, rule 9).

Rule (PREREG.md section 5): for one tuning sweep (one model family) and one arm, each variant
scores the mean over the tuning seeds of its pooled validation macro-F1 (``result.validation``;
the arm's model is ``metrics.validation.ARM_MODEL``). The winner is the cheapest variant within
``tolerance`` (0.005) of the best mean; cheaper = fewer local epochs, then FedAvg over FedProx,
then no server optimizer, then fewer (centralized / local-only) epochs. Remaining ties: higher
mean, then lower variant index.

Only the ``validation`` subtree of each result is ever loaded into the selection (``_load_val``),
so a test score cannot influence it (tested by scrambling every test field).

``emit_sweep`` writes the next sweep from a selection: stage B of the federated grid (template
axes on top of the stage A winner) or the evaluation sweep (winner, evaluation seeds).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fedastro.config import ExperimentConfig
from fedastro.grid import Variant, load_grid
from fedastro.manifest import make_run_id
from fedastro.metrics.validation import ARM_MODEL, SELECTION_METRIC

TOLERANCE = 0.005
HIER_ONLY = ("hier_fl", "secagg", "updates", "stream")  # config blocks of the hierarchical arm only
# The arms each selected arm is run with (the hierarchical arm needs local_only).
ARM_RUN = {"centralized": ["centralized"], "local_only": ["local_only"], "federated": ["federated"],
           "hierarchical": ["local_only", "hierarchical"]}


def _load_val(path: Path) -> dict[str, Any]:
    """Only the validation section of a result file."""
    result = json.loads(path.read_text())
    if not result.get("validation"):
        raise ValueError(f"{path}: no result.validation (selection needs a hierarchical topology, Phase H8+)")
    return result["validation"]


def cost_key(cfg: ExperimentConfig) -> tuple:
    return (cfg.fl.local_epochs, cfg.fl.strategy == "fedprox", cfg.fl.server_opt != "none", cfg.train.epochs)


def variant_scores(v: Variant, out_dir: Path, arm: str) -> list[float]:
    model = ARM_MODEL[arm]
    scores = []
    for seed in v.config.seeds:
        path = out_dir / make_run_id(v.config, seed) / "result.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing tuning run {path} (variant {v.name}, seed {seed}); "
                                    f"selection needs every seed of every variant")
        val = _load_val(path)["models"]
        if model not in val:
            raise ValueError(f"{path}: no validation scores for {model!r} (arm {arm} not run?)")
        scores.append(float(val[model]["pooled"][SELECTION_METRIC]))
    return scores


def select(variants: list[Variant], out_dir: Path, arm: str, tolerance: float = TOLERANCE) -> dict[str, Any]:
    if arm not in ARM_MODEL:
        raise ValueError(f"arm must be one of {sorted(ARM_MODEL)}")
    rows = []
    for v in variants:
        s = variant_scores(v, out_dir, arm)
        rows.append({"index": v.index, "name": v.name, "overrides": v.overrides, "scores": s,
                     "mean": float(np.mean(s)), "std": float(np.std(s, ddof=1)) if len(s) > 1 else None,
                     "cost": list(cost_key(v.config))})
    best = max(r["mean"] for r in rows)
    near = [r for r in rows if r["mean"] >= best - tolerance]
    win = min(near, key=lambda r: (tuple(r["cost"]), -r["mean"], r["index"]))
    return {"arm": arm, "model": ARM_MODEL[arm], "metric": f"validation {SELECTION_METRIC}, pooled over nodes",
            "seeds": list(variants[0].config.seeds), "tolerance": tolerance,
            "rule": "cheapest variant within tolerance of the best mean (cost: local epochs, fedprox, server "
                    "optimizer, epochs); ties: higher mean, lower index",
            "best_mean": best, "winner": win, "candidates": rows}


def emit_sweep(stage_path: Path, selection: dict[str, Any], out_path: Path, name: str,
               template: dict[str, Any] | None = None, seeds: list[int] | None = None) -> dict[str, Any]:
    """A sweep for the selected arm: the stage's base and fixed overrides plus the winner's,
    with the template's ``axes`` / ``variants`` (stage B) or the winner alone (evaluation)."""
    spec = yaml.safe_load(stage_path.read_text())
    base = (stage_path.parent / spec["base"]).resolve()
    fixed = {**(spec.get("fixed") or {}), **selection["winner"]["overrides"], "arms": ARM_RUN[selection["arm"]]}
    if "hierarchical" not in fixed["arms"]:  # hierarchical-arm-only blocks are invalid without it
        fixed = {k: v for k, v in fixed.items() if k.split(".")[0] not in HIER_ONLY}
    out: dict[str, Any] = {"name": name, "base": os.path.relpath(base, out_path.parent.resolve())}
    if seeds is not None:
        out["seeds"] = list(seeds)
    elif "seeds" in spec:
        out["seeds"] = spec["seeds"]
    if template:
        out["fixed"] = fixed
        for key in ("axes", "variants"):
            if template.get(key):
                out[key] = template[key]
    else:
        out["variants"] = [fixed]
    header = (f"# GENERATED by `fedastro select` from {stage_path.name} (arm {selection['arm']}, winner "
              f"{selection['winner']['name']}, validation {SELECTION_METRIC} {selection['winner']['mean']:.4f}).\n"
              f"# Do not edit by hand; regenerate instead.\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + yaml.safe_dump(out, sort_keys=False))
    load_grid(out_path)  # must load
    return out


def run_select(sweep: Path, out_dir: Path, arms: list[str]) -> dict[str, Any]:
    name, variants = load_grid(sweep)
    return {"sweep": name, "selections": {arm: select(variants, out_dir, arm) for arm in arms}}


__all__ = ["select", "run_select", "emit_sweep", "cost_key", "TOLERANCE", "ARM_RUN"]
