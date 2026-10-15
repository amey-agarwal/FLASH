"""Multi-seed aggregation of result.json files into mean +/- std summaries.

Every number that goes on the poster comes from here (CLAUDE.md rule 3: >= 5 seeds).
std is the sample standard deviation (ddof=1) across seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fedastro.metrics.classification import METRICS


def _mean_std(vals: list[float]) -> dict[str, float | int | None]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        "n": len(vals),
    }


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("no results to summarise")
    names = {r["config"]["name"] for r in results}
    hashes = {r["run_id"].rsplit("-", 1)[-1] for r in results}
    if len(names) != 1 or len(hashes) != 1:
        raise ValueError(f"results mix experiments: names={names}, config hashes={hashes}")
    seeds = sorted(r["seed"] for r in results)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {seeds}")
    arms = results[0]["arms"]
    summary: dict[str, Any] = {
        "experiment": names.pop(),
        "config_hash": hashes.pop(),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "provenance": results[0]["provenance"],
        "arms": {},
    }
    if arms["centralized"]["history"]:
        summary["arms"]["centralized"] = {
            m: _mean_std([r["arms"]["centralized"]["final"][m] for r in results]) for m in METRICS
        }
    if arms["local_only"]["per_silo"]:
        lo: dict[str, Any] = {
            "mean_across_silos": {
                m: _mean_std([r["arms"]["local_only"]["mean"][m] for r in results]) for m in METRICS
            },
            "per_silo": {},
        }
        for silo in arms["local_only"]["per_silo"]:
            lo["per_silo"][silo] = {
                m: _mean_std([r["arms"]["local_only"]["per_silo"][silo]["final"][m] for r in results])
                for m in METRICS
            }
        summary["arms"]["local_only"] = lo
    if arms["federated"]["rounds"]:
        summary["arms"]["federated"] = _summarise_federated(results)
    if {"centralized", "local_only", "federated"} <= set(summary["arms"]):
        summary["h1"] = _h1_tally(results)
    return summary


def _summarise_federated(results: list[dict[str, Any]]) -> dict[str, Any]:
    feds = [r["arms"]["federated"] for r in results]
    n_rounds = {len(f["rounds"]) for f in feds}
    if len(n_rounds) != 1:
        raise ValueError(f"federated runs have different round counts: {n_rounds}")
    curve = []
    for i in range(n_rounds.pop()):
        curve.append({
            "round": feds[0]["rounds"][i]["round"],
            **{m: _mean_std([f["rounds"][i]["global"][m] for f in feds]) for m in METRICS},
        })
    out: dict[str, Any] = {
        "final": {m: _mean_std([f["final"][m] for f in feds]) for m in METRICS},
        "curve": curve,
        "total_bytes": _mean_std([f["total_bytes"] for f in feds]),
        "rounds_to_target": {},
    }
    for key in feds[0]["rounds_to_target"]:
        reached = [f["rounds_to_target"][key]["round"] for f in feds]
        out["rounds_to_target"][key] = {
            "round": _mean_std([r for r in reached if r is not None]),
            "n_reached": sum(r is not None for r in reached),
            "n_runs": len(reached),
        }
    return out


def _h1_tally(results: list[dict[str, Any]], metric: str = "accuracy") -> dict[str, Any]:
    """Per seed: is federated > local_only silo mean, and <= centralized? Reported as
    counts; a federated loss to local_only is a result, not a failure."""
    beats_local = between = 0
    for r in results:
        a = r["arms"]
        fed, lo, cen = a["federated"]["final"][metric], a["local_only"]["mean"][metric], a["centralized"]["final"][metric]
        beats_local += fed > lo
        between += lo < fed <= cen
    return {"metric": metric, "n_seeds": len(results), "federated_beats_local_only_mean": beats_local,
            "federated_between_local_only_and_centralized": between}


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(Path(p).read_text()) for p in paths]


def format_summary(summary: dict[str, Any]) -> str:
    def fmt(ms: dict) -> str:
        if ms["mean"] is None:
            return "n/a"
        return f"{ms['mean']:.3f} ± {ms['std']:.3f}" if ms["std"] is not None else f"{ms['mean']:.3f}"

    lines = [f"{summary['experiment']} ({summary['n_seeds']} seeds: {summary['seeds']})",
             f"{'arm':<28}" + "".join(f"{m:>20}" for m in METRICS)]
    arms = summary["arms"]
    if "centralized" in arms:
        lines.append(f"{'centralized':<28}" + "".join(f"{fmt(arms['centralized'][m]):>20}" for m in METRICS))
    if "local_only" in arms:
        lo = arms["local_only"]
        lines.append(f"{'local_only (silo mean)':<28}" + "".join(f"{fmt(lo['mean_across_silos'][m]):>20}" for m in METRICS))
        for silo, ms in lo["per_silo"].items():
            lines.append(f"{'  ' + silo:<28}" + "".join(f"{fmt(ms[m]):>20}" for m in METRICS))
    if "federated" in arms:
        fed = arms["federated"]
        lines.append(f"{'federated (final round)':<28}" + "".join(f"{fmt(fed['final'][m]):>20}" for m in METRICS))
        lines.append(f"federated total bytes: {fmt(fed['total_bytes'])}")
        for key, v in fed["rounds_to_target"].items():
            lines.append(f"rounds to target ({key}): {fmt(v['round'])}, reached in {v['n_reached']}/{v['n_runs']} runs")
    if "h1" in summary:
        h = summary["h1"]
        lines.append(f"H1 ({h['metric']}): federated > local_only mean in {h['federated_beats_local_only_mean']}/{h['n_seeds']} seeds; "
                     f"between local_only and centralized in {h['federated_between_local_only_and_centralized']}/{h['n_seeds']}")
    return "\n".join(lines)
