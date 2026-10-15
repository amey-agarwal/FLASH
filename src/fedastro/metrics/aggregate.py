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
    return summary


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
    return "\n".join(lines)
