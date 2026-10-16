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
    fairness = {}
    for arm in ("centralized", "local_only", "federated"):
        if all(arm in r.get("fairness", {}) for r in results):
            fairness[arm] = {k: _mean_std([r["fairness"][arm][k] for r in results])
                             for k in ("spread", "std", "min", "mean")}
    if fairness:
        summary["fairness"] = fairness
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
    if "fairness" in summary:
        lines.append("fairness (label-mix holdout accuracy across silos; spread = max - min, lower is fairer):")
        for arm, f in summary["fairness"].items():
            lines.append(f"  {arm:<14} spread {fmt(f['spread'])}   worst silo {fmt(f['min'])}   silo mean {fmt(f['mean'])}")
    if "h1" in summary:
        h = summary["h1"]
        lines.append(f"H1 ({h['metric']}): federated > local_only mean in {h['federated_beats_local_only_mean']}/{h['n_seeds']} seeds; "
                     f"between local_only and centralized in {h['federated_between_local_only_and_centralized']}/{h['n_seeds']}")
    return "\n".join(lines)


def grid_report(variants, out_dir: Path) -> tuple[list[dict[str, Any]], str]:
    """One row per grid variant from its ``<name>-<hash>-summary.json`` (missing -> skipped)."""
    rows, lines = [], []

    def ms(d: dict | None) -> str:
        if not d or d.get("mean") is None:
            return "n/a"
        return f"{d['mean']:.3f}±{d['std']:.3f}" if d.get("std") is not None else f"{d['mean']:.3f}"

    head = (f"{'variant':<46}{'fed acc':>13}{'fed F1':>13}{'local acc':>13}{'cent acc':>13}"
            f"{'fed-local':>10}{'spread fed':>13}{'spread loc':>13}{'H1':>6}")
    lines += [head, "-" * len(head)]
    for v in variants:
        path = out_dir / f"{v.config.name}-{v.config.config_hash()[:10]}-summary.json"
        if not path.exists():
            lines.append(f"{v.name:<46}  (missing: {path.name})")
            continue
        s = json.loads(path.read_text())
        arms, fair = s["arms"], s.get("fairness", {})
        fed = arms.get("federated", {}).get("final", {})
        lo = arms.get("local_only", {}).get("mean_across_silos", {})
        cen = arms.get("centralized", {})
        gap = (fed["accuracy"]["mean"] - lo["accuracy"]["mean"]) if fed and lo else None
        h1 = s.get("h1")
        row = {
            "variant": v.name, "overrides": v.overrides, "n_seeds": s["n_seeds"],
            "federated_accuracy": fed.get("accuracy"), "federated_macro_f1": fed.get("macro_f1"),
            "local_only_accuracy": lo.get("accuracy"), "centralized_accuracy": cen.get("accuracy"),
            "fed_minus_local_accuracy": gap,
            "fairness_spread_federated": fair.get("federated", {}).get("spread"),
            "fairness_spread_local_only": fair.get("local_only", {}).get("spread"),
            "h1": h1,
        }
        rows.append(row)
        h1_txt = f"{h1['federated_beats_local_only_mean']}/{h1['n_seeds']}" if h1 else "n/a"
        lines.append(
            f"{v.name:<46}{ms(row['federated_accuracy']):>13}{ms(row['federated_macro_f1']):>13}"
            f"{ms(row['local_only_accuracy']):>13}{ms(row['centralized_accuracy']):>13}"
            f"{(f'{gap:+.3f}' if gap is not None else 'n/a'):>10}"
            f"{ms(row['fairness_spread_federated']):>13}{ms(row['fairness_spread_local_only']):>13}{h1_txt:>6}")
    return rows, "\n".join(lines)
