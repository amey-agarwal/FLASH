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
    from fedastro.metrics.stats import holdout_draw, paired_comparisons

    keys = sorted((r["seed"], holdout_draw(r)) for r in results)
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate (seed, holdout draw) runs: {keys}")
    seeds = sorted({s for s, _ in keys})
    draws = sorted({d for _, d in keys})
    arms = results[0]["arms"]
    summary: dict[str, Any] = {
        "experiment": names.pop(),
        "config_hash": hashes.pop(),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "provenance": results[0]["provenance"],
        "arms": {},
    }
    if draws != [0]:  # Phase 12 holdout redraws: every (seed, draw) is one run
        summary.update(holdout_draws=draws, n_runs=len(results))
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
    if "ensemble_of_locals" in arms:
        summary["arms"]["ensemble_of_locals"] = {
            m: _mean_std([r["arms"]["ensemble_of_locals"]["final"][m] for r in results]) for m in METRICS}
    if "hierarchical" in arms:  # Phase H3: per region, every model on that region's own test data
        summary["arms"]["hierarchical"] = {
            key: {m: _mean_std([r["arms"]["hierarchical"][key][m] for r in results]) for m in METRICS}
            for key in ("final", "regional_final")}
        regions = results[0]["hierarchy"]["per_region"]
        models = [k for k in regions[next(iter(regions))] if k != "n_test"]
        summary["hierarchy"] = {
            "per_region": {reg: {mod: {m: _mean_std([r["hierarchy"]["per_region"][reg][mod][m] for r in results])
                                       for m in METRICS} for mod in models} for reg in regions},
            "disagreement": {k: _mean_std([r["disagreement"][k] for r in results
                                           if r["disagreement"][k] is not None]) for k in ("acc_agree", "acc_disagree")},
        }
    if {"centralized", "local_only", "federated"} <= set(summary["arms"]):
        summary["h1"] = _h1_tally(results)
    fairness = {}
    for arm in ("centralized", "local_only", "federated", "ensemble_of_locals"):
        if all(arm in r.get("fairness", {}) for r in results):
            fairness[arm] = {k: _mean_std([r["fairness"][arm][k] for r in results])
                             for k in ("spread", "std", "min", "mean")}
            # First-class worst-silo and best-to-worst gap (Phase 12). The gap is the
            # spread; the worst silo is named per run and counted.
            worst = [min(r["fairness"][arm]["per_silo"], key=r["fairness"][arm]["per_silo"].get) for r in results]
            fairness[arm]["worst_silo"] = {"value": fairness[arm]["min"],
                                           "silo_counts": {s: worst.count(s) for s in sorted(set(worst))}}
            fairness[arm]["gap"] = fairness[arm]["spread"]
    if fairness:
        summary["fairness"] = fairness
    stats = paired_comparisons(results, (results[0]["config"].get("evaluation") or {}).get("primary", ""))
    if stats:
        summary["stats"] = stats
    privacy = _summarise_privacy(results)
    if privacy:
        summary["privacy"] = privacy
    if all("model_report" in r for r in results):  # Phase H6
        summary["model_report"] = _summarise_model_report(results)
    return summary


MODEL_REPORT_METRICS = ("accuracy", "macro_f1", "auc", "ece", "worst_node_accuracy")


def _summarise_model_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    first = results[0]["model_report"]
    rows = {}
    for model in first["rows"]:
        rows[model] = {m: _mean_std([r["model_report"]["rows"][model][m] for r in results
                                     if r["model_report"]["rows"][model][m] is not None])
                       for m in MODEL_REPORT_METRICS}
        worst = [r["model_report"]["rows"][model]["worst_node"] for r in results]
        rows[model]["worst_node_counts"] = {n: worst.count(n) for n in sorted({w for w in worst if w})}
    return {"arch": first["arch"], "n_params": first["n_params"], "payload_bytes": first["payload_bytes"],
            "scope": first["scope"], "rows": rows}


ATTACK_METRICS = ("mse", "rel_mse", "corr", "nn_identified")


def _summarise_privacy(results: list[dict[str, Any]]) -> dict[str, Any]:
    """epsilon spent (weakest model in the run) and, per attack cell, the across-seed
    mean +/- std of each seed's trial mean, plus a hierarchical bootstrap 95% CI over
    seeds and trials (``bootstrap_ci``) for the attack and its chance level."""
    out: dict[str, Any] = {}
    if all(r["privacy"]["enabled"] for r in results):
        out["target_epsilon"] = results[0]["privacy"]["target_epsilon"]
        out["epsilon"] = _mean_std([r["privacy"]["epsilon"] for r in results])
        out["delta_max"] = max(r["privacy"]["delta"] for r in results)
    if all("silos" in r["privacy"]["attack"] for r in results):
        cells = {}
        for silo, entry in results[0]["privacy"]["attack"]["silos"].items():
            for i, c in enumerate(entry["cells"]):
                per_seed = [r["privacy"]["attack"]["silos"][silo]["cells"][i] for r in results]
                ident = (c["state"], c["batch_size"], c["labels"])
                if any((p["state"], p["batch_size"], p["labels"]) != ident for p in per_seed):
                    raise ValueError("attack cells differ across seeds")
                metrics = ATTACK_METRICS + (("label_accuracy",) if "label_accuracy" in c else ())
                cell = {
                    "silo": silo, "state": c["state"], "batch_size": c["batch_size"], "labels": c["labels"],
                    "noise_multiplier": _mean_std([p["noise_multiplier"] for p in per_seed]),
                    **{m: _mean_std([p[m]["mean"] for p in per_seed]) for m in metrics},
                    "ci95": {m: bootstrap_ci([[t[m] for t in p["trials"]] for p in per_seed]) for m in metrics},
                    "chance": {m: _mean_std([p["chance"][m] for p in per_seed]) for m in ATTACK_METRICS},
                    "chance_ci95": {m: bootstrap_ci([[t["chance"][m] for t in p["trials"]] for p in per_seed])
                                    for m in ATTACK_METRICS},
                    "n_trials": sum(len(p["trials"]) for p in per_seed),
                }
                cells[f"{silo}/{c['state']}-B{c['batch_size']}-{c['labels']}"] = cell
        out["attack"] = {"headline": results[0]["privacy"]["attack"]["headline"], "cells": cells}
    return out


def bootstrap_ci(groups: list[list[float]], n_boot: int = 2000, level: float = 0.95,
                 seed: int = 0) -> list[float] | None:
    """Hierarchical percentile bootstrap of the mean: resample seeds (groups) with
    replacement, then trials within each drawn seed. Deterministic (fixed rng seed).
    With few seeds the interval is approximate; it is reported alongside mean +/- std."""
    groups = [np.asarray(g, dtype=np.float64) for g in groups if len(g)]
    if not groups:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, len(groups), size=len(groups))
        means[i] = np.mean(np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in (groups[j] for j in picked)]))
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return [float(lo), float(hi)]


def _summarise_federated(results: list[dict[str, Any]]) -> dict[str, Any]:
    feds = [r["arms"]["federated"] for r in results]
    n_rounds = [len(f["rounds"]) for f in feds]
    if len(set(n_rounds)) != 1 and not any(f.get("stalled") for f in feds):
        raise ValueError(f"federated runs have different round counts: {n_rounds}")
    timed = any(f.get("sim_wallclock_s_total") is not None for f in feds)
    curve = []
    for i in range(min(n_rounds)):  # stalled runs end early; the curve covers rounds all seeds reached
        curve.append({
            "round": feds[0]["rounds"][i]["round"],
            **{m: _mean_std([f["rounds"][i]["global"][m] for f in feds]) for m in METRICS},
            **({"sim_wallclock_s": _mean_std([f["rounds"][i]["sim_wallclock_s"] for f in feds])} if timed else {}),
        })
    out: dict[str, Any] = {
        "final": {m: _mean_std([f["final"][m] for f in feds]) for m in METRICS},
        "curve": curve,
        "total_bytes": _mean_std([f["total_bytes"] for f in feds]),
        "rounds_to_target": {},
        "n_rounds": n_rounds,
        "n_stalled": sum(bool(f.get("stalled")) for f in feds),
    }
    if timed:
        out["sim_wallclock_s_total"] = _mean_std([f["sim_wallclock_s_total"] for f in feds])
    for key in feds[0]["rounds_to_target"]:
        reached = [f["rounds_to_target"][key]["round"] for f in feds]
        out["rounds_to_target"][key] = {
            "round": _mean_std([r for r in reached if r is not None]),
            "sim_wallclock_s": _mean_std([f["rounds_to_target"][key].get("sim_wallclock_s") for f in feds
                                          if f["rounds_to_target"][key]["round"] is not None]),
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
    # n_seeds counts runs; it equals the seed count unless there are holdout redraws.
    return {"metric": metric, "n_seeds": len(results), "federated_beats_local_only_mean": beats_local,
            "federated_between_local_only_and_centralized": between}


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(Path(p).read_text()) for p in paths]


def format_summary(summary: dict[str, Any]) -> str:
    def fmt(ms: dict) -> str:
        if ms["mean"] is None:
            return "n/a"
        return f"{ms['mean']:.3f} ± {ms['std']:.3f}" if ms["std"] is not None else f"{ms['mean']:.3f}"

    draws = f", holdout draws {summary['holdout_draws']} = {summary['n_runs']} runs" if "n_runs" in summary else ""
    lines = [f"{summary['experiment']} ({summary['n_seeds']} seeds: {summary['seeds']}{draws})",
             f"{'arm':<28}" + "".join(f"{m:>20}" for m in METRICS)]
    arms = summary["arms"]
    if "centralized" in arms:
        lines.append(f"{'centralized':<28}" + "".join(f"{fmt(arms['centralized'][m]):>20}" for m in METRICS))
    if "local_only" in arms:
        lo = arms["local_only"]
        lines.append(f"{'local_only (silo mean)':<28}" + "".join(f"{fmt(lo['mean_across_silos'][m]):>20}" for m in METRICS))
        for silo, ms in lo["per_silo"].items():
            lines.append(f"{'  ' + silo:<28}" + "".join(f"{fmt(ms[m]):>20}" for m in METRICS))
    if "ensemble_of_locals" in arms:
        lines.append(f"{'ensemble_of_locals':<28}" + "".join(f"{fmt(arms['ensemble_of_locals'][m]):>20}" for m in METRICS))
    if "federated" in arms:
        fed = arms["federated"]
        lines.append(f"{'federated (final round)':<28}" + "".join(f"{fmt(fed['final'][m]):>20}" for m in METRICS))
        lines.append(f"federated total bytes: {fmt(fed['total_bytes'])}")
        if "sim_wallclock_s_total" in fed:
            lines.append(f"federated simulated wall-clock total (s): {fmt(fed['sim_wallclock_s_total'])}"
                         f"; stalled runs: {fed['n_stalled']}/{len(fed['n_rounds'])}")
        for key, v in fed["rounds_to_target"].items():
            sim_t = v.get("sim_wallclock_s")
            sim_txt = f", simulated time {fmt(sim_t)} s" if sim_t and sim_t["mean"] is not None else ""
            lines.append(f"rounds to target ({key}): {fmt(v['round'])}{sim_txt}, reached in {v['n_reached']}/{v['n_runs']} runs")
    if "hierarchical" in arms:
        hi = arms["hierarchical"]
        lines.append(f"{'hierarchical global':<28}" + "".join(f"{fmt(hi['final'][m]):>20}" for m in METRICS))
        lines.append(f"{'hierarchical regional':<28}" + "".join(f"{fmt(hi['regional_final'][m]):>20}" for m in METRICS))
        lines.append("per region, each model on that region's own test data (macro-F1):")
        for reg, mods in summary["hierarchy"]["per_region"].items():
            lines.append(f"  {reg:<8}" + "   ".join(f"{mod} {fmt(ms['macro_f1'])}" for mod, ms in mods.items()))
        d = summary["hierarchy"]["disagreement"]
        lines.append(f"regional model accuracy where local/regional/global agree {fmt(d['acc_agree'])}, "
                     f"disagree {fmt(d['acc_disagree'])} (a flag, not an independent opinion)")
    if "fairness" in summary:
        lines.append("fairness (label-mix holdout accuracy across silos; spread = max - min, lower is fairer):")
        for arm, f in summary["fairness"].items():
            counts = ", ".join(f"{s} x{n}" for s, n in f["worst_silo"]["silo_counts"].items())
            lines.append(f"  {arm:<18} gap {fmt(f['gap'])}   worst silo {fmt(f['worst_silo']['value'])} "
                         f"({counts})   silo mean {fmt(f['mean'])}")
    if "privacy" in summary:
        pr = summary["privacy"]
        if "epsilon" in pr:
            lines.append(f"DP: target epsilon {pr['target_epsilon']}, spent {fmt(pr['epsilon'])} "
                         f"(weakest model), max delta {pr['delta_max']:.2e}")
        else:
            lines.append("DP: off (epsilon = inf)")
        if "attack" in pr:
            def ci(v):
                return f"[{v[0]:.3f}, {v[1]:.3f}]" if v else "n/a"

            lines.append("gradient inversion (rel_mse < 1 = leakage; nn_id = window re-identified; "
                         "[95% CI over seeds x trials]; chance = random guess):")
            for key, c in pr["attack"]["cells"].items():
                la = f"  labels {fmt(c['label_accuracy'])}" if "label_accuracy" in c else ""
                lines.append(f"  {key:<34} rel_mse {fmt(c['rel_mse'])} {ci(c['ci95']['rel_mse'])}  "
                             f"nn_id {fmt(c['nn_identified'])} {ci(c['ci95']['nn_identified'])}  "
                             f"chance nn_id {fmt(c['chance']['nn_identified'])}{la}")
    if "model_report" in summary:
        mr = summary["model_report"]
        lines.append(f"model report: arch {mr['arch']}, {mr['n_params']} parameters, payload {mr['payload_bytes']} B "
                     f"({mr['scope']})")
        lines.append(f"  {'model':<20}" + "".join(f"{m:>22}" for m in MODEL_REPORT_METRICS))
        for model, row in mr["rows"].items():
            lines.append(f"  {model:<20}" + "".join(f"{fmt(row[m]):>22}" for m in MODEL_REPORT_METRICS))
    if "h1" in summary:
        h = summary["h1"]
        lines.append(f"H1 ({h['metric']}): federated > local_only mean in {h['federated_beats_local_only_mean']}/{h['n_seeds']} runs; "
                     f"between local_only and centralized in {h['federated_between_local_only_and_centralized']}/{h['n_seeds']}")
    if "stats" in summary:
        st = summary["stats"]
        lines.append(f"paired differences federated - X over runs (95% bootstrap CI); "
                     f"primary: {st['primary'] or 'none'}; {st['n_comparisons']} comparisons:")
        for c in st["comparisons"]:
            ci = f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]" if c["ci95"] else "n/a"
            lines.append(f"  {'PRIMARY ' if c['role'] == 'primary' else ''}{c['b']:<20} {c['metric']:<18} "
                         f"{c['mean_diff']:+.3f} {ci}  federated better in {c['n_a_better']}/{c['n_pairs']}")
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
            "privacy": s.get("privacy"),
            "sim_wallclock_s_total": arms.get("federated", {}).get("sim_wallclock_s_total"),
            "time_to_target_sim_s": {k: v.get("sim_wallclock_s") for k, v in
                                     arms.get("federated", {}).get("rounds_to_target", {}).items()},
        }
        if "model_report" in s:  # Phase H6: the family comparison
            row["model_report"] = s["model_report"]
            mr = s["model_report"]
            lines.append(f"{'':<46}  {mr['arch']} ({mr['n_params']} params, {mr['payload_bytes']} B): " + "  ".join(
                f"{m} acc {ms(r['accuracy'])} F1 {ms(r['macro_f1'])} ECE {ms(r['ece'])} worst {ms(r['worst_node_accuracy'])}"
                for m, r in mr["rows"].items()))
        rows.append(row)
        h1_txt = f"{h1['federated_beats_local_only_mean']}/{h1['n_seeds']}" if h1 else "n/a"
        if row["sim_wallclock_s_total"]:
            ttt = "  ".join(f"{k} {ms(v)} s" for k, v in row["time_to_target_sim_s"].items() if v)
            fedr = arms["federated"]
            lines.append(f"{'':<46}  simulated total {ms(row['sim_wallclock_s_total'])} s  "
                         f"time-to-target {ttt or 'n/a'}  stalled {fedr['n_stalled']}/{len(fedr['n_rounds'])}")
        pr = s.get("privacy") or {}
        if pr:
            eps = ms(pr["epsilon"]) if "epsilon" in pr else "inf"
            hl = pr.get("attack", {}).get("headline")
            atk = ""
            if hl:
                k = f"{hl['silo']}/{hl['state']}-B{hl['batch_size']}-{hl['labels']}"
                c = pr["attack"]["cells"][k]
                atk = (f"headline {k}: rel {ms(c['rel_mse'])} nn {ms(c['nn_identified'])} "
                       f"(chance nn {ms(c['chance']['nn_identified'])})")
            lines.append(f"{'':<46}  eps spent {eps}  {atk}")
        lines.append(
            f"{v.name:<46}{ms(row['federated_accuracy']):>13}{ms(row['federated_macro_f1']):>13}"
            f"{ms(row['local_only_accuracy']):>13}{ms(row['centralized_accuracy']):>13}"
            f"{(f'{gap:+.3f}' if gap is not None else 'n/a'):>10}"
            f"{ms(row['fairness_spread_federated']):>13}{ms(row['fairness_spread_local_only']):>13}{h1_txt:>6}")
    return rows, "\n".join(lines)
