"""The PREREG.md section 2 decision rules for H1-H4 (CLAUDE.md Phase H9).

Input: per model family, each arm's evaluation results (one run per evaluation seed), keyed by
shift setting ``(kind, magnitude)`` (``result.shift``). Each arm comes from its own run at its own
validation-selected configuration (Phase H8), so arms are paired by seed.

Rule (one comparison): the paired difference over seeds, its mean and the hierarchical percentile
bootstrap 95% CI (metrics/aggregate.bootstrap_ci, one value per seed).
- CI above 0: ``supported``; CI below 0: ``not supported``; CI contains 0 (or < 2 seeds):
  ``inconclusive``. No multiple-comparison correction (disclosed in PREREG section 2).

- H1: federated (final model) minus local-only (mean over nodes), accuracy, no shift
  (magnitude 0, the control: bit-identical to unshifted data).
- H2: at the primary shift, on the shifted regions pooled (n_test-weighted):
  (a) regional minus local-only and (b) regional minus global. Supported iff both are
  supported; not supported if either is not supported; else inconclusive.
  Control clause, magnitude 0, every region pooled: ``supported`` unless regional minus global
  has its CI above 0 (regional clearly better than global), then ``not supported``.
- H3 (descriptive): accuracy where local, regional and global agree minus where they disagree
  (``result.disagreement``, the headline model), primary shift. The three models share
  training data, so their agreement is correlated: a flag, never independent confirmation.
- H4: the H1 and H2 verdicts per family. ``holds in every family`` iff all supported; ``does
  not hold`` if any family is not supported; else ``mixed / inconclusive``.

Accuracy is primary (continuity with the locked flat H1); macro-F1 is reported alongside.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fedastro.metrics.aggregate import bootstrap_ci

SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE = "supported", "not supported", "inconclusive"
PRIMARY_METRIC = "accuracy"
SECONDARY_METRIC = "macro_f1"
Setting = tuple[str, float]


def verdict(ci: list[float] | None) -> str:
    if ci is None:
        return INCONCLUSIVE
    if ci[0] > 0:
        return SUPPORTED
    if ci[1] < 0:
        return NOT_SUPPORTED
    return INCONCLUSIVE


def setting_of(result: dict[str, Any]) -> Setting:
    sh = result.get("shift") or {}
    if not sh.get("enabled"):
        return ("none", 0.0)
    return (str(sh["kind"]), float(sh["magnitude"]))


def is_control(s: Setting) -> bool:
    return s[1] == 0.0


def by_setting(results: list[dict[str, Any]]) -> dict[Setting, dict[int, dict[str, Any]]]:
    out: dict[Setting, dict[int, dict[str, Any]]] = {}
    for r in results:
        cell = out.setdefault(setting_of(r), {})
        if r["seed"] in cell:
            raise ValueError(f"two runs for seed {r['seed']} at shift {setting_of(r)} "
                             f"({r['run_id']}, {cell[r['seed']]['run_id']})")
        cell[r["seed"]] = r
    return out


def _control(runs: dict[Setting, dict[int, dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    """Every control cell pooled into one: kind does not matter at magnitude 0 (identity)."""
    cells = [c for s, c in runs.items() if is_control(s)]
    if len(cells) != 1:
        raise ValueError(f"need exactly one magnitude-0 (control) setting, have {[s for s in runs if is_control(s)]}")
    return cells[0]


def compare(a: dict[int, dict[str, Any]], b: dict[int, dict[str, Any]], fa, fb,
            label: str) -> dict[str, Any]:
    """Paired by seed: fa(result_a) - fb(result_b). Seeds must match."""
    if set(a) != set(b):
        raise ValueError(f"{label}: seeds {sorted(a)} != {sorted(b)}")
    pairs = [(fa(a[s]), fb(b[s])) for s in sorted(a)]
    diffs = [x - y for x, y in pairs if x is not None and y is not None]
    ci = bootstrap_ci([[d] for d in diffs]) if len(diffs) > 1 else None
    return {"comparison": label, "a": float(np.mean([x for x, _ in pairs if x is not None])) if diffs else None,
            "b": float(np.mean([y for _, y in pairs if y is not None])) if diffs else None,
            "diff_mean": float(np.mean(diffs)) if diffs else None,
            "diff_std": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
            "ci95": ci, "n_seeds": len(diffs), "verdict": verdict(ci)}


# --- scores -----------------------------------------------------------------------------

def _region_pool(result: dict[str, Any], model: str, regions: list[str], metric: str) -> float | None:
    """n_test-weighted mean over regions of hierarchy.per_region[r][model][metric]. Exact for
    accuracy (a pooled mean); for macro-F1 it is the weighted mean of region scores."""
    pr = result["hierarchy"]["per_region"]
    vals = [(pr[r][model][metric], pr[r]["n_test"]) for r in regions]
    if any(v is None for v, _ in vals):
        return None
    return float(sum(v * n for v, n in vals) / sum(n for _, n in vals))


def _local_region_pool(result: dict[str, Any], regions: list[str], metric: str) -> float | None:
    """The local-only arm's nodes in these regions, each on its own test data, n_test-weighted."""
    per = result["arms"]["local_only"]["per_silo"]
    vals = [(per[s["name"]]["final"][metric], s["n_test"]) for s in result["silos"] if s["region"] in regions]
    if not vals or any(v is None for v, _ in vals):
        return None
    return float(sum(v * n for v, n in vals) / sum(n for _, n in vals))


def _all_regions(result: dict[str, Any]) -> list[str]:
    return list(result["hierarchy"]["per_region"])


# --- hypotheses --------------------------------------------------------------------------

def h1(runs: dict[str, dict[Setting, dict[int, dict[str, Any]]]]) -> dict[str, Any]:
    fed, lo = _control(runs["federated"]), _control(runs["local_only"])
    rows = [compare(fed, lo, lambda r, m=m: r["arms"]["federated"]["final"][m],
                    lambda r, m=m: r["arms"]["local_only"]["mean"][m], f"federated - local_only ({m})")
            for m in (PRIMARY_METRIC, SECONDARY_METRIC)]
    return {"setting": "no shift (magnitude 0)", "rows": rows, "verdict": rows[0]["verdict"]}


def h2(runs: dict[str, dict[Setting, dict[int, dict[str, Any]]]], primary: Setting,
       shifted_regions: list[str]) -> dict[str, Any]:
    hi, lo = runs["hierarchical"][primary], runs["local_only"][primary]
    rows = []
    for m in (PRIMARY_METRIC, SECONDARY_METRIC):
        rows.append(compare(hi, lo, lambda r, m=m: _region_pool(r, "regional", shifted_regions, m),
                            lambda r, m=m: _local_region_pool(r, shifted_regions, m),
                            f"(a) regional - local_only, regions {shifted_regions} ({m})"))
        rows.append(compare(hi, hi, lambda r, m=m: _region_pool(r, "regional", shifted_regions, m),
                            lambda r, m=m: _region_pool(r, "global", shifted_regions, m),
                            f"(b) regional - global, regions {shifted_regions} ({m})"))
    a, b = rows[0]["verdict"], rows[1]["verdict"]  # primary metric
    v = SUPPORTED if a == b == SUPPORTED else NOT_SUPPORTED if NOT_SUPPORTED in (a, b) else INCONCLUSIVE
    hc = _control(runs["hierarchical"])
    ctrl = compare(hc, hc, lambda r: _region_pool(r, "regional", _all_regions(r), PRIMARY_METRIC),
                   lambda r: _region_pool(r, "global", _all_regions(r), PRIMARY_METRIC),
                   f"control: regional - global, every region ({PRIMARY_METRIC})")
    ctrl_v = NOT_SUPPORTED if ctrl["ci95"] is not None and ctrl["ci95"][0] > 0 else SUPPORTED
    ctrl["verdict"] = ctrl_v
    ctrl["rule"] = "supported unless the CI is above 0 (regional clearly better than global)"
    return {"setting": f"shift {primary[0]} magnitude {primary[1]}", "rows": rows, "verdict": v,
            "control": ctrl}


def h3(runs: dict[str, dict[Setting, dict[int, dict[str, Any]]]], primary: Setting) -> dict[str, Any]:
    hi = runs["hierarchical"][primary]
    row = compare(hi, hi, lambda r: r["disagreement"]["acc_agree"], lambda r: r["disagreement"]["acc_disagree"],
                  "accuracy where local/regional/global agree - where they disagree")
    model = next(iter(hi.values()))["disagreement"].get("headline_model")
    return {"setting": f"shift {primary[0]} magnitude {primary[1]}", "rows": [row], "verdict": row["verdict"],
            "model": model, "note": "descriptive; the three models share training data, so agreement is "
                                    "correlated: a flag for review, not an independent confirmation"}


def h4(per_family: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for h in ("H1", "H2"):
        vs = {f: per_family[f][h]["verdict"] for f in per_family}
        overall = ("holds in every family" if all(v == SUPPORTED for v in vs.values())
                   else "does not hold" if NOT_SUPPORTED in vs.values() else "mixed / inconclusive")
        out[h] = {"per_family": vs, "verdict": overall}
    return out


def evaluate_hypotheses(families: dict[str, dict[str, list[dict[str, Any]]]], primary: Setting,
                        shifted_regions: list[str]) -> dict[str, Any]:
    """``families[family][arm]``: that arm's evaluation results (arms: local_only, federated,
    hierarchical), every shift setting together."""
    per_family = {}
    for fam, arms in families.items():
        runs = {arm: by_setting(res) for arm, res in arms.items()}
        per_family[fam] = {"H1": h1(runs), "H2": h2(runs, primary, shifted_regions), "H3": h3(runs, primary)}
    return {"primary_shift": {"kind": primary[0], "magnitude": primary[1]}, "shifted_regions": shifted_regions,
            "rule": "paired by seed; hierarchical percentile bootstrap 95% CI; above 0 supported, below 0 not "
                    "supported, else inconclusive; no multiple-comparison correction",
            "families": per_family, "H4": h4(per_family)}


def format_hypotheses(rep: dict[str, Any]) -> str:
    def ci(r):
        return "      n/a       " if r["ci95"] is None else f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"

    def num(x):
        return "  n/a " if x is None else f"{x:+.3f}"

    lines = [f"primary shift: {rep['primary_shift']['kind']} {rep['primary_shift']['magnitude']}; "
             f"shifted regions {rep['shifted_regions']} (shift SIMULATED)"]
    for fam, hs in rep["families"].items():
        lines.append(f"== {fam}")
        for h in ("H1", "H2", "H3"):
            lines.append(f"  {h}: {hs[h]['verdict'].upper()}  ({hs[h]['setting']})")
            for r in hs[h]["rows"] + ([hs[h]["control"]] if "control" in hs[h] else []):
                lines.append(f"    {r['comparison']:<62} {num(r['diff_mean'])} {ci(r)} n={r['n_seeds']}  {r['verdict']}")
    for h, v in rep["H4"].items():
        lines.append(f"H4 ({h} across families): {v['verdict'].upper()}  {v['per_family']}")
    lines.append(rep["rule"])
    return "\n".join(lines)


__all__ = ["evaluate_hypotheses", "format_hypotheses", "verdict", "compare", "by_setting",
           "SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"]
