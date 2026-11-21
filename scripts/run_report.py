#!/usr/bin/env python3
"""Self-contained HTML report for one multi-seed experiment: the three-arm comparison.

Reads only the run artifacts (``result.json`` and ``manifest.json`` per run, the
CLAUDE.md §6 contract) and never recomputes training. Standard library only, so it runs
on the cluster or the laptop without extra dependencies. Charts are inline SVG and the
page needs no network.

Means are over seeds; std is the sample std (ddof=1), as in ``fedastro summarize``.

Usage:
    python scripts/run_report.py runs/wesad_by_subject-seed*-ce31e25dda \\
        --out reports/wesad_by_subject.html --label "PACE ICE, SLURM job 6665444"
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import statistics
import sys
from pathlib import Path

METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "auc")
METRIC_NAMES = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
                "macro_f1": "Macro-F1", "auc": "AUC (macro OvR)"}
ARMS = ("centralized", "federated", "local_only", "ensemble_of_locals")
ARM_NAMES = {"centralized": "Centralized (upper bound)", "federated": "Federated",
             "local_only": "Local-only, silo mean (lower bound)",
             "ensemble_of_locals": "Ensemble of local models (no federation)"}
ARM_SHORT = {"centralized": "Centralized", "federated": "Federated", "local_only": "Local-only",
             "ensemble_of_locals": "Ensemble of locals"}
ARM_VAR = {"federated": "--series-1", "centralized": "--series-2", "local_only": "--series-3",
           "ensemble_of_locals": "--series-4"}
PACKAGES = ("torch", "flwr", "ray", "opacus", "numpy", "scikit-learn")
# provenance.domain (CLAUDE2.md rule 10) for artifacts written before Phase 11, which have
# no domain field. A copy of the registry in fedastro/data/__init__.py (this script is
# standard-library only); a test keeps the two in step.
LEGACY_DOMAINS = {"SYNTHETIC": "synthetic", "WESAD": "terrestrial_proxy", "OSD-435": "terrestrial_analog"}


def domain(prov: dict) -> str:
    return prov.get("domain") or LEGACY_DOMAINS.get(prov.get("dataset"), "not recorded")


# -- loading -------------------------------------------------------------------------
def load_runs(paths: list[str]) -> list[dict]:
    runs = []
    for p in map(Path, paths):
        res = p if p.name == "result.json" else p / "result.json"
        if not res.is_file():
            continue
        man = res.parent / "manifest.json"
        runs.append({"result": json.loads(res.read_text()), "dir": res.parent,
                     "manifest": json.loads(man.read_text()) if man.is_file() else None})
    return sorted(runs, key=lambda r: (r["result"]["seed"], draw(r["result"])))


def draw(r: dict) -> int:
    """Holdout draw of a run (Phase 12); 0 for runs without redraws."""
    return (r.get("evaluation") or {}).get("holdout_draw", 0)


def run_label(r: dict) -> str:
    return f"seed {r['seed']}" + (f" · draw {draw(r)}" if "evaluation" in r and r["evaluation"].get("holdout_draws", 1) > 1 else "")


def load_summary(runs: list[dict]) -> dict | None:
    """The sweep summary (<name>-<hash>-summary.json) next to the run dirs, for the paired
    statistics; None if it is not there."""
    r0 = runs[0]["result"]
    path = runs[0]["dir"].parent / f"{experiment_name(r0['run_id'])}-{config_hash(r0['run_id'])}-summary.json"
    return json.loads(path.read_text()) if path.is_file() else None


def config_hash(run_id: str) -> str:
    return run_id.rsplit("-", 1)[-1]


def experiment_name(run_id: str) -> str:
    return run_id.rsplit("-seed", 1)[0]  # also for <name>-seed<s>-draw<d>-<hash>


# -- statistics ----------------------------------------------------------------------
def mean_std(vals: list[float | None]) -> tuple[float | None, float | None]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else None)


def fmt(v: float | None, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def fmt_ms(ms: tuple[float | None, float | None]) -> str:
    m, s = ms
    return "n/a" if m is None else (fmt(m) if s is None else f"{fmt(m)} ± {fmt(s)}")


def arm_final(r: dict, arm: str) -> dict | None:
    a = r["arms"].get(arm)
    if not a:
        return None
    return a.get("mean") if arm == "local_only" else a.get("final")


def esc(x) -> str:
    return html.escape(str(x))


# -- SVG primitives ------------------------------------------------------------------
W, H = 680, 300
ML, MR, MT, MB = 52, 140, 18, 40
WIDE = 1000  # full-width charts get a wider canvas so their text stays the same size


def nice_ticks(lo: float, hi: float) -> list[float]:
    span = hi - lo
    step = 0.05 if span <= 0.35 else 0.1 if span <= 0.8 else 0.2
    t, out = round(lo / step) * step, []
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            out.append(round(t, 4))
        t += step
    return out


def y_domain(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    pad = max(0.03, (hi - lo) * 0.08)
    lo, hi = max(0.0, lo - pad), min(1.0, hi + pad)
    step = 0.05
    return (int(lo / step) * step, min(1.0, (int(hi / step) + 1) * step))


def end_labels(items: list[tuple[str, float, str]], min_gap: float = 15) -> list[tuple[str, float, str]]:
    """Direct end labels, kept only where they don't collide (legend + tooltip cover the rest)."""
    kept: list[tuple[str, float, str]] = []
    for text, y, var in sorted(items, key=lambda i: i[1]):
        if all(abs(y - k[1]) >= min_gap for k in kept):
            kept.append((text, y, var))
    return kept


def line_chart(runs: list[dict], metric: str) -> str:
    """Federated per-round mean ± std band, with the centralized and local-only finals as
    horizontal references. Rounds and centralized epochs are different units, so the
    centralized curve is not drawn on this axis; only its final value is."""
    per_round: dict[int, list[float]] = {}
    for run in runs:
        for rd in run["result"]["arms"]["federated"]["rounds"]:
            v = rd["global"].get(metric)
            if v is not None:
                per_round.setdefault(rd["round"], []).append(v)
    rounds = sorted(per_round)
    stats = {r: mean_std(per_round[r]) for r in rounds}
    refs = {arm: mean_std([(arm_final(x["result"], arm) or {}).get(metric) for x in runs])[0]
            for arm in ("centralized", "local_only")}
    vals = [m + (s or 0) for m, s in stats.values()] + [m - (s or 0) for m, s in stats.values()]
    vals += [v for v in refs.values() if v is not None]
    y0, y1 = y_domain(vals)
    pw, ph = W - ML - MR, H - MT - MB
    xr = lambda r: ML + (r - rounds[0]) / max(1, rounds[-1] - rounds[0]) * pw
    yv = lambda v: MT + (1 - (v - y0) / (y1 - y0)) * ph

    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
           f'aria-label="Federated {esc(METRIC_NAMES[metric])} per round with reference lines">']
    for t in nice_ticks(y0, y1):
        out.append(f'<line class="grid" x1="{ML}" x2="{ML + pw}" y1="{yv(t):.1f}" y2="{yv(t):.1f}"/>'
                   f'<text class="tick" x="{ML - 8}" y="{yv(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
    xt = [r for r in rounds if r == rounds[0] or r % 5 == 0 or r == rounds[-1]]
    for r in xt:
        out.append(f'<text class="tick" x="{xr(r):.1f}" y="{MT + ph + 18}" text-anchor="middle">{r}</text>')
    out.append(f'<line class="axis" x1="{ML}" x2="{ML + pw}" y1="{MT + ph}" y2="{MT + ph}"/>')
    out.append(f'<text class="axis-title" x="{ML + pw / 2:.1f}" y="{H - 4}" text-anchor="middle">'
               f'Federated round</text>')

    upper = [(xr(r), yv(stats[r][0] + (stats[r][1] or 0))) for r in rounds]
    lower = [(xr(r), yv(stats[r][0] - (stats[r][1] or 0))) for r in reversed(rounds)]
    out.append('<polygon class="band" style="fill:var(--series-1)" points="'
               + " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower) + '"/>')
    labels = []
    for arm, v in refs.items():
        if v is None:
            continue
        out.append(f'<line class="ref" style="stroke:var({ARM_VAR[arm]})" x1="{ML}" x2="{ML + pw}" '
                   f'y1="{yv(v):.1f}" y2="{yv(v):.1f}"/>')
        labels.append((f"{ARM_SHORT[arm]} {v:.3f}", yv(v), ARM_VAR[arm]))
    pts = " ".join(f"{xr(r):.1f},{yv(stats[r][0]):.1f}" for r in rounds)
    out.append(f'<polyline class="series" style="stroke:var(--series-1)" points="{pts}"/>')
    last = rounds[-1]
    out.append(f'<circle class="dot" style="fill:var(--series-1)" cx="{xr(last):.1f}" '
               f'cy="{yv(stats[last][0]):.1f}" r="4.5"/>')
    labels.append((f"Federated {stats[last][0]:.3f}", yv(stats[last][0]), "--series-1"))
    for text, y, var in end_labels(labels):
        out.append(f'<line style="stroke:var({var})" class="key" x1="{ML + pw + 8}" x2="{ML + pw + 20}" '
                   f'y1="{y:.1f}" y2="{y:.1f}"/><text class="label" x="{ML + pw + 25}" y="{y + 4:.1f}">'
                   f'{esc(text)}</text>')

    out.append(f'<line class="xh" x1="0" x2="0" y1="{MT}" y2="{MT + ph}"/>')
    half = pw / max(1, len(rounds) - 1) / 2
    for r in rounds:
        m, s = stats[r]
        tip = (f"Round {r}<br>Federated {fmt(m)}" + (f" ± {fmt(s)}" if s is not None else "")
               + f" (n={len(per_round[r])} runs)")
        for arm, v in refs.items():
            if v is not None:
                tip += f"<br>{ARM_SHORT[arm]} (final) {fmt(v)}"
        out.append(f'<rect class="hit" x="{xr(r) - half:.1f}" y="{MT}" width="{2 * half:.1f}" height="{ph}" '
                   f'data-cx="{xr(r):.1f}" data-tip="{esc(tip)}"/>')
    out.append("</svg>")
    return "".join(out)


def interval_chart(rows: list[tuple[str, list[tuple[str, list[float], str]]]],
                   refs: list[tuple[str, float, str]], aria: str) -> str:
    """Rows of mean ± std intervals (one or more per row), with optional vertical refs.
    rows: [(row label, [(series name, per-seed values, css var), ...]), ...]."""
    all_vals = [v for _, ser in rows for _, vals, _ in ser for v in vals] + [v for _, v, _ in refs]
    x0, x1 = y_domain(all_vals)
    ml, mr, mt = 150, 24, 30 if refs else 12
    n_sub = max(len(ser) for _, ser in rows)
    row_h = 22 + 14 * (n_sub - 1)
    h = mt + row_h * len(rows) + 34
    pw = WIDE - ml - mr
    xv = lambda v: ml + (v - x0) / (x1 - x0) * pw
    out = [f'<svg viewBox="0 0 {WIDE} {h}" class="chart" role="img" aria-label="{esc(aria)}">']
    for t in nice_ticks(x0, x1):
        out.append(f'<line class="grid" x1="{xv(t):.1f}" x2="{xv(t):.1f}" y1="{mt}" y2="{h - 34}"/>'
                   f'<text class="tick" x="{xv(t):.1f}" y="{h - 16}" text-anchor="middle">{t:.2f}</text>')
    for name, v, var in refs:
        out.append(f'<line class="ref" style="stroke:var({var})" x1="{xv(v):.1f}" x2="{xv(v):.1f}" '
                   f'y1="{mt - 6}" y2="{h - 34}"/>')
    for name, v, var in refs:
        anchor = "end" if xv(v) > ml + pw * 0.7 else "start"
        dx = -5 if anchor == "end" else 5
        out.append(f'<text class="label" x="{xv(v) + dx:.1f}" y="{mt - 10}" text-anchor="{anchor}">'
                   f'{esc(name)} {v:.3f}</text>')
    for i, (label, ser) in enumerate(rows):
        yc = mt + i * row_h + row_h / 2
        out.append(f'<text class="label" x="{ml - 10}" y="{yc + 4:.1f}" text-anchor="end">{esc(label)}</text>')
        for j, (sname, vals, var) in enumerate(ser):
            y = yc + (j - (len(ser) - 1) / 2) * 14
            m, s = mean_std(vals)
            if m is None:
                continue
            if s is not None:
                out.append(f'<line class="whisker" style="stroke:var({var})" x1="{xv(max(x0, m - s)):.1f}" '
                           f'x2="{xv(min(x1, m + s)):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
            tip = (f"{esc(label)} · {esc(sname)}<br>{fmt_ms((m, s))} over {len(vals)} runs"
                   f"<br>per run: {', '.join(fmt(v) for v in vals)}")
            out.append(f'<circle class="dot" style="fill:var({var})" cx="{xv(m):.1f}" cy="{y:.1f}" r="5"/>'
                       f'<circle class="hit" cx="{xv(m):.1f}" cy="{y:.1f}" r="12" data-tip="{esc(tip)}"/>')
    out.append("</svg>")
    return "".join(out)


def legend(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:var({var})"></i>{esc(name)}</span>' for name, var in items) + "</div>"


# -- page sections -------------------------------------------------------------------
def table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def integrity(runs: list[dict]) -> tuple[list[tuple[str, str, str]], bool]:
    """(check, status, detail) rows; status is ok / warn / fail. Shown verbatim."""
    rs = [r["result"] for r in runs]
    rows = []
    names = {experiment_name(r["run_id"]) for r in rs}
    hashes = {config_hash(r["run_id"]) for r in rs}
    rows.append(("One experiment and config", "ok" if len(names) == 1 and len(hashes) == 1 else "fail",
                 f"experiment {', '.join(sorted(names))}; config hash {', '.join(sorted(hashes))}"))
    keys = [(r["seed"], draw(r)) for r in rs]
    seeds = sorted({s for s, _ in keys})
    rows.append(("Distinct runs (seed, holdout draw)", "ok" if len(set(keys)) == len(keys) else "fail",
                 f"{len(keys)} runs: " + ", ".join(run_label(r) for r in rs)))
    rows.append(("At least 5 seeds (poster rule)", "ok" if len(seeds) >= 5 else "warn", f"{len(seeds)} seeds"))
    shas = {r["git_sha"] for r in rs}
    dirty = any(s.endswith("-dirty") or s == "unknown" for s in shas)
    rows.append(("Same, clean git commit", "ok" if len(shas) == 1 and not dirty else "fail",
                 ", ".join(sorted(shas))))
    kinds = {p.get("kind", "?") for r in rs for p in r.get("provenance", [])}
    rows.append(("Real data (not SYNTHETIC)", "ok" if kinds == {"REAL"} else "fail",
                 ", ".join(sorted(kinds)) or "no provenance recorded"))
    sums = {(p.get("dataset"), p.get("checksum")) for r in rs for p in r.get("provenance", [])}
    rows.append(("Same dataset checksum in every run", "ok" if len(sums) == 1 else "fail",
                 "; ".join(f"{d}: {c}" for d, c in sorted(sums, key=str))))
    by_draw: dict[int, set] = {}
    for r in rs:
        by_draw.setdefault(draw(r), set()).add(json.dumps(r.get("holdout", {}).get("subjects")))
    rows.append(("Same held-out subjects in every run of a draw", "ok" if all(len(h) == 1 for h in by_draw.values()) else "warn",
                 "; ".join(f"draw {d}: {', '.join(sorted(h))}" for d, h in sorted(by_draw.items()))))
    leak = []
    for r in rs:
        held = set(r.get("holdout", {}).get("subjects") or [])
        seen: dict[int, str] = {}
        for s in r.get("silos", []):
            for subj in s.get("subjects") or []:
                if subj in held or subj in seen:
                    leak.append(f"seed {r['seed']}: subject {subj}")
                seen[subj] = s["name"]
    rows.append(("Subjects disjoint across silos and held-out set", "fail" if leak else "ok",
                 "; ".join(leak) or "no subject appears twice"))
    if any("groups" in r for r in rs):
        group_leak = [f"seed {r['seed']}: {g} in {s['name']}" for r in rs
                      for s in r.get("silos", []) for g in s.get("groups") or []
                      if g in (r.get("groups") or {}).get("holdout_groups", [])]
        rows.append(("Held-out groups contribute no training data", "fail" if group_leak else "ok",
                     "; ".join(group_leak) or "no held-out group appears in a silo"))
    missing = [r["run_id"] for r in runs_without_manifest(runs)]
    rows.append(("manifest.json present", "warn" if missing else "ok", ", ".join(missing) or "all runs"))
    return rows, not any(s == "fail" for _, s, _ in rows)


def runs_without_manifest(runs: list[dict]) -> list[dict]:
    return [r["result"] for r in runs if r["manifest"] is None]


def h1_rows(rs: list[dict]) -> tuple[list[list[str]], int, int]:
    rows, beat, between = [], 0, 0
    for r in rs:
        f, lo, c = (arm_final(r, a) or {} for a in ("federated", "local_only", "centralized"))
        fa, la, ca = f.get("accuracy"), lo.get("accuracy"), c.get("accuracy")
        b = fa is not None and la is not None and fa > la
        bt = b and ca is not None and fa <= ca
        beat, between = beat + b, between + bt
        rows.append([esc(run_label(r)), fmt(ca), fmt(fa), fmt(la), fmt(f.get("macro_f1")), fmt(lo.get("macro_f1")),
                     fmt(c.get("macro_f1")), "yes" if b else "no", "yes" if bt else "no"])
    return rows, beat, between


def build(runs: list[dict], title: str, label: str | None) -> str:
    rs = [r["result"] for r in runs]
    checks, all_ok = integrity(runs)
    exp = experiment_name(rs[0]["run_id"])
    chash = config_hash(rs[0]["run_id"])
    arms_present = [a for a in ARMS if all(a in r["arms"] for r in rs)]
    finals = {a: {m: mean_std([(arm_final(r, a) or {}).get(m) for r in rs]) for m in METRICS}
              for a in arms_present}

    # stat tiles
    tiles = "".join(
        f'<div class="tile"><div class="tile-label"><i style="background:var({ARM_VAR[a]})"></i>'
        f'{esc(ARM_NAMES[a])}</div><div class="tile-value">{fmt(finals[a]["accuracy"][0])}</div>'
        f'<div class="tile-sub">accuracy ± {fmt(finals[a]["accuracy"][1])} (std over {len(rs)} runs)</div></div>'
        for a in ARMS if a in finals)

    h1, beat, between = h1_rows(rs) if {"federated", "local_only", "centralized"} <= set(arms_present) else ([], 0, 0)

    sections = []
    # integrity
    badge = {"ok": "✓ ok", "warn": "! warning", "fail": "✗ fail"}
    sections.append("<h2>Run integrity</h2><p class=\"muted\">Automatic checks on the artifacts. A "
                    "<b>fail</b> means these files should not be reported as one official run.</p>"
                    + table(["Check", "Status", "Detail"],
                            [[esc(c), f'<span class="st st-{s}">{badge[s]}</span>', esc(d)] for c, s, d in checks]))

    # three-arm table
    sections.append("<h2>Three-arm comparison (final, held-out set)</h2>"
                    "<p class=\"muted\">Mean ± sample std over runs (seeds × holdout draws). Local-only is each "
                    "silo's own model on the same global held-out set, averaged across silos. The ensemble "
                    "averages those local models' predicted probabilities, with no federation.</p>"
                    + table(["Arm"] + [METRIC_NAMES[m] for m in METRICS],
                            [[f'<i class="sw" style="background:var({ARM_VAR[a]})"></i>{esc(ARM_NAMES[a])}']
                             + [fmt_ms(finals[a][m]) for m in METRICS] for a in arms_present], "num"))
    if h1:
        sections.append(f"<p><b>H1, per run:</b> federated beats the local-only mean in <b>{beat}/{len(rs)}</b> "
                        f"runs, and lies between local-only and centralized in <b>{between}/{len(rs)}</b>.</p>")
    summary = load_summary(runs)
    stats = (summary or {}).get("stats")
    if stats:
        sections.append(
            "<h2>Paired differences (federated − comparator)</h2><p class=\"muted\">One pair per run: both arms "
            f"on the same silos and held-out set. {esc(stats['ci_method'])}. Primary comparison (pre-registered): "
            f"<b>{esc(stats['primary'] or 'none')}</b>; every other row is exploratory. "
            f"{stats['n_comparisons']} comparisons.</p>"
            + table(["Comparator", "Metric", "Role", "Mean difference", "95% CI", "Federated better"],
                    [[esc(ARM_SHORT.get(c["b"], c["b"])), esc(METRIC_NAMES[c["metric"]]), esc(c["role"]),
                      f'{c["mean_diff"]:+.3f}',
                      f'[{c["ci95"][0]:+.3f}, {c["ci95"][1]:+.3f}]' if c["ci95"] else "n/a",
                      f'{c["n_a_better"]}/{c["n_pairs"]}'] for c in stats["comparisons"]], "num"))
    elif len(rs) > 1:
        sections.append("<p class=\"muted\">Paired statistics: no sweep summary found next to these runs "
                        "(run <code>fedastro sweep</code> or <code>fedastro summarize</code>).</p>")

    # charts
    if "federated" in arms_present:
        lg = legend([(ARM_SHORT[a] + (" (mean ± std band)" if a == "federated" else " (final)"), ARM_VAR[a])
                     for a in ("federated", "centralized", "local_only") if a in arms_present])
        sections.append("<h2>Federated training curve</h2><p class=\"muted\">Global model on the held-out set "
                        "after each round. The band is ± 1 std over seeds. Centralized and local-only are their "
                        "<i>final</i> values, drawn as references: their training runs in epochs, not rounds.</p>"
                        + lg + '<div class="grid2">'
                        + "".join(f'<figure><figcaption>{esc(METRIC_NAMES[m])}</figcaption>{line_chart(runs, m)}</figure>'
                                  for m in ("accuracy", "macro_f1")) + "</div>")

    rows = [(METRIC_NAMES[m], [(ARM_SHORT[a], [(arm_final(r, a) or {}).get(m) for r in rs if (arm_final(r, a) or {}).get(m) is not None],
                                ARM_VAR[a]) for a in arms_present])
            for m in METRICS]
    sections.append("<h2>All metrics by arm</h2><p class=\"muted\">Dot = mean over runs, whisker = ± 1 std. "
                    "Hover a dot for per-run values.</p>"
                    + legend([(ARM_SHORT[a] + (" (silo mean)" if a == "local_only" else ""), ARM_VAR[a])
                              for a in arms_present])
                    + interval_chart(rows, [], "Final metrics by arm"))

    fair_arms = [a for a in ARMS if all(a in r.get("fairness", {}) for r in rs)]
    if fair_arms:
        def worst(a: str) -> str:
            names = [min(r["fairness"][a]["per_silo"], key=r["fairness"][a]["per_silo"].get) for r in rs]
            return ", ".join(f"{s} ×{names.count(s)}" for s in sorted(set(names)))

        sections.append(
            "<h2>Silo fairness</h2><p class=\"muted\">Held-out recall re-weighted to each silo's training label "
            "mix (label-mix heterogeneity only). Worst silo = the lowest silo; gap = best − worst, lower is "
            "fairer. Mean ± std over runs; the worst silo is named per run.</p>"
            + table(["Arm", "Worst silo", "Worst silo (which, per run)", "Gap (best − worst)", "Silo mean"],
                    [[esc(ARM_SHORT[a]), fmt_ms(mean_std([r["fairness"][a]["min"] for r in rs])), esc(worst(a)),
                      fmt_ms(mean_std([r["fairness"][a]["spread"] for r in rs])),
                      fmt_ms(mean_std([r["fairness"][a]["mean"] for r in rs]))] for a in fair_arms], "num"))

    per_group = [r["evaluation"]["per_holdout_group"] for r in rs if (r.get("evaluation") or {}).get("per_holdout_group")]
    if per_group:
        gnames = list(per_group[0])
        grp_arms = [a for a in ARMS if a in per_group[0][gnames[0]]]
        indiv = [r["evaluation"]["held_out_individuals"] for r in rs if r["evaluation"].get("held_out_individuals")]
        blocks = [(f"held-out group {g}", [p[g] for p in per_group]) for g in gnames]
        if indiv:
            blocks.append(("held-out individuals (training groups)", indiv))
        sections.append(
            "<h2>Held-out groups</h2><p class=\"muted\">Each arm on each whole held-out group, and on the held-out "
            "individuals of the training groups, separately (accuracy and macro-F1, mean ± std over runs).</p>"
            + table(["Held-out set", "n"] + [f"{ARM_SHORT[a]} acc / F1" for a in grp_arms],
                    [[esc(name), str(rows_[0][grp_arms[0]]["n"])]
                     + [f'{fmt_ms(mean_std([x[a]["accuracy"] for x in rows_]))} / '
                        f'{fmt_ms(mean_std([x[a]["macro_f1"] for x in rows_]))}' for a in grp_arms]
                     for name, rows_ in blocks], "num"))

    if "local_only" in arms_present:
        silo_names = [s["name"] for s in rs[0]["silos"]]
        per_silo = {s: {m: [r["arms"]["local_only"]["per_silo"][s]["final"][m] for r in rs] for m in METRICS}
                    for s in silo_names}
        refs = [(ARM_SHORT[a], finals[a]["accuracy"][0], ARM_VAR[a]) for a in ("federated", "centralized")
                if a in finals and finals[a]["accuracy"][0] is not None]
        sections.append("<h2>Each silo alone vs the federation</h2><p class=\"muted\">Accuracy on the global "
                        "held-out set of the model each silo trains on its own data (mean ± std over seeds), "
                        "against the federated and centralized means.</p>"
                        + legend([("Local-only, this silo", "--series-3")] + [(n, v) for n, _, v in refs])
                        + interval_chart([(s, [("local-only", per_silo[s]["accuracy"], "--series-3")]) for s in silo_names],
                                         refs, "Local-only accuracy per silo")
                        + table(["Silo"] + [METRIC_NAMES[m] for m in METRICS],
                                [[esc(s)] + [fmt_ms(mean_std(per_silo[s][m])) for m in METRICS] for s in silo_names], "num"))

    if h1:
        sections.append("<h2>Per-run results</h2>" + table(
            ["Run", "Centralized acc", "Federated acc", "Local-only acc", "Federated F1", "Local-only F1",
             "Centralized F1", "Fed > local", "Between bounds"], h1, "num"))

    # data
    silo_rows = []
    for s in [x["name"] for x in rs[0]["silos"]]:
        cells = []
        for r in rs:
            info = next(x for x in r["silos"] if x["name"] == s)
            lc = info.get("label_counts", {})
            cells.append(f'{info["n_train"]}<span class="muted"> · subj {",".join(map(str, info.get("subjects") or []))}'
                         f'<br>{"/".join(str(lc.get(k, 0)) for k in sorted(lc))}</span>')
        silo_rows.append([esc(s)] + cells)
    labels = sorted(rs[0]["silos"][0].get("label_counts", {}))
    hold = rs[0].get("holdout", {})
    sections.append("<h2>Data</h2><p class=\"muted\">Training windows per silo and seed (silo membership is "
                    f"re-drawn per seed). Label counts as {'/'.join(labels)}.</p>"
                    + table(["Silo"] + [esc(run_label(r).capitalize()) for r in rs], silo_rows)
                    + f"<p>Held-out set: <b>{hold.get('n_test', 'n/a')}</b> windows from subjects "
                    f"{esc(hold.get('subjects'))}; label counts {esc(hold.get('label_counts'))}.</p>")
    groups = rs[0].get("groups")
    if groups:
        sections.append(table(["Silo", "Groups"], [[esc(s), esc(", ".join(g))] for s, g in groups["silo_to_groups"].items()])
                        + f"<p>Held-out groups: <b>{esc(', '.join(groups['holdout_groups']) or 'none')}</b>; "
                        f"unused groups: {esc(', '.join(groups['unused_groups']) or 'none')}.</p>")

    # comms
    fed = rs[0]["arms"].get("federated")
    if fed:
        sections.append("<h2>Communication</h2>" + table(
            ["Strategy", "Rounds", "Parameters", "Payload per model", "Total bytes (mean over seeds)"],
            [[esc(fed.get("strategy")), str(len(fed.get("rounds", []))), str(fed.get("n_params")),
              f'{fed.get("payload_bytes", 0):,} B',
              f'{statistics.fmean(r["arms"]["federated"].get("total_bytes", 0) for r in rs):,.0f} B']], "num"))

    # provenance + environment
    prov = rs[0].get("provenance", [{}])[0]
    env_rows = []
    for run in runs:
        m = run["manifest"] or {}
        e = m.get("environment", {})
        pk = e.get("packages", {})
        env_rows.append([esc(run["result"]["run_id"]), esc(e.get("hostname", "n/a")), esc(e.get("platform", "n/a")),
                         esc(e.get("python", "n/a")), esc(" ".join(f"{p} {pk[p]}" for p in PACKAGES if p in pk)),
                         esc(m.get("started_utc", run["result"].get("timestamp_utc"))),
                         f'{m.get("real_elapsed_s", float("nan")):.0f} s' if m else "n/a"])
    sections.append("<h2>Provenance and environment</h2>" + table(["Field", "Value"], [
        ["Experiment", esc(exp)], ["Config hash", esc(chash)], ["Git commit", esc(", ".join(sorted({r["git_sha"] for r in rs})))],
        ["Dataset", esc(f'{prov.get("dataset")} ({prov.get("kind")})')], ["Domain", esc(domain(prov))],
        ["Version", esc(prov.get("version"))],
        ["Archive checksum", esc(prov.get("checksum"))], ["Processed checksum", esc(prov.get("processed_checksum"))],
        ["Source", esc(prov.get("source"))]]) + table(
        ["Run", "Host", "Platform", "Python", "Packages", "Started (UTC)", "Real time"], env_rows))

    sections.append("<h2>Scope of these results</h2><ul class=\"muted\">"
                    "<li>A simulated federation on one machine; silo names are scenario labels, not real agencies "
                    "or deployments.</li><li>WESAD is terrestrial laboratory data used as a stated proxy; it is "
                    "not spaceflight data.</li><li>These numbers demonstrate the mechanism under simulated "
                    "conditions; they do not validate an operational system.</li></ul>")

    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = ('<div class="banner ok">✓ All integrity checks passed</div>' if all_ok else
              '<div class="banner fail">✗ Integrity checks failed: see “Run integrity” before reporting these numbers</div>')
    n_seeds = len({r["seed"] for r in rs})
    sub = (f"{esc(exp)} · config {esc(chash)} · {n_seeds} seeds"
           + (f" × {len({draw(r) for r in rs})} holdout draws = {len(rs)} runs" if len(rs) != n_seeds else "")
           + (f" · {esc(label)}" if label else ""))
    return PAGE.format(title=esc(title), sub=sub, status=status, tiles=tiles, body="".join(sections),
                       generated=generated, argv=esc(" ".join(sys.argv)))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light; --page:#f9f9f7; --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a; --series-4:#8a5fd1; --good:#0ca30c; --warn:#fab219; --crit:#d03b3b; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --page:#0d0d0d;
  --surface-1:#1a1a19; --text-primary:#f0efec; --text-secondary:#c3c2b7; --grid:#2c2c2a; --axis:#383835;
  --border:rgba(255,255,255,0.10); --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --series-4:#9a72e0; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --page:#0d0d0d; --surface-1:#1a1a19; --text-primary:#f0efec;
  --text-secondary:#c3c2b7; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--page); color:var(--text-primary);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:36px 0 8px; }}
.muted {{ color:var(--text-secondary); }} .sub {{ color:var(--text-secondary); margin:0 0 16px; }}
.banner {{ padding:10px 14px; border-radius:8px; border:1px solid var(--border); background:var(--surface-1);
  margin:12px 0 20px; font-weight:600; }}
.banner.ok {{ border-left:4px solid var(--good); }} .banner.fail {{ border-left:4px solid var(--crit); }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }}
.tile {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }}
.tile-label {{ color:var(--text-secondary); font-size:13px; display:flex; align-items:center; gap:8px; }}
.tile-label i, .legend i, .sw {{ display:inline-block; width:10px; height:10px; border-radius:50%; flex:none; }}
.sw {{ margin-right:8px; vertical-align:-1px; }}
.tile-value {{ font-size:32px; font-weight:600; margin-top:4px; }} .tile-sub {{ color:var(--muted); font-size:13px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 18px; color:var(--text-secondary); font-size:13px; margin:8px 0; }}
.legend span {{ display:inline-flex; align-items:center; gap:6px; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
figure {{ margin:0; background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:10px; }}
figcaption {{ font-size:13px; color:var(--text-secondary); margin:0 0 4px 4px; }}
svg.chart {{ width:100%; height:auto; display:block; background:var(--surface-1); border-radius:10px; }}
.grid {{ stroke:var(--grid); stroke-width:1; }} .axis {{ stroke:var(--axis); stroke-width:1; }}
.tick {{ fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }}
.axis-title {{ fill:var(--text-secondary); font-size:12px; }} .label {{ fill:var(--text-secondary); font-size:12px; }}
.series {{ fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
.ref {{ stroke-width:2; opacity:.9; }} .key {{ stroke-width:2; stroke-linecap:round; }}
.band {{ opacity:.12; }} .whisker {{ stroke-width:2; stroke-linecap:round; }}
.dot {{ stroke:var(--surface-1); stroke-width:2; }}
.hit {{ fill:transparent; cursor:crosshair; }} circle.hit {{ cursor:pointer; }}
.xh {{ stroke:var(--axis); stroke-width:1; visibility:hidden; pointer-events:none; }}
.table-wrap {{ overflow-x:auto; margin:8px 0; }}
table {{ border-collapse:collapse; width:100%; background:var(--surface-1); border:1px solid var(--border);
  border-radius:10px; font-size:13px; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--grid); vertical-align:top; }}
th {{ color:var(--text-secondary); font-weight:600; }}
table.num td {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
.st {{ font-weight:600; white-space:nowrap; }} .st-ok {{ color:var(--good); }} .st-fail {{ color:var(--crit); }}
.st-warn {{ color:var(--text-primary); }}
#tip {{ position:fixed; pointer-events:none; background:var(--surface-1); color:var(--text-primary);
  border:1px solid var(--border); border-radius:8px; padding:8px 10px; font-size:12px; line-height:1.45;
  box-shadow:0 4px 16px rgba(0,0,0,.12); display:none; z-index:10; max-width:280px; }}
footer {{ margin-top:40px; color:var(--muted); font-size:12px; word-break:break-all; }}
</style></head>
<body><main>
<h1>{title}</h1><p class="sub">{sub}</p>
{status}
<div class="tiles">{tiles}</div>
{body}
<footer>Generated {generated} from run artifacts only (result.json, manifest.json); nothing was retrained.<br>
Command: {argv}</footer>
</main><div id="tip"></div>
<script>
(() => {{
  const tip = document.getElementById("tip");
  document.querySelectorAll("[data-tip]").forEach(el => {{
    const svg = el.closest("svg"), xh = svg && svg.querySelector(".xh");
    el.addEventListener("mousemove", e => {{
      tip.innerHTML = el.getAttribute("data-tip"); tip.style.display = "block";
      const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
      tip.style.left = x + "px"; tip.style.top = (e.clientY + 14) + "px";
      if (xh && el.dataset.cx) {{ xh.setAttribute("x1", el.dataset.cx); xh.setAttribute("x2", el.dataset.cx);
        xh.style.visibility = "visible"; }}
    }});
    el.addEventListener("mouseleave", () => {{ tip.style.display = "none"; if (xh) xh.style.visibility = "hidden"; }});
  }});
}})();
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories or result.json files (one experiment, several seeds)")
    ap.add_argument("--out", default="report.html", help="output HTML path")
    ap.add_argument("--title", help="page title (default: the experiment name)")
    ap.add_argument("--label", help="free-text context shown under the title, e.g. cluster and job id")
    args = ap.parse_args()
    runs = load_runs(args.runs)
    if not runs:
        print("no result.json found in the given paths", file=sys.stderr)
        return 1
    title = args.title or f"Run report: {experiment_name(runs[0]['result']['run_id'])}"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(runs, title, args.label), encoding="utf-8")
    print(f"wrote {out} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
