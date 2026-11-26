#!/usr/bin/env python3
"""Slide-ready figures for the ASGSR 2026 presentation (Phase H11, CLAUDE.md §5).

Reads run artifacts under ``runs/`` only (``*-summary.json``, ``result.json``, and the
``hypotheses_*.json`` / ``gap_*.json`` files written by the ``fedastro`` CLI from them) and
never trains or recomputes a model. Each figure is written three ways:

- ``figN_<name>.png`` (16:9, 1920 x 1080 px) and ``figN_<name>.pdf``;
- ``figN_<name>.json``, a sidecar with every plotted number and the files it came from, so
  every number in a figure traces to a result file (CLAUDE.md H11 acceptance).

The style is greyscale-safe: grey fills of distinct lightness plus a hatch per series.

``--tables`` also writes ``tables_<dataset>.txt``: the all-nodes, full-architecture and
stream tables per dataset (the same definitions as the WESAD ``h9_summary.txt``).

Usage (from the repo root):
    uv run --frozen python scripts/make_figures.py --tables
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAMILIES = ("logreg", "mlp", "cnn1d")
FAMILY_NAMES = {"logreg": "Logistic regression", "mlp": "MLP", "cnn1d": "1D-CNN"}
DATASET_NAMES = {"cinc15": "CinC 2021 (20k subset, 15 nodes)", "wesad12": "WESAD (12 nodes)"}
REGIONS = ("leo", "moon", "mars")
REGION_NAMES = {"leo": "LEO", "moon": "Moon", "mars": "Mars"}
# Shift settings of the H9 sweeps: (kind, magnitude tag, label).
SHIFTS = (("both", "0p0", "both 0.0"), ("both", "0p5", "both 0.5"), ("both", "1p0", "both 1.0"),
          ("both", "2p0", "both 2.0"), ("covariate", "1p0", "covariate 1.0"),
          ("concept", "1p0", "concept 1.0"))
# One look per series, used across figures: (face grey, hatch).
STYLE = {
    "local": ("#f7f7f7", "//"), "local_only": ("#f7f7f7", "//"),
    "regional": ("#969696", ""), "hier_regional": ("#969696", ""),
    "global": ("#d9d9d9", ".."), "hier_global": ("#d9d9d9", ".."),
    "centralized": ("#252525", ""), "federated": ("#636363", "xx"),
    "agree": ("#bdbdbd", ""), "disagree": ("#f7f7f7", "\\\\"),
    "off": ("#d9d9d9", ""), "on": ("#525252", "//"),
}
LABELS = {"local": "Local-only model", "regional": "Regional model", "global": "Global model",
          "centralized": "Centralized", "federated": "Flat federated",
          "hier_global": "Hier. global", "hier_regional": "Hier. regional",
          "local_only": "Local-only"}
SIM_NOTE = "Location shift is SIMULATED (no real location-dependent physiology in these datasets)."
CINC_ARTIFACT = ("CinC: the concept part of the shift makes the binary task near-separable in Moon and "
                 "Mars, so H2/H3 margins there are mostly an artifact of the shift.")
W, H, DPI = 9.6, 5.4, 200  # 1920 x 1080 px

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
    "hatch.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": DPI,
})


class FigureInputError(RuntimeError):
    """A figure's input file is missing or ambiguous."""


# -- inputs --------------------------------------------------------------------------
def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FigureInputError(f"missing input: {path}")
    return json.loads(path.read_text())


def one(root: Path, pattern: str) -> Path:
    hits = sorted(root.glob(pattern))
    if len(hits) != 1:
        raise FigureInputError(f"expected exactly 1 file for {root}/{pattern}, found {len(hits)}")
    return hits[0]


def many(root: Path, pattern: str) -> list[Path]:
    hits = sorted(root.glob(pattern))
    if not hits:
        raise FigureInputError(f"no files for {root}/{pattern}")
    return hits


def sweep_summary(root: Path, dataset: str, family: str, arm: str, kind: str, mag: str) -> Path:
    return one(root, f"h9_{dataset}_{family}_{arm}-enabledTrue-kind{kind}-magnitude{mag}-*-summary.json")


def ms(d: dict) -> tuple[float, float]:
    """(mean, std) from a summary's {"mean", "std", "n"} cell."""
    return d["mean"], d["std"]


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)


# -- output --------------------------------------------------------------------------
def new_fig(nrows: int = 1, ncols: int = 1, **kw):
    return plt.subplots(nrows, ncols, figsize=(W, H), **kw)


def save(fig, out: Path, name: str, data: dict, sources: list[Path], caption: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=8, wrap=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    side = {"figure": name, "caption": caption, "data": data,
            "sources": sorted({rel(s) for s in sources})}
    (out / f"{name}.json").write_text(json.dumps(side, indent=2, sort_keys=True) + "\n")
    return side


def bar(ax, x, h, w, key, err=None, label=None):
    face, hatch = STYLE[key]
    return ax.bar(x, h, w, yerr=err, color=face, hatch=hatch, edgecolor="black", linewidth=0.7,
                  capsize=2.5, error_kw={"elinewidth": 0.8}, label=label)


# -- figure 1: accuracy per region ------------------------------------------------------
def fig_regions(root: Path, dataset: str, mag: str, out: Path) -> dict:
    """Local / regional / global per region, plus the all-nodes arms, per family."""
    fig, axes = new_fig(1, 3, sharey=True)
    data, sources = {}, []
    for ax, fam in zip(axes, FAMILIES):
        hier = sweep_summary(root, dataset, fam, "hierarchical", "both", mag)
        paths = {a: sweep_summary(root, dataset, fam, a, "both", mag)
                 for a in ("centralized", "federated", "local_only")}
        sources += [hier, *paths.values()]
        hs, ss = read_json(hier), {a: read_json(p) for a, p in paths.items()}
        d = {}
        for r in REGIONS:
            pr = hs["hierarchy"]["per_region"][r]
            d[r] = {m: ms(pr[m]["accuracy"]) for m in ("local", "regional", "global")}
        d["all"] = {
            "centralized": ms(ss["centralized"]["arms"]["centralized"]["accuracy"]),
            "federated": ms(ss["federated"]["arms"]["federated"]["final"]["accuracy"]),
            "hier_global": ms(hs["arms"]["hierarchical"]["final"]["accuracy"]),
            "hier_regional": ms(hs["arms"]["hierarchical"]["regional_final"]["accuracy"]),
            "local_only": ms(ss["local_only"]["arms"]["local_only"]["mean_across_silos"]["accuracy"]),
        }
        data[fam] = d
        w, x0 = 0.26, 0
        for r in REGIONS:
            for i, m in enumerate(("local", "regional", "global")):
                mean, std = d[r][m]
                bar(ax, x0 + (i - 1) * w, mean, w, m, std, LABELS[m] if (r == "leo" and fam == "logreg") else None)
            x0 += 1
        x0 += 0.15
        w2 = 0.17
        for i, m in enumerate(("centralized", "federated", "hier_global", "hier_regional", "local_only")):
            mean, std = d["all"][m]
            lab = LABELS[m] if fam == "logreg" and m in ("centralized", "federated") else None
            bar(ax, x0 + (i - 2) * w2, mean, w2, m, std, lab)
        ax.set_xticks([0, 1, 2, x0], [REGION_NAMES[r] for r in REGIONS] + ["All nodes"])
        ax.set_title(FAMILY_NAMES[fam])
        ax.set_ylim(0, 1.05)
        ax.axvline(2.58, color="black", linewidth=0.5, linestyle=":")
    axes[0].set_ylabel("Test accuracy (mean ± std, 5 seeds)")
    shift = "no shift (magnitude 0, control)" if mag == "0p0" else f"shift both {mag.replace('p', '.')} on Moon and Mars"
    fig.suptitle(f"{DATASET_NAMES[dataset]}: accuracy per region, {shift}", fontsize=13)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    caption = ("All nodes: centralized, flat federated, hierarchical global and regional, local-only "
               "(silo mean). Bars per region are the hierarchical arm's three models. ")
    if mag != "0p0":
        caption += SIM_NOTE + (" " + CINC_ARTIFACT if dataset == "cinc15" else "")
    name = f"fig1_regions_{dataset}_shift{'0' if mag == '0p0' else mag.replace('p0', '')}"
    return save(fig, out, name, data, sources, caption)


# -- figure 2: H3 disagreement ----------------------------------------------------------
def fig_h3(hyps: dict[str, Path], out: Path) -> dict:
    fig, axes = new_fig(1, len(hyps), sharey=True, squeeze=False)
    data = {}
    for ax, (ds, path) in zip(axes[0], hyps.items()):
        h = read_json(path)
        d = {}
        for i, fam in enumerate(FAMILIES):
            row = h["families"][fam]["H3"]["rows"][0]
            d[fam] = {"acc_agree": row["a"], "acc_disagree": row["b"], "diff_mean": row["diff_mean"],
                      "ci95": row["ci95"], "verdict": row["verdict"]}
            bar(ax, i - 0.18, row["a"], 0.36, "agree", label="Models agree" if i == 0 else None)
            bar(ax, i + 0.18, row["b"], 0.36, "disagree", label="Models disagree" if i == 0 else None)
            lo, hi = row["ci95"]
            ax.text(i, max(row["a"], row["b"]) + 0.03, f"+{row['diff_mean']:.3f}\n[{lo:+.3f}, {hi:+.3f}]",
                    ha="center", fontsize=8.5)
        data[ds] = d
        ax.set_xticks(range(3), [FAMILY_NAMES[f] for f in FAMILIES])
        ax.set_title(f"{DATASET_NAMES[ds]}\n{h['families']['logreg']['H3']['setting']}")
        ax.set_ylim(0, 1.15)
    axes[0][0].set_ylabel("Test accuracy (mean over 5 seeds)")
    axes[0][0].legend(loc="lower left", frameon=True)
    fig.suptitle("H3: accuracy where local, regional and global models agree vs disagree", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    caption = ("Label: difference (agree − disagree) and paired-bootstrap 95% CI. Agreement is a flag for "
               "clinician review, not independent confirmation: the models share training data. "
               + SIM_NOTE + (" " + CINC_ARTIFACT if "cinc15" in hyps else ""))
    return save(fig, out, "fig2_h3_disagreement", data, list(hyps.values()), caption)


# -- figure 3: H4 model comparison (forest) ----------------------------------------------
def _acc_row(rows: list[dict], prefix: str) -> dict:
    hits = [r for r in rows if r["comparison"].startswith(prefix) and r["comparison"].endswith("(accuracy)")]
    if len(hits) != 1:
        raise FigureInputError(f"expected one accuracy row starting {prefix!r}, found {len(hits)}")
    return hits[0]


def fig_h4(hyps: dict[str, Path], out: Path) -> dict:
    comps = (("H1", "federated − local-only (no shift)"), ("H2a", "H2(a) regional − local-only"),
             ("H2b", "H2(b) regional − global"), ("ctrl", "H2 control: regional − global, no shift"))
    markers = {"logreg": "o", "mlp": "s", "cnn1d": "^"}
    fig, axes = new_fig(1, len(hyps), sharey=True, squeeze=False)
    data = {}
    for ax, (ds, path) in zip(axes[0], hyps.items()):
        h = read_json(path)
        d = {}
        for fi, fam in enumerate(FAMILIES):
            f = h["families"][fam]
            rows = {"H1": _acc_row(f["H1"]["rows"], "federated"), "H2a": _acc_row(f["H2"]["rows"], "(a)"),
                    "H2b": _acc_row(f["H2"]["rows"], "(b)"), "ctrl": f["H2"]["control"]}
            d[fam] = {k: {"diff_mean": r["diff_mean"], "ci95": r["ci95"], "verdict": r["verdict"]}
                      for k, r in rows.items()}
            for ci, (k, _) in enumerate(comps):
                y = ci + (fi - 1) * 0.22
                r = rows[k]
                lo, hi = r["ci95"]
                ax.plot([lo, hi], [y, y], color="black", linewidth=1.2)
                ax.plot(r["diff_mean"], y, markers[fam], color="black",
                        markerfacecolor=["white", "#969696", "black"][fi], markersize=6.5,
                        label=FAMILY_NAMES[fam] if ci == 0 else None)
        data[ds] = d
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(range(len(comps)), [c[1] for c in comps])
        ax.invert_yaxis()
        ax.set_xlabel("Accuracy difference, 95% CI (5 seeds)")
        ax.set_title(DATASET_NAMES[ds])
    axes[0][0].legend(loc="lower right", frameon=True)
    fig.suptitle("H4: does the federated advantage hold across model families?", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    caption = ("PREREG §2 rule: paired by seed, hierarchical bootstrap 95% CI; above 0 = supported "
               "(control: supported unless the CI is above 0). H2 rows at shift both 1.0 on Moon and Mars. "
               + SIM_NOTE)
    return save(fig, out, "fig3_h4_model_comparison", data, list(hyps.values()), caption)


# -- figure 4: gap ---------------------------------------------------------------------
GAP_COMPS = (("centralized - federated", "federated"), ("centralized - hierarchical", "hier_global"),
             ("centralized - hierarchical_regional", "hier_regional"))


def fig_gap(gaps: dict[str, Path], out: Path) -> dict:
    fig, axes = new_fig(1, len(gaps), sharey=True, squeeze=False)
    data = {}
    for ax, (ds, path) in zip(axes[0], gaps.items()):
        rows = read_json(path)["rows"]
        d = {}
        for i, fam in enumerate(FAMILIES):
            d[fam] = {}
            for j, (comp, key) in enumerate(GAP_COMPS):
                hits = [r for r in rows if r["family"] == fam and r["comparison"] == comp and r["metric"] == "accuracy"]
                if len(hits) != 1:
                    raise FigureInputError(f"{path}: expected one {fam} {comp} accuracy row, found {len(hits)}")
                r = hits[0]
                lo, hi = r["ci95"]
                d[fam][comp] = {"gap_mean": r["gap_mean"], "ci95": r["ci95"]}
                bar(ax, i + (j - 1) * 0.26, r["gap_mean"], 0.26, key,
                    err=[[r["gap_mean"] - lo], [hi - r["gap_mean"]]],
                    label=f"centralized − {LABELS[key].lower()}" if i == 0 else None)
        data[ds] = d
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(3), [FAMILY_NAMES[f] for f in FAMILIES])
        ax.set_title(DATASET_NAMES[ds])
    axes[0][0].set_ylabel("Accuracy gap (above 0: centralized better)")
    axes[0][0].legend(loc="best", frameon=True)
    fig.suptitle("Remaining centralized vs federated gap after validation-only tuning (Phase H8)", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    caption = ("Test accuracy, 5 seeds, 95% CI; no shift. Settings selected on validation scores only "
               "(PREREG). A negative gap means federated or hierarchical beat centralized.")
    return save(fig, out, "fig4_gap", data, list(gaps.values()), caption)


# -- figure 5: secure-aggregation overhead ------------------------------------------------
def total_bytes(r: dict) -> int:
    return sum(v.get("bytes", 0) for rr in r["hierarchy"]["rounds"] for v in rr["regions"].values()) + \
        sum(rr.get("global_bytes") or 0 for rr in r["hierarchy"]["rounds"])


def fig_secagg(secagg_root: Path, cinc_root: Path | None, out: Path) -> dict:
    presets = ("near", "mean", "far")
    data, sources = {}, []
    for p in presets:
        data[p] = {}
        for state, flag in (("off", "False"), ("on", "True")):
            path = one(secagg_root, f"h4secagg-mars{p}-enabled{flag}-seed0-*/result.json")
            sources.append(path)
            r = read_json(path)
            data[p][state] = {"sim_end_h": r["hierarchy"]["rounds"][-1]["sim_wallclock_s"] / 3600,
                              "bytes_mb": total_bytes(r) / 1e6,
                              "dataset": r["provenance"][0]["dataset"]}
    real = None
    if cinc_root is not None:
        real = {}
        for fam in FAMILIES:
            paths = many(cinc_root, f"h9_cinc15_{fam}_full-*-seed*-*/result.json")
            sources += paths
            rs = [read_json(p) for p in paths]
            real[fam] = {"overhead_sim_h_mean": statistics.mean(r["secagg"]["overhead_sim_s"] for r in rs) / 3600,
                         "overhead_mb_mean": statistics.mean(r["secagg"]["overhead_bytes"] for r in rs) / 1e6,
                         "n_runs": len(rs)}
        data["cinc15_full_mars_mean"] = real
    fig, (a1, a2) = new_fig(1, 2)
    for ax, key, ylab in ((a1, "sim_end_h", "Simulated run time (h)"), (a2, "bytes_mb", "Total bytes sent (MB)")):
        for i, p in enumerate(presets):
            for j, state in enumerate(("off", "on")):
                v = data[p][state][key]
                bar(ax, i + (j - 0.5) * 0.36, v, 0.36, state,
                    label=f"Secure aggregation {state}" if i == 0 else None)
                ax.text(i + (j - 0.5) * 0.36, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(range(3), ["Mars near\n(182 s one-way)", "Mars mean\n(751 s)", "Mars far\n(1337 s)"])
        ax.set_ylabel(ylab)
    a1.legend(loc="upper left", frameon=False)
    a1.set_title("Time: extra key and share round trips")
    a2.set_title("Bytes: masked words and protocol messages")
    fig.suptitle("Simulated secure aggregation: overhead vs Mars light-time preset (SYNTHETIC data)", fontsize=13)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    caption = ("SYNTHETIC smoke data, seed 0, 6 rounds, 3 global merges, no dropouts, no LOS (Phase H4). "
               "All times SIMULATED, not wall-clock. Secure aggregation is a protocol simulation "
               "(seeded PRG masks), not cryptography.")
    if real:
        caption += (" Real data, CinC full architecture at Mars mean (30 rounds, LOS on): overhead "
                    + ", ".join(f"{FAMILY_NAMES[f]} {real[f]['overhead_sim_h_mean']:.1f} h / "
                                f"{real[f]['overhead_mb_mean']:.2f} MB" for f in FAMILIES) + " per run.")
    return save(fig, out, "fig5_secagg_overhead", data, sources, caption)


# -- figure 6: threshold-fallback timeline -------------------------------------------------
def fig_fallback(result_path: Path, out: Path) -> dict:
    r = read_json(result_path)
    rounds = r["hierarchy"]["rounds"]
    n = len(rounds)
    grid = {reg: [] for reg in REGIONS}
    for rr in rounds:
        for reg in REGIONS:
            grid[reg].append("fallback" if rr["regions"][reg]["fallback"] else "aggregated")
    merges = [rr["round"] for rr in rounds if rr.get("global_aggregated")]
    sim_h = [rr["sim_wallclock_s"] / 3600 for rr in rounds]
    fig, ax = new_fig()
    for yi, reg in enumerate(REGIONS):
        for xi, state in enumerate(grid[reg]):
            fb = state == "fallback"
            ax.add_patch(plt.Rectangle((xi + 0.5, yi - 0.4), 1, 0.8, facecolor="#252525" if fb else "#f0f0f0",
                                       hatch="" if fb else "", edgecolor="white", linewidth=1))
    for m in merges:
        ax.plot(m, len(REGIONS) - 0.35, "v", color="black", markersize=7)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(-0.6, len(REGIONS) - 0.1)
    ax.set_yticks(range(len(REGIONS)), [REGION_NAMES[x] for x in REGIONS])
    ticks = [t for t in range(1, n + 1) if t == 1 or t % 5 == 0]
    ax.set_xticks(ticks, [f"r{t}\n{sim_h[t - 1]:.1f} h" for t in ticks])
    ax.set_xlabel("Regional round  /  simulated time at the end of the round (SIMULATED, not wall-clock)")
    ax.spines["left"].set_visible(False)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(facecolor="#f0f0f0", edgecolor="black", label="Regional aggregation ran"),
                       Patch(facecolor="#252525", label="Fallback to global (below min_participants)"),
                       Line2D([], [], marker="v", color="black", linestyle="", label="Global merge")],
              loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    cfg = r["config"]
    fam = cfg.get("model", {}).get("arch", "?")
    fig.suptitle(f"Threshold fallback during loss of signal: {r['topology']['name']}, {fam}, seed {r['seed']}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    data = {"rounds": list(range(1, n + 1)), "sim_end_h": sim_h, "state": grid, "global_merges": merges,
            "n_fallbacks": {reg: grid[reg].count("fallback") for reg in REGIONS}, "run_id": r["run_id"]}
    caption = ("Full architecture: Mars mean light-time, loss-of-signal windows, 120 s regional deadline, "
               "secure aggregation and update gate on. A region below min_participants (5) uses the global "
               "model for that round; the fallback is logged in result.json.")
    return save(fig, out, "fig6_fallback_timeline", data, [result_path], caption)


# -- figure 7: DP collapse (Phase 5) -----------------------------------------------------
def eps_of(experiment: str) -> float:
    m = re.search(r"-eps(\d+)p(\d+)", experiment)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    if experiment.endswith("-base"):
        return float("inf")
    raise FigureInputError(f"cannot read epsilon from experiment {experiment!r}")


def fig_dp(dp_root: Path, out: Path) -> dict:
    paths = many(dp_root, "p5epsc-*-summary.json")
    rows = []
    for p in paths:
        s = read_json(p)
        a = s["arms"]
        cells = s["privacy"]["attack"]["cells"]
        att = {}
        for b in (1, 32):
            sel = [c for k, c in cells.items() if k.endswith(f"/final-B{b}-known")]
            if not sel:
                raise FigureInputError(f"{p}: no final-B{b}-known attack cells")
            att[f"B{b}"] = {"nn_identified": statistics.mean(c["nn_identified"]["mean"] for c in sel),
                            "chance": statistics.mean(c["chance"]["nn_identified"]["mean"] for c in sel),
                            "n_silos": len(sel)}
        rows.append({"epsilon": eps_of(s["experiment"]),
                     "federated": ms(a["federated"]["final"]["accuracy"]),
                     "federated_macro_f1": ms(a["federated"]["final"]["macro_f1"]),
                     "local_only": ms(a["local_only"]["mean_across_silos"]["accuracy"]),
                     "centralized": ms(a["centralized"]["accuracy"]), "attack": att})
    rows.sort(key=lambda r: r["epsilon"])
    res = one(dp_root, "p5epsc-base-seed0-*/result.json")
    counts = read_json(res)["holdout"]["label_counts"]
    majority = max(counts.values()) / sum(counts.values())
    xs = range(len(rows))
    xl = ["∞\n(no DP)" if r["epsilon"] == float("inf") else f"{r['epsilon']:g}" for r in rows]
    fig, (a1, a2) = new_fig(1, 2)
    for key, mk, face in (("centralized", "s", "black"), ("federated", "o", "white"), ("local_only", "^", "#969696")):
        mean = [r[key][0] for r in rows]
        std = [r[key][1] for r in rows]
        a1.errorbar(xs, mean, yerr=std, marker=mk, color="black", markerfacecolor=face, capsize=2.5,
                    linewidth=1, label=LABELS[key] if key != "federated" else "Federated")
    a1.axhline(majority, color="black", linestyle=":", linewidth=1)
    a1.text(0, majority + 0.01, f"majority class ({majority:.3f})", fontsize=8.5)
    a1.set_xticks(xs, xl)
    a1.set_xlabel("Record-level ε per silo")
    a1.set_ylabel("Holdout accuracy (mean ± std, 5 seeds)")
    a1.set_title("Utility: federated collapses toward the majority class")
    a1.legend(loc="lower right", frameon=True)
    for b, mk, face in (("B1", "o", "black"), ("B32", "s", "white")):
        a2.plot(xs, [r["attack"][b]["nn_identified"] for r in rows], marker=mk, color="black",
                markerfacecolor=face, linewidth=1, label=f"Attack, batch {b[1:]}")
        a2.plot(xs, [r["attack"][b]["chance"] for r in rows], color="black", linestyle="--" if b == "B1" else ":",
                linewidth=1, label=f"Chance, batch {b[1:]}")
    a2.set_xticks(xs, xl)
    a2.set_xlabel("Record-level ε per silo")
    a2.set_ylabel("Window re-identified (fraction of trials)")
    a2.set_title("Gradient-inversion attack on one released update")
    a2.legend(loc="upper left", frameon=True)
    fig.suptitle("Phase 5 discussion: DP-SGD drives the attack to chance but collapses federated utility",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    caption = ("Flat five-silo WESAD, PACE runs (p5epsc), 5 seeds. Attack: final model, known labels, mean over "
               "5 silos. DP-SGD is set aside in this branch; this is a discussion point, not a headline. "
               "Without DP there is no formal privacy guarantee.")
    data = {"rows": [{**r, "epsilon": "inf" if r["epsilon"] == float("inf") else r["epsilon"]} for r in rows],
            "majority_class_accuracy": majority, "holdout_label_counts": counts}
    return save(fig, out, "fig7_dp_collapse", data, [*paths, res], caption)


# -- tables ------------------------------------------------------------------------------
def fmt(m: tuple[float, float]) -> str:
    return f"{m[0]:.3f} ± {m[1]:.3f}"


def tables(root: Path, dataset: str) -> str:
    lines = [f"H9-style tables, {dataset}, from {rel(root)}. Test set, 5 seeds, mean ± std. Shift is SIMULATED;",
             "shifted regions moon, mars; 'both' = covariate + concept.", ""]
    for metric in ("accuracy", "macro_f1"):
        lines.append(f"== Table A: test {metric} over all nodes")
        for fam in FAMILIES:
            lines.append(f"-- {fam}")
            lines.append(f"{'shift':<16}" + "".join(f"{c:<17}" for c in
                         ("centralized", "flat FL", "hier global", "hier regional", "local-only")))
            for kind, mag, lab in SHIFTS:
                s = {a: read_json(sweep_summary(root, dataset, fam, a, kind, mag))["arms"]
                     for a in ("centralized", "federated", "hierarchical", "local_only")}
                vals = (s["centralized"]["centralized"][metric], s["federated"]["federated"]["final"][metric],
                        s["hierarchical"]["hierarchical"]["final"][metric],
                        s["hierarchical"]["hierarchical"]["regional_final"][metric],
                        s["local_only"]["local_only"]["mean_across_silos"][metric])
                lines.append(f"{lab:<16}" + "".join(f"{fmt(ms(v)):<17}" for v in vals))
        lines.append("")
    lines.append("== Full architecture (Mars mean, LOS, 120 s regional deadline, secagg + update gate; shift both 1.0)")
    lines.append("family  hier global      hier regional    local-only       gate rejected / offered (mean/run)  "
                 "secagg MB  secagg sim h  fallbacks/run  sim end h")
    for fam in FAMILIES:
        summ = read_json(one(root, f"h9_{dataset}_{fam}_full-*-summary.json"))
        rs = [read_json(p) for p in many(root, f"h9_{dataset}_{fam}_full-*-seed*-*/result.json")]
        hier = summ["arms"]["hierarchical"]
        rej = statistics.mean(r["updates"]["n_rejected"] for r in rs)
        off = statistics.mean(r["updates"]["n_offered"] for r in rs)
        fb = statistics.mean(sum(1 for rr in r["hierarchy"]["rounds"] for v in rr["regions"].values() if v["fallback"])
                             for r in rs)
        lines.append(f"{fam:<8}{fmt(ms(hier['final']['accuracy'])):<17}{fmt(ms(hier['regional_final']['accuracy'])):<17}"
                     f"{fmt(ms(summ['arms']['local_only']['mean_across_silos']['accuracy'])):<17}"
                     f"{rej:.1f} / {off:.1f} ({100 * rej / off:.1f}%){'':<14}"
                     f"{statistics.mean(r['secagg']['overhead_bytes'] for r in rs) / 1e6:<11.2f}"
                     f"{statistics.mean(r['secagg']['overhead_sim_s'] for r in rs) / 3600:<14.1f}"
                     f"{fb:<15.1f}{statistics.mean(r['hierarchy']['rounds'][-1]['sim_wallclock_s'] for r in rs) / 3600:.1f}"
                     f"   (n={len(rs)})")
    lines.append("All times SIMULATED. Gate rejections are real updates unless n_injected > 0.")
    lines.append("")
    lines.append("== Test-then-train stream: accuracy on examples scored before their label was used (local / regional / global)")
    for fam in FAMILIES:
        cells = []
        for frac, delay in (("0p1", 0), ("0p1", 1), ("1p0", 0), ("1p0", 1)):
            s = read_json(one(root, f"h9_{dataset}_{fam}_stream-label_fraction{frac}-label_delay_rounds{delay}-*-summary.json"))
            mo = s["stream"]["models"]
            cells.append(f"{frac.replace('p', '.')} labels, delay {delay}: "
                         + " / ".join(f"{mo[m]['accuracy']['mean']:.3f}" for m in ("local", "regional", "global")))
        lines.append(f"{fam:<8}" + "  |  ".join(cells))
    return "\n".join(lines) + "\n"


# -- main --------------------------------------------------------------------------------
def build(args) -> list[str]:
    out = Path(args.out)
    done = []
    hyps = {"cinc15": Path(args.cinc) / "hypotheses_cinc15.json", "wesad12": Path(args.wesad) / "hypotheses_wesad12.json"}
    gaps = {"cinc15": Path(args.cinc_gap), "wesad12": Path(args.wesad_gap)}
    jobs = [
        ("fig1", lambda: [fig_regions(Path(root), ds, mag, out) for ds, root in (("cinc15", args.cinc), ("wesad12", args.wesad))
                          for mag in ("0p0", "1p0")]),
        ("fig2", lambda: fig_h3(hyps, out)),
        ("fig3", lambda: fig_h4(hyps, out)),
        ("fig4", lambda: fig_gap(gaps, out)),
        ("fig5", lambda: fig_secagg(Path(args.secagg), Path(args.cinc), out)),
        ("fig6", lambda: fig_fallback(one(Path(args.cinc), f"h9_cinc15_{args.fallback_family}_full-*-seed{args.fallback_seed}-*/result.json"), out)),
        ("fig7", lambda: fig_dp(Path(args.dp), out)),
    ]
    for key, fn in jobs:
        if args.only and key not in args.only:
            continue
        fn()
        done.append(key)
    if args.tables:
        for ds, root in (("cinc15", args.cinc), ("wesad12", args.wesad)):
            (out / f"tables_{ds}.txt").write_text(tables(Path(root), ds))
            done.append(f"tables_{ds}")
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cinc", default="runs/h10-af34071", help="CinC H9-style runs (H10c step 3)")
    ap.add_argument("--wesad", default="runs/h9-84a40ec", help="WESAD H9 runs")
    ap.add_argument("--cinc-gap", default="runs/h10-f7db96a/gap_cinc15.json")
    ap.add_argument("--wesad-gap", default="runs/h8-62cf33b/gap_wesad12.json")
    ap.add_argument("--secagg", default="runs/h4secagg", help="Phase H4 Mars-preset sweep (SYNTHETIC)")
    ap.add_argument("--dp", default="runs/p5epsc-pace", help="Phase 5 DP sweep from PACE")
    ap.add_argument("--fallback-family", default="logreg", choices=FAMILIES)
    ap.add_argument("--fallback-seed", default=0, type=int)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", nargs="*", help="e.g. --only fig1 fig4")
    ap.add_argument("--tables", action="store_true", help="also write tables_<dataset>.txt")
    args = ap.parse_args(argv)
    try:
        done = build(args)
    except FigureInputError as e:
        print(f"make_figures: {e}", file=sys.stderr)
        return 2
    print(f"wrote {', '.join(done)} to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
