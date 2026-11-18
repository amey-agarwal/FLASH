"""Terminal peek (CLAUDE.md Phase H5): ``fedastro peek runs/<run_id> [--follow]``.

Read-only and on request only: it opens files for reading and never writes, so it cannot
change a run's results (tested). Per hierarchical round it shows:
- simulated time (SIMULATED seconds, never real time; "comms off" when links are not modelled);
- each node's state: training / fallback / in transit / loss of signal;
- the regional model's accuracy per region, and when a merge happened, the global model's;
- the local, regional and global models' accuracy and their disagreement rate (every node's
  test set pooled; a diagnostic, never used to select anything);
- update-gate events: offered, accepted, and every rejection with its scores.

Sources: a finished run is read from ``result.json`` only. ``--follow`` tails
``progress.jsonl`` (one line per completed round, written by ``fedastro run``) until the run
ends; this is the one place the view reads something other than result.json (Phase H5
decision 6).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, TextIO

PROGRESS_FILE = "progress.jsonl"  # written by pipeline.ProgressLog
STATES = ("training", "fallback", "in_transit", "loss_of_signal")
_STATE_LABEL = {"training": "training", "fallback": "fallback", "in_transit": "in transit",
                "loss_of_signal": "loss of signal"}


def _f(x: Any, nd: int = 3) -> str:
    return "  -  " if x is None else f"{x:.{nd}f}"


def _run_dir(path: str | Path) -> Path:
    p = Path(path)
    return p.parent if p.name in ("result.json", PROGRESS_FILE) else p


def render_header(h: dict[str, Any]) -> str:
    regions = "; ".join(f"{r}: {', '.join(nodes)}" for r, nodes in h["regions"].items())
    lines = [f"run {h['run_id']}",
             f"  {h['rounds']} rounds, global merge every {h['global_every']}; data {h['data']}; "
             f"simulated links {'on' if h['comms'] else 'off'}",
             f"  regions (scenario labels) - {regions}"]
    if h.get("labels"):
        lines.append(f"  {h['labels']}")
    return "\n".join(lines)


def _states(states: dict[str, str]) -> str:
    by: dict[str, list[str]] = {}
    for node, st in states.items():
        by.setdefault(st, []).append(node)
    parts = []
    for st in STATES:
        nodes = by.get(st)
        if not nodes:
            continue
        parts.append(f"{_STATE_LABEL[st]} x{len(nodes)}" if st == "training"
                     else f"{_STATE_LABEL[st]}: {', '.join(nodes)}")
    return " | ".join(parts)


def render_round(rec: dict[str, Any], updates: list[dict[str, Any]]) -> str:
    t = rec.get("sim_wallclock_s")
    when = "comms off" if t is None else f"sim t {t:,.1f} s (SIMULATED)"
    merge = ""
    if rec.get("global_aggregated"):
        merge = (f" | global merge v{rec.get('global_version', '?')} of {', '.join(rec.get('global_participants', []))}"
                 f"; received by {', '.join(rec.get('global_received', [])) or 'none'}")
    elif "global_participants" in rec:
        merge = " | merge round: no global merge"
    lines = [f"-- round {rec['round']} -- {when}{merge}"]
    for name, reg in rec["regions"].items():
        flag = "FALLBACK" if reg["fallback"] else "        "
        acc = (reg.get("regional") or {}).get("accuracy")
        states = reg.get("node_state")
        detail = _states(states) if states else f"participants: {', '.join(reg['participants'])}"
        lines.append(f"  {name:<6} {flag} regional acc {_f(acc)} | {detail}")
    if rec.get("global"):
        lines.append(f"  global model acc {_f(rec['global'].get('accuracy'))} (pooled test sets)")
    d = rec.get("diagnostic")
    if d:
        a = d["accuracy"]
        lines.append(f"  acc local {_f(a['local'])}  regional {_f(a['regional'])}  global {_f(a['global'])}"
                     f"  | disagreement {_f(d['disagreement_rate'])}  (test sets, diagnostic only)")
    if updates:
        acc_n = sum(e["accepted"] for e in updates)
        lines.append(f"  updates: {len(updates)} offered, {acc_n} accepted, {len(updates) - acc_n} rejected")
        for e in updates:
            if not e["accepted"]:
                lines.append(f"    REJECTED {e['node']} {e['model']} v{e['version']}: val macro-F1 "
                             f"{_f(e['val_after'])} < {_f(e['val_before'])} - {_f(e['tolerance_used'])}; keeps "
                             f"v{e['version_before']}" + (" [injected fault, SCRIPTED]" if e.get("injected") else ""))
    return "\n".join(lines)


def _labels(result: dict[str, Any]) -> str:
    tags = []
    if any(p.get("kind") == "SYNTHETIC" for p in result.get("provenance", [])):
        tags.append("SYNTHETIC data")
    if result.get("shift", {}).get("enabled"):
        tags.append("SIMULATED location shift")
    if result.get("secagg", {}).get("enabled"):
        tags.append("SIMULATED secure aggregation")
    return ", ".join(tags)


def from_result(result: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[dict, list]], str]:
    """(header, [(round record, its update events)], footer) from a finished result."""
    cfg = result.get("config", {})
    regions = {r["name"]: [n["name"] for n in r["nodes"]] for r in result["topology"]["regions"]}
    header = {"run_id": result["run_id"], "rounds": cfg.get("fl", {}).get("rounds"),
              "global_every": cfg.get("hier_fl", {}).get("global_every"), "data": cfg.get("data", {}).get("name"),
              "comms": bool(result["hierarchy"].get("comms")), "regions": regions, "labels": _labels(result)}
    events: dict[int, list] = {}
    for e in result.get("updates", {}).get("events", []):
        events.setdefault(e["round"], []).append(e)
    rounds = [(rec, events.get(rec["round"], [])) for rec in result["hierarchy"]["rounds"]]
    arm = result["arms"].get("hierarchical", {})
    up = result.get("updates", {})
    foot = [f"final (each node's own test set pooled): global acc {_f(arm.get('final', {}).get('accuracy'))}, "
            f"regional acc {_f(arm.get('regional_final', {}).get('accuracy'))}; "
            f"{arm.get('n_merges', 0)} merges, {arm.get('n_fallbacks', 0)} fallbacks"]
    if up.get("enabled"):
        foot.append(f"updates: {up['n_offered']} offered, {up['n_rejected']} rejected; deployed versions "
                    + ", ".join(f"{n} r{v['regional']}/g{v['global']}" for n, v in up["deployed"].items()))
    st = result.get("stream", {})
    if st.get("enabled"):
        foot.append(render_stream(st))
    return header, rounds, "\n".join(foot)


def render_stream(st: dict[str, Any]) -> str:
    """Test-then-train view (Phase H7): per round, each model's accuracy on the examples that
    arrived, scored before any training on them (n = how many that model scored)."""
    tt = st["test_then_train"]
    lines = [f"test-then-train ({st['order']} order, {st['order_provenance']}; labels {st['label_fraction']:g}, "
             f"delay {st['label_delay_rounds']} round(s), warm-up {st['warmup_frac']:g}):"]
    for r in tt["rounds"]:
        accs = "  ".join(f"{m} {_f(r['accuracy'][m])} (n {r['n_scored'][m]})" for m in tt["models"])
        lines.append(f"  round {r['round']:>3}: arrived {r['n_arrived']:>4} | {accs}")
    g = tt["global"]
    lines.append("  whole stream: " + "  ".join(
        f"{m} acc {_f((g[m] or {}).get('accuracy'))}" for m in tt["models"]))
    return "\n".join(lines)


def peek(path: str | Path, out: TextIO = sys.stdout) -> int:
    """Print every round of a run (result.json if it exists, else progress.jsonl so far)."""
    run_dir = _run_dir(path)
    res = run_dir / "result.json"
    if res.is_file():
        result = json.loads(res.read_text())
        if not result.get("hierarchy", {}).get("rounds"):
            print(f"{res}: no hierarchical rounds in this run (flat run); nothing to show", file=out)
            return 2
        header, rounds, foot = from_result(result)
        print(render_header(header), file=out)
        for rec, ups in rounds:
            print(render_round(rec, ups), file=out)
        print(foot, file=out)
        return 0
    prog = run_dir / PROGRESS_FILE
    if prog.is_file():
        status = None
        for line in prog.read_text().splitlines():
            status = _handle(json.loads(line), out) or status
        if status is None:
            print("(run in progress; use --follow to keep watching)", file=out)
        return 0 if status in (None, "done") else 1
    print(f"{run_dir}: no result.json or {PROGRESS_FILE}", file=out)
    return 2


def _handle(record: dict[str, Any], out: TextIO) -> str | None:
    kind = record["kind"]
    if kind == "start":
        print(render_header({**record, "labels": ""}), file=out)
        print("  (flat arms train first; rounds appear as the hierarchical arm runs)", file=out)
    elif kind == "round":
        print(render_round(record["round"], record["updates"]), file=out)
    elif kind == "done":
        print("run complete; the full result is in result.json", file=out)
        return "done"
    elif kind == "failed":
        print(f"run FAILED: {record['error']}", file=out)
        return "failed"
    return None


def follow(path: str | Path, out: TextIO = sys.stdout, interval: float = 1.0, timeout: float | None = None,
           sleep: Callable[[float], None] = time.sleep) -> int:
    """Tail ``progress.jsonl`` until the run ends (0 = done, 1 = failed, 3 = timeout). A run
    with no progress file but a result.json (a flat run, or one that predates H5) is shown
    once, as ``peek`` does."""
    run_dir = _run_dir(path)
    prog = run_dir / PROGRESS_FILE
    pos, buf, waited, said = 0, "", 0.0, False
    while True:
        if prog.is_file():
            if prog.stat().st_size < pos:  # the run was restarted: start over
                pos, buf = 0, ""
            with prog.open("r") as f:
                f.seek(pos)
                buf += f.read()
                pos = f.tell()
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                status = _handle(json.loads(line), out) if line.strip() else None
                if status == "done":
                    return 0
                if status == "failed":
                    return 1
            out.flush()
        elif (run_dir / "result.json").is_file():
            return peek(run_dir, out)
        elif not said:
            print(f"waiting for {prog} ...", file=out)
            said = True
        if timeout is not None and waited >= timeout:
            print("(timeout; the run has not finished)", file=out)
            return 3
        sleep(interval)
        waited += interval
