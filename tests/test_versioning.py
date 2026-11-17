"""Phase H5: on-the-fly model updates with a validation gate (fl/versioning.py) and the
read-only terminal peek (peek.py)."""

import dataclasses
import hashlib
import io
import json
import os
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from fedastro.cli import main
from fedastro.config import ConfigError, UpdateInject, UpdatesConfig, load_config
from fedastro.fl.hierarchy import HierSim, run_rounds
from fedastro.fl.versioning import UpdateGate, degrade, macro_f1
from fedastro.metrics.schema import validate_result
from fedastro.peek import follow, peek, render_round

from test_hierarchy_fl import COMMS, _cfg, _fake_train, _n_train

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "configs" / "experiments"


# -- the gate -------------------------------------------------------------------------


def _gate(cfg=None, y=None, local=None, predict=None, nodes=("a",)):
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1]) if y is None else y
    local = y.copy() if local is None else local  # the local model is perfect on validation
    return UpdateGate(cfg or UpdatesConfig(enabled=True), 0, 3, list(nodes), {n: y for n in nodes},
                      {n: local for n in nodes}, predict or (lambda w, n: w[0]), ["r"])


def test_gate_accepts_at_the_boundary_and_rejects_below():
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    worse = y.copy()
    worse[0] = 1  # one mistake
    after = macro_f1(y, worse, 3)
    g = _gate(y=y)
    cur = g.deployed["a"]["regional"]
    assert cur.version == 0 and cur.score == 1.0 and cur.weights is None
    cur.tolerance = 1.0 - after  # exactly at the boundary: accepted
    (ev,) = g.offer(1, "regional", "r", 1, [worse], ["a"])
    assert ev["accepted"] and g.deployed["a"]["regional"].version == 1
    g = _gate(y=y)
    g.deployed["a"]["regional"].tolerance = (1.0 - after) - 1e-9  # just below: rejected
    (ev,) = g.offer(1, "regional", "r", 1, [worse], ["a"])
    assert not ev["accepted"]
    assert {"round", "node", "model", "version", "accepted", "val_before", "val_after"} <= set(ev)


def test_a_rejected_version_is_never_deployed():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    g = _gate(y=y)
    bad = [np.zeros_like(y)]  # predicts class 0 everywhere
    (ev,) = g.offer(1, "global", "r", 1, bad, ["a"])
    assert not ev["accepted"] and ev["version_before"] == 0
    d = g.deployed["a"]["global"]
    assert d.version == 0 and d.weights is None  # still the local model
    assert g.deployed["a"]["regional"].version == 0  # tracks are separate
    good = [y.copy()]
    (ev,) = g.offer(2, "global", "r", 2, good, ["a"])
    assert ev["accepted"] and g.deployed["a"]["global"].weights is good
    assert g.section()["n_rejected"] == 1 and g.section()["deployed"] == {"a": {"regional": 0, "global": 2}}


def test_tolerance_is_k_times_the_bootstrap_se_and_seeded():
    y = np.array([0, 1, 2] * 5)
    local = y.copy()
    local[:4] = (local[:4] + 1) % 3
    a, b = _gate(y=y, local=local), _gate(y=y, local=local)
    assert a.deployed["a"]["regional"].tolerance == b.deployed["a"]["regional"].tolerance > 0
    k3 = _gate(UpdatesConfig(enabled=True, k=3.0), y=y, local=local)
    assert k3.deployed["a"]["regional"].tolerance == pytest.approx(1.5 * a.deployed["a"]["regional"].tolerance)


def test_degrade_is_a_seeded_copy():
    w = [np.arange(6, dtype=np.float32).reshape(2, 3), np.ones(3, dtype=np.float32)]
    keep = [x.copy() for x in w]
    n1, n2 = degrade(w, "noise", 5.0, 1), degrade(w, "noise", 5.0, 1)
    assert all(np.array_equal(a, b) for a, b in zip(n1, n2))
    assert all(np.array_equal(a, b) for a, b in zip(w, keep))  # input untouched
    assert not np.allclose(n1[0], w[0]) and n1[0].dtype == np.float32
    assert all(not z.any() for z in degrade(w, "zero", 1.0, 0))


def test_an_empty_validation_set_is_an_error():
    with pytest.raises(ValueError, match="validation set at every node"):
        _gate(y=np.array([], dtype=int), local=np.array([], dtype=int))


# -- the gate in the schedule (no Flower) ---------------------------------------------


def _schedule_gate(cfg, inject=()):
    """A gate whose 'model' is the 1-element weight vector: it predicts class 1 when the
    weight is positive, else class 0; validation labels are all 1 at every node."""
    names = [n.name for n in cfg.hierarchy.nodes]
    y = {n: np.ones(6, dtype=int) for n in names}
    local = {n: np.array([1, 1, 1, 0, 0, 0]) for n in names}
    ucfg = UpdatesConfig(enabled=True, inject=tuple(inject))
    return UpdateGate(ucfg, 0, 2, names, y, local, lambda w, n: np.full(6, int(w[0][0] > 0)),
                      [r.name for r in cfg.hierarchy.regions])


def _run(cfg, gate=None, sim=None, **kw):
    n_train = _n_train(cfg)
    return run_rounds(cfg, _fake_train([]), [np.zeros(1)], n_train, sim or HierSim(cfg, 0, n_train), 0.0,
                      lambda w, r: {"accuracy": float(w[0][0])}, gate=gate, **kw)


def test_the_gate_changes_no_training_and_rejects_an_injected_fault():
    cfg = _cfg(rounds=4, global_every=2, offline={"leo_nasa": (3,)})
    plain = _run(cfg)
    gate = _schedule_gate(cfg, [UpdateInject(round=3, region="moon", track="regional", kind="zero")])
    gated = _run(cfg, gate=gate)
    strip = lambda rounds: json.dumps(rounds, sort_keys=True)  # noqa: E731
    assert strip(plain["rounds"]) == strip(gated["rounds"])
    assert np.array_equal(plain["global"][0], gated["global"][0])
    assert all(np.array_equal(plain["regional"][r][0], gated["regional"][r][0]) for r in plain["regional"])
    ev = gated["update_events"]
    inj = [e for e in ev if e["injected"]]
    assert len(inj) == 5 and not any(e["accepted"] for e in inj)  # zero weights -> class 0 -> F1 0
    assert all(e["accepted"] for e in ev if not e["injected"])
    # recipients: regional versions to the round's participants only; LEO fell back in round 3
    r3 = [e for e in ev if e["round"] == 3 and e["model"] == "regional"]
    assert {e["region"] for e in r3} == {"moon", "mars"}
    r1 = [e for e in ev if e["round"] == 1]
    assert len(r1) == 15 and {e["version"] for e in r1} == {1}
    glob = [e for e in ev if e["model"] == "global"]
    assert {e["round"] for e in glob} == {2, 4} and {e["version"] for e in glob} == {1, 2}
    moon = gate.deployed["moon_nasa"]["regional"]
    assert moon.version == 4  # v3 (injected) rejected, v4 (round 4) accepted


def test_node_states_per_round():
    cfg = _cfg(rounds=2, global_every=2, offline={"leo_nasa": (1,)})
    out = _run(cfg)
    leo = out["rounds"][0]["regions"]["leo"]["node_state"]
    assert leo["leo_nasa"] == "loss_of_signal"
    assert {leo[n] for n in leo if n != "leo_nasa"} == {"fallback"}
    assert set(out["rounds"][0]["regions"]["moon"]["node_state"].values()) == {"training"}
    cfg = _cfg(rounds=1, global_every=1, comms=COMMS, region_deadline_s=60.0)
    sim = HierSim(cfg, 0, _n_train(cfg))
    sim.node_links["moon_esa"].forced = (0.0, 1e12)
    out = _run(cfg, sim=sim)
    assert out["rounds"][0]["regions"]["moon"]["node_state"]["moon_esa"] == "loss_of_signal"
    cfg = _cfg(rounds=1, global_every=1, comms=COMMS, region_deadline_s=1e-6)  # every update late
    out = _run(cfg)
    assert set(out["rounds"][0]["regions"]["mars"]["node_state"].values()) == {"in_transit"}


def test_on_round_sees_every_round_once_and_diag_fills_the_record():
    cfg = _cfg(rounds=3, global_every=2)
    seen = []
    out = _run(cfg, gate=_schedule_gate(cfg), diag=lambda reg, g: {"has_global": g is not None},
               on_round=lambda rec, ups: seen.append((rec["round"], len(ups))))
    assert [r for r, _ in seen] == [1, 2, 3] and all(n > 0 for _, n in seen)
    assert [rec["diagnostic"]["has_global"] for rec in out["rounds"]] == [False, True, True]


# -- config ---------------------------------------------------------------------------


def test_config_rules():
    cfg = load_config(EXP / "smoke_hier_updates.yaml")
    assert "updates" in cfg.experiment_dict()
    assert "updates" not in load_config(EXP / "smoke_hier_fl.yaml").experiment_dict()  # run ids unchanged
    up = cfg.updates
    bad = [
        (dataclasses.replace(up, inject=(UpdateInject(round=3, region="venus"),)), "not a region"),
        (dataclasses.replace(up, inject=(UpdateInject(round=99, region="moon"),)), "<= fl.rounds"),
        (dataclasses.replace(up, inject=(UpdateInject(round=3, region="moon", track="global"),)), "merge round"),
        (dataclasses.replace(up, k=0.0), "k must be > 0"),
        (dataclasses.replace(up, metric="accuracy"), "macro_f1"),
    ]
    for u, match in bad:
        with pytest.raises(ConfigError, match=match):
            dataclasses.replace(cfg, updates=u).validate()
    flat = load_config(EXP / "smoke.yaml")
    with pytest.raises(ConfigError, match="only implemented for the 'hierarchical' arm"):
        dataclasses.replace(flat, updates=UpdatesConfig(enabled=True)).validate()
    dataclasses.replace(cfg, updates=dataclasses.replace(up, enabled=False, k=0.5)).validate()  # ignored when off


# -- through Flower, and the peek -----------------------------------------------------


def _run_cli(out_dir):
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        assert main(["run", str(EXP / "smoke_hier_updates.yaml"), "--seed", "0", "--output-dir", str(out_dir)]) == 0
    finally:
        del os.environ["SOURCE_DATE_EPOCH"]
    (path,) = Path(out_dir).glob("*/result.json")
    return path


@pytest.fixture(scope="module")
def updates_run(tmp_path_factory):
    return _run_cli(tmp_path_factory.mktemp("updates_smoke"))


def test_a_scripted_run_rejects_the_injected_updates(updates_run):
    r = json.loads(updates_run.read_text())
    validate_result(r)
    u = r["updates"]
    assert u["enabled"] and u["tolerance"]["rule"] == "bootstrap_se" and u["tolerance"]["k"] == 2.0
    inj = [e for e in u["events"] if e["injected"]]
    assert {(e["round"], e["model"]) for e in inj} == {(3, "regional"), (4, "global")}
    assert any(not e["accepted"] for e in inj)  # CLAUDE.md H5 acceptance
    assert r["arms"]["hierarchical"]["n_updates_rejected"] == u["n_rejected"] > 0
    for m in r["hierarchy"]["per_node"].values():
        assert {"local", "regional", "global", "deployed_regional", "deployed_global"} <= set(m)
    for rec in r["hierarchy"]["rounds"]:
        assert rec["diagnostic"]["accuracy"]["local"] is not None
        assert set().union(*(reg["node_state"].values() for reg in rec["regions"].values())) <= {
            "training", "fallback", "in_transit", "loss_of_signal"}
    assert r["hierarchy"]["rounds"][1]["regions"]["leo"]["node_state"]["leo_roscosmos"] == "loss_of_signal"
    assert "SCRIPTED" in r["notes"]
    lines = (updates_run.parent / "progress.jsonl").read_text().splitlines()
    kinds = [json.loads(x)["kind"] for x in lines]
    assert kinds == ["start"] + ["round"] * 6 + ["done"]


def _snapshot(d: Path):
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in d.iterdir()}


def test_peek_shows_the_run_and_writes_nothing(updates_run):
    run_dir = updates_run.parent
    before = _snapshot(run_dir)
    buf = io.StringIO()
    assert peek(run_dir, buf) == 0
    text = buf.getvalue()
    assert text.count("-- round ") == 6 and "REJECTED" in text and "[injected fault, SCRIPTED]" in text
    assert "loss of signal: leo_roscosmos" in text and "FALLBACK" in text and "disagreement" in text
    assert "SYNTHETIC data" in text
    buf = io.StringIO()
    assert follow(run_dir, buf, interval=0.0, timeout=1.0) == 0  # finished run: reads to "done"
    assert buf.getvalue().count("-- round ") == 6 and "run complete" in buf.getvalue()
    assert main(["peek", str(updates_run)]) == 0
    assert _snapshot(run_dir) == before  # read-only


def test_peek_on_a_run_in_progress(updates_run, tmp_path):
    lines = (updates_run.parent / "progress.jsonl").read_text().splitlines(keepends=True)
    live = tmp_path / "live"
    live.mkdir()
    prog = live / "progress.jsonl"
    prog.write_text("".join(lines[:3]))  # start + two rounds so far
    buf = io.StringIO()
    assert peek(live, buf) == 0 and buf.getvalue().count("-- round ") == 2 and "in progress" in buf.getvalue()
    rest = iter(lines[3:])

    def sleep(_):  # the "run" writes one more line between polls
        line = next(rest, None)
        if line is not None:
            with prog.open("a") as f:
                f.write(line)

    buf = io.StringIO()
    assert follow(live, buf, interval=0.01, timeout=60.0, sleep=sleep) == 0
    assert buf.getvalue().count("-- round ") == 6 and "run complete" in buf.getvalue()
    assert sorted(p.name for p in live.iterdir()) == ["progress.jsonl"]


def test_render_round_labels_simulated_time():
    rec = {"round": 1, "sim_wallclock_s": 12.5, "global_aggregated": False,
           "regions": {"leo": {"participants": ["a"], "fallback": False, "bytes": 0, "regional": {"accuracy": 0.5},
                               "node_state": {"a": "training", "b": "in_transit"}}}}
    text = render_round(rec, [])
    assert "sim t 12.5 s (SIMULATED)" in text and "in transit: b" in text and "training x1" in text


@pytest.mark.slow
def test_results_are_identical_with_and_without_peek(updates_run, tmp_path):
    """A second run with ``peek --follow`` tailing it the whole time writes the same bytes."""
    out_dir = tmp_path / "followed"
    from fedastro.manifest import make_run_id

    cfg = load_config(EXP / "smoke_hier_updates.yaml")
    run_dir = out_dir / make_run_id(cfg, 0)
    buf = io.StringIO()
    status = {}
    t = threading.Thread(target=lambda: status.setdefault("rc", follow(run_dir, buf, interval=0.2, timeout=1800)))
    t.start()
    again = _run_cli(out_dir)
    t.join(timeout=120)
    assert status.get("rc") == 0 and buf.getvalue().count("-- round ") == 6
    assert again.read_bytes() == updates_run.read_bytes()
    shutil.rmtree(out_dir)
