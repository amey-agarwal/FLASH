"""Phase H3: hierarchical FL (fl/hierarchy.py). The schedule is tested without Flower
(fast, exact); the Flower path end to end, including the flat-FedAvg equivalence."""

import dataclasses
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from fedastro.cli import main
from fedastro.config import ConfigError, load_config
from fedastro.fl.hierarchy import HierSim, run_rounds
from fedastro.fl.strategy import ClientUpdate
from fedastro.metrics.schema import validate_result
from fedastro.pipeline import run_experiment

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "configs" / "experiments"


def _cfg(rounds=6, comms=None, **hier_fl):
    cfg = load_config(EXP / "smoke_hier_fl.yaml")
    hf = dataclasses.replace(cfg.hier_fl, **{"offline": {}, **hier_fl})
    cfg = dataclasses.replace(cfg, fl=dataclasses.replace(cfg.fl, rounds=rounds), hier_fl=hf)
    if comms:
        cfg = dataclasses.replace(cfg, comms=dataclasses.replace(cfg.comms, enabled=True, **comms))
    cfg.validate()
    return cfg


def _n_train(cfg):
    return {n.name: 10 * (i + 1) for i, n in enumerate(cfg.hierarchy.nodes)}


def _fake_train(log):
    """Each node returns its weights plus (index + 1): exact, Flower-free updates."""
    def train(dispatch, rnd):
        log.append((rnd, sorted(dispatch)))
        return {i: ClientUpdate(silo_index=i, silo=str(i), arrays=[w[0] + (i + 1)], num_examples=10 * (i + 1),
                                train_loss=None) for i, w in dispatch.items()}
    return train


def _run(cfg, sim=None, log=None):
    n_train = _n_train(cfg)
    return run_rounds(cfg, _fake_train([] if log is None else log), [np.zeros(1)], n_train,
                      sim or HierSim(cfg, 0, n_train), 0.0, lambda w, r: {"accuracy": float(w[0][0])})


def _fedavg(idxs, w):
    n = np.array([10 * (i + 1) for i in idxs], dtype=float)
    return float(np.dot(n / n.sum(), [w + i + 1 for i in idxs]))


# -- schedule (no Flower) ---------------------------------------------------------------


def test_regional_fedavg_every_round_and_weighted_global_merge():
    cfg = _cfg(rounds=2, global_every=2)
    out = _run(cfg)
    idx = {r.name: [i for i, n in enumerate(cfg.hierarchy.nodes) if cfg.hierarchy.region_of(n.name) == r.name]
           for r in cfg.hierarchy.regions}
    reg = {r: _fedavg(ix, _fedavg(ix, 0.0)) for r, ix in idx.items()}  # two regional rounds from 0
    for r, v in reg.items():
        assert out["regional"][r][0][0] == pytest.approx(v)
    ex = {r: sum(10 * (i + 1) for i in ix) for r, ix in idx.items()}
    want = sum(ex[r] * reg[r] for r in reg) / sum(ex.values())
    assert out["global"][0][0] == pytest.approx(want)
    rec = out["rounds"][1]
    assert rec["global_aggregated"] and rec["global_participants"] == ["leo", "moon", "mars"]
    assert not out["rounds"][0]["global_aggregated"] and out["n_merges"] == 1
    assert all(rec["sim_wallclock_s"] is None for rec in out["rounds"])  # comms off


def test_regional_models_reset_to_the_global_after_a_merge():
    cfg = _cfg(rounds=3, global_every=2)
    log = []
    out = _run(cfg, log=log)
    g = out["global"][0][0]
    # round 3 trains every node from the merged global model
    idx = list(range(5))  # leo
    assert out["regional"]["leo"][0][0] == pytest.approx(_fedavg(idx, g))
    assert log[2] == (3, list(range(15)))


def test_one_region_with_global_every_1_is_flat_fedavg():
    cfg = _cfg(rounds=4, global_every=1)
    one = dataclasses.replace(cfg.hierarchy, regions=(dataclasses.replace(
        cfg.hierarchy.regions[0], nodes=tuple(n for r in cfg.hierarchy.regions for n in r.nodes)),),
        min_participants=1)
    cfg = dataclasses.replace(cfg, hierarchy=one)
    out = _run(cfg)
    w = 0.0
    for _ in range(4):
        w = _fedavg(list(range(15)), w)
    assert out["global"][0][0] == pytest.approx(w, rel=1e-6)
    assert out["regional"]["leo"][0][0] == pytest.approx(w, rel=1e-6)


def test_a_node_offline_drops_its_region_below_threshold_and_falls_back():
    cfg = _cfg(rounds=4, global_every=2, offline={"leo_nasa": (2, 3)})
    log = []
    out = _run(cfg, log=log)
    leo = [rec["regions"]["leo"] for rec in out["rounds"]]
    assert [r["fallback"] for r in leo] == [False, True, True, False]
    assert leo[1]["offline"] == ["leo_nasa"] and len(leo[1]["participants"]) == 4
    assert all(not rec["regions"]["moon"]["fallback"] for rec in out["rounds"])
    assert not set(log[1][1]) & set(range(5))  # no LEO training while it falls back
    falls = [e for e in out["events"] if e["kind"] == "fallback"]
    assert [(e["round"], e["region"]) for e in falls] == [(2, "leo"), (3, "leo")]
    # round 2: LEO holds the initial model and has nothing new, so it sits out the merge
    assert out["rounds"][1]["global_participants"] == ["moon", "mars"]
    assert out["rounds"][1]["global_received"] == ["leo", "moon", "mars"]
    assert any(e["kind"] == "merge_skipped" and e["region"] == "leo" for e in out["events"])
    # round 4: LEO trained in round 4 from the round-2 global and merges again
    assert out["rounds"][3]["global_participants"] == ["leo", "moon", "mars"]


def test_a_region_cut_off_at_merge_time_keeps_its_model():
    cfg = _cfg(rounds=2, global_every=2, region_offline={"mars": (2,)})
    out = _run(cfg)
    rec = out["rounds"][1]
    assert rec["global_participants"] == ["leo", "moon"] and rec["global_received"] == ["leo", "moon"]
    assert out["regional"]["mars"][0][0] != pytest.approx(out["global"][0][0])
    assert any(e["kind"] == "merge_skipped" and "scripted" in e["reason"] for e in out["events"])


def test_no_region_reachable_means_no_merge_and_no_crash():
    cfg = _cfg(rounds=2, global_every=2, region_offline={"leo": (2,), "moon": (2,), "mars": (2,)})
    out = _run(cfg)
    assert not out["rounds"][1]["global_aggregated"] and out["n_merges"] == 0
    assert out["global"][0][0] == 0.0  # still the initial model


def test_schedule_is_deterministic():
    cfg = _cfg(rounds=4, global_every=2, offline={"moon_esa": (3,)})
    a, b = _run(cfg), _run(cfg)
    assert json.dumps(a["rounds"], sort_keys=True) == json.dumps(b["rounds"], sort_keys=True)


# -- simulated time ------------------------------------------------------------------


COMMS = dict(los=False, jitter_s=0.0, compute_s_per_example=0.0)


def test_simulated_time_follows_the_links_and_mars_merges_take_light_time():
    cfg = _cfg(rounds=2, global_every=2, comms=COMMS, region_deadline_s=60.0)
    out = _run(cfg)
    t1, t2 = (rec["sim_wallclock_s"] for rec in out["rounds"])
    assert 0 < t1 < 60  # regional rounds: short node links
    # the merge waits for Mars up (751 s) and back down (751 s), plus transmission
    assert t2 - 2 * t1 > 2 * 751.0
    assert all(not r["fallback"] for rec in out["rounds"] for r in rec["regions"].values())


def test_a_node_in_loss_of_signal_misses_the_deadline_and_its_region_falls_back():
    cfg = _cfg(rounds=4, global_every=2, comms=COMMS, region_deadline_s=60.0)
    sim = HierSim(cfg, 0, _n_train(cfg))
    sim.node_links["moon_esa"].forced = (0.0, 1e12)  # dark for the whole run
    out = _run(cfg, sim=sim)
    assert all(rec["regions"]["moon"]["fallback"] for rec in out["rounds"])
    assert out["rounds"][0]["regions"]["moon"]["missed_deadline"] == ["moon_esa"]
    assert not any(rec["regions"]["leo"]["fallback"] for rec in out["rounds"])
    assert all("moon" not in rec.get("global_participants", []) for rec in out["rounds"])


def test_solar_conjunction_cuts_mars_out_of_the_merge():
    cfg = _cfg(rounds=2, global_every=2, comms=COMMS, region_deadline_s=60.0, global_deadline_s=3_600.0,
               conjunction_region="mars", conjunction_from_s=0.0, conjunction_duration_s=14 * 86_400.0)
    out = _run(cfg)
    rec = out["rounds"][1]
    assert rec["global_participants"] == ["leo", "moon"] and "mars" not in rec["global_received"]
    reasons = [e["reason"] for e in out["events"] if e["region"] == "mars"]
    assert any("did not reach the global aggregator" in r for r in reasons)


# -- config rules ------------------------------------------------------------------


def test_config_rules():
    cfg = load_config(EXP / "smoke_hier_fl.yaml")
    assert "hier_fl" in cfg.experiment_dict()
    assert "hier_fl" not in load_config(EXP / "smoke.yaml").experiment_dict()  # locked run ids unchanged
    bad = [
        (dict(arms=("centralized", "federated", "hierarchical")), "needs the 'local_only' arm"),
        (dict(hier_fl=dataclasses.replace(cfg.hier_fl, global_every=99)), "global_every must be <= fl.rounds"),
        (dict(hier_fl=dataclasses.replace(cfg.hier_fl, offline={"leo_venus": (1,)})), "unknown nodes"),
        (dict(hier_fl=dataclasses.replace(cfg.hier_fl, offline={"leo_nasa": (99,)})), "<= fl.rounds"),
        (dict(hier_fl=dataclasses.replace(cfg.hier_fl, region_offline={"venus": (2,)})), "unknown regions"),
        (dict(hier_fl=dataclasses.replace(cfg.hier_fl, conjunction_region="mars", conjunction_duration_s=10.0)),
         "needs comms.enabled"),
        (dict(hier_fl=dataclasses.replace(cfg.hier_fl, region_deadline_s=5.0)), "deadlines need comms"),
        (dict(comms=dataclasses.replace(cfg.comms, enabled=True)), "region_deadline_s > 0"),
        (dict(fl=dataclasses.replace(cfg.fl, strategy="fedbuff")), "supports fl.strategy"),
        (dict(privacy=dataclasses.replace(cfg.privacy, enabled=True, target_epsilon=1.0)), "not supported"),
    ]
    for kw, match in bad:
        with pytest.raises(ConfigError, match=match):
            dataclasses.replace(cfg, **kw).validate()
    flat = load_config(EXP / "smoke.yaml")
    with pytest.raises(ConfigError, match="needs a hierarchy"):
        dataclasses.replace(flat, arms=("local_only", "hierarchical")).validate()
    with pytest.raises(ConfigError, match="only meaningful with the 'hierarchical' arm"):
        dataclasses.replace(flat, hier_fl=dataclasses.replace(cfg.hier_fl, global_every=3)).validate()


# -- through Flower ------------------------------------------------------------------


def _one_region_root(tmp_path):
    """A one-region topology with five nodes, for the flat-equivalence check."""
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    topo = yaml.safe_load((REPO / "configs" / "topologies" / "cinc15.yaml").read_text())
    leo = topo["regions"][0]
    leo["nodes"][2] = {"name": "leo_cmsa2", "agency": "cmsa2", "sources": ["chapman"]}  # INCART is tiny
    topo.update(name="one_region", min_participants=5, regions=[leo])
    (tmp_path / "configs" / "topologies" / "one_region.yaml").write_text(yaml.safe_dump(topo))
    exp = yaml.safe_load((EXP / "smoke_hier_fl.yaml").read_text())
    exp.update(name="one_region", hierarchy="one_region", arms=["local_only", "federated", "hierarchical"])
    exp["shift"] = {"enabled": False}
    exp["fl"] = {"rounds": 3}
    exp["hier_fl"] = {"global_every": 1}
    path = tmp_path / "configs" / "experiments" / "one_region.yaml"
    path.write_text(yaml.safe_dump(exp))
    return path


def test_one_region_hierarchy_reproduces_flat_fedavg_through_flower(tmp_path):
    r = run_experiment(load_config(_one_region_root(tmp_path)), 0)
    assert r["arms"]["hierarchical"]["final"] == r["arms"]["federated"]["final"]
    for node, m in r["hierarchy"]["per_node"].items():
        assert m["global"] == m["flat_federated"] == m["regional"], node
    assert r["disagreement"]["per_node"]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("hier_smoke")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        t0 = time.perf_counter()
        assert main(["run", str(EXP / "smoke_hier_fl.yaml"), "--seed", "0", "--output-dir", str(out)]) == 0
        elapsed = time.perf_counter() - t0
    finally:
        del os.environ["SOURCE_DATE_EPOCH"]
    (path,) = out.glob("*/result.json")
    return path, elapsed


def test_one_command_produces_every_arm_and_the_hierarchy(smoke_run):
    path, elapsed = smoke_run
    assert elapsed < 600  # CLAUDE.md H3: synthetic, laptop CPU, < 10 min
    r = json.loads(path.read_text())
    validate_result(r)
    assert r["schema_version"] == "2" and r["shift"]["enabled"] and r["topology"]["name"] == "cinc15"
    for arm in ("centralized", "local_only", "federated", "hierarchical"):
        assert r["arms"][arm]
    h = r["hierarchy"]
    assert len(h["rounds"]) == 6 and len(h["per_node"]) == 15 and set(h["per_region"]) == {"leo", "moon", "mars"}
    for m in h["per_node"].values():
        assert {"local", "regional", "global", "flat_federated", "centralized"} <= set(m)
    assert [(e["round"], e["region"]) for e in h["events"] if e["kind"] == "fallback"] == [(2, "leo"), (3, "leo")]
    assert r["arms"]["hierarchical"]["n_fallbacks"] == 2 and r["arms"]["hierarchical"]["n_merges"] == 3
    d = r["disagreement"]
    assert 0 <= d["global"]["rate"] <= 1 and d["headline_model"] == "regional"
    assert d["acc_agree"] is not None and d["acc_disagree"] is not None
    assert "SIMULATED location shift" in r["notes"] and "SYNTHETIC" in r["notes"]
    silo = r["silos"][0]
    assert {"region", "n_val", "n_test", "val_label_counts"} <= set(silo)


@pytest.mark.slow
def test_hierarchical_run_is_byte_identical(smoke_run, tmp_path):
    from fedastro.manifest import dumps  # noqa: F401  (same writer as the run)
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        assert main(["run", str(EXP / "smoke_hier_fl.yaml"), "--seed", "0", "--output-dir", str(tmp_path)]) == 0
    finally:
        del os.environ["SOURCE_DATE_EPOCH"]
    (again,) = tmp_path.glob("*/result.json")
    assert again.read_bytes() == smoke_run[0].read_bytes()


def test_summary_reports_the_hierarchy(smoke_run):
    from fedastro.metrics.aggregate import format_summary, load_results, summarise
    s = summarise(load_results([smoke_run[0]]))
    assert set(s["hierarchy"]["per_region"]) == {"leo", "moon", "mars"}
    assert "hierarchical regional" in format_summary(s)
