"""Phase H4: simulated pairwise-mask secure aggregation (fl/secagg.py). A protocol
simulation: these tests check the data flow and the cost, not cryptographic security."""

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from fedastro.cli import main
from fedastro.config import ConfigError, load_config
from fedastro.fl import hierarchy, secagg
from fedastro.fl.hierarchy import HierSim, run_rounds
from fedastro.fl.strategy import ClientUpdate, FedAvg
from fedastro.metrics.schema import validate_result

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "configs" / "experiments"
COMMS = dict(los=False, jitter_s=0.0, compute_s_per_example=0.0)
TOL = 1e-6


def _updates(rng, k, shapes=((3, 4), (4,))):
    return {i: ([rng.normal(0, 1, s).astype(np.float32) for s in shapes], int(rng.integers(5, 500)))
            for i in range(k)}


def _fedavg(ups):
    return FedAvg().aggregate([ClientUpdate(i, str(i), a, n, None) for i, (a, n) in ups.items()])


def _close(a, b):
    return all(np.allclose(x, y, rtol=0, atol=TOL) for x, y in zip(a, b))


# -- the primitive ------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5, 12])
def test_masks_cancel_and_the_sum_equals_fedavg(k):
    rng = np.random.default_rng(k)
    ups = _updates(rng, k)
    group = list(ups)
    masked = {i: secagg.mask_update(a, n, i, group, 7, "regional", 3, 24) for i, (a, n) in ups.items()}
    out = secagg.unmask_sum(masked, group, 7, "regional", 3, 24, ups[0][0], min_group=2)
    assert out.total_examples == sum(n for _, n in ups.values()) and out.recovered == ()
    assert _close(out.weights, _fedavg(ups))
    # the pairwise masks alone sum to zero in the ring
    zero = [np.zeros(s, np.float32) for s in ((3, 4), (4,))]
    zs = [secagg.mask_update(zero, 0, i, group, 7, "regional", 3, 24) for i in group]
    assert not np.any(np.sum(zs, axis=0, dtype=np.uint64))


def test_one_masked_vector_reveals_nothing_without_the_others():
    rng = np.random.default_rng(0)
    ups = _updates(rng, 5)
    a, n = ups[2]
    plain = secagg.encode(a, n, 24, 5)
    masked = secagg.mask_update(a, n, 2, list(ups), 7, "regional", 1, 24)
    assert np.mean(masked == plain) < 0.01
    decoded = masked[:-1].view(np.int64).astype(np.float64) / 2.0 ** 24  # decoding it alone: noise
    assert np.max(np.abs(decoded)) > 1e9
    # a new round or level gives different masks
    assert not np.array_equal(masked, secagg.mask_update(a, n, 2, list(ups), 7, "regional", 2, 24))
    assert not np.array_equal(masked, secagg.mask_update(a, n, 2, list(ups), 7, "global", 1, 24))


def test_dropout_recovery_rebuilds_the_dropped_masks():
    rng = np.random.default_rng(1)
    ups = _updates(rng, 5)
    group = list(ups)
    masked = {i: secagg.mask_update(a, n, i, group, 3, "regional", 1, 24) for i, (a, n) in ups.items()}
    del masked[1], masked[3]  # joined the group, then never uploaded
    out = secagg.unmask_sum(masked, group, 3, "regional", 1, 24, ups[0][0], min_group=3)
    assert out.recovered == (1, 3)
    assert _close(out.weights, _fedavg({i: ups[i] for i in (0, 2, 4)}))


def test_too_few_survivors_and_overflow_raise():
    rng = np.random.default_rng(2)
    ups = _updates(rng, 3)
    masked = {i: secagg.mask_update(a, n, i, list(ups), 0, "regional", 1, 24) for i, (a, n) in ups.items()}
    with pytest.raises(secagg.SecAggError, match="fewer than the group minimum"):
        secagg.unmask_sum({0: masked[0]}, list(ups), 0, "regional", 1, 24, ups[0][0], min_group=2)
    with pytest.raises(secagg.SecAggError, match="overflow"):
        secagg.encode([np.array([1e12], np.float32)], 10_000, 24, 5)
    with pytest.raises(ValueError, match="not in its mask group"):
        secagg.mask_update(ups[0][0], 1, 9, [0, 1], 0, "regional", 1, 24)


# -- the schedule (no Flower) ---------------------------------------------------------


def _cfg(secagg_on=True, comms=None, min_participants=None, rounds=4, dropout=None, **hier_fl):
    cfg = load_config(EXP / "smoke_hier_secagg.yaml")
    hf = dataclasses.replace(cfg.hier_fl, **{"region_deadline_s": 120.0 if comms else 0.0, **hier_fl})
    sa = dataclasses.replace(cfg.secagg, enabled=secagg_on, dropout=dropout or {})
    h = cfg.hierarchy if min_participants is None else dataclasses.replace(cfg.hierarchy, min_participants=min_participants)
    cfg = dataclasses.replace(cfg, fl=dataclasses.replace(cfg.fl, rounds=rounds), hier_fl=hf, secagg=sa,
                              hierarchy=h, comms=dataclasses.replace(cfg.comms, enabled=bool(comms), **(comms or {})))
    cfg.validate()
    return cfg


def _n_train(cfg):
    return {n.name: 10 * (i + 1) for i, n in enumerate(cfg.hierarchy.nodes)}


def _train(cfg, seen=None):
    """Node i returns its weights + (i + 1) + a sentinel offset; with mask groups it masks the
    update itself (as the Flower client does) and returns only the masked vector."""
    def train(dispatch, rnd, groups=None):
        out = {}
        for i, w in dispatch.items():
            upd = [w[0] + (i + 1) + 1e-3 * rnd]
            if seen is not None:
                seen.append(upd[0].copy())
            if groups is None:
                out[i] = ClientUpdate(i, str(i), upd, 10 * (i + 1), None)
            else:
                out[i] = secagg.mask_update(upd, 10 * (i + 1), i, groups[i], 0, "regional", rnd, cfg.secagg.frac_bits)
        return out
    return train


def _run(cfg, seen=None, sim=None):
    n = _n_train(cfg)
    return run_rounds(cfg, _train(cfg, seen), [np.zeros(2)], n, sim or HierSim(cfg, 0, n), 0.0,
                      lambda w, r: {"accuracy": float(w[0][0])}, seed=0)


@pytest.mark.parametrize("comms", [None, COMMS])
def test_schedule_aggregates_equal_with_secagg_on_and_off(comms):
    on, off = _run(_cfg(True, comms)), _run(_cfg(False, comms))
    for r in off["regional"]:
        assert _close(on["regional"][r], off["regional"][r])
    assert _close(on["global"], off["global"])
    assert on["n_merges"] == off["n_merges"] == 2
    assert [e["kind"] for e in on["events"]] == [e["kind"] for e in off["events"]]
    for a, b in zip(on["rounds"], off["rounds"]):
        assert a.get("global_participants") == b.get("global_participants")
        assert [x["participants"] for x in a["regions"].values()] == [x["participants"] for x in b["regions"].values()]


def test_scripted_dropout_runs_recovery_and_matches_the_survivor_aggregate():
    # min_participants 4 of 5, so one dropout leaves enough survivors to recover
    on = _run(_cfg(True, min_participants=4, dropout={"moon_esa": (2,)}))
    off = _run(_cfg(False, min_participants=4, offline={"moon_esa": (2,)}))
    rec = on["rounds"][1]["regions"]["moon"]
    assert rec["secagg"]["recovery"] and rec["secagg"]["recovered"] == ["moon_esa"]
    assert "moon_esa" not in rec["participants"] and not rec["fallback"]
    assert _close(on["regional"]["moon"], off["regional"]["moon"]) and _close(on["global"], off["global"])
    assert any(e["kind"] == "secagg_recovery" and e["region"] == "moon" for e in on["events"])


def test_a_dropout_below_the_threshold_falls_back_and_unmasks_nothing(monkeypatch):
    calls = []
    real = secagg.unmask_sum
    monkeypatch.setattr(secagg, "unmask_sum", lambda masked, group, *a, **k: calls.append(sorted(masked)) or
                        real(masked, group, *a, **k))
    out = _run(_cfg(True, dropout={"leo_nasa": (1,)}, rounds=2, global_every=2))
    leo = out["rounds"][0]["regions"]["leo"]
    assert leo["fallback"] and not leo["secagg"]["recovery"] and leo["secagg"]["dropped"] == ["leo_nasa"]
    assert len(calls) == 2 + 3 + 1  # round 1: moon and mars; round 2: all three; the merge
    assert all(len(c) >= 2 for c in calls)


def test_global_group_below_minimum_skips_the_merge():
    # leo and mars offline at merge time: one region left, its "sum" would be its own model
    out = _run(_cfg(True, rounds=2, global_every=2, region_offline={"leo": (2,), "mars": (2,)}))
    assert not out["rounds"][1]["global_aggregated"] and out["n_merges"] == 0
    assert any(e["kind"] == "secagg_group_too_small" for e in out["events"])
    plain = _run(_cfg(False, rounds=2, global_every=2, region_offline={"leo": (2,), "mars": (2,)}))
    assert plain["n_merges"] == 1  # the protocol's cost: without secagg one region merges alone


def _walk_arrays(obj):
    if isinstance(obj, np.ndarray):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_arrays(v)
    elif isinstance(obj, (list, tuple)):
        if obj and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
            yield np.asarray(obj, dtype=float)
        for v in obj:
            yield from _walk_arrays(v)


def test_no_unmasked_update_is_stored_or_logged(monkeypatch):
    seen: list[np.ndarray] = []
    received: list[np.ndarray] = []
    real = secagg.unmask_sum

    def spy(masked, *a, **k):
        received.extend(masked.values())
        return real(masked, *a, **k)

    monkeypatch.setattr(secagg, "unmask_sum", spy)
    out = _run(_cfg(True, COMMS, min_participants=4, dropout={"mars_esa": (3,)}), seen=seen)
    assert seen and received
    # the aggregator only ever received masked ring vectors
    assert all(v.dtype == np.uint64 for v in received)
    for v in received:
        assert not any(np.allclose(v[:-1].view(np.int64) / 2.0 ** 24 / max(int(v[-1]), 1), u, atol=1e-3)
                       for u in seen)
    # nothing in the schedule output equals any node's update
    stored = list(_walk_arrays({k: v for k, v in out.items() if k != "sim"}))
    for u in seen:
        assert not any(a.shape == u.shape and np.allclose(a, u, atol=1e-9) for a in stored)


def test_overhead_is_reported_per_round_and_grows_with_mars_distance():
    ends, over = {}, {}
    for preset in ("mars_near", "mars_mean", "mars_far"):
        on = _cfg(True, dict(COMMS, preset=preset), rounds=2, global_every=2)
        off = _cfg(False, dict(COMMS, preset=preset), rounds=2, global_every=2)
        r_on, r_off = _run(on), _run(off)
        ends[preset] = r_on["sim_end_s"] - r_off["sim_end_s"]
        sec = hierarchy.secagg_section(on, r_on)
        over[preset] = sec["overhead_sim_s"]
        assert len(sec["per_round"]) == 2 and sec["overhead_bytes"] > 0
        assert all(row["overhead_bytes"] > 0 for row in sec["per_round"])
        assert sec["per_round"][1]["global"]["overhead_sim_s"] > 0
    assert 0 < ends["mars_near"] < ends["mars_mean"] < ends["mars_far"]
    assert over["mars_near"] < over["mars_mean"] < over["mars_far"]
    # the overhead accounts for the extra time: key agreement (2 round trips) + share forwarding
    assert over["mars_mean"] == pytest.approx(ends["mars_mean"], rel=0.05)


def test_config_rules():
    cfg = load_config(EXP / "smoke_hier_secagg.yaml")
    assert "secagg" in cfg.experiment_dict()
    assert "secagg" not in load_config(EXP / "smoke_hier_fl.yaml").experiment_dict()  # H3 run ids unchanged
    sa = cfg.secagg
    bad = [
        (dataclasses.replace(sa, levels=("edge",)), "secagg.levels"),
        (dataclasses.replace(sa, key_agreement="once"), "per_round"),
        (dataclasses.replace(sa, global_min_group=1), "global_min_group"),
        (dataclasses.replace(sa, frac_bits=60), "frac_bits"),
        (dataclasses.replace(sa, dropout={"leo_venus": (1,)}), "unknown nodes or regions"),
        (dataclasses.replace(sa, dropout={"leo_nasa": (99,)}), "<= fl.rounds"),
    ]
    for s, match in bad:
        with pytest.raises(ConfigError, match=match):
            dataclasses.replace(cfg, secagg=s).validate()
    flat = load_config(EXP / "smoke.yaml")
    with pytest.raises(ConfigError, match="only implemented for the 'hierarchical' arm"):
        dataclasses.replace(flat, secagg=dataclasses.replace(sa, enabled=True)).validate()


# -- through Flower ------------------------------------------------------------------


def test_one_region_flat_equivalence_holds_with_secagg(tmp_path):
    """The H3 one-region check with regional secure aggregation on: the masked hierarchy
    still equals flat FedAvg (within float tolerance), and the server only got masked vectors."""
    import yaml
    from test_hierarchy_fl import _one_region_root

    from fedastro.pipeline import run_experiment

    path = _one_region_root(tmp_path)
    exp = yaml.safe_load(path.read_text())
    exp["secagg"] = {"enabled": True, "levels": ["regional"]}
    path.write_text(yaml.safe_dump(exp))
    r = run_experiment(load_config(path), 0)
    a, b = r["arms"]["hierarchical"]["final"], r["arms"]["federated"]["final"]
    for k in ("accuracy", "macro_f1"):
        assert a[k] == pytest.approx(b[k], abs=1e-3)
    assert r["secagg"]["enabled"] and r["secagg"]["overhead_sim_s"] is None  # comms off
    validate_result(r)


@pytest.fixture(scope="module")
def secagg_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("secagg_smoke")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        assert main(["run", str(EXP / "smoke_hier_secagg.yaml"), "--seed", "0", "--output-dir", str(out)]) == 0
    finally:
        del os.environ["SOURCE_DATE_EPOCH"]
    (path,) = out.glob("*/result.json")
    return json.loads(path.read_text()), path


def test_smoke_run_reports_secagg(secagg_run):
    r, path = secagg_run
    validate_result(r)
    s = r["secagg"]
    assert s["enabled"] and s["simulated"] and s["levels"] == ["regional", "global"]
    assert s["overhead_bytes"] > 0 and s["overhead_sim_s"] > 0 and s["dropout_recoveries"] == 1
    assert len(s["per_round"]) == 6 and "not cryptographically secure" in s["note"]
    rounds = r["hierarchy"]["rounds"]
    assert rounds[2]["regions"]["leo"]["fallback"]  # leo_nasa dropped after masking, 4 < 5
    assert rounds[3]["global_secagg"]["recovered"] == ["moon"]
    assert rounds[3]["global_participants"] == ["leo", "mars"]
    assert "SIMULATED secure aggregation" in r["notes"]
