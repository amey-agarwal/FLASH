"""Phase 6: simulated links, buffered-async aggregation, silos going dark."""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from fedastro.cli import main
from fedastro.config import ConfigError, LOSProfile, SiloConfig, load_config
from fedastro.fl.comms import INF, Link, build_links, resolve_latency
from fedastro.fl.strategy import BufferedUpdate, ClientUpdate, FedBuff
from fedastro.grid import load_grid
from fedastro.metrics.schema import validate_result

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "configs/experiments/smoke_comms.yaml"


def _link(profile=LOSProfile(), one_way=10.0, bw=8_000.0, jitter=0.0, forced=None, seed=0, **buffer):
    return Link("s", 0, one_way, bw, jitter, profile, np.random.default_rng(seed),
                np.random.default_rng(seed + 1), forced, **buffer)


OUTAGE_50_80 = LOSProfile(kind="periodic", period_s=1000.0, phase_s=50.0, duration_s=30.0)


def test_newer_message_supersedes_one_still_waiting():
    link = _link(OUTAGE_50_80)
    link.transfer(55.0, 1000, "down")  # waits for the outage to end at 80
    assert link.transfer(60.0, 1000, "down") == pytest.approx(80.0 + 1.0 + 10.0)
    assert [(d["reason"], d["handed_s"]) for d in link.drops] == [("superseded", 55.0)]
    link.transfer(62.0, 1000, "up")  # other direction: independent buffer
    assert len(link.drops) == 1


def test_no_supersede_once_sent_or_when_disabled():
    link = _link(OUTAGE_50_80)
    link.transfer(10.0, 1000)  # transmits immediately
    link.transfer(20.0, 1000)
    assert link.drops == []
    off = _link(OUTAGE_50_80, supersede=False)
    off.transfer(55.0, 1000)
    off.transfer(60.0, 1000)
    assert off.drops == []


def test_expiry_drops_messages_that_would_arrive_too_late():
    link = _link(OUTAGE_50_80, ttl_s=20.0)
    assert link.transfer(10.0, 1000) == pytest.approx(21.0)  # 11 s <= ttl
    assert link.transfer(55.0, 1000) == INF  # would arrive at 91, 36 s after hand-over
    assert link.drops[-1]["reason"] == "expired" and link.drops[-1]["dropped_s"] == 75.0
    far = _link(one_way=30.0, ttl_s=20.0)  # light time alone exceeds the lifetime
    assert far.transfer(0.0, 1000) == INF


def test_cache_cap_drops_only_messages_that_must_wait():
    link = _link(OUTAGE_50_80, cache_bytes=500)
    assert link.transfer(10.0, 1000) < INF  # link up: sent straight away, no buffering
    assert link.transfer(55.0, 1000) == INF
    assert link.drops[-1]["reason"] == "cache_full"
    assert _link(OUTAGE_50_80, cache_bytes=5000).transfer(55.0, 1000) < INF


# --- link model -------------------------------------------------------------------

def test_transfer_is_transmit_plus_light_time_when_link_is_up():
    assert _link().transfer(100.0, 1000) == pytest.approx(100.0 + 1.0 + 10.0)  # 8000 bits at 8 kbps


def test_transmission_waits_for_outage_to_end():
    link = _link(LOSProfile(kind="periodic", period_s=100.0, phase_s=50.0, duration_s=30.0))
    assert link.transfer(60.0, 1000) == pytest.approx(80.0 + 1.0 + 10.0)  # starts after the outage
    # an outage starting mid-transmission restarts it after the outage (store-and-forward)
    assert link.transfer(49.5, 1000) == pytest.approx(80.0 + 1.0 + 10.0)
    assert link.transfer(10.0, 1000) == pytest.approx(21.0)


def test_permanent_dark_makes_silo_unreachable():
    link = _link(forced=(500.0, INF))
    assert link.transfer(100.0, 1000) < INF
    assert link.transfer(600.0, 1000) == INF
    assert link.transfer(499.5, 1000) == INF  # transmission would overlap the outage


def test_outage_schedule_is_independent_of_traffic():
    prof = LOSProfile(kind="poisson", mean_interval_s=1000.0, duration_s=100.0)
    busy, idle = _link(prof, jitter=1.0), _link(prof, jitter=1.0)
    for t in range(0, 20_000, 37):
        busy.transfer(float(t), 500)
    assert busy.outages(20_000.0) == idle.outages(20_000.0)


def test_presets_move_only_the_deep_space_silo():
    mars = SiloConfig("mars_transit", 0.08, 751.0, "long_blackout", {"near": 182.0, "mean": 751.0, "far": 1337.0})
    iss = SiloConfig("iss_commercial", 0.2, 0.25, "short_occasional")
    assert [resolve_latency(mars, p) for p in ("earth", "moon", "mars_near", "mars_mean", "mars_far")] == \
        [0.0, 1.3, 182.0, 751.0, 1337.0]
    assert resolve_latency(iss, "mars_far") == 0.25


def test_los_off_and_forced_dark_via_config():
    cfg = load_config(SMOKE, {"comms.los": False, "comms.dark.silo": "lunar_habitat", "comms.dark.from_s": 10.0})
    links = build_links(cfg, 0)
    assert all(l.profile.kind == "none" for l in links.values())
    assert links["lunar_habitat"].forced == (10.0, INF)


def test_dark_by_round_leaves_window_to_the_server():
    cfg = load_config(SMOKE, {"comms.dark.silo": "lunar_habitat", "comms.dark.from_round": 2})
    assert build_links(cfg, 0)["lunar_habitat"].forced is None


@pytest.mark.parametrize("overrides, message", [
    ({"fl.strategy": "fedbuff", "comms.enabled": False}, "needs comms.enabled"),
    ({"fl.strategy": "fedbuff", "privacy.enabled": True, "privacy.target_epsilon": 1.0}, "fedbuff"),
    ({"fl.strategy": "fedbuff", "fl.buffer_k": 9}, "buffer_k"),
    ({"comms.preset": "venus"}, "comms.preset"),
    ({"comms.dark.silo": "pluto"}, "not a silo"),
    ({"comms.los_profiles.periodic.duration_s": 9e9}, "periodic needs"),
    ({"comms.ttl_s": -1.0}, "ttl_s"),
    ({"comms.cache_bytes": -5}, "cache_bytes"),
    ({"comms.dark.silo": "mars_transit", "comms.dark.from_round": 2, "comms.dark.from_s": 5.0}, "not both"),
    ({"comms.dark.from_round": 2}, "needs comms.dark.silo"),
    ({"comms.dark.silo": "mars_transit", "comms.dark.from_round": 99}, "<= fl.rounds"),
])
def test_comms_config_validation(overrides, message):
    with pytest.raises(ConfigError, match=message):
        load_config(SMOKE, overrides)


def test_comms_absent_from_identity_when_off():
    off = load_config(REPO / "configs/experiments/smoke_fl.yaml")
    d = off.experiment_dict()
    assert "comms" not in d and "buffer_k" not in d["fl"]
    assert "real_sleep_scale" not in d


# --- FedBuff --------------------------------------------------------------------

def _upd(i, value, n):
    return ClientUpdate(i, f"s{i}", [np.full(3, value, np.float32)], n, 0.0)


def test_fedbuff_applies_staleness_weighted_example_weighted_deltas():
    current = [np.zeros(3, np.float32)]
    start = [np.zeros(3, np.float32)]
    buf = [BufferedUpdate(_upd(0, 1.0, 30), start, 0), BufferedUpdate(_upd(1, 1.0, 10), start, 3)]
    out = FedBuff(server_lr=1.0).aggregate(current, buf)
    np.testing.assert_allclose(out[0], 0.75 * 1.0 + 0.25 * 0.5, rtol=1e-6)  # 1/sqrt(1+3) = 0.5


def test_fedbuff_no_change_when_deltas_are_zero():
    w = [np.arange(3, dtype=np.float32)]
    buf = [BufferedUpdate(ClientUpdate(0, "a", [w[0].copy()], 5, 0.0), [w[0].copy()], 2)]
    assert np.array_equal(FedBuff(1.0).aggregate(w, buf)[0], w[0])


def test_grid_variants_list():
    _, v = load_grid(REPO / "configs/sweeps/phase6_dark.yaml")
    assert [x.name for x in v] == ["p6dark-fedavg", "p6dark-fedavg-dl3600", "p6dark-fedbuff"]
    assert all(x.config.comms.dark.silo == "mars_transit" for x in v)
    assert all(x.config.comms.dark.from_round == 16 and x.config.comms.dark.from_s == 0 for x in v)
    assert v[1].config.comms.sync_deadline_s == 3600


# --- end to end (real Flower/Ray simulation) --------------------------------------

def _run(overrides):
    from fedastro.pipeline import run_experiment

    return run_experiment(load_config(SMOKE, {"arms": ["federated"], **overrides}), 0)


@pytest.fixture(scope="module")
def runs():
    # setup ends ~4.7e3 s and a Mars-far round lasts ~2.7e3 s: dark from mid-round 2
    dark = {"comms.dark.silo": "mars_transit", "comms.dark.from_s": 9000.0}
    by_round = {"comms.dark.silo": "mars_transit", "comms.dark.from_round": 3}
    return {
        "off": _run({"comms.enabled": False}),
        "sync": _run({}),
        "buff": _run({"fl.strategy": "fedbuff"}),
        "dark_sync": _run(dark),
        "dark_deadline": _run({**dark, "comms.sync_deadline_s": 3600.0}),
        "dark_buff": _run({**dark, "fl.strategy": "fedbuff"}),
        "round_sync": _run(by_round),
        "round_buff": _run({**by_round, "fl.strategy": "fedbuff"}),
    }


def test_comms_does_not_change_sync_learning(runs):
    """Same participants every round -> identical weights; comms only adds time."""
    off, on = runs["off"]["arms"]["federated"], runs["sync"]["arms"]["federated"]
    assert [r["global"] for r in off["rounds"]] == [r["global"] for r in on["rounds"]]
    assert all(r["sim_wallclock_s"] is None for r in off["rounds"])
    t = [r["sim_wallclock_s"] for r in on["rounds"]]
    assert all(a < b for a, b in zip(t, t[1:])) and t[0] > 2 * 1337  # Mars-far round trip


def test_comms_section_and_schema(runs):
    r = runs["sync"]
    validate_result(r)
    c = r["comms"]
    assert c["preset"] == "mars_far" and c["enabled"] and c["links"]["mars_transit"]["one_way_s"] == 1337.0
    assert all(e["end_s"] is None or e["end_s"] > e["start_s"] for e in c["los_events"])
    assert r["arms"]["federated"]["sim_wallclock_s_total"] == c["sim_end_s"]
    assert isinstance(c["drops"], list) and c["buffer"]["ttl_s"] == 21600.0


def test_fedbuff_runs_matched_updates_in_arrival_order(runs):
    f = runs["buff"]["arms"]["federated"]
    assert len(f["rounds"]) == f["target_aggregations"] == math.ceil(4 * 5 / 3)
    t = [r["sim_wallclock_s"] for r in f["rounds"]]
    assert all(a <= b for a, b in zip(t, t[1:]))
    assert all(len(r["participants"]) == 3 for r in f["rounds"])
    validate_result(runs["buff"])


def test_dark_silo_stalls_sync_without_crashing(runs):
    f = runs["dark_sync"]["arms"]["federated"]
    assert f["stalled"] and f["stalled"]["unreachable"] == ["mars_transit"]
    assert 0 < len(f["rounds"]) < 4
    assert "STALLED" in runs["dark_sync"]["notes"]


def test_dark_silo_with_deadline_keeps_training(runs):
    f = runs["dark_deadline"]["arms"]["federated"]
    assert not f["stalled"] and len(f["rounds"]) == 4
    later = [r for r in f["rounds"] if r["sim_wallclock_s"] > 9000 + 3600]
    assert later and all("mars_transit" not in r["participants"] for r in later)


def test_silo_dark_from_the_start_is_a_result_not_a_crash():
    r = _run({"comms.dark.silo": "mars_transit", "comms.dark.from_s": 0.0})
    f = r["arms"]["federated"]
    assert f["rounds"] == [] and f["stalled"]["round"] == 1
    assert f["final"]["accuracy"] is not None
    validate_result(r)


def test_short_ttl_drops_mars_messages_and_federation_continues():
    """TTL below the Mars-far light time: every Mars message expires; with a deadline the
    other silos carry on, and the drops are logged."""
    r = _run({"comms.ttl_s": 1000.0, "comms.sync_deadline_s": 3600.0})
    f, c = r["arms"]["federated"], r["comms"]
    assert len(f["rounds"]) == 4 and all("mars_transit" not in x["participants"] for x in f["rounds"])
    reasons = {(d["silo"], d["reason"]) for d in c["drops"]}
    assert ("mars_transit", "expired") in reasons
    assert c["buffer"] == {"supersede": True, "ttl_s": 1000.0, "cache_bytes": None}


def _schedule(rounds):
    return [(r["sim_wallclock_s"], r["participants"]) for r in rounds]


def test_dark_by_round_sync_goes_dark_when_that_round_is_sent(runs):
    f, ref = runs["round_sync"]["arms"]["federated"], runs["sync"]["arms"]["federated"]
    dark = runs["round_sync"]["comms"]["dark"]
    assert dark["from_round"] == 3 and dark["from_s"] == ref["rounds"][1]["sim_wallclock_s"]
    assert _schedule(f["rounds"]) == _schedule(ref["rounds"][:2])  # pilot schedule == real one
    assert f["stalled"]["round"] == 3 and f["stalled"]["unreachable"] == ["mars_transit"]
    validate_result(runs["round_sync"])


def test_dark_by_round_fedbuff_goes_dark_at_the_matched_share_of_updates(runs):
    f, ref = runs["round_buff"]["arms"]["federated"], runs["buff"]["arms"]["federated"]
    done = (3 - 1) * 5 // 3  # aggregations before the onset
    onset = runs["round_buff"]["comms"]["dark"]["from_s"]
    assert onset == ref["rounds"][done - 1]["sim_wallclock_s"]
    assert _schedule(f["rounds"][:done]) == _schedule(ref["rounds"][:done])
    assert not f["stalled"] and len(f["rounds"]) == f["target_aggregations"]
    # anything mars_transit sent before the onset has landed within one light time
    assert all("mars_transit" not in r["participants"] for r in f["rounds"] if r["sim_wallclock_s"] > onset + 1400)


def test_dark_silo_fedbuff_keeps_aggregating(runs):
    f = runs["dark_buff"]["arms"]["federated"]
    assert not f["stalled"] and len(f["rounds"]) == f["target_aggregations"]


def test_comms_run_is_byte_identical(tmp_path):
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        for sub in ("a", "b"):
            assert main(["run", str(SMOKE), "--seed", "0", "--output-dir", str(tmp_path / sub)]) == 0
    finally:
        del os.environ["SOURCE_DATE_EPOCH"]
    a, b = (next((tmp_path / s).glob("*/result.json")).read_bytes() for s in ("a", "b"))
    assert a == b
    assert json.loads(a)["comms"]["preset"] == "mars_far"


def test_partial_los_profile_override_keeps_other_defaults():
    cfg = load_config(SMOKE, {"comms.los_profiles.periodic.duration_s": 600.0})
    assert cfg.comms.los_profiles["periodic"].duration_s == 600.0
    assert cfg.comms.los_profiles["periodic"].period_s == 7200.0
    assert cfg.comms.los_profiles["long_blackout"].duration_s == 7200.0
