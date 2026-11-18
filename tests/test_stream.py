"""Phase H7: test-then-train evaluation (data/stream.py, fl/stream.py, metrics/stream.py)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from fedastro.baselines.common import init_model
from fedastro.config import ConfigError, StreamConfig, load_config
from fedastro.data.base import SYNTHETIC, DatasetBundle, Provenance
from fedastro.data.stream import NEVER, make_stream, make_streams
from fedastro.fl.client import get_weights, set_weights
from fedastro.fl.hierarchy import HierSim, run_rounds
from fedastro.fl.stream import StreamRun
from fedastro.fl.strategy import ClientUpdate
from fedastro.data.preprocess import Standardizer
from fedastro.metrics.schema import validate_result
from fedastro.metrics.stream import stream_report
from fedastro.pipeline import build_hier_federation, run_experiment
from fedastro.training import predict_proba

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "configs" / "experiments"
SMOKE = EXP / "smoke_hier_stream.yaml"


def _bundle(n_patients=10, per_patient=6, seed=0) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n = n_patients * per_patient
    prov = Provenance(dataset="SYNTHETIC", version="t", checksum="x", source="test", kind=SYNTHETIC)
    return DatasetBundle(X=rng.normal(size=(n, 3)).astype(np.float32), y=rng.integers(0, 2, n).astype(np.int64),
                         subject_id=np.repeat(np.arange(n_patients), per_patient).astype(np.int64),
                         class_names=("a", "b"), provenance=prov)


# -- config ---------------------------------------------------------------------------------


def test_config_rules():
    cfg = load_config(SMOKE)
    assert "stream" in cfg.experiment_dict()
    with pytest.raises(ConfigError, match="needs the 'hierarchical' arm"):
        load_config(SMOKE, {"arms": ["local_only", "hierarchical_stream"]})
    with pytest.raises(ConfigError, match="fl.rounds >= 2"):
        load_config(SMOKE, {"fl.rounds": 1, "hier_fl.global_every": 1})
    with pytest.raises(ConfigError, match="only meaningful with the 'hierarchical_stream' arm"):
        load_config(SMOKE, {"arms": ["local_only", "hierarchical"], "stream.label_fraction": 0.5})
    with pytest.raises(ConfigError, match="by_source nodes"):
        load_config(SMOKE, {"stream.order": "time"})
    for bad in ({"stream.label_fraction": 0.0}, {"stream.label_delay_rounds": -1},
                {"stream.warmup_frac": 1.0}, {"stream.order": "random"}):
        with pytest.raises(ConfigError):
            load_config(SMOKE, bad)
    # without the arm the block is out of the run id, so earlier run ids are unchanged
    plain = load_config(EXP / "smoke_hier_updates.yaml")
    assert "stream" not in plain.experiment_dict()


# -- schedule -------------------------------------------------------------------------------


@pytest.mark.parametrize("p,d", [(0.1, 1), (0.5, 0), (1.0, 0), (0.3, 3)])
def test_schedule_counts_labels_and_delays(p, d):
    data = _bundle()
    R = 6
    s = make_stream(data, StreamConfig(label_fraction=p, label_delay_rounds=d), "time", R, seed=3, node_index=2)
    n = len(data)
    assert sorted(s.order.tolist()) == list(range(n))
    n_w = max(1, round(0.1 * n))
    assert s.n_warmup == n_w and (s.usable_from[s.arrival == 0] == 1).all()
    assert set(np.unique(s.arrival)) <= set(range(R + 1)) and not (s.arrival == 1).any()
    arrived = labelled = 0
    for r in range(2, R + 1):
        chunk = s.arriving(r)
        assert (s.arrival[chunk] == r).all()
        arrived += len(chunk)
        lab = chunk[s.usable_from[chunk] < NEVER]
        labelled += len(lab)
        assert labelled == int(np.floor(p * arrived + 0.5))  # cumulative rounding
        assert (s.usable_from[lab] == r + d).all()
    assert arrived == n - n_w
    # deterministic in the seed, and the seed matters
    again = make_stream(data, StreamConfig(label_fraction=p, label_delay_rounds=d), "time", R, 3, 2)
    assert np.array_equal(again.usable_from, s.usable_from)
    if p < 1:
        other = make_stream(data, StreamConfig(label_fraction=p, label_delay_rounds=d), "time", R, 4, 2)
        assert not np.array_equal(other.usable_from, s.usable_from)


def test_a_small_fraction_still_yields_labels_over_the_stream():
    s = make_stream(_bundle(5, 6), StreamConfig(label_fraction=0.1), "time", 11, 0, 0)
    sizes = s.summary()["chunk_sizes"]
    assert max(sizes) * 0.1 < 0.5  # no single chunk would round to one label
    assert s.summary()["n_stream_labelled"] == round(0.1 * sum(sizes)) > 0


def test_full_labels_zero_delay_trains_on_everything_by_the_last_round():
    s = make_stream(_bundle(), StreamConfig(label_fraction=1.0, label_delay_rounds=0), "time", 5, 0, 0)
    assert s.train_mask(5).all()
    stream = s.arrival > 0
    assert (s.usable_from[stream] == s.arrival[stream]).all()


def test_arrival_orders():
    data = _bundle()
    t = make_stream(data, StreamConfig(), "time", 4, 0, 0)
    assert np.array_equal(t.order, np.arange(len(data)))
    sh = make_stream(data, StreamConfig(), "patient_shuffle", 4, 0, 0)
    pid = data.subject_id[sh.order]
    assert (np.diff(pid) != 0).sum() == len(np.unique(pid)) - 1  # each patient's windows stay together
    assert not np.array_equal(sh.order, t.order)
    for p in np.unique(pid):  # within a patient, index (time) order
        assert np.all(np.diff(sh.order[pid == p]) > 0)


def test_unlabelled_examples_are_never_trainable():
    s = make_stream(_bundle(), StreamConfig(label_fraction=0.2), "time", 6, 0, 0)
    never = s.usable_from == NEVER
    assert never.any()
    assert not any(s.train_mask(r)[never].any() for r in range(1, 50))


# -- the stream hooks, Flower-free -------------------------------------------------------------


def _hooked(overrides=None):
    """run_rounds over the smoke federation with a fake trainer that emulates the clients:
    each node 'trains' on exactly the labels its schedule allows and returns its weights."""
    cfg = load_config(SMOKE, {"fl.rounds": 6, "train.epochs": 1, **(overrides or {})})
    _, hfed, _ = build_hier_federation(cfg, 0)
    names = [n.name for n in cfg.hierarchy.nodes]
    train = {n: hfed.nodes[n].train for n in names}
    streams = make_streams(cfg, train, 0)
    first = train[names[0]]
    model = init_model(cfg, first.X.shape[1:], len(first.class_names), 0)
    warm = np.concatenate([train[n].X[streams[n].arrival == 0] for n in names])
    scaler = Standardizer.fit(warm)

    def fed_proba(w, X):
        set_weights(model, w)
        return predict_proba(model, X)

    st = StreamRun(cfg, 0, "cpu", train, streams, scaler, fed_proba)
    seen = []

    def fake_train(dispatch, rnd):
        out = {}
        for i, w in dispatch.items():
            k = int(streams[names[i]].train_mask(rnd).sum())
            seen.append((rnd, names[i], k))
            out[i] = ClientUpdate(silo_index=i, silo=names[i], arrays=[a + 0.01 * rnd for a in w],
                                  num_examples=k, train_loss=None)
        return out

    deliveries = []
    orig = st.deliver
    st.deliver = lambda track, w, rec: (deliveries.append((st.last_round, track, tuple(rec))), orig(track, w, rec))
    n_train = {n: len(train[n]) for n in names}
    sched = run_rounds(cfg, fake_train, get_weights(model), n_train, HierSim(cfg, 0, n_train), 0.0,
                       lambda w, r: {"accuracy": 0.0}, stream=st)
    return cfg, streams, st, sched, seen, deliveries


@pytest.fixture(scope="module")
def hooked():
    return _hooked()


def test_no_example_is_scored_after_a_model_trained_on_it(hooked):
    cfg, streams, st, _, seen, _ = hooked
    audit = st.audit()
    assert audit["ok"] and audit["n_scored_after_local_training"] == 0 and audit["n_scored_after_client_training"] == 0
    assert audit["n_scored"] == sum(int((s.arrival > 0).sum()) for s in streams.values())
    # every example scored exactly once, in its arrival round
    for n, recs in st.records.items():
        idx = np.concatenate([r["idx"] for r in recs])
        assert len(idx) == len(np.unique(idx)) == int((streams[n].arrival > 0).sum())
        for r in recs:
            assert (streams[n].arrival[r["idx"]] == r["round"]).all()
    # the (emulated) clients train on exactly the schedule's usable labels
    assert all(k == int(streams[n].train_mask(r).sum()) for r, n, k in seen)


def test_the_audit_is_not_vacuous():
    _, streams, st, _, _, _ = _hooked({"fl.rounds": 3, "hier_fl.global_every": 1, "hier_fl.offline": {}})
    n = next(iter(st.records))
    rec = st.records[n][0]
    st.trained_log[n].append((rec["round"] - 1, rec["idx"][:1]))  # pretend it was trained on earlier
    assert not st.audit()["ok"]


def test_models_a_node_has_not_received_score_nothing(hooked):
    cfg, _, st, _, _, _ = hooked
    tt = stream_report(st.records, {n.name: r.name for r in cfg.hierarchy.regions for n in r.nodes}, cfg.fl.rounds)
    r2 = tt["rounds"][0]
    assert r2["round"] == 2 and r2["n_scored"]["global"] == 0  # first merge is at the end of round 2
    assert r2["n_scored"]["local"] == r2["n_arrived"] > 0
    assert tt["rounds"][1]["n_scored"]["global"] > 0
    assert tt["disagreement"]["n_compared"] == tt["global"]["n_scored"]["global"]


def test_a_node_offline_keeps_its_older_model(hooked):
    cfg, _, st, sched, _, deliveries = hooked
    leo = [n.name for n in cfg.hierarchy.regions[0].nodes]
    assert "leo_roscosmos" in leo
    assert any(e["round"] == 3 and e["region"] == "leo" and e["kind"] == "fallback" for e in sched["events"])
    assert not any(r == 3 and t == "regional" and set(rec) & set(leo) for r, t, rec in deliveries)
    assert any(r == 2 and t == "regional" and "leo_roscosmos" in rec for r, t, rec in deliveries)


def test_merge_weights_follow_the_usable_labels(hooked):
    _, streams, st, _, _, _ = hooked
    assert st.n_train(1) == {n: s.n_warmup for n, s in streams.items()}
    assert all(st.n_train(6)[n] >= st.n_train(2)[n] for n in streams)


# -- end to end (Flower) ------------------------------------------------------------------------


@pytest.mark.slow
def test_stream_arm_end_to_end_leaves_the_fixed_split_results_unchanged():
    cfg = load_config(SMOKE)
    with_stream = run_experiment(cfg, 0)
    validate_result(with_stream)
    st = with_stream["stream"]
    assert st["enabled"] and st["audit"]["ok"] and st["order"] == "patient_shuffle"
    assert st["test_then_train"]["global"]["n_arrived"] == st["audit"]["n_scored"] > 0
    assert set(st["comparison"]["stream_final"]) == {"local", "regional", "global"}
    plain = run_experiment(dataclasses.replace(cfg, arms=("local_only", "hierarchical"),
                                               stream=StreamConfig()), 0)
    for key in ("arms", "hierarchy", "disagreement", "calibration", "model_report"):
        assert with_stream[key] == plain[key], key
    assert not plain["stream"]["enabled"]
