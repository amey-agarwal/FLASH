"""Flower ServerApp and the federated arm runner.

Round protocol (all traffic goes through Flower messages):
0. Setup: ``query.stats`` to every node; combine sufficient statistics into the global
   scaler (identical to fitting on pooled data); ``query.set_scaler`` to every node.
1..R. ``train`` to every node -> aggregate (strategy) -> score the new global model on
   the global holdout (server side) -> ``evaluate`` to every node (global model on each
   silo's own data).

Per round and per silo the result records:
- ``update_on_holdout``: that silo's post-training local update, scored on the holdout.
- ``global_on_local``: the aggregated global model, scored on that silo's training
  data (optimistic: that data was trained on).
- ``train_loss`` and ``n_train``.

Links (Phase 6, fl/comms.py). With ``comms.enabled`` off, links are synchronous and
lossless and ``sim_wallclock_s`` is null. With it on, every exchange is also timed in
SIMULATED seconds by the link model (delay, jitter, bandwidth, loss of signal), and:
- synchronous strategies wait for the last update, or drop silos that miss
  ``comms.sync_deadline_s``; without a deadline, a silo that can never be reached
  stalls the federation and the run stops there (recorded in ``stalled``);
- ``fedbuff`` aggregates in simulated arrival order, every ``fl.buffer_k`` updates;
- a silo forced dark by round (``comms.dark.from_round``) gets its onset time from a
  timing-only pilot of the same run (``_dark_onset_s``).
Flower still carries every real message; the real Flower exchange itself is always
synchronous and lossless, and a real failed reply is still an error. Only the
participants' updates are evaluated on their local data (diagnostic, not timed).
"""

from __future__ import annotations

import heapq
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.simulation import run_simulation

from fedastro.baselines.common import ARM_IDS, holdout_evaluator, init_model
from fedastro.config import ExperimentConfig
from fedastro.data.partition import Federation
from fedastro.data.preprocess import Standardizer, Stats
from fedastro.fl.client import get_weights, make_client_app, set_weights
from fedastro.fl.comms import INF, build_links, compute_s, dark_window, describe_links, link_drops, los_events
from fedastro.fl.strategy import BufferedUpdate, ClientUpdate, FedBuff, build_strategy
from fedastro.metrics.classification import METRICS
from fedastro.metrics.comms import payload_bytes, rounds_to_target
from fedastro.models.model import count_parameters
from fedastro.training import derive_seed


def _wait_for_nodes(grid: Grid, n: int, timeout_s: float) -> list[int]:
    deadline = time.monotonic() + timeout_s
    while len(nodes := sorted(grid.get_node_ids())) < n:
        if time.monotonic() > deadline:
            raise RuntimeError(f"only {len(nodes)}/{n} simulated silos came online")
        time.sleep(0.1)
    return nodes


def _exchange(grid: Grid, nodes: list[int], message_type: str, content: RecordDict,
              timeout_s: float) -> list[Message]:
    """Send to every node and return replies sorted by silo index. Any missing or
    failed reply raises (no silent drop-outs in the synchronous, lossless setting)."""
    messages = [Message(content, dst_node_id=n, message_type=message_type) for n in nodes]
    replies = list(grid.send_and_receive(messages, timeout=timeout_s))
    errors = [r for r in replies if r.has_error()]
    if errors:
        reasons = "; ".join(f"node {r.metadata.src_node_id}: {r.error.reason}" for r in errors)
        raise RuntimeError(f"{message_type}: client error(s): {reasons}")
    if len(replies) != len(nodes):
        raise RuntimeError(f"{message_type}: {len(replies)}/{len(nodes)} replies within {timeout_s}s")
    return sorted(replies, key=lambda r: int(r.content["metrics"]["silo_index"]))


def _opt_float(rec, key: str) -> float | None:
    return float(rec[key]) if key in rec else None


def _metrics_from_record(rec) -> dict[str, float | None]:
    return {m: (float(rec[m]) if m in rec else None) for m in METRICS}


class _Sim:
    """Simulated clock and link model for one run; inert (no time) when comms is off."""

    def __init__(self, cfg: ExperimentConfig, seed: int, fed: Federation):
        self.cfg = cfg
        self.on = cfg.comms.enabled
        self.links = build_links(cfg, seed) if self.on else {}
        self.n_train = {name: len(split.data) for name, split in fed.silos.items()}
        self.real_sleep = cfg.real_sleep_scale

    def round_trip(self, silo: str, t: float, down_bytes: int, up_bytes: int, train: bool) -> tuple[float, float]:
        """(download arrival, upload arrival) for a message sent to ``silo`` at ``t``."""
        link = self.links[silo]
        down = link.transfer(t, down_bytes, "down")
        if down == INF:
            return INF, INF
        ready = down + (compute_s(self.cfg, self.n_train[silo]) if train else 0.0)
        return down, link.transfer(ready, up_bytes, "up")

    def sleep(self, dt: float) -> None:
        if self.real_sleep > 0 and 0 < dt < INF:
            time.sleep(dt * self.real_sleep)


def make_server_app(fed: Federation, cfg: ExperimentConfig, seed: int, device: str,
                    out: dict[str, Any]) -> ServerApp:
    app = ServerApp()
    silo_names = list(fed.silos)
    first = fed.silos[silo_names[0]].data
    input_shape, n_classes = first.X.shape[1:], len(first.class_names)
    timeout = cfg.fl.timeout_s
    sim = _Sim(cfg, seed, fed)

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        try:
            nodes = _wait_for_nodes(grid, len(silo_names), timeout)

            # --- setup: federated standardisation -------------------------------
            replies = _exchange(grid, nodes, "query.stats", RecordDict({"config": ConfigRecord({})}), timeout)
            node_of = {int(r.content["metrics"]["silo_index"]): r.metadata.src_node_id for r in replies}
            if sorted(node_of) != list(range(len(silo_names))):
                raise RuntimeError(f"silo indices reported by nodes: {sorted(node_of)}")
            stats_list, setup_bytes, stats_bytes = [], 0, 0
            for r in replies:
                total, total_sq = r.content["arrays"].to_numpy_ndarrays()
                stats_list.append(Stats(float(r.content["metrics"]["count"]), total, total_sq))
                stats_bytes = payload_bytes([total, total_sq])
                setup_bytes += stats_bytes
            combined = stats_list[0]
            for st in stats_list[1:]:
                combined = combined + st
            scaler = Standardizer.from_stats(combined)
            scaler_arrays = [scaler.mean, scaler.std]
            _exchange(grid, nodes, "query.set_scaler", RecordDict({"arrays": ArrayRecord(scaler_arrays)}), timeout)
            setup_bytes += payload_bytes(scaler_arrays) * len(nodes)

            ordered_nodes = [node_of[i] for i in range(len(silo_names))]
            model = init_model(cfg, input_shape, n_classes, derive_seed(seed, ARM_IDS["federated"]))
            weights = get_weights(model)
            t0, dark_from_s = 0.0, None
            if sim.on:
                dark = cfg.comms.dark
                if dark.from_round:
                    dark_from_s = _dark_onset_s(cfg, seed, fed, silo_names, weights, stats_bytes,
                                                payload_bytes(scaler_arrays))
                    if dark_from_s is not None:  # set before any simulated transfer
                        sim.links[dark.silo].forced = dark_window(cfg.comms, dark_from_s)
                elif dark.silo:
                    dark_from_s = dark.from_s
                t0 = _setup_sim_s(sim, silo_names, stats_bytes, payload_bytes(scaler_arrays))
                sim.sleep(t0)
            out["global_weights"] = {"init": weights}
            out["scaler"] = scaler
            holdout_eval = holdout_evaluator(fed.holdout, scaler, device)
            ctx = _RunCtx(grid, ordered_nodes, silo_names, model, holdout_eval, timeout, sim)
            out["init_metrics"] = ctx.score_global(weights)
            if cfg.fl.strategy == "fedbuff":
                rounds, weights, extra = _run_fedbuff(ctx, cfg, weights, t0)
            else:
                rounds, weights, extra = _run_sync(ctx, cfg, weights, t0)
            out["global_weights"]["final"] = weights
            out.update(rounds=rounds, payload_bytes=payload_bytes(weights), setup_bytes=setup_bytes,
                       n_params=count_parameters(model), setup_sim_s=t0 if sim.on else None,
                       dark_from_s=dark_from_s, sim=sim, **extra)
        except BaseException as e:  # surfaced by run_federated; Flower would only log it
            out["error"] = e
            raise

    return app


@dataclass
class _RunCtx:
    grid: Grid
    nodes: list[int]  # Flower node id per silo index
    silo_names: list[str]
    model: Any
    holdout_eval: Any
    timeout: float
    sim: _Sim

    def train(self, idxs: list[int], weights: list[np.ndarray], dispatch_no: dict[int, int]) -> list[ClientUpdate]:
        """Real local training on the given silos via Flower (one message each).
        ``dispatch_no`` is each silo's dispatch counter, sent as "server-round" (it
        seeds the client's training; in synchronous runs it equals the round)."""
        messages = [Message(RecordDict({"arrays": ArrayRecord(weights),
                                        "config": ConfigRecord({"server-round": dispatch_no[i]})}),
                            dst_node_id=self.nodes[i], message_type=MessageType.TRAIN) for i in idxs]
        replies = list(self.grid.send_and_receive(messages, timeout=self.timeout))
        errors = [r for r in replies if r.has_error()]
        if errors or len(replies) != len(idxs):
            reasons = "; ".join(f"node {r.metadata.src_node_id}: {r.error.reason}" for r in errors)
            raise RuntimeError(f"train: {len(replies)}/{len(idxs)} replies; {reasons}")
        ups = []
        for r in sorted(replies, key=lambda r: int(r.content["metrics"]["silo_index"])):
            m = r.content["metrics"]
            idx = int(m["silo_index"])
            ups.append(ClientUpdate(
                silo_index=idx, silo=self.silo_names[idx], arrays=r.content["arrays"].to_numpy_ndarrays(),
                num_examples=int(m["num-examples"]), train_loss=_opt_float(m, "train_loss"),
                epsilon=_opt_float(m, "epsilon"),
            ))
        return ups

    def score_update(self, upd: ClientUpdate) -> dict[str, Any]:
        set_weights(self.model, upd.arrays)
        entry = {"n_train": upd.num_examples, "train_loss": upd.train_loss,
                 "update_on_holdout": self.holdout_eval(self.model)}
        if upd.epsilon is not None:
            entry["epsilon"] = upd.epsilon
        return entry

    def score_global(self, weights: list[np.ndarray]) -> dict[str, Any]:
        set_weights(self.model, weights)
        return self.holdout_eval(self.model)

    def evaluate(self, idxs: list[int], weights: list[np.ndarray], rnd: int) -> dict[str, dict]:
        """The global model on each listed silo's own data (diagnostic, not timed)."""
        content = RecordDict({"arrays": ArrayRecord(weights), "config": ConfigRecord({"server-round": rnd})})
        replies = _exchange(self.grid, [self.nodes[i] for i in idxs], MessageType.EVALUATE, content, self.timeout)
        return {self.silo_names[int(r.content["metrics"]["silo_index"])]: _metrics_from_record(r.content["metrics"])
                for r in replies}


@dataclass
class _TimingCtx:
    """Stands in for _RunCtx in a timing-only pilot: no Flower traffic, no training, no
    scoring. Simulated time depends only on the links, payload sizes and n_train, never
    on what is learned, so the pilot reproduces the real run's schedule."""

    silo_names: list[str]
    sim: _Sim

    def train(self, idxs: list[int], weights: list[np.ndarray], dispatch_no: dict[int, int]) -> list[ClientUpdate]:
        return [ClientUpdate(silo_index=i, silo=self.silo_names[i], arrays=weights,
                             num_examples=self.sim.n_train[self.silo_names[i]], train_loss=None) for i in idxs]

    def score_update(self, upd: ClientUpdate) -> dict[str, Any]:
        return {}

    def score_global(self, weights: list[np.ndarray]) -> dict[str, Any]:
        return {}

    def evaluate(self, idxs: list[int], weights: list[np.ndarray], rnd: int) -> dict[str, dict]:
        return {}


def _setup_sim_s(sim: _Sim, silo_names: list[str], stats_bytes: int, scaler_bytes: int) -> float:
    """Simulated setup: stats request -> reply, then scaler -> ack, per silo. The
    federation starts when every reachable silo has its scaler."""
    ends = [sim.round_trip(n, 0.0, 64, stats_bytes, train=False)[1] for n in silo_names]
    t_scaler = max([e for e in ends if e < INF], default=0.0)
    acks = [sim.round_trip(n, t_scaler, scaler_bytes, 64, train=False)[0] for n in silo_names]
    return max([a for a in acks if a < INF], default=t_scaler)


def _dark_onset_s(cfg: ExperimentConfig, seed: int, fed: Federation, silo_names: list[str],
                  weights: list[np.ndarray], stats_bytes: int, scaler_bytes: int) -> float | None:
    """Simulated time at which ``comms.dark.from_round`` starts. Sync: when that round
    is sent. FedBuff: once (from_round - 1) x n_silos client updates have been
    aggregated, rounded down to whole aggregations of K (the same share of the matched
    updates). Taken from a timing-only pilot of this run without the outage: nothing
    that happens before the onset can depend on it, so the pilot's schedule up to the
    onset is the real run's. None = the pilot stopped first, so the silo never goes dark."""
    pilot = _Sim(cfg, seed, fed)
    pilot.real_sleep = 0.0
    t0 = _setup_sim_s(pilot, silo_names, stats_bytes, scaler_bytes)
    ctx = _TimingCtx(silo_names, pilot)
    r = cfg.comms.dark.from_round
    if cfg.fl.strategy == "fedbuff":
        rounds, _, _ = _run_fedbuff(ctx, cfg, weights, t0)
        done = (r - 1) * len(silo_names) // cfg.fl.buffer_k
    else:
        rounds, _, _ = _run_sync(ctx, cfg, weights, t0)
        done = r - 1
    if done == 0:
        return t0
    return rounds[done - 1]["sim_wallclock_s"] if len(rounds) >= done else None


def _run_sync(ctx: _RunCtx, cfg: ExperimentConfig, weights: list[np.ndarray], t0: float):
    """Synchronous FedAvg/FedProx. With comms on, a round lasts until the last
    participating update arrives (or the deadline). Without a deadline, a silo that
    can never be reached stalls the federation: the run stops there and says so."""
    strategy = build_strategy(cfg.fl)
    sim, n = ctx.sim, len(ctx.silo_names)
    payload = payload_bytes(weights)
    deadline = cfg.comms.sync_deadline_s if sim.on else 0.0
    rounds, t, stalled = [], t0, None
    for rnd in range(1, cfg.fl.rounds + 1):
        if sim.on:
            trips = [sim.round_trip(name, t, payload, payload, train=True) for name in ctx.silo_names]
            limit = t + deadline if deadline > 0 else INF
            idxs = [i for i, (_, up) in enumerate(trips) if up <= limit and up < INF]
            if deadline == 0 and len(idxs) < n:
                stalled = {"round": rnd, "sim_wallclock_s": t,
                           "unreachable": [ctx.silo_names[i] for i in range(n) if i not in idxs],
                           "reason": "no sync deadline and a silo's message will never arrive "
                                     "(unreachable, expired or dropped)"}
                break
            t_end = max(trips[i][1] for i in idxs) if len(idxs) == n else limit
            delivered = sum(down < INF for down, _ in trips)
        else:
            idxs, t_end, delivered = list(range(n)), None, n
        per_silo: dict[str, Any] = {}
        if idxs:
            updates = ctx.train(idxs, weights, {i: rnd for i in idxs})
            for upd in updates:
                per_silo[upd.silo] = ctx.score_update(upd)
            weights = strategy.aggregate(updates)
        global_metrics = ctx.score_global(weights)
        if idxs:
            for silo, metrics in ctx.evaluate(idxs, weights, rnd).items():
                per_silo[silo]["global_on_local"] = metrics
        if sim.on:
            sim.sleep(t_end - t)
            t = t_end
        rounds.append({
            "round": rnd, "global": global_metrics, "per_silo": per_silo,
            "bytes": payload * (delivered + len(idxs)),
            "sim_wallclock_s": t_end,
            "participants": [ctx.silo_names[i] for i in idxs],
        })
    return rounds, weights, {"stalled": stalled, "sim_end_s": t if sim.on else None}


def _run_fedbuff(ctx: _RunCtx, cfg: ExperimentConfig, weights: list[np.ndarray], t0: float):
    """Buffered asynchronous aggregation (FedBuff). Each silo trains on the latest
    model it was sent and uploads; updates are queued by simulated arrival time; every
    ``buffer_k`` arrivals the server applies the staleness-weighted update and sends
    each just-arrived silo the current model. A silo that can never be reached simply
    stops contributing. Runs ceil(rounds * n_silos / buffer_k) aggregations."""
    strategy = FedBuff(cfg.fl.server_lr)
    sim, n, k = ctx.sim, len(ctx.silo_names), cfg.fl.buffer_k
    payload = payload_bytes(weights)
    target = math.ceil(cfg.fl.rounds * n / k)
    version, seq = 0, 0
    dispatches = {i: 0 for i in range(n)}
    queue: list[tuple[float, int, int, ClientUpdate, list[np.ndarray], int]] = []
    lost: list[dict] = []

    def dispatch(idxs: list[int], t: float) -> None:
        nonlocal seq
        reach = []
        for i in idxs:
            _, up = sim.round_trip(ctx.silo_names[i], t, payload, payload, train=True)
            if up == INF:
                lost.append({"silo": ctx.silo_names[i], "sent_s": t})
            else:
                reach.append((i, up))
        if not reach:
            return
        for i, _ in reach:
            dispatches[i] += 1
        updates = {u.silo_index: u for u in ctx.train([i for i, _ in reach], weights, dispatches)}
        for i, up in reach:
            seq += 1
            heapq.heappush(queue, (up, i, seq, updates[i], weights, version))

    dispatch(list(range(n)), t0)
    rounds, buffer, t = [], [], t0
    while len(rounds) < target and queue:
        t, i, _, upd, start, v_start = heapq.heappop(queue)
        buffer.append(BufferedUpdate(upd, start, version - v_start))
        if len(buffer) == k:
            weights = strategy.aggregate(weights, buffer)
            version += 1
            per_silo: dict[str, Any] = {}
            for j, b in enumerate(buffer):
                per_silo[f"{b.update.silo}#{j}" if b.update.silo in per_silo else b.update.silo] = {
                    **ctx.score_update(b.update), "staleness": b.staleness}
            prev = rounds[-1]["sim_wallclock_s"] if rounds else t0
            sim.sleep(t - prev)
            rounds.append({
                "round": version, "global": ctx.score_global(weights), "per_silo": per_silo,
                "bytes": payload * 2 * k, "sim_wallclock_s": t,
                "participants": [b.update.silo for b in buffer],
            })
            buffer = []
        dispatch([i], t)
    stalled = None
    if len(rounds) < target:
        stalled = {"round": len(rounds) + 1, "sim_wallclock_s": t,
                   "reason": "no reachable silo left to deliver updates"}
    return rounds, weights, {"stalled": stalled, "sim_end_s": t, "lost_dispatches": lost,
                             "dispatches": {ctx.silo_names[i]: d for i, d in dispatches.items()},
                             "target_aggregations": target}


def run_federated(fed: Federation, cfg: ExperimentConfig, seed: int, device: str,
                  centralized_final: dict | None = None,
                  capture: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the federated arm. If ``capture`` is a dict it receives the global weights
    sent at round 1 (``global_weights["init"]``) and after the last round
    (``["final"]``) plus the federated ``scaler``, for the privacy attack."""
    silos = [split.data for split in fed.silos.values()]
    out: dict[str, Any] = {}
    server_app = make_server_app(fed, cfg, seed, device, out)
    client_app = make_client_app(silos, cfg, seed, device)
    logging.getLogger("flwr").setLevel(logging.WARNING)
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=len(silos),
        backend_name="ray",
        backend_config={
            "client_resources": {"num_cpus": cfg.resources.client_num_cpus,
                                 "num_gpus": cfg.resources.client_num_gpus},
            "init_args": {"log_to_driver": False, "include_dashboard": False,
                          "logging_level": logging.ERROR},
        },
    )
    if "error" in out:
        raise RuntimeError("federated arm failed") from out["error"]
    if "rounds" not in out:
        raise RuntimeError("federated arm produced no result (ServerApp did not complete)")

    if capture is not None:
        capture.update(global_weights=out["global_weights"], scaler=out["scaler"])
        sim = out["sim"]
        if sim.on:
            capture["comms"] = {
                "preset": cfg.comms.preset, "los_events": los_events(sim.links, out["sim_end_s"]),
                "drops": link_drops(sim.links, out["sim_end_s"]),
                "buffer": {"supersede": cfg.comms.supersede, "ttl_s": cfg.comms.ttl_s or None,
                           "cache_bytes": cfg.comms.cache_bytes or None},
                "enabled": True, "los": cfg.comms.los, "links": describe_links(sim.links),
                "sync_deadline_s": cfg.comms.sync_deadline_s or None,
                "compute_s_per_example": cfg.comms.compute_s_per_example,
                "dark": {"silo": cfg.comms.dark.silo or None, "from_round": cfg.comms.dark.from_round or None,
                         "from_s": out["dark_from_s"], "duration_s": cfg.comms.dark.duration_s or None},
                "setup_sim_s": out["setup_sim_s"], "sim_end_s": out["sim_end_s"],
                "time_unit": "simulated seconds (never real elapsed time)",
            }
    rounds = out["rounds"]
    targets = {}
    if centralized_final is not None and centralized_final.get("accuracy") is not None:
        targets["relative_to_centralized"] = {
            "fraction": cfg.fl.relative_target,
            **rounds_to_target(rounds, cfg.fl.relative_target * centralized_final["accuracy"]),
        }
    if cfg.fl.absolute_target > 0:
        targets["absolute"] = rounds_to_target(rounds, cfg.fl.absolute_target)
    result = {
        "rounds": rounds,
        # A run that stalled before its first round reports the untrained global model.
        "final": dict(rounds[-1]["global"] if rounds else out["init_metrics"]),
        "strategy": cfg.fl.strategy,
        "proximal_mu": cfg.fl.proximal_mu if cfg.fl.strategy == "fedprox" else None,
        "local_epochs": cfg.fl.local_epochs,
        "n_params": out["n_params"],
        "payload_bytes": out["payload_bytes"],
        "setup_bytes": out["setup_bytes"],
        "total_bytes": out["setup_bytes"] + sum(r["bytes"] for r in rounds),
        "rounds_to_target": targets,
        "stalled": out["stalled"],
        "sim_wallclock_s_total": out["sim_end_s"],
    }
    for key in ("dispatches", "lost_dispatches", "target_aggregations"):
        if key in out:
            result[key] = out[key]
    if cfg.privacy.enabled:
        from fedastro.baselines.common import dp_plan
        from fedastro.privacy.dp import records_per_subject

        total = cfg.fl.rounds * cfg.fl.local_epochs
        result["privacy"] = {
            name: dp_plan(cfg, len(split.data), total).to_dict(records_per_subject(split.data.subject_id))
            for name, split in fed.silos.items()
        }
    return result
