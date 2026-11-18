"""Hierarchical federated learning (CLAUDE.md Phase H3): nodes -> regional aggregators ->
one global aggregator.

Protocol, per regional round r = 1..fl.rounds (every exchange is a real Flower message):
1. Each region's aggregator sends its regional model to its nodes; reachable nodes train
   locally (the same client code and seeds as the flat federated arm) and reply.
2. If at least ``min_participants`` of a region's nodes reported, the regional model
   becomes their FedAvg (example-weighted). Otherwise the region **falls back**: its model
   for the round is the latest global model it has received, the round's node updates are
   discarded, and the event is logged. (Discarded updates are not computed: a client's
   training seed depends only on its index and the round, so skipping them changes nothing
   else.)
3. Every ``hier_fl.global_every`` rounds the global aggregator merges the regional models it
   receives, weighted by each region's training examples, and sends the result back. A
   region that receives it resets its regional model to it (client-edge-cloud hierarchical
   FedAvg). A region cut off at merge time is left out and keeps its regional model. A
   region with no successful regional round since it last received a global model is also
   left out: its model is that global model, so merging it would only pull the new global
   back toward the old one (logged as ``merge_skipped``).

With one region and ``global_every: 1`` this is exactly flat FedAvg (tested): the arm uses
the flat arm's initial model and client seed stream on purpose (common random numbers), so
the two arms differ only in how updates are combined.

Models evaluated per node on its own test set (``evaluate_hierarchy``):
- ``local``: the node's local-only model (its own scaler), from the local_only arm.
- ``regional``: the node's region's model before the last reset, i.e. after its final
  stretch of region-only training.
- ``global``: the last merged global model.
Regional and global models share the federated scaler (Phase H3 decision 2).

Simulated time (``comms.enabled``; else every time field is null): node links use the
region's ``node_link``, regional-to-global links its ``global_link`` (latency relative to
the receiving aggregator; Mars presets from ``comms.preset``), plus an optional
solar-conjunction blackout. Regions run synchronised rounds; a round ends when the last
region finishes (its last expected update, or ``hier_fl.region_deadline_s``). Time is
SIMULATED seconds, never real time.

Per-round regional and global scores on the test sets are diagnostics for the log and the
terminal view. They are never used to select anything (CLAUDE.md rule 9).

Phase H5 adds, without changing training or aggregation: each node's state per round
(``node_state``), a per-round test-set diagnostic (``diagnostic``), the version gate
(fl/versioning.py; each node's deployed regional and global versions) and a per-round
callback for the progress file read by ``fedastro peek --follow``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.simulation import run_simulation

from fedastro.baselines.common import ARM_IDS, init_model
from fedastro.config import ExperimentConfig, HierarchyConfig, LinkConfig, LOSProfile, SiloConfig
from fedastro.data.base import DatasetBundle
from fedastro.data.partition import HierFederation
from fedastro.data.preprocess import Standardizer
from fedastro.fl import secagg
from fedastro.fl.client import get_weights, make_client_app, set_weights
from fedastro.fl.comms import COMMS_ID, INF, Link, compute_s, los_events, resolve_latency
from fedastro.fl.server import _opt_float, federated_scaler
from fedastro.fl.strategy import ClientUpdate, FedAvg, build_server_opt, build_strategy
from fedastro.metrics.classification import evaluate
from fedastro.metrics.comms import payload_bytes
from fedastro.models.model import count_parameters
from fedastro.training import derive_seed, predict_proba

HIER_LINK_ID = 6  # derive_seed namespace for regional-to-global links (COMMS_ID = 5 is node links)


def _brief(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if k != "recall_per_class"}


# -- test sets ------------------------------------------------------------------------


def region_test_sets(fed: HierFederation, hierarchy: HierarchyConfig) -> dict[str, DatasetBundle]:
    """Each region's test data: its nodes' own test sets pooled, or (shared held-out set)
    the region's view of it, once."""
    from fedastro.data.base import concat_bundles

    out = {}
    for region in hierarchy.regions:
        names = [n.name for n in region.nodes]
        if fed.holdout is None:
            out[region.name] = concat_bundles([fed.nodes[n].test for n in names])
        else:
            out[region.name] = fed.test_of(names[0])
    return out


def pooled_test_set(fed: HierFederation, hierarchy: HierarchyConfig) -> DatasetBundle:
    """The test set of the flat arms on a hierarchical topology: every region's test data
    once (the shared held-out set once when it is not shifted)."""
    from fedastro.data.base import concat_bundles

    if fed.holdout is not None and not fed.region_holdout:
        return fed.holdout
    return concat_bundles(list(region_test_sets(fed, hierarchy).values()))


# -- simulated links ------------------------------------------------------------------


def _link(name: str, index: int, spec: LinkConfig, cfg: ExperimentConfig, seed: int, namespace: int) -> Link:
    c = cfg.comms
    as_silo = SiloConfig(name=name, data_share=1.0, one_way_latency_s=spec.one_way_latency_s,
                         los_profile=spec.los_profile, latency_presets_s=spec.latency_presets_s)
    return Link(silo=name, index=index, one_way_s=resolve_latency(as_silo, c.preset),
                bandwidth_bps=c.bandwidth_bps, jitter_s=c.jitter_s,
                profile=c.los_profiles[spec.los_profile] if c.los else LOSProfile(),
                los_rng=np.random.default_rng(derive_seed(seed, namespace, index, 1)),
                jitter_rng=np.random.default_rng(derive_seed(seed, namespace, index, 2)),
                supersede=c.supersede, ttl_s=c.ttl_s, cache_bytes=c.cache_bytes)


class HierSim:
    """Simulated clock for the hierarchy; inert when comms is off."""

    def __init__(self, cfg: ExperimentConfig, seed: int, n_train: dict[str, int]):
        self.cfg, self.on, self.n_train = cfg, cfg.comms.enabled, n_train
        h, hf = cfg.hierarchy, cfg.hier_fl
        self.node_links: dict[str, Link] = {}
        self.global_links: dict[str, Link] = {}
        if not self.on:
            return
        i = 0
        for j, region in enumerate(h.regions):
            for node in region.nodes:  # same seed stream as the flat arm's link of this node index
                self.node_links[node.name] = _link(node.name, i, region.node_link, cfg, seed, COMMS_ID)
                i += 1
            link = _link(region.aggregator, j, region.global_link, cfg, seed, HIER_LINK_ID)
            if hf.conjunction_region == region.name:
                link.forced = (hf.conjunction_from_s, hf.conjunction_from_s + hf.conjunction_duration_s)
            self.global_links[region.name] = link

    def node_round_trip(self, node: str, t: float, nbytes: int) -> tuple[float, float]:
        return self.node_round_trip_sized(node, t, nbytes, nbytes)

    def node_round_trip_sized(self, node: str, t: float, down_bytes: int, up_bytes: int) -> tuple[float, float]:
        """Model down, local training, update up: (arrival at the node, back at the aggregator)."""
        link = self.node_links[node]
        down = link.transfer(t, down_bytes, "down")
        if down == INF:
            return INF, INF
        return down, link.transfer(down + compute_s(self.cfg, self.n_train[node]), up_bytes, "up")

    def setup_s(self, stats_bytes: int, scaler_bytes: int) -> float:
        """Stats request -> reply, then scaler -> ack, over every node link."""
        ends = [l.transfer(l.transfer(0.0, 64, "down"), stats_bytes, "up") for l in self.node_links.values()]
        t = max([e for e in ends if e < INF], default=0.0)
        acks = [l.transfer(t, scaler_bytes, "down") for l in self.node_links.values()]
        return max([a for a in acks if a < INF], default=t)

    def events(self, until: float) -> list[dict]:
        return los_events({**self.node_links, **self.global_links}, until)


# -- the arm ----------------------------------------------------------------------------


@dataclass
class _Ctx:
    grid: Grid
    nodes: list[int]  # Flower node id per node index
    names: list[str]
    timeout: float

    frac_bits: int = 24

    def train(self, dispatch: dict[int, list[np.ndarray]], rnd: int,
              groups: dict[int, list[int]] | None = None) -> dict[int, Any]:
        """Train the listed nodes, each from its own (regional) weights, in one exchange.
        With ``groups`` (simulated secure aggregation) each node masks its update before
        replying, and only the masked vectors are returned: no unmasked node update ever
        reaches this process."""
        if not dispatch:
            return {}

        def config(i):
            c = {"server-round": rnd}
            if groups is not None:
                c.update({"secagg-group": [int(g) for g in groups[i]], "secagg-frac-bits": self.frac_bits})
            return ConfigRecord(c)

        messages = [Message(RecordDict({"arrays": ArrayRecord(w), "config": config(i)}),
                            dst_node_id=self.nodes[i], message_type=MessageType.TRAIN) for i, w in dispatch.items()]
        replies = list(self.grid.send_and_receive(messages, timeout=self.timeout))
        errors = [r for r in replies if r.has_error()]
        if errors or len(replies) != len(dispatch):
            reasons = "; ".join(f"node {r.metadata.src_node_id}: {r.error.reason}" for r in errors)
            raise RuntimeError(f"hierarchical train: {len(replies)}/{len(dispatch)} replies; {reasons}")
        out = {}
        for r in replies:
            m = r.content["metrics"]
            idx = int(m["silo_index"])
            if groups is not None:
                (vec,) = r.content["arrays"].to_numpy_ndarrays()
                if vec.dtype != np.uint64:
                    raise RuntimeError(f"secagg: node {idx} replied without masking")
                out[idx] = vec
                continue
            out[idx] = ClientUpdate(silo_index=idx, silo=self.names[idx], arrays=r.content["arrays"].to_numpy_ndarrays(),
                                    num_examples=int(m["num-examples"]), train_loss=_opt_float(m, "train_loss"))
        return out


@dataclass
class _Plan:
    """One region's round, decided before training."""

    parts: list[int]  # nodes whose update reaches the regional aggregator in time
    scripted: list[int]  # scripted offline
    fallback: bool
    nbytes: int  # every byte of the round on the node links (secagg overhead included)
    reason: str = ""  # fallback reason
    group: list[int] | None = None  # secagg mask group (None = secagg off at this level)
    secagg: dict[str, Any] | None = None
    lost: list[int] = field(default_factory=list)  # in loss of signal (an exchange could never complete)

    def node_states(self, names: list[str], idxs: list[int]) -> dict[str, str]:
        """Each node's state this round, for the terminal view: ``training`` (its update was
        aggregated), ``fallback`` (in contact, but its region fell back), ``in_transit`` (its
        exchange was still under way at the deadline) or ``loss_of_signal``."""
        out = {}
        for i in idxs:
            if i in self.scripted or i in self.lost:
                out[names[i]] = "loss_of_signal"
            elif i in self.parts or (self.fallback and i in (self.group or ())):
                out[names[i]] = "fallback" if self.fallback else "training"
            else:
                out[names[i]] = "in_transit"
        return out


def _rt(link, t: float, down_bytes: int, up_bytes: int, pause: float = 0.0) -> float:
    """Aggregator -> member -> aggregator; arrival back at the aggregator (or INF)."""
    down = link.transfer(t, down_bytes, "down")
    return INF if down == INF else link.transfer(down + pause, up_bytes, "up")


def _key_agreement(links: dict[int, Any], members: list[int], t: float,
                   limit: float) -> tuple[list[int], float, list[int]]:
    """Advertise keys, then share keys (two round trips, secagg steps 1-2). Returns the mask
    group (members that completed both by ``limit``), the time the group is formed and the
    members whose exchange could never complete (loss of signal). The
    aggregator moves on once the last member that will answer has answered (an oracle
    timeout: a real deployment needs a timeout here, which would only add time)."""
    if links is None:
        return list(members), t, []
    k1 = {i: _rt(links[i], t, secagg.REQUEST_BYTES, secagg.KEY_BYTES) for i in members}
    joined = [i for i in members if k1[i] <= limit]
    t1 = max([k1[i] for i in joined], default=t)
    g = len(joined)
    k2 = {i: _rt(links[i], t1, secagg.KEY_BYTES * g, secagg.SHARE_BYTES * (g - 1)) for i in joined}
    group = [i for i in joined if k2[i] <= limit]
    lost = [i for i in members if k1[i] == INF or (i in k2 and k2[i] == INF)]
    return group, max([k2[i] for i in group], default=t1), lost


def _recovery(links: dict[int, Any] | None, survivors: list[int], n_dropped: int, t: float) -> tuple[list[int], float]:
    """Secagg step 4: survivors send their shares of the dropped members' seeds."""
    if links is None:
        return list(survivors), t
    back = {i: _rt(links[i], t, secagg.REQUEST_BYTES, secagg.SHARE_BYTES * n_dropped) for i in survivors}
    ok = [i for i in survivors if back[i] < INF]
    return ok, max([back[i] for i in ok], default=t)


def _plan_plain(cfg, sim, names, idxs, scripted, t, payload) -> tuple[_Plan, float]:
    h, hf = cfg.hierarchy, cfg.hier_fl
    end = t
    if sim.on:
        trips = {i: sim.node_round_trip(names[i], t, payload) for i in idxs if i not in scripted}
        limit = t + hf.region_deadline_s
        parts = [i for i, (_, up) in trips.items() if up <= limit]
        delivered = sum(down < INF for down, _ in trips.values())
        end = max(trips[i][1] for i in parts) if len(parts) == len(idxs) else limit
        lost = [i for i, (_, up) in trips.items() if up == INF]
    else:
        parts = [i for i in idxs if i not in scripted]
        delivered = len(parts)
        lost = []
    fallback = len(parts) < h.min_participants
    reason = (f"{len(parts)} of {len(idxs)} nodes reported, fewer than min_participants = "
              f"{h.min_participants}; the region uses its latest global model")
    return _Plan(parts, scripted, fallback, payload * (delivered + len(parts)), reason, lost=lost), end


def _plan_secagg(cfg, sim, names, idxs, scripted, t, rnd, weights) -> tuple[_Plan, float]:
    """A regional round under simulated secure aggregation (module docstring of fl/secagg.py)."""
    h, hf, sa = cfg.hierarchy, cfg.hier_fl, cfg.secagg
    payload, masked = payload_bytes(weights), secagg.masked_bytes(weights)
    cand = [i for i in idxs if i not in scripted]
    drops = {i for i in cand if rnd in sa.dropout.get(names[i], ())}
    links = {i: sim.node_links[names[i]] for i in cand} if sim.on else None
    limit = t + hf.region_deadline_s if sim.on else INF
    group, t_group, lost = _key_agreement(links, cand, t, limit)
    g = len(group)
    nbytes = secagg.key_bytes(g, len(cand))
    info = {"group": [names[i] for i in group], "dropped": [], "recovered": [], "recovery": False}
    min_p = h.min_participants
    if g < min_p:
        reason = (f"mask group of {g} of {len(idxs)} nodes, fewer than min_participants = {min_p}; "
                  f"no masked round; the region uses its latest global model")
        info.update(overhead_bytes=nbytes, overhead_sim_s=(t_group - t) if sim.on else None)
        return _Plan([], scripted, True, nbytes, reason, group, info, lost), t_group
    end = t_group
    if sim.on:  # model + forwarded shares down, local training, masked update up
        trips = {i: sim.node_round_trip_sized(names[i], t_group, payload + secagg.SHARE_BYTES * (g - 1), masked)
                 for i in group}
        delivered = sum(down < INF for down, _ in trips.values())
        parts = [i for i in group if i not in drops and trips[i][1] <= limit]
        lost += [i for i in group if trips[i][1] == INF]
        end = max(trips[i][1] for i in parts) if len(parts) == len(idxs) else limit
    else:
        delivered = g
        parts = [i for i in group if i not in drops]
    base = payload * (delivered + len(parts))
    nbytes += base + len(parts) * (masked - payload)
    dropped = [i for i in group if i not in parts]
    fallback = len(parts) < min_p
    reason = (f"{len(parts)} of {len(idxs)} nodes reported, fewer than min_participants = {min_p}; "
              f"the region uses its latest global model")
    rec_s = 0.0
    if dropped and not fallback:
        answered, rec_end = _recovery(links, parts, len(dropped), end)
        nbytes += secagg.recovery_bytes(len(parts), len(dropped))
        rec_s, end = rec_end - end, rec_end
        info.update(recovery=True, dropped=[names[i] for i in dropped])
        if len(answered) < min_p:
            fallback = True
            reason = (f"dropout recovery: {len(answered)} survivors answered, fewer than "
                      f"min_participants = {min_p}; nothing unmasked; the region uses its latest global model")
        else:
            info["recovered"] = [names[i] for i in dropped]
    elif dropped:
        info["dropped"] = [names[i] for i in dropped]
    if sim.on:
        extra_tx = (masked - payload + secagg.SHARE_BYTES * (g - 1)) * 8.0 / cfg.comms.bandwidth_bps
        info["overhead_sim_s"] = (t_group - t) + rec_s + extra_tx
    else:
        info["overhead_sim_s"] = None
    info["overhead_bytes"] = nbytes - base
    lost += [i for i in drops if i in group and i not in lost]  # scripted: its masked upload never arrives
    return _Plan(parts, scripted, fallback, nbytes, reason, group, info, lost), end


def _opt_step(opt, current: list[np.ndarray], aggregate: list[np.ndarray]) -> list[np.ndarray]:
    return aggregate if opt is None else opt.step(current, aggregate)


def run_rounds(cfg: ExperimentConfig, train, weights: list[np.ndarray], n_train: dict[str, int],
               sim: HierSim, t0: float, score, seed: int = 0, gate=None, diag=None,
               on_round=None, stream=None) -> dict[str, Any]:
    """The regional/global schedule. ``train(dispatch, rnd)`` trains nodes (by index) from
    the given weights and returns their updates; with secure aggregation at the regional
    level it is called as ``train(dispatch, rnd, groups=...)`` and returns each node's masked
    vector instead (``fl/secagg.mask_update``, computed on the node). ``score(weights, region
    | None)`` gives diagnostic metrics. Kept free of Flower so the schedule itself is
    unit-testable.

    Optional hooks, none of which changes training or aggregation (Phase H5):
    - ``gate`` (fl/versioning.UpdateGate): offered each new regional version (to the nodes
      that took part) and each new global version (to the nodes in contact in the regions
      that received it); its accept/reject events go into the output's ``update_events``.
    - ``diag(regional, global | None)``: per-round test-set diagnostic for the log and the
      terminal view (``rec["diagnostic"]``), never used to select anything.
    - ``on_round(rec, update_events)``: called once each round is complete (progress file).
    - ``stream`` (Phase H7, fl/stream.StreamRun; the hierarchical_stream arm): at the start of
      each round ``stream.begin_round(rnd)`` predicts the examples arriving that round, then
      trains the local models; ``stream.n_train(rnd)`` gives each node's usable labels, which
      weight the global merge and drive simulated compute time; every new regional / global
      version is handed to ``stream.deliver`` with the same recipients as the gate's."""
    h, hf, sa = cfg.hierarchy, cfg.hier_fl, cfg.secagg
    sa_regional = sa.enabled and "regional" in sa.levels
    strategy = build_strategy(cfg.fl)
    # Phase H8: one server optimizer per regional aggregator (own state, kept across merges);
    # the global merge stays a weighted average.
    server_opt = {r.name: build_server_opt(cfg.fl) for r in h.regions}
    names = [n.name for n in h.nodes]
    region_idx = {r.name: [names.index(n.name) for n in r.nodes] for r in h.regions}
    region_examples = {r.name: sum(n_train[n.name] for n in r.nodes) for r in h.regions}
    payload = payload_bytes(weights)
    global_w = weights
    regional = {r.name: weights for r in h.regions}  # current regional model
    last_regional = dict(regional)  # regional model before its last reset (evaluated)
    received = {r.name: weights for r in h.regions}  # latest global model each region holds
    fresh = {r.name: False for r in h.regions}  # aggregated since it last received a global model
    rounds, events, t = [], [], t0
    n_merges = 0
    versions = {r.name: 0 for r in h.regions}  # regional versions published so far
    update_events: list[dict[str, Any]] = []
    for rnd in range(1, cfg.fl.rounds + 1):
        if stream is not None:
            counts = stream.n_train(rnd)
            region_examples = {r.name: sum(counts[n.name] for n in r.nodes) for r in h.regions}
            sim.n_train = counts
            stream.begin_round(rnd)
        round_updates: list[dict[str, Any]] = []
        rec: dict[str, Any] = {"round": rnd, "regions": {}, "global_aggregated": False}
        plan, dispatch, groups, t_end = {}, {}, {}, t
        for region in h.regions:
            idxs = region_idx[region.name]
            scripted = [i for i in idxs if rnd in hf.offline.get(names[i], ())]
            if sa_regional:
                p, end = _plan_secagg(cfg, sim, names, idxs, scripted, t, rnd, weights)
            else:
                p, end = _plan_plain(cfg, sim, names, idxs, scripted, t, payload)
            t_end = max(t_end, end)
            plan[region.name] = p
            if not p.fallback:
                dispatch.update({i: regional[region.name] for i in p.parts})
                if sa_regional:
                    groups.update({i: p.group for i in p.parts})
        updates = train(dispatch, rnd, groups=groups) if sa_regional else train(dispatch, rnd)
        for region in h.regions:
            p = plan[region.name]
            missed = [names[i] for i in region_idx[region.name] if i not in p.parts and i not in p.scripted]
            if p.fallback:
                regional[region.name] = received[region.name]
                fresh[region.name] = False
                events.append({"round": rnd, "region": region.name, "kind": "fallback", "reason": p.reason})
            elif sa_regional:
                out = secagg.unmask_sum({i: updates[i] for i in p.parts}, p.group, seed, "regional", rnd,
                                        sa.frac_bits, regional[region.name], h.min_participants)
                regional[region.name] = _opt_step(server_opt[region.name], regional[region.name], out.weights)
                fresh[region.name] = True
            else:
                regional[region.name] = _opt_step(server_opt[region.name], regional[region.name],
                                                  strategy.aggregate([updates[i] for i in p.parts]))
                fresh[region.name] = True
            if p.secagg is not None and p.secagg["recovery"]:
                events.append({"round": rnd, "region": region.name, "kind": "secagg_recovery",
                               "reason": f"dropout after masking: {', '.join(p.secagg['dropped'])}; "
                                         f"survivors sent their shares, masks rebuilt"
                                         + ("" if p.secagg["recovered"] else "; too few answered")})
            last_regional[region.name] = regional[region.name]
            rec["regions"][region.name] = {
                "participants": [names[i] for i in p.parts], "offline": [names[i] for i in p.scripted],
                "missed_deadline": missed, "fallback": p.fallback, "bytes": p.nbytes,
                "regional": _brief(score(regional[region.name], region.name)),
                "node_state": p.node_states(names, region_idx[region.name]),
            }
            if not p.fallback:
                versions[region.name] += 1
                if stream is not None:
                    stream.deliver("regional", regional[region.name], [names[i] for i in p.parts])
                if gate is not None:
                    round_updates += gate.offer(rnd, "regional", region.name, versions[region.name],
                                                regional[region.name], [names[i] for i in p.parts])
            if p.secagg is not None:
                rec["regions"][region.name]["secagg"] = p.secagg
        rec["global"] = None
        if rnd % hf.global_every == 0:
            t_end = _merge(cfg, rnd, rec, events, regional, received, fresh, region_examples, sim, t_end,
                           payload, seed)
            if rec["global_aggregated"]:
                n_merges += 1
                global_w = rec.pop("_weights")
                rec["global"] = _brief(score(global_w, None))
                rec["global_version"] = n_merges
                if stream is not None:
                    for r in rec["global_received"]:
                        stream.deliver("global", global_w, [names[i] for i in plan[r].parts])
                if gate is not None:
                    for r in rec["global_received"]:
                        round_updates += gate.offer(rnd, "global", r, n_merges, global_w,
                                                    [names[i] for i in plan[r].parts])
        rec["sim_wallclock_s"] = t_end if sim.on else None
        if diag is not None:
            rec["diagnostic"] = diag(last_regional, global_w if n_merges else None)
        t = t_end
        rounds.append(rec)
        update_events += round_updates
        if on_round is not None:
            on_round(rec, round_updates)
    out = {"rounds": rounds, "events": events, "global": global_w, "regional": last_regional,
           "n_merges": n_merges, "sim_end_s": t if sim.on else None}
    if gate is not None:
        out["update_events"] = update_events
    return out


def _merge_members_secagg(cfg, rnd, rec, events, regions, eligible, regional, region_examples, sim, t,
                          payload, seed) -> tuple[list[str], float, list[np.ndarray] | None, int]:
    """The global merge under simulated secure aggregation: key agreement over the
    regional-to-global links, masked regional models up, recovery if one is missing.
    Returns (merged regions, time, merged weights or None, bytes)."""
    hf, sa = cfg.hier_fl, cfg.secagg
    template = regional[regions[0]]
    masked_b = secagg.masked_bytes(template)
    ids = {r: regions.index(r) for r in regions}
    members = [ids[r] for r in eligible]
    links = {ids[r]: sim.global_links[r] for r in eligible} if sim.on else None
    limit = t + hf.global_deadline_s if (sim.on and hf.global_deadline_s) else INF
    group, t_group, _ = _key_agreement(links, members, t, limit)
    g = len(group)
    nbytes = secagg.key_bytes(g, len(members))
    info: dict[str, Any] = {"group": [regions[i] for i in group], "dropped": [], "recovered": [],
                            "recovery": False}
    rec["global_secagg"] = info
    drops = {ids[r] for r in eligible if rnd in sa.dropout.get(r, ())}
    if g < sa.global_min_group:
        info.update(overhead_bytes=nbytes, overhead_sim_s=(t_group - t) if sim.on else None)
        events.append({"round": rnd, "region": None, "kind": "secagg_group_too_small",
                       "reason": f"global mask group of {g}, fewer than secagg.global_min_group = "
                                 f"{sa.global_min_group}; no global merge this round"})
        return [], t_group, None, nbytes
    if sim.on:  # shares down to each regional aggregator, masked model up
        up = {i: _rt(links[i], t_group, secagg.SHARE_BYTES * (g - 1), masked_b) for i in group}
        parts = [i for i in group if i not in drops and up[i] <= limit]
        t_up = max([up[i] for i in parts], default=t_group)
    else:
        parts, t_up = [i for i in group if i not in drops], t
    nbytes += secagg.SHARE_BYTES * (g - 1) * g + masked_b * len(parts)
    dropped = [i for i in group if i not in parts]
    rec_s = 0.0
    ok = len(parts) >= sa.global_min_group
    if dropped and ok:
        answered, rec_end = _recovery(links, parts, len(dropped), t_up)
        nbytes += secagg.recovery_bytes(len(parts), len(dropped))
        rec_s, t_up = rec_end - t_up, rec_end
        info.update(recovery=True, dropped=[regions[i] for i in dropped])
        ok = len(answered) >= sa.global_min_group
        if ok:
            info["recovered"] = [regions[i] for i in dropped]
        events.append({"round": rnd, "region": None, "kind": "secagg_recovery",
                       "reason": f"global level: dropout after masking: {', '.join(info['dropped'])}; "
                                 f"survivors sent their shares" + ("" if ok else "; too few answered")})
    elif dropped:
        info["dropped"] = [regions[i] for i in dropped]
    info["overhead_bytes"] = nbytes - payload * len(parts)
    if sim.on:
        info["overhead_sim_s"] = (t_group - t) + rec_s + (
            max(sim.global_links[regions[i]].one_way_s for i in group)
            + (masked_b - payload + secagg.SHARE_BYTES * (g - 1)) * 8.0 / cfg.comms.bandwidth_bps)
    else:
        info["overhead_sim_s"] = None
    if not ok:
        events.append({"round": rnd, "region": None, "kind": "secagg_group_too_small",
                       "reason": f"{len(parts)} masked regional models arrived, fewer than "
                                 f"secagg.global_min_group = {sa.global_min_group}; no global merge this round"})
        return [], t_up, None, nbytes
    masked = {i: secagg.mask_update(regional[regions[i]], region_examples[regions[i]], i, group, seed,
                                    "global", rnd, sa.frac_bits) for i in parts}
    out = secagg.unmask_sum(masked, group, seed, "global", rnd, sa.frac_bits, template, sa.global_min_group)
    return [regions[i] for i in parts], t_up, out.weights, nbytes


def _merge(cfg, rnd, rec, events, regional, received, fresh, region_examples, sim, t, payload,
           seed: int = 0) -> float:
    """One global merge at simulated time ``t``; returns the time it completes."""
    h, hf, sa = cfg.hierarchy, cfg.hier_fl, cfg.secagg
    regions = [r.name for r in h.regions]
    cut = [r for r in regions if rnd in hf.region_offline.get(r, ())]
    stale = [r for r in regions if not fresh[r] and r not in cut]
    up = {r: INF for r in cut + stale}
    t_done, nbytes = t, 0
    merged_w = None
    if sa.enabled and "global" in sa.levels:
        eligible = [r for r in regions if r not in cut and r not in stale]
        merged, t, merged_w, nbytes = _merge_members_secagg(cfg, rnd, rec, events, regions, eligible, regional,
                                                            region_examples, sim, t, payload, seed)
        t_done = t
    elif sim.on:
        for r in regions:
            if r not in cut and r not in stale:
                up[r] = sim.global_links[r].transfer(t, payload, "up")
        limit = t + hf.global_deadline_s if hf.global_deadline_s else INF
        merged = [r for r in regions if up[r] < INF and up[r] <= limit]
        nbytes += payload * len(merged)
    else:
        merged = [r for r in regions if r not in cut and r not in stale]
    rec["global_participants"] = merged
    for r in regions:
        if r not in merged:
            why = ("scripted loss of signal" if r in cut else
                   "no regional update since its last global model" if r in stale else
                   "secure-aggregation dropout after masking" if r in rec.get("global_secagg", {}).get("dropped", ()) else
                   "its model did not reach the global aggregator in time")
            events.append({"round": rnd, "region": r, "kind": "merge_skipped",
                           "reason": f"left out of the global merge ({why}); keeps its regional model"})
    if not merged:
        rec["global_received"] = []
        rec["global_bytes"] = nbytes
        return t
    rec["global_aggregated"] = True
    secure = merged_w is not None  # already the unmasked sum (simulated secure aggregation)
    if not secure:
        merged_w = FedAvg().aggregate([
            ClientUpdate(silo_index=regions.index(r), silo=r, arrays=regional[r],
                         num_examples=region_examples[r], train_loss=None) for r in merged])
    rec["_weights"] = merged_w
    got = [r for r in regions if r not in cut]  # stale regions still receive the new global
    if sim.on:
        t_merge = t if secure else max(up[r] for r in merged)
        limit = t_merge + hf.global_deadline_s if hf.global_deadline_s else INF
        down = {r: sim.global_links[r].transfer(t_merge, payload, "down") for r in regions if r not in cut}
        got = [r for r, a in down.items() if a < INF and a <= limit]
        nbytes += payload * len(got)
        t_done = max([down[r] for r in got], default=t_merge)
        for r in merged:
            if r not in got:
                events.append({"round": rnd, "region": r, "kind": "global_missed",
                               "reason": "the merged global model did not reach the region in time; "
                                         "it keeps its regional model"})
    for r in got:
        received[r] = merged_w
        regional[r] = merged_w
        fresh[r] = False
    rec["global_received"] = got
    rec["global_bytes"] = nbytes
    return t_done


def run_hierarchical(fed: HierFederation, cfg: ExperimentConfig, seed: int, device: str,
                     local_models: dict[str, tuple[Any, Standardizer]] | None = None,
                     on_round=None, streams: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the hierarchical arm through Flower. Returns the schedule output plus the
    federated scaler, payload, setup bytes and the simulated-link description.

    With ``local_models`` (node -> (local-only model, its scaler)) each round also gets a
    test-set diagnostic (accuracy of the local, regional and global models and their
    disagreement rate, every node's test set pooled) and, with ``updates.enabled``, the
    version gate runs (fl/versioning.py; ``out["gate"]``). ``on_round`` is passed to
    ``run_rounds`` (progress file for ``fedastro peek --follow``).

    With ``streams`` (node -> data/stream.NodeStream; the hierarchical_stream arm, Phase H7)
    the same schedule runs test-then-train: nodes train only on usable labels, the scaler
    comes from the warm-up blocks, and ``out["stream"]`` (fl/stream.StreamRun) holds every
    on-arrival prediction. Pass ``local_models=None`` then: the stream trains its own local
    models, and the gate (``updates.enabled``) uses the warm-up local model as version 0."""
    h = cfg.hierarchy
    names = [n.name for n in h.nodes]
    region_of = {n.name: r.name for r in h.regions for n in r.nodes}
    silos = [fed.nodes[n].train for n in names]
    n_train = {n: len(fed.nodes[n].train) for n in names}
    first = silos[0]
    input_shape, n_classes = first.X.shape[1:], len(first.class_names)
    tests = region_test_sets(fed, h)
    pooled = pooled_test_set(fed, h)
    out: dict[str, Any] = {}
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        try:
            setup = federated_scaler(grid, len(names), cfg.fl.timeout_s)
            model = init_model(cfg, input_shape, n_classes, derive_seed(seed, ARM_IDS["federated"])).to(device)
            weights = get_weights(model)
            sim = HierSim(cfg, seed, n_train)
            t0 = sim.setup_s(setup.stats_bytes, setup.scaler_bytes) if sim.on else 0.0
            X_test = {r: setup.scaler.transform(d.X) for r, d in tests.items()}
            X_pool = setup.scaler.transform(pooled.X)

            def score(w, region):
                set_weights(model, w)
                if region is None:
                    return evaluate(pooled.y, predict_proba(model, X_pool, device))
                return evaluate(tests[region].y, predict_proba(model, X_test[region], device))

            def classes(w, X):
                set_weights(model, w)
                return predict_proba(model, X, device).argmax(axis=1)

            gate = diag = stream = None
            if streams is not None:
                from fedastro.fl.stream import StreamRun

                def fed_proba(w, X):
                    set_weights(model, w)
                    return predict_proba(model, X, device)

                stream = StreamRun(cfg, seed, device, {n: fed.nodes[n].train for n in names}, streams,
                                   setup.scaler, fed_proba)
                if cfg.updates.enabled:
                    from fedastro.fl.versioning import UpdateGate

                    vals = {n: fed.nodes[n].val for n in names}
                    X_val = {n: setup.scaler.transform(v.X) for n, v in vals.items()}
                    gate = UpdateGate(cfg.updates, seed, n_classes, names, {n: v.y for n, v in vals.items()},
                                      {n: stream.local_v0_classes(n, vals[n].X) for n in names},
                                      lambda w, n: classes(w, X_val[n]), [r.name for r in h.regions])
                    stream.gate = gate
            if local_models is not None:
                from fedastro.fl.versioning import UpdateGate

                node_test = {n: fed.test_of(n) for n in names}
                X_node = {n: setup.scaler.transform(d.X) for n, d in node_test.items()}
                y_all = np.concatenate([node_test[n].y for n in names])
                local_test = np.concatenate([predict_proba(local_models[n][0], local_models[n][1].transform(
                    node_test[n].X), device).argmax(axis=1) for n in names])

                def diag(regional, global_w):
                    reg = np.concatenate([classes(regional[region_of[n]], X_node[n]) for n in names])
                    acc = {"local": float(np.mean(local_test == y_all)), "regional": float(np.mean(reg == y_all)),
                           "global": None}
                    rate = None
                    if global_w is not None:
                        glob = np.concatenate([classes(global_w, X_node[n]) for n in names])
                        acc["global"] = float(np.mean(glob == y_all))
                        rate = float(np.mean(~((local_test == reg) & (reg == glob))))
                    return {"accuracy": acc, "disagreement_rate": rate,
                            "scope": "every node's test set pooled; diagnostic only"}

                if cfg.updates.enabled:
                    vals = {n: fed.nodes[n].val for n in names}
                    X_val = {n: setup.scaler.transform(v.X) for n, v in vals.items()}
                    local_val = {n: predict_proba(local_models[n][0], local_models[n][1].transform(vals[n].X),
                                                  device).argmax(axis=1) for n in names}
                    gate = UpdateGate(cfg.updates, seed, n_classes, names, {n: v.y for n, v in vals.items()},
                                      local_val, lambda w, n: classes(w, X_val[n]), [r.name for r in h.regions])

            ctx = _Ctx(grid, setup.ordered_nodes, names, cfg.fl.timeout_s, cfg.secagg.frac_bits)
            sched = run_rounds(cfg, ctx.train, weights, n_train, sim, t0, score, seed=seed, gate=gate,
                               diag=diag, on_round=on_round, stream=stream)
            out["gate"] = gate
            out["stream"] = stream
            out.update(sched, scaler=setup.scaler, payload_bytes=payload_bytes(weights),
                       setup_bytes=setup.setup_bytes, setup_sim_s=t0 if sim.on else None,
                       n_params=count_parameters(model), sim=sim)
        except BaseException as e:  # surfaced below; Flower would only log it
            out["error"] = e
            raise

    logging.getLogger("flwr").setLevel(logging.WARNING)
    run_simulation(
        server_app=app, num_supernodes=len(silos),
        client_app=make_client_app(silos, cfg, seed, device,
                                   streams=None if streams is None else [streams[n] for n in names]),
        backend_name="ray",
        backend_config={"client_resources": {"num_cpus": cfg.resources.client_num_cpus,
                                             "num_gpus": cfg.resources.client_num_gpus},
                        "init_args": {"log_to_driver": False, "include_dashboard": False,
                                      "logging_level": logging.ERROR}},
    )
    if "error" in out:
        raise RuntimeError("hierarchical arm failed") from out["error"]
    if "rounds" not in out:
        raise RuntimeError("hierarchical arm produced no result (ServerApp did not complete)")
    return out


# -- evaluation -------------------------------------------------------------------------


def _model_proba(cfg, weights, X, input_shape, n_classes, device) -> np.ndarray:
    model = init_model(cfg, input_shape, n_classes, seed=0).to(device)  # weights replaced
    set_weights(model, weights)
    return predict_proba(model, X, device)


def evaluate_hierarchy(fed: HierFederation, cfg: ExperimentConfig, run: dict[str, Any],
                       local_models: dict[str, tuple[Any, Standardizer]], device: str,
                       extra: dict[str, tuple[list[np.ndarray] | Any, Standardizer]] | None = None) -> dict[str, Any]:
    """Score the local, regional and global models on every node's test set.

    ``extra`` adds other single models scored the same way (e.g. the flat federated model,
    as weights, and the centralized model, as a module), never used in the disagreement
    analysis. Returns per_node, per_region and global metrics and the per-node
    (y, predictions) the disagreement analysis needs."""
    from fedastro.metrics.disagreement import disagreement_report

    h = cfg.hierarchy
    first = next(iter(fed.nodes.values())).train
    input_shape, n_classes = first.X.shape[1:], len(first.class_names)
    scaler: Standardizer = run["scaler"]
    probas: dict[str, dict[str, np.ndarray]] = {}
    ys: dict[str, np.ndarray] = {}
    region_of = {n.name: r.name for r in h.regions for n in r.nodes}
    for node in (n.name for n in h.nodes):
        test = fed.test_of(node)
        ys[node] = test.y
        local, local_scaler = local_models[node]
        p = {"local": predict_proba(local, local_scaler.transform(test.X), device)}
        X = scaler.transform(test.X)
        p["regional"] = _model_proba(cfg, run["regional"][region_of[node]], X, input_shape, n_classes, device)
        p["global"] = _model_proba(cfg, run["global"], X, input_shape, n_classes, device)
        if run.get("gate") is not None:  # the versions the node deployed (Phase H5); local if none accepted
            for track, d in run["gate"].deployed[node].items():
                p[f"deployed_{track}"] = (p["local"] if d.weights is None else
                                          _model_proba(cfg, d.weights, X, input_shape, n_classes, device))
        for name, (m, m_scaler) in (extra or {}).items():
            Xm = m_scaler.transform(test.X)
            p[name] = (_model_proba(cfg, m, Xm, input_shape, n_classes, device) if isinstance(m, list)
                       else predict_proba(m, Xm, device))
        probas[node] = p
    models = list(next(iter(probas.values())))

    def pooled(nodes: list[str]) -> dict[str, Any]:
        y = np.concatenate([ys[n] for n in nodes])
        return {"n_test": int(len(y)),
                **{m: evaluate(y, np.concatenate([probas[n][m] for n in nodes])) for m in models}}

    per_node = {n: {"region": region_of[n], "n_test": int(len(ys[n])),
                    **{m: evaluate(ys[n], probas[n][m]) for m in models}} for n in probas}
    per_region = {r.name: pooled([n.name for n in r.nodes]) for r in h.regions}
    preds = {n: {m: probas[n][m].argmax(axis=1) for m in ("local", "regional", "global")} for n in probas}
    y_all = np.concatenate([ys[n] for n in probas])
    pooled_proba = {m: (y_all, np.concatenate([probas[n][m] for n in probas])) for m in models}
    return {"per_node": per_node, "per_region": per_region, "global": pooled(list(probas)),
            "disagreement": disagreement_report(ys, preds, region_of), "pooled_proba": pooled_proba}


def arm_summary(cfg: ExperimentConfig, run: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """``result.arms.hierarchical``: headline numbers; the details are in ``result.hierarchy``."""
    rounds = run["rounds"]
    node_bytes = sum(r["bytes"] for rec in rounds for r in rec["regions"].values())
    global_bytes = sum(rec.get("global_bytes", 0) for rec in rounds)
    return {
        "final": metrics["global"]["global"],  # the global model on the pooled test sets
        "regional_final": metrics["global"]["regional"],  # each node scored with its region's model
        "global_every": cfg.hier_fl.global_every,
        "min_participants": cfg.hierarchy.min_participants,
        "n_merges": run["n_merges"],
        "n_fallbacks": sum(e["kind"] == "fallback" for e in run["events"]),
        "n_params": run["n_params"],
        "payload_bytes": run["payload_bytes"],
        "setup_bytes": run["setup_bytes"],
        "total_bytes": run["setup_bytes"] + node_bytes + global_bytes,
        "sim_wallclock_s_total": run["sim_end_s"],
        "setup_sim_s": run["setup_sim_s"],
    }


def comms_section(cfg: ExperimentConfig, run: dict[str, Any]) -> dict[str, Any] | None:
    sim: HierSim = run["sim"]
    if not sim.on:
        return None
    hf = cfg.hier_fl
    return {"enabled": True, "preset": cfg.comms.preset, "los": cfg.comms.los,
            "region_deadline_s": hf.region_deadline_s, "global_deadline_s": hf.global_deadline_s or None,
            "conjunction": ({"region": hf.conjunction_region, "from_s": hf.conjunction_from_s,
                             "duration_s": hf.conjunction_duration_s} if hf.conjunction_region else None),
            "los_events": sim.events(run["sim_end_s"] if run["sim_end_s"] is not None else 0.0),
            "time_unit": "simulated seconds (never real elapsed time)"}


__all__ = ["run_hierarchical", "run_rounds", "evaluate_hierarchy", "arm_summary", "pooled_test_set",
           "region_test_sets", "HierSim", "comms_section"]


def secagg_section(cfg: ExperimentConfig, run: dict[str, Any]) -> dict[str, Any] | None:
    """``result.secagg`` for a run with simulated secure aggregation (None = keep the
    disabled default). Per round: each region's and the global merge's mask group, dropouts,
    recoveries and overhead. A round's overhead time is the largest regional overhead plus
    the global merge's: regions run in parallel, the merge after them."""
    sa = cfg.secagg
    if not sa.enabled:
        return None
    on = cfg.comms.enabled
    per_round, total_b, total_s, recoveries = [], 0, 0.0, 0
    for rec in run["rounds"]:
        regions = {r: reg["secagg"] for r, reg in rec["regions"].items() if "secagg" in reg}
        glob = rec.get("global_secagg")
        parts = list(regions.values()) + ([glob] if glob else [])
        nbytes = sum(p["overhead_bytes"] for p in parts)
        sim_s = None
        if on:
            sim_s = max([p["overhead_sim_s"] for p in regions.values()], default=0.0) + \
                (glob["overhead_sim_s"] if glob else 0.0)
            total_s += sim_s
        total_b += nbytes
        recoveries += sum(p["recovery"] for p in parts)
        per_round.append({"round": rec["round"], "overhead_bytes": nbytes, "overhead_sim_s": sim_s,
                          "regions": regions, "global": glob})
    return {
        "enabled": True, "simulated": True,
        "levels": list(sa.levels), "key_agreement": sa.key_agreement, "frac_bits": sa.frac_bits,
        "min_group": {"regional": cfg.hierarchy.min_participants, "global": sa.global_min_group},
        "message_bytes": {"request": secagg.REQUEST_BYTES, "keys": secagg.KEY_BYTES,
                          "share_per_peer": secagg.SHARE_BYTES, "masked_word": secagg.WORD_BYTES},
        "overhead_bytes": int(total_b), "overhead_sim_s": total_s if on else None,
        "dropout_recoveries": int(recoveries), "per_round": per_round,
        "note": ("SIMULATED secure aggregation: pairwise masks from a seeded PRG, no real key agreement "
                 "or secret sharing; not cryptographically secure. Message sizes are scenario assumptions. "
                 "No formal privacy guarantee; the final model can still leak (membership inference)."),
    }
