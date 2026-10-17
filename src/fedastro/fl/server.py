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

Phase 3 assumes synchronous, lossless links: a missing or failed reply is an error.
Link delay, loss of signal and stragglers arrive in Phase 6, until then
``sim_wallclock_s`` is null.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.simulation import run_simulation

from fedastro.baselines.common import ARM_IDS, holdout_evaluator, init_model
from fedastro.config import ExperimentConfig
from fedastro.data.partition import Federation
from fedastro.data.preprocess import Standardizer, Stats
from fedastro.fl.client import get_weights, make_client_app, set_weights
from fedastro.fl.strategy import ClientUpdate, build_strategy
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


def make_server_app(fed: Federation, cfg: ExperimentConfig, seed: int, device: str,
                    out: dict[str, Any]) -> ServerApp:
    app = ServerApp()
    silo_names = list(fed.silos)
    first = fed.silos[silo_names[0]].data
    input_shape, n_classes = first.X.shape[1:], len(first.class_names)
    strategy = build_strategy(cfg.fl)
    timeout = cfg.fl.timeout_s

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        try:
            nodes = _wait_for_nodes(grid, len(silo_names), timeout)

            # --- setup: federated standardisation -------------------------------
            replies = _exchange(grid, nodes, "query.stats", RecordDict({"config": ConfigRecord({})}), timeout)
            node_of = {int(r.content["metrics"]["silo_index"]): r.metadata.src_node_id for r in replies}
            if sorted(node_of) != list(range(len(silo_names))):
                raise RuntimeError(f"silo indices reported by nodes: {sorted(node_of)}")
            stats_list, setup_bytes = [], 0
            for r in replies:
                total, total_sq = r.content["arrays"].to_numpy_ndarrays()
                stats_list.append(Stats(float(r.content["metrics"]["count"]), total, total_sq))
                setup_bytes += payload_bytes([total, total_sq])
            combined = stats_list[0]
            for st in stats_list[1:]:
                combined = combined + st
            scaler = Standardizer.from_stats(combined)
            scaler_arrays = [scaler.mean, scaler.std]
            _exchange(grid, nodes, "query.set_scaler", RecordDict({"arrays": ArrayRecord(scaler_arrays)}), timeout)
            setup_bytes += payload_bytes(scaler_arrays) * len(nodes)

            # --- rounds --------------------------------------------------------
            ordered_nodes = [node_of[i] for i in range(len(silo_names))]
            model = init_model(cfg, input_shape, n_classes, derive_seed(seed, ARM_IDS["federated"]))
            weights = get_weights(model)
            out["global_weights"] = {"init": weights}
            out["scaler"] = scaler
            holdout_eval = holdout_evaluator(fed.holdout, scaler, device)
            payload = payload_bytes(weights)
            rounds = []
            for rnd in range(1, cfg.fl.rounds + 1):
                content = RecordDict({"arrays": ArrayRecord(weights),
                                      "config": ConfigRecord({"server-round": rnd})})
                replies = _exchange(grid, ordered_nodes, MessageType.TRAIN, content, timeout)
                updates, per_silo = [], {}
                for r in replies:
                    idx = int(r.content["metrics"]["silo_index"])
                    upd = ClientUpdate(
                        silo_index=idx, silo=silo_names[idx],
                        arrays=r.content["arrays"].to_numpy_ndarrays(),
                        num_examples=int(r.content["metrics"]["num-examples"]),
                        train_loss=_opt_float(r.content["metrics"], "train_loss"),
                    )
                    updates.append(upd)
                    set_weights(model, upd.arrays)
                    per_silo[upd.silo] = {"n_train": upd.num_examples, "train_loss": upd.train_loss,
                                          "update_on_holdout": holdout_eval(model)}
                    if "epsilon" in r.content["metrics"]:
                        per_silo[upd.silo]["epsilon"] = float(r.content["metrics"]["epsilon"])
                weights = strategy.aggregate(updates)
                set_weights(model, weights)
                global_metrics = holdout_eval(model)

                content = RecordDict({"arrays": ArrayRecord(weights),
                                      "config": ConfigRecord({"server-round": rnd})})
                for r in _exchange(grid, ordered_nodes, MessageType.EVALUATE, content, timeout):
                    silo = silo_names[int(r.content["metrics"]["silo_index"])]
                    per_silo[silo]["global_on_local"] = _metrics_from_record(r.content["metrics"])

                rounds.append({
                    "round": rnd,
                    "global": global_metrics,
                    "per_silo": per_silo,
                    "bytes": 2 * payload * len(updates),
                    "sim_wallclock_s": None,
                    "participants": [u.silo for u in sorted(updates, key=lambda u: u.silo_index)],
                })
            out["global_weights"]["final"] = weights
            out.update(rounds=rounds, payload_bytes=payload, setup_bytes=setup_bytes,
                       n_params=count_parameters(model))
        except BaseException as e:  # surfaced by run_federated; Flower would only log it
            out["error"] = e
            raise

    return app


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
        "final": dict(rounds[-1]["global"]),
        "strategy": cfg.fl.strategy,
        "proximal_mu": cfg.fl.proximal_mu if cfg.fl.strategy == "fedprox" else None,
        "local_epochs": cfg.fl.local_epochs,
        "n_params": out["n_params"],
        "payload_bytes": out["payload_bytes"],
        "setup_bytes": out["setup_bytes"],
        "total_bytes": out["setup_bytes"] + sum(r["bytes"] for r in rounds),
        "rounds_to_target": targets,
    }
    if cfg.privacy.enabled:
        from fedastro.baselines.common import dp_plan

        total = cfg.fl.rounds * cfg.fl.local_epochs
        result["privacy"] = {name: dp_plan(cfg, len(split.data), total).to_dict()
                             for name, split in fed.silos.items()}
    return result
