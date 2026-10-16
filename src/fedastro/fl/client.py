"""Flower ClientApp: one simulated silo per node (partition-id = silo index).

Handlers (message types):
- ``query.stats``: send standardisation sufficient statistics (count, sum, sum of
  squares) of the local training data. No raw data leaves the silo.
- ``query.set_scaler``: receive the global scaler; kept in the node's context state.
- ``train``: receive global weights, run the shared ``train_local`` for
  ``fl.local_epochs``, return the updated weights and example count.
- ``evaluate``: score the received global weights on the silo's own training data.

Flower 1.23's Message API replaces the older NumPyClient get_parameters/fit/evaluate
methods; ``train`` and ``evaluate`` are their equivalents, and initial parameters come
from the server, so no get_parameters call exists.
"""

from __future__ import annotations

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch import nn

from fedastro.baselines.common import ARM_IDS, init_model
from fedastro.config import ExperimentConfig
from fedastro.data.base import DatasetBundle
from fedastro.data.preprocess import Standardizer, Stats
from fedastro.fl.strategy import client_proximal_mu
from fedastro.metrics.classification import evaluate
from fedastro.training import derive_seed, predict_proba, train_local


def get_weights(model: nn.Module) -> list[np.ndarray]:
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, arrays: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    if len(keys) != len(arrays):
        raise ValueError(f"expected {len(keys)} arrays, got {len(arrays)}")
    model.load_state_dict({k: torch.from_numpy(np.asarray(a)) for k, a in zip(keys, arrays)})


def metric_record(metrics: dict) -> MetricRecord:
    """MetricRecord cannot hold None; undefined metrics are omitted and restored as None."""
    return MetricRecord({k: v for k, v in metrics.items() if v is not None})


def make_client_app(silos: list[DatasetBundle], cfg: ExperimentConfig, seed: int, device: str) -> ClientApp:
    app = ClientApp()
    n_classes = len(silos[0].class_names)

    def _silo(context: Context) -> tuple[int, DatasetBundle]:
        idx = int(context.node_config["partition-id"])
        return idx, silos[idx]

    def _scaler(context: Context) -> Standardizer:
        if "scaler" not in context.state:
            raise RuntimeError("scaler not set on this node; server must send query.set_scaler first")
        mean, std = context.state["scaler"].to_numpy_ndarrays()
        return Standardizer(mean, std)

    def _model(data: DatasetBundle, arrays: list[np.ndarray]) -> nn.Module:
        model = init_model(cfg, data.X.shape[1:], n_classes, seed=0)  # weights overwritten next
        set_weights(model, arrays)
        return model

    @app.query("stats")
    def stats(msg: Message, context: Context) -> Message:
        idx, data = _silo(context)
        st = Stats.of(data.X)
        content = RecordDict({
            "arrays": ArrayRecord([st.total, st.total_sq]),
            "metrics": MetricRecord({"silo_index": idx, "count": st.count}),
        })
        return Message(content, reply_to=msg)

    @app.query("set_scaler")
    def set_scaler(msg: Message, context: Context) -> Message:
        idx, _ = _silo(context)
        context.state["scaler"] = msg.content["arrays"]
        return Message(RecordDict({"metrics": MetricRecord({"silo_index": idx})}), reply_to=msg)

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        torch.set_num_threads(1)
        idx, data = _silo(context)
        server_round = int(msg.content["config"]["server-round"])
        model = _model(data, msg.content["arrays"].to_numpy_ndarrays())
        X = _scaler(context).transform(data.X)
        history = train_local(
            model, X, data.y, cfg.train, n_classes,
            seed=derive_seed(seed, ARM_IDS["federated"], idx, server_round),
            device=device, epochs=cfg.fl.local_epochs,
            proximal_mu=client_proximal_mu(cfg.fl),
        )
        content = RecordDict({
            "arrays": ArrayRecord(get_weights(model)),
            "metrics": MetricRecord({
                "silo_index": idx, "num-examples": len(data), "train_loss": history[-1]["train_loss"],
            }),
        })
        return Message(content, reply_to=msg)

    @app.evaluate()
    def evaluate_local(msg: Message, context: Context) -> Message:
        torch.set_num_threads(1)
        idx, data = _silo(context)
        model = _model(data, msg.content["arrays"].to_numpy_ndarrays())
        X = _scaler(context).transform(data.X)
        m = evaluate(data.y, predict_proba(model, X, device))
        rec = metric_record(m)
        rec["silo_index"] = idx
        rec["num-examples"] = len(data)
        return Message(RecordDict({"metrics": rec}), reply_to=msg)

    return app
