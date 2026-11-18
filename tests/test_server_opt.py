"""Phase H8: server optimizers (fl/strategy.ServerOpt) in the flat and hierarchical arms."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from fedastro.config import ConfigError, load_config
from fedastro.fl.hierarchy import HierSim, run_rounds
from fedastro.fl.strategy import ClientUpdate, ServerOpt, build_server_opt

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "configs" / "experiments"


def _w(seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(3, 2)).astype(np.float32), rng.normal(size=4).astype(np.float32)]


def test_momentum_with_beta0_lr1_is_plain_aggregation():
    opt = ServerOpt("momentum", lr=1.0, beta1=0.0, beta2=0.99, tau=1e-3)
    cur = _w(0)
    for k in range(3):
        agg = _w(k + 1)
        new = opt.step(cur, agg)
        for a, b in zip(new, agg):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
            assert a.dtype == b.dtype
        cur = new


def test_momentum_accumulates():
    opt = ServerOpt("momentum", lr=1.0, beta1=0.9, beta2=0.99, tau=1e-3)
    cur = [np.zeros(2)]
    w1 = opt.step(cur, [np.ones(2)])  # v = 1
    np.testing.assert_allclose(w1[0], 1.0)
    w2 = opt.step(w1, [np.full(2, 2.0)])  # delta 1, v = 0.9 + 1
    np.testing.assert_allclose(w2[0], 1.0 + 1.9)


def test_adam_matches_a_hand_computation():
    lr, b1, b2, tau = 0.1, 0.9, 0.99, 1e-3
    opt = ServerOpt("adam", lr, b1, b2, tau)
    cur = [np.array([1.0, -2.0])]
    d1 = np.array([0.5, 0.2])
    out = opt.step(cur, [cur[0] + d1])
    m, v = (1 - b1) * d1, (1 - b2) * d1 ** 2
    np.testing.assert_allclose(out[0], cur[0] + lr * m / (np.sqrt(v) + tau))
    d2 = np.array([-0.1, 0.3])
    out2 = opt.step(out, [out[0] + d2])
    m, v = b1 * m + (1 - b1) * d2, b2 * v + (1 - b2) * d2 ** 2
    np.testing.assert_allclose(out2[0], out[0] + lr * m / (np.sqrt(v) + tau))


def test_config_rules():
    base = EXP / "smoke_hier_fl.yaml"
    assert build_server_opt(load_config(base).fl) is None
    with pytest.raises(ConfigError, match="server_opt_"):
        load_config(base, {"fl.server_opt_lr": 0.1})
    with pytest.raises(ConfigError):
        load_config(base, {"fl.server_opt": "sgd"})
    cfg = load_config(base, {"fl.server_opt": "adam", "fl.server_opt_lr": 0.01})
    assert cfg.experiment_dict()["fl"]["server_opt"] == "adam"
    assert "server_opt" not in load_config(base).experiment_dict()["fl"]  # out of the run id at default


def _hier(server_opt=None, **kw):
    cfg = load_config(EXP / "smoke_hier_fl.yaml")
    fl = dataclasses.replace(cfg.fl, rounds=6, **({"server_opt": server_opt, **kw} if server_opt else {}))
    cfg = dataclasses.replace(cfg, fl=fl, hier_fl=dataclasses.replace(cfg.hier_fl, offline={}))
    cfg.validate()
    n_train = {n.name: 10 * (i + 1) for i, n in enumerate(cfg.hierarchy.nodes)}

    def train(dispatch, rnd):
        return {i: ClientUpdate(silo_index=i, silo=str(i), arrays=[w[0] + (i + 1)], num_examples=10 * (i + 1),
                                train_loss=None) for i, w in dispatch.items()}

    return run_rounds(cfg, train, [np.zeros(1)], n_train, HierSim(cfg, 0, n_train), 0.0,
                      lambda w, r: {"accuracy": float(w[0][0])})


def test_regional_server_optimizer_in_the_hierarchy():
    plain = _hier()
    same = _hier("momentum", server_beta1=0.0)  # lr 1, beta 0 = plain FedAvg
    for r in plain["regional"]:
        np.testing.assert_allclose(same["regional"][r][0], plain["regional"][r][0], atol=1e-9)
    np.testing.assert_allclose(same["global"][0], plain["global"][0], atol=1e-9)
    mom = _hier("momentum")
    assert not np.allclose(mom["global"][0], plain["global"][0])  # momentum changes the trajectory
    # each region keeps its own state: regions differ in their node sizes, so their models differ
    vals = [float(mom["regional"][r][0][0]) for r in mom["regional"]]
    assert len(set(vals)) == len(vals)
