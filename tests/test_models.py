import numpy as np
import pytest
import torch
from opacus.validators import ModuleValidator

from fedastro.config import ModelConfig, TrainConfig
from fedastro.data.preprocess import Standardizer, Stats
from fedastro.metrics.classification import evaluate
from fedastro.models.model import build_model, count_parameters
from fedastro.training import derive_seed, train_local


@pytest.mark.parametrize("arch,shape", [("mlp", (50,)), ("cnn1d", (10, 240))])
def test_models_small_shaped_and_opacus_compatible(arch, shape):
    model = build_model(ModelConfig(arch=arch), shape, 3)
    out = model(torch.zeros(4, *shape))
    assert out.shape == (4, 3)
    assert count_parameters(model) < 20_000, "keep the model small (CLAUDE.md rule 6)"
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    assert ModuleValidator.validate(model, strict=False) == []


def test_wrong_input_rank_rejected():
    with pytest.raises(ValueError):
        build_model(ModelConfig(arch="cnn1d"), (50,), 3)


@pytest.mark.parametrize("shape", [(100, 6), (100, 4, 12)])
def test_combined_stats_equal_pooled_fit(shape):
    rng = np.random.default_rng(0)
    X = rng.normal(3.0, 2.0, size=shape).astype(np.float32)
    a, b = X[:30], X[30:]
    pooled = Standardizer.fit(X)
    combined = Standardizer.from_stats(Stats.of(a) + Stats.of(b))
    np.testing.assert_allclose(pooled.mean, combined.mean, rtol=1e-10)
    np.testing.assert_allclose(pooled.std, combined.std, rtol=1e-8)
    Z = pooled.transform(X)
    axes = (0,) if Z.ndim == 2 else (0, 2)
    np.testing.assert_allclose(Z.mean(axis=axes), 0, atol=1e-5)
    np.testing.assert_allclose(Z.std(axis=axes), 1, atol=1e-4)


def test_constant_feature_does_not_produce_nan():
    X = np.ones((10, 3), dtype=np.float32)
    assert np.isfinite(Standardizer.fit(X).transform(X)).all()


def test_metrics_perfect_and_degenerate():
    y = np.array([0, 1, 2, 0])
    perfect = np.eye(3)[y]
    m = evaluate(y, perfect)
    assert m["accuracy"] == m["macro_f1"] == m["balanced_accuracy"] == m["auc"] == 1.0
    assert evaluate(np.zeros(4, dtype=int), np.full((4, 3), 1 / 3))["auc"] is None


def _toy(seed=0, n=120):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = (np.eye(3)[y] * 3 + rng.normal(size=(n, 3))).astype(np.float32)
    return X, y.astype(np.int64)


def test_train_local_deterministic_and_learns():
    X, y = _toy()
    cfg = TrainConfig(epochs=15)

    def run():
        torch.manual_seed(1)
        model = build_model(ModelConfig(), (3,), 3)
        hist = train_local(model, X, y, cfg, 3, seed=5)
        return hist, torch.cat([p.detach().flatten() for p in model.parameters()])

    (h1, w1), (h2, w2) = run(), run()
    assert torch.equal(w1, w2)
    assert h1 == h2
    assert h1[-1]["train_loss"] < h1[0]["train_loss"]


def test_derive_seed_stable_and_distinct():
    assert derive_seed(0, 1) == derive_seed(0, 1)
    assert len({derive_seed(0, 1), derive_seed(0, 2), derive_seed(1, 1), derive_seed(0, 2, 0)}) == 4
    assert 0 <= derive_seed(123, 4, 5) < 2**63
