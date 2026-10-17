"""Record-level differential privacy for local training: DP-SGD via Opacus.

Unit of privacy: one training record (one WESAD window). Every model trained in a run
gets its own (epsilon, delta) budget over its own training data. This is NOT
subject-level privacy (a subject contributes ~70 overlapping windows) and it does
not cover the standardisation statistics each silo shares; PROGRESS.md Phase 5 says
what that means for any claim.

Calibration. ``PrivacyEngine.make_private_with_epsilon`` is ``get_noise_multiplier``
followed by ``make_private``. The two steps are split here because a federated client
calls local training once per round with a fresh optimizer, while its budget must
cover all rounds: the noise multiplier is calibrated once for the total number of
steps (rounds x local epochs x steps per epoch), then every round runs
``make_private`` with it. The RDP accountant composes identical Poisson-subsampled
Gaussian steps by adding step counts, so the epsilon spent after k epochs is computed
in closed form by ``DPPlan.epsilon_after``. tests/test_privacy.py checks that this
matches ``make_private_with_epsilon`` + ``get_epsilon`` for a single training call.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fedastro.config import PrivacyConfig

ACCOUNTANT = "rdp"

# Known, irrelevant Opacus warnings: the secure RNG is off by design (runs must be
# reproducible), and the non-full backward hook warning does not apply to these models.
warnings.filterwarnings("ignore", message=r".*[Ss]ecure RNG.*")
warnings.filterwarnings("ignore", message=r".*non-full backward hook.*")
warnings.filterwarnings("ignore", message=r".*Optimal order is the largest alpha.*")


@dataclass(frozen=True)
class DPPlan:
    """DP-SGD settings for one trained model over its own data."""

    n_train: int
    batch_size: int
    epochs: int  # total epochs the budget covers
    target_epsilon: float
    delta: float
    max_grad_norm: float
    noise_multiplier: float

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.n_train / self.batch_size)

    @property
    def sample_rate(self) -> float:
        # Opacus's Poisson DPDataLoader samples each record with prob 1/len(loader).
        return 1.0 / self.steps_per_epoch

    def epsilon_after(self, epochs_done: int) -> float:
        if epochs_done <= 0:
            return 0.0
        acc = RDPAccountant()
        acc.history = [(self.noise_multiplier, self.sample_rate, epochs_done * self.steps_per_epoch)]
        return float(acc.get_epsilon(delta=self.delta))

    def to_dict(self, epochs_done: int | None = None) -> dict[str, Any]:
        done = self.epochs if epochs_done is None else epochs_done
        return {
            "n_train": self.n_train,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
            "steps": done * self.steps_per_epoch,
            "epsilon": self.epsilon_after(done),
        }


@lru_cache(maxsize=256)
def _noise_multiplier(target_epsilon: float, delta: float, sample_rate: float, steps: int) -> float:
    return float(get_noise_multiplier(
        target_epsilon=target_epsilon, target_delta=delta, sample_rate=sample_rate,
        steps=steps, accountant=ACCOUNTANT,
    ))


def plan_dp(cfg: PrivacyConfig, n_train: int, batch_size: int, epochs: int) -> DPPlan:
    if not cfg.enabled:
        raise ValueError("plan_dp called with privacy disabled")
    if n_train < 1:
        raise ValueError("DP needs at least one training record")
    steps_per_epoch = math.ceil(n_train / batch_size)
    delta = cfg.delta_factor / n_train
    return DPPlan(
        n_train=n_train, batch_size=batch_size, epochs=epochs,
        target_epsilon=cfg.target_epsilon, delta=delta, max_grad_norm=cfg.max_grad_norm,
        noise_multiplier=_noise_multiplier(cfg.target_epsilon, delta, 1.0 / steps_per_epoch,
                                           epochs * steps_per_epoch),
    )


def make_private(model: nn.Module, optimizer: Optimizer, loader: DataLoader, plan: DPPlan):
    """Wrap for DP-SGD with Poisson sampling. Randomness (sampling and noise) comes from
    the global torch RNG, which ``train_local`` seeds, so runs stay reproducible and no
    CPU generator has to follow the model onto CUDA."""
    if len(loader) != plan.steps_per_epoch:
        raise ValueError(f"loader has {len(loader)} batches, plan expects {plan.steps_per_epoch}")
    engine = PrivacyEngine(accountant=ACCOUNTANT)
    return engine.make_private(
        module=model, optimizer=optimizer, data_loader=loader,
        noise_multiplier=plan.noise_multiplier, max_grad_norm=plan.max_grad_norm,
        poisson_sampling=True,
    )


def privacy_section(cfg: PrivacyConfig, accounting: dict[str, Any]) -> dict[str, Any]:
    """The result's ``privacy`` block. ``accounting`` maps arm -> per-model dicts from
    ``DPPlan.to_dict``. ``epsilon``/``delta`` report the WEAKEST guarantee among all
    models in the run (largest epsilon, largest delta)."""
    models = [m for arm in accounting.values() for m in _flatten(arm)]
    return {
        "enabled": cfg.enabled,
        "epsilon": max(m["epsilon"] for m in models) if models else None,
        "delta": max(m["delta"] for m in models) if models else None,
        "max_grad_norm": cfg.max_grad_norm if cfg.enabled else None,
        "target_epsilon": cfg.target_epsilon if cfg.enabled else None,
        "delta_rule": f"delta = {cfg.delta_factor} / N_train per trained model" if cfg.enabled else None,
        "accountant": ACCOUNTANT if cfg.enabled else None,
        "unit": "record (one window), per trained model" if cfg.enabled else None,
        "not_covered": ([
            "subject-level privacy (each subject contributes many correlated windows)",
            "standardisation statistics (count, sum, sum of squares) shared by each silo",
            "number of training records per silo (num-examples weights FedAvg)",
        ] if cfg.enabled else []),
        "accounting": accounting,
        "attack": {"mse": None},
    }


def _flatten(arm: dict[str, Any]) -> list[dict[str, Any]]:
    if "epsilon" in arm:
        return [arm]
    return [m for v in arm.values() for m in _flatten(v)]
