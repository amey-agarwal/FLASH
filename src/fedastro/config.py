"""Config loading and validation.

An experiment YAML is deep-merged over ``configs/base.yaml``. The ``silos`` key names
a topology file in ``configs/silos/`` which is expanded inline, so the resolved config
recorded in every result is fully self-contained.

Unknown keys are rejected: a typo in a config must fail loudly, never silently fall
back to a default.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or fails validation."""


DATASETS = ("synthetic", "wesad")  # Phase 9 adds "osd435", "eicu".
PARTITION_METHODS = ("by_subject", "by_site", "dirichlet", "dirichlet_quantity_control")
DIRICHLET_METHODS = ("dirichlet", "dirichlet_quantity_control")
WESAD_DEVICES = ("chest", "wrist")
ARCHS = ("mlp", "cnn1d")
OPTIMIZERS = ("adam", "sgd")
CLASS_WEIGHTING = ("none", "balanced")
ARMS = ("centralized", "local_only", "federated")
FL_STRATEGIES = ("fedavg", "fedprox", "fedbuff")
SYNC_STRATEGIES = ("fedavg", "fedprox")
REPRESENTATIONS = ("features", "sequence")
DEVICES = ("cpu", "cuda")
LOS_PROFILES = ("none", "rare", "short_occasional", "periodic", "long_blackout")
LOS_KINDS = ("none", "poisson", "periodic")
COMMS_PRESETS = ("earth", "moon", "mars_near", "mars_mean", "mars_far")
ATTACK_STATES = ("init", "final")
ATTACK_LABEL_MODES = ("known", "inferred")


@dataclass(frozen=True)
class SyntheticParams:
    n_subjects: int = 20
    windows_per_subject: int = 30
    n_features: int = 8
    n_classes: int = 3
    class_sep: float = 1.0
    subject_shift: float = 0.5
    n_sites: int = 0  # 0 = no site ids; >0 assigns subject i to site i % n_sites

    def validate(self) -> None:
        for name in ("n_subjects", "windows_per_subject", "n_features", "n_classes"):
            if getattr(self, name) < 1:
                raise ConfigError(f"data.synthetic.{name} must be >= 1")
        if self.n_classes < 2:
            raise ConfigError("data.synthetic.n_classes must be >= 2")
        if self.class_sep < 0 or self.subject_shift < 0:
            raise ConfigError("data.synthetic.class_sep and subject_shift must be >= 0")
        if self.n_sites < 0:
            raise ConfigError("data.synthetic.n_sites must be >= 0")


@dataclass(frozen=True)
class WesadParams:
    """WESAD windowing and feature extraction. See PROGRESS.md Phase 1 for rationale."""

    devices: tuple[str, ...] = ("chest", "wrist")
    window_s: float = 60.0
    stride_s: float = 30.0
    representation: str = "features"  # "features": per-channel stats; "sequence": downsampled signals
    sequence_hz: float = 4.0
    subjects: tuple[int, ...] = ()  # empty = all 15 subjects

    def validate(self) -> None:
        if not self.devices or any(d not in WESAD_DEVICES for d in self.devices):
            raise ConfigError(f"data.wesad.devices must be a non-empty subset of {WESAD_DEVICES}")
        if len(set(self.devices)) != len(self.devices):
            raise ConfigError("data.wesad.devices has duplicates")
        if self.window_s <= 0 or self.stride_s <= 0:
            raise ConfigError("data.wesad.window_s and stride_s must be > 0")
        if self.representation not in REPRESENTATIONS:
            raise ConfigError(f"data.wesad.representation must be one of {REPRESENTATIONS}")
        if self.sequence_hz <= 0 or self.sequence_hz > 4.0:
            raise ConfigError("data.wesad.sequence_hz must be in (0, 4] (wrist EDA/TEMP are 4 Hz)")
        if (self.window_s * self.sequence_hz) % 1:
            raise ConfigError("data.wesad.window_s * sequence_hz must be an integer")


@dataclass(frozen=True)
class DataConfig:
    name: str = "synthetic"
    synthetic: SyntheticParams = field(default_factory=SyntheticParams)
    wesad: WesadParams = field(default_factory=WesadParams)

    def validate(self) -> None:
        if self.name not in DATASETS:
            raise ConfigError(f"data.name must be one of {DATASETS}, got {self.name!r}")
        self.synthetic.validate()
        self.wesad.validate()

    @property
    def is_synthetic(self) -> bool:
        return self.name == "synthetic"


@dataclass(frozen=True)
class SiloConfig:
    name: str
    data_share: float
    one_way_latency_s: float
    los_profile: str
    latency_presets_s: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("silo name must be non-empty")
        if self.data_share <= 0:
            raise ConfigError(f"silo {self.name}: data_share must be > 0")
        if self.one_way_latency_s < 0:
            raise ConfigError(f"silo {self.name}: one_way_latency_s must be >= 0")
        if self.los_profile not in LOS_PROFILES:
            raise ConfigError(f"silo {self.name}: los_profile must be one of {LOS_PROFILES}")
        for k, v in self.latency_presets_s.items():
            if v < 0:
                raise ConfigError(f"silo {self.name}: latency preset {k!r} must be >= 0")


@dataclass(frozen=True)
class TopologyConfig:
    name: str
    silos: tuple[SiloConfig, ...]

    def validate(self) -> None:
        if not self.silos:
            raise ConfigError("topology must define at least one silo")
        names = [s.name for s in self.silos]
        if len(set(names)) != len(names):
            raise ConfigError(f"duplicate silo names in topology {self.name!r}: {names}")
        for s in self.silos:
            s.validate()


@dataclass(frozen=True)
class PartitionConfig:
    method: str = "by_subject"
    # Dirichlet concentration; required (> 0) for the two dirichlet methods.
    # "dirichlet_quantity_control" reproduces the per-silo subjects and window counts of
    # "dirichlet" at the same alpha and seed, but subsamples uniformly (natural labels).
    alpha: float = 0.0
    holdout_subjects: int = 3  # subjects reserved for the global test set; no silo trains on them
    # The test set is drawn from this seed, NOT the run seed, so every run and arm of an
    # experiment family is evaluated on the same held-out subjects.
    holdout_seed: int = 0

    def validate(self) -> None:
        if self.method not in PARTITION_METHODS:
            raise ConfigError(f"partition.method must be one of {PARTITION_METHODS}")
        if self.method in DIRICHLET_METHODS and self.alpha <= 0:
            raise ConfigError(f"partition.alpha must be > 0 for method {self.method!r}")
        if self.method not in DIRICHLET_METHODS and self.alpha != 0:
            raise ConfigError("partition.alpha is only meaningful for the dirichlet methods")
        if self.holdout_subjects < 1:
            raise ConfigError("partition.holdout_subjects must be >= 1")
        if self.holdout_seed < 0:
            raise ConfigError("partition.holdout_seed must be >= 0")


@dataclass(frozen=True)
class ModelConfig:
    """Small by design (CLAUDE.md rule 6). Normalisation is GroupNorm, never BatchNorm,
    so models are Opacus-compatible without modification (Phase 5)."""

    arch: str = "mlp"
    hidden: tuple[int, ...] = (64, 32)  # mlp only
    cnn_channels: tuple[int, ...] = (16, 32)  # cnn1d only
    kernel_size: int = 5  # cnn1d only
    dropout: float = 0.2

    def validate(self) -> None:
        if self.arch not in ARCHS:
            raise ConfigError(f"model.arch must be one of {ARCHS}")
        if not 1 <= len(self.hidden) <= 3 or any(h < 1 for h in self.hidden):
            raise ConfigError("model.hidden must list 1-3 positive layer widths")
        if not 1 <= len(self.cnn_channels) <= 3 or any(c < 1 for c in self.cnn_channels):
            raise ConfigError("model.cnn_channels must list 1-3 positive channel counts")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ConfigError("model.kernel_size must be a positive odd integer")
        if not 0 <= self.dropout < 1:
            raise ConfigError("model.dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainConfig:
    """Local training settings, shared verbatim by every arm (and later FL clients)."""

    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    class_weighting: str = "none"

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ConfigError("train.epochs and train.batch_size must be >= 1")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ConfigError("train.lr must be > 0 and train.weight_decay >= 0")
        if self.optimizer not in OPTIMIZERS:
            raise ConfigError(f"train.optimizer must be one of {OPTIMIZERS}")
        if self.class_weighting not in CLASS_WEIGHTING:
            raise ConfigError(f"train.class_weighting must be one of {CLASS_WEIGHTING}")


@dataclass(frozen=True)
class FLConfig:
    """Federated arm. Defaults: 30 rounds x 1 local epoch, so every client passes over
    its data 30 times, matching the 30 epochs of the baseline arms."""

    strategy: str = "fedavg"
    rounds: int = 30
    local_epochs: int = 1
    # FedProx proximal coefficient; only used (and only recorded) when strategy is
    # "fedprox". 0.01 was fixed a priori (Phase 4), not tuned on the holdout.
    proximal_mu: float = 0.01
    # rounds-to-target thresholds on global holdout accuracy:
    relative_target: float = 0.95  # fraction of the centralized arm's final accuracy
    absolute_target: float = 0.0  # 0 disables the absolute threshold
    timeout_s: float = 600.0  # per round, real seconds; a missing reply is an error
    # FedBuff (buffered asynchronous, Nguyen et al. 2022); used only when strategy is
    # "fedbuff". The server applies an update once ``buffer_k`` client updates have
    # arrived. It runs rounds * n_silos / buffer_k aggregations, so the total number of
    # client updates matches the synchronous run with the same ``rounds``.
    buffer_k: int = 3
    server_lr: float = 1.0

    def validate(self) -> None:
        if self.buffer_k < 1 or self.server_lr <= 0:
            raise ConfigError("fl.buffer_k must be >= 1 and fl.server_lr > 0")
        if self.strategy not in FL_STRATEGIES:
            raise ConfigError(f"fl.strategy must be one of {FL_STRATEGIES}")
        if self.rounds < 1 or self.local_epochs < 1:
            raise ConfigError("fl.rounds and fl.local_epochs must be >= 1")
        if self.proximal_mu <= 0:
            raise ConfigError("fl.proximal_mu must be > 0 (use strategy fedavg for mu = 0)")
        if not 0 < self.relative_target <= 1:
            raise ConfigError("fl.relative_target must be in (0, 1]")
        if not 0 <= self.absolute_target <= 1:
            raise ConfigError("fl.absolute_target must be in [0, 1]")
        if self.timeout_s <= 0:
            raise ConfigError("fl.timeout_s must be > 0")


@dataclass(frozen=True)
class AttackConfig:
    """Gradient inversion against one client's single-step update (privacy/attack.py).

    Threat model: an honest-but-curious server that knows the architecture, the global
    weights it sent, the batch size and the clipping norm, and observes one released
    gradient. Labels are either known or inferred from the gradient (``label_modes``).
    See PROGRESS.md Phase 5 for why each is assumed."""

    enabled: bool = False
    silos: tuple[str, ...] = ()  # silos to attack; empty = every silo of the topology
    label_modes: tuple[str, ...] = ("known", "inferred")
    model_states: tuple[str, ...] = ("init", "final")  # global model the update is computed at
    batch_sizes: tuple[int, ...] = (1, 4, 32)
    trials: int = 3  # batches attacked per (state, batch size); drawn by seed, never picked
    iterations: int = 1000
    lr: float = 0.1
    prior_weight: float = 0.01  # Gaussian prior on standardised inputs (weight * mean(x^2))
    # Chance level per trial: score this many random N(0, 1) guesses (the attack's own
    # starting point) exactly like a reconstruction. Attack scores are read against it.
    chance_draws: int = 50

    def validate(self) -> None:
        if not self.model_states or any(s not in ATTACK_STATES for s in self.model_states):
            raise ConfigError(f"privacy.attack.model_states must be a non-empty subset of {ATTACK_STATES}")
        if not self.batch_sizes or any(b < 1 for b in self.batch_sizes):
            raise ConfigError("privacy.attack.batch_sizes must list positive integers")
        if not self.label_modes or any(m not in ATTACK_LABEL_MODES for m in self.label_modes):
            raise ConfigError(f"privacy.attack.label_modes must be a non-empty subset of {ATTACK_LABEL_MODES}")
        for name in ("batch_sizes", "model_states", "label_modes", "silos"):
            vals = getattr(self, name)
            if len(set(vals)) != len(vals):
                raise ConfigError(f"privacy.attack.{name} must not repeat")
        if self.trials < 1 or self.iterations < 1 or self.chance_draws < 1:
            raise ConfigError("privacy.attack.trials, iterations and chance_draws must be >= 1")
        if self.lr <= 0 or self.prior_weight < 0:
            raise ConfigError("privacy.attack.lr must be > 0 and prior_weight >= 0")


@dataclass(frozen=True)
class PrivacyConfig:
    """Record-level DP-SGD (Opacus) for every model trained in the run (privacy/dp.py).

    Each trained model (the centralized model, each local-only model, each federated
    client) gets its own (target_epsilon, delta) budget over its own training data,
    with delta = delta_factor / N_train of that data. delta_factor < 1 keeps
    delta < 1/N_train (CLAUDE.md section 2)."""

    enabled: bool = False
    target_epsilon: float = 0.0  # required (> 0) when enabled
    delta_factor: float = 0.1
    max_grad_norm: float = 1.0
    attack: AttackConfig = field(default_factory=AttackConfig)

    def validate(self) -> None:
        if self.enabled and self.target_epsilon <= 0:
            raise ConfigError("privacy.target_epsilon must be > 0 when privacy is enabled")
        if not self.enabled and self.target_epsilon != 0:
            raise ConfigError("privacy.target_epsilon is only meaningful when privacy.enabled is true")
        if not 0 < self.delta_factor < 1:
            raise ConfigError("privacy.delta_factor must be in (0, 1) so that delta < 1/N_train")
        if self.max_grad_norm <= 0:
            raise ConfigError("privacy.max_grad_norm must be > 0")
        self.attack.validate()


@dataclass(frozen=True)
class LOSProfile:
    """Loss-of-signal process for one silo's link (fl/comms.py).

    kind "poisson": outages start at exponential intervals (mean ``mean_interval_s``)
    and last ``duration_s``. kind "periodic": an outage of ``duration_s`` every
    ``period_s``, starting at ``phase_s``. kind "none": always up."""

    kind: str = "none"
    mean_interval_s: float = 0.0
    period_s: float = 0.0
    phase_s: float = 0.0
    duration_s: float = 0.0

    def validate(self, where: str) -> None:
        if self.kind not in LOS_KINDS:
            raise ConfigError(f"{where}.kind must be one of {LOS_KINDS}")
        if self.kind == "poisson" and (self.mean_interval_s <= 0 or self.duration_s <= 0):
            raise ConfigError(f"{where}: poisson needs mean_interval_s > 0 and duration_s > 0")
        if self.kind == "periodic" and not (0 < self.duration_s < self.period_s and self.phase_s >= 0):
            raise ConfigError(f"{where}: periodic needs 0 < duration_s < period_s and phase_s >= 0")


def _default_los_profiles() -> dict[str, LOSProfile]:
    # Scenario assumptions, not measurements, except where a source is named.
    # 450 s = the ~7.5 min loss-of-signal pause logged by the FLUID ISS deployment
    # (CLAUDE.md section 4), used as the calibration outage length.
    return {
        "none": LOSProfile(),
        # terrestrial analog site: about one 7.5 min outage per day
        "rare": LOSProfile(kind="poisson", mean_interval_s=86_400.0, duration_s=450.0),
        # ISS: about one 7.5 min gap per ~92 min orbit (orbital period is standard; the
        # one-gap-per-orbit rate is a scenario assumption)
        "short_occasional": LOSProfile(kind="poisson", mean_interval_s=5_520.0, duration_s=450.0),
        # lunar habitat relayed by an orbiter in a ~2 h orbit, out of view ~48 min per orbit
        # (Apollo-era far-side LOS was roughly 45-48 min; confirm before poster use)
        "periodic": LOSProfile(kind="periodic", period_s=7_200.0, phase_s=1_800.0, duration_s=2_880.0),
        # Mars transit vehicle: multi-hour gaps in ground-station coverage, about 4 per day
        # (scenario assumption; solar conjunction, ~2 weeks, is far longer than any run)
        "long_blackout": LOSProfile(kind="poisson", mean_interval_s=21_600.0, duration_s=7_200.0),
    }


@dataclass(frozen=True)
class DarkConfig:
    """Force one silo's link down for ``duration_s`` (0 = for the rest of the run): the
    "silo goes dark" test. Onset is either ``from_s`` (simulated seconds) or
    ``from_round``: sync, when that round is sent; FedBuff, once (from_round - 1) x
    n_silos client updates have been aggregated, the same share of the matched updates
    (fl/server.py, _dark_onset_s). ``from_round`` makes "mid-run" mean mid-run for every
    strategy; an absolute time can fall after a fast strategy has finished."""

    silo: str = ""  # "" = no forced outage
    from_s: float = 0.0
    from_round: int = 0  # 0 = use from_s
    duration_s: float = 0.0

    def validate(self) -> None:
        if self.from_s < 0 or self.duration_s < 0 or self.from_round < 0:
            raise ConfigError("comms.dark.from_s, from_round and duration_s must be >= 0")
        if self.from_round and self.from_s:
            raise ConfigError("comms.dark: set from_s or from_round, not both")
        if self.from_round and not self.silo:
            raise ConfigError("comms.dark.from_round needs comms.dark.silo")


@dataclass(frozen=True)
class CommsConfig:
    """Simulated deep-space links (fl/comms.py). Time is modelled, never slept
    (except with the CLI's --real-sleep, for the live demo only)."""

    enabled: bool = False
    # Moves every silo that defines latency_presets_s (the deep-space silo) to Earth,
    # Moon or a Mars distance. Other silos keep their own one_way_latency_s.
    preset: str = "mars_mean"
    los: bool = True  # False = links never drop (outages switched off)
    los_profiles: dict[str, LOSProfile] = field(default_factory=_default_los_profiles)
    jitter_s: float = 0.05  # std of |N(0, jitter_s)| added to each transfer (scenario assumption)
    bandwidth_bps: float = 256_000.0  # per link, both directions (scenario assumption)
    # Simulated local compute: seconds per training example per local epoch (scenario assumption)
    compute_s_per_example: float = 0.01
    # Synchronous strategies: 0 = wait for every silo; > 0 = a silo whose update has not
    # arrived this many simulated seconds after the round started is dropped for that round.
    sync_deadline_s: float = 0.0
    dark: DarkConfig = field(default_factory=DarkConfig)
    # Store-and-forward buffer at each sender (fl/comms.py), per link direction:
    # supersede: a newer message handed to the link replaces an older one still waiting
    #   in the buffer (an outdated model or update is worthless).
    # ttl_s: message lifetime from hand-over; a message that would arrive later is dropped
    #   (DTN bundle lifetime). 6 h exceeds the longest modelled outage (2 h) plus Mars-far
    #   light time; scenario assumption. 0 = never expire.
    # cache_bytes: buffer capacity; a message that must wait and does not fit is dropped
    #   (drop-tail). 0 = unlimited (default, off).
    supersede: bool = True
    ttl_s: float = 21_600.0
    cache_bytes: int = 0

    def validate(self) -> None:
        if self.preset not in COMMS_PRESETS:
            raise ConfigError(f"comms.preset must be one of {COMMS_PRESETS}")
        if self.jitter_s < 0 or self.bandwidth_bps <= 0 or self.compute_s_per_example < 0:
            raise ConfigError("comms.jitter_s >= 0, bandwidth_bps > 0, compute_s_per_example >= 0 required")
        if self.sync_deadline_s < 0 or self.ttl_s < 0 or self.cache_bytes < 0:
            raise ConfigError("comms.sync_deadline_s, ttl_s and cache_bytes must be >= 0")
        missing = set(LOS_PROFILES) - set(self.los_profiles)
        if missing:
            raise ConfigError(f"comms.los_profiles is missing {sorted(missing)}")
        for name, prof in self.los_profiles.items():
            prof.validate(f"comms.los_profiles.{name}")
        self.dark.validate()


@dataclass(frozen=True)
class ResourcesConfig:
    """Per-client Ray resources for the simulation engine. Execution detail only: it
    changes speed, not results, so it is excluded from the run id."""

    client_num_cpus: float = 1.0
    client_num_gpus: float = 0.0

    def validate(self) -> None:
        if self.client_num_cpus <= 0 or self.client_num_gpus < 0:
            raise ConfigError("resources.client_num_cpus must be > 0, client_num_gpus >= 0")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    seeds: tuple[int, ...]
    output_dir: str
    device: str
    data: DataConfig
    topology: TopologyConfig
    partition: PartitionConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    arms: tuple[str, ...] = ("centralized", "local_only", "federated")
    fl: FLConfig = field(default_factory=FLConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    comms: CommsConfig = field(default_factory=CommsConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    notes: str = ""
    # Live demo only: sleep simulated link time x this factor (0 = never sleep).
    # Execution detail, excluded from the run id.
    real_sleep_scale: float = 0.0

    def validate(self) -> None:
        if self.real_sleep_scale < 0:
            raise ConfigError("real_sleep_scale must be >= 0")
        if not self.name or not all(c.isalnum() or c in "-_" for c in self.name):
            raise ConfigError("name must be non-empty and use only [A-Za-z0-9_-]")
        if self.seed < 0 or any(s < 0 for s in self.seeds):
            raise ConfigError("seeds must be non-negative integers")
        if self.device not in DEVICES:
            raise ConfigError(f"device must be one of {DEVICES}")
        self.data.validate()
        self.topology.validate()
        self.partition.validate()
        self.model.validate()
        self.train.validate()
        self.fl.validate()
        self.privacy.validate()
        self.comms.validate()
        self.resources.validate()
        if self.comms.dark.silo and self.comms.dark.silo not in [x.name for x in self.topology.silos]:
            raise ConfigError(f"comms.dark.silo {self.comms.dark.silo!r} is not a silo of the topology")
        if self.comms.dark.from_round > self.fl.rounds:
            raise ConfigError("comms.dark.from_round must be <= fl.rounds")
        if self.fl.strategy == "fedbuff":
            if not self.comms.enabled:
                raise ConfigError("fl.strategy 'fedbuff' needs comms.enabled (arrival order comes from the link model)")
            if self.privacy.enabled:
                # a fast silo trains more often than a slow one, so per-silo DP budgets
                # cannot be fixed in advance from rounds x local epochs
                raise ConfigError("privacy.enabled is not supported with fl.strategy 'fedbuff'")
            if self.privacy.attack.enabled:
                raise ConfigError("privacy.attack is only wired for synchronous strategies")
            if self.fl.buffer_k > len(self.topology.silos):
                raise ConfigError("fl.buffer_k must be <= the number of silos")
        if self.device == "cuda" and "federated" in self.arms and self.resources.client_num_gpus == 0:
            raise ConfigError("device 'cuda' with the federated arm needs resources.client_num_gpus > 0")
        if any(a not in ARMS for a in self.arms) or len(set(self.arms)) != len(self.arms):
            raise ConfigError(f"arms must be distinct entries from {ARMS}")
        sequence_input = self.data.name == "wesad" and self.data.wesad.representation == "sequence"
        if self.model.arch == "cnn1d" and not sequence_input:
            raise ConfigError("model.arch 'cnn1d' needs data.wesad.representation 'sequence'")
        if self.model.arch == "mlp" and sequence_input:
            raise ConfigError("model.arch 'mlp' needs feature input, not 'sequence'")
        if self.privacy.enabled:
            if self.train.class_weighting != "none":
                # balanced weights are a function of the whole dataset's label counts,
                # so one record changes every sample's loss: DP-SGD's per-sample
                # sensitivity bound would no longer hold.
                raise ConfigError("privacy.enabled needs train.class_weighting 'none'")
            if "federated" in self.arms and self.fl.strategy == "fedprox":
                # Opacus builds the update from per-sample gradients only; the proximal
                # term's gradient would be silently dropped.
                raise ConfigError("privacy.enabled is not supported with fl.strategy 'fedprox'")
        attack = self.privacy.attack
        if attack.enabled:
            if "federated" not in self.arms:
                raise ConfigError("privacy.attack needs the federated arm (it attacks a client update)")
            unknown = set(attack.silos) - {s.name for s in self.topology.silos}
            if unknown:
                raise ConfigError(f"privacy.attack.silos {sorted(unknown)} are not silos of the topology")
        n_silos = len(self.topology.silos)
        if self.data.is_synthetic:
            needed = n_silos + self.partition.holdout_subjects
            if self.data.synthetic.n_subjects < needed:
                raise ConfigError(
                    f"synthetic n_subjects must be >= silos + holdout_subjects ({needed})")

    # Execution details: where to write and which seeds to iterate. They do not change
    # what a single run computes, so they are excluded from the recorded experiment
    # definition and from the run id (they go into manifest.json instead).
    EXECUTION_KEYS = ("seed", "seeds", "output_dir", "resources", "real_sleep_scale")

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))

    def experiment_dict(self) -> dict[str, Any]:
        """Config minus execution keys and minus parameter blocks of unused datasets."""
        d = self.to_dict()
        for k in self.EXECUTION_KEYS:
            d.pop(k)
        for other in DATASETS:
            if other != self.data.name:
                d["data"].pop(other, None)
        if self.fl.strategy != "fedprox":
            d["fl"].pop("proximal_mu")
        if self.fl.strategy != "fedbuff":
            d["fl"].pop("buffer_k")
            d["fl"].pop("server_lr")
        if not self.comms.enabled:
            d.pop("comms")
        # Absent when unused, so configs that predate Phase 5 keep their run ids.
        if not self.privacy.enabled and not self.privacy.attack.enabled:
            d.pop("privacy")
        elif not self.privacy.attack.enabled:
            d["privacy"].pop("attack")
        return d

    def config_hash(self) -> str:
        """Stable hash of the experiment definition (execution keys excluded)."""
        blob = json.dumps(self.experiment_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def with_overrides(self, **kw: Any) -> "ExperimentConfig":
        cfg = dataclasses.replace(self, **{k: v for k, v in kw.items() if v is not None})
        cfg.validate()
        return cfg


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def find_config_root(path: Path) -> Path:
    """Walk upward from an experiment file to the directory holding base.yaml."""
    for parent in [path.parent, *path.parents]:
        if (parent / "base.yaml").is_file():
            return parent
    raise ConfigError(f"no base.yaml found above {path}")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at top level")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_topology(name: str, config_root: Path) -> TopologyConfig:
    raw = _read_yaml(config_root / "silos" / f"{name}.yaml")
    raw.setdefault("name", name)
    return _build(TopologyConfig, raw, "topology")


def dotted_to_nested(overrides: dict[str, Any]) -> dict[str, Any]:
    """{"partition.alpha": 0.1} -> {"partition": {"alpha": 0.1}}"""
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        cur = out
        *parents, leaf = key.split(".")
        for part in parents:
            cur = cur.setdefault(part, {})
        cur[leaf] = value
    return out


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load an experiment; ``overrides`` (dotted keys) are applied last, before validation."""
    path = Path(path).resolve()
    root = find_config_root(path)
    merged = deep_merge(_read_yaml(root / "base.yaml"), _read_yaml(path))
    if overrides:
        merged = deep_merge(merged, dotted_to_nested(overrides))
    comms = merged.get("comms")
    if isinstance(comms, dict) and isinstance(comms.get("los_profiles"), dict):
        # Partial overrides change only the named profiles/fields; the rest keep defaults.
        defaults = {k: dataclasses.asdict(v) for k, v in _default_los_profiles().items()}
        comms["los_profiles"] = deep_merge(defaults, comms["los_profiles"])
    topo_name = merged.pop("silos", None)
    if not isinstance(topo_name, str):
        raise ConfigError("'silos' must name a topology file in configs/silos/")
    merged["topology"] = dataclasses.asdict(load_topology(topo_name, root))
    seeds = merged.get("seeds")
    if seeds is None:
        merged["seeds"] = [merged.get("seed", 0)]
    cfg = _build(ExperimentConfig, merged, "")
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Strict dict -> dataclass construction
# ---------------------------------------------------------------------------


def _build(cls: type, raw: Any, where: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where or 'config'}: expected a mapping, got {type(raw).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise ConfigError(f"{where or 'config'}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for name, f in fields.items():
        sub = f"{where}.{name}" if where else name
        if name not in raw:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise ConfigError(f"missing required key: {sub}")
            continue
        kwargs[name] = _coerce(hints[name], raw[name], sub)
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    if origin is tuple:
        (inner, _ellipsis) = typing.get_args(tp)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list")
        return tuple(_coerce(inner, v, f"{where}[{i}]") for i, v in enumerate(value))
    if origin is dict:
        kt, vt = typing.get_args(tp)
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        return {_coerce(kt, k, where): _coerce(vt, v, f"{where}.{k}") for k, v in value.items()}
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported config type {tp}")


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj
