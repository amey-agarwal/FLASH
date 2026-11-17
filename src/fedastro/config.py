"""Config loading and validation.

An experiment YAML is deep-merged over ``configs/base.yaml``. The ``silos`` key names
a topology file in ``configs/silos/`` which is expanded inline, so the resolved config
recorded in every result is fully self-contained.

``hierarchy`` (default off) names a hierarchical topology in ``configs/topologies/``:
regions, regional aggregators, one global aggregator and nodes (CLAUDE.md section 4). Its
nodes then also act as the silo list, so the flat arms run over the same nodes. Region,
aggregator, agency and node names are SCENARIO LABELS, never real deployments.

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

from fedastro.data import dataset_names


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or fails validation."""


DATASETS = dataset_names()  # from the registry in fedastro/data/__init__.py
PARTITION_METHODS = ("by_subject", "by_site", "dirichlet", "dirichlet_quantity_control", "by_group")
DIRICHLET_METHODS = ("dirichlet", "dirichlet_quantity_control")
WESAD_DEVICES = ("chest", "wrist")
ARCHS = ("mlp", "cnn1d")
OPTIMIZERS = ("adam", "sgd")
CLASS_WEIGHTING = ("none", "balanced")
# "hierarchical" (Phase H3, fl/hierarchy.py) needs a hierarchy and the local_only arm.
ARMS = ("centralized", "local_only", "federated", "ensemble_of_locals", "hierarchical")
# Paired comparisons (metrics/stats.py): federated minus each of these.
COMPARATORS = ("local_only", "ensemble_of_locals", "centralized")
STAT_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "auc")  # = metrics.classification.METRICS
FL_STRATEGIES = ("fedavg", "fedprox", "fedbuff")
SYNC_STRATEGIES = ("fedavg", "fedprox")
REPRESENTATIONS = ("features", "sequence")
DEVICES = ("cpu", "cuda")
LOS_PROFILES = ("none", "rare", "short_occasional", "periodic", "long_blackout")
LOS_KINDS = ("none", "poisson", "periodic")
# How a hierarchical topology hands data to its nodes (data/partition.py):
# "by_source": a node holds patients of its listed dataset groups; per-node train/val/test.
# "one_subject_per_node": one training subject per node; validation is the time-ordered
#   tail of that subject; the test set is the shared held-out subjects.
HIER_PARTITIONS = ("by_source", "one_subject_per_node")
SHIFT_KINDS = ("covariate", "concept", "both")
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
    # 0 = ungrouped (default; left out of the run id). >0 = contiguous subject groups
    # group_0.. with their own mean shift: a grouped bundle, for the by-group path.
    n_groups: int = 0
    # Optional names for the groups (default group_0..), so synthetic data can stand in
    # for a real dataset's sources when smoke-testing a hierarchical topology.
    group_names: tuple[str, ...] = ()

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
        if not 0 <= self.n_groups <= self.n_subjects:
            raise ConfigError("data.synthetic.n_groups must be in [0, n_subjects]")
        if self.group_names and (len(self.group_names) != self.n_groups
                                 or len(set(self.group_names)) != self.n_groups):
            raise ConfigError("data.synthetic.group_names must be n_groups distinct names")


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
class LinkConfig:
    """One link of a hierarchical topology. Scenario assumptions unless sourced."""

    one_way_latency_s: float = 0.0
    los_profile: str = "none"
    latency_presets_s: dict[str, float] = field(default_factory=dict)

    def validate(self, where: str) -> None:
        if self.one_way_latency_s < 0:
            raise ConfigError(f"{where}: one_way_latency_s must be >= 0")
        if self.los_profile not in LOS_PROFILES:
            raise ConfigError(f"{where}: los_profile must be one of {LOS_PROFILES}")
        for k, v in self.latency_presets_s.items():
            if v < 0:
                raise ConfigError(f"{where}: latency preset {k!r} must be >= 0")


@dataclass(frozen=True)
class NodeConfig:
    name: str
    agency: str  # agency analog, a scenario label
    # by_source only: the dataset groups (e.g. CinC sources) this node's patients come from.
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegionConfig:
    name: str
    aggregator: str
    nodes: tuple[NodeConfig, ...]
    node_link: LinkConfig = field(default_factory=LinkConfig)  # node <-> regional aggregator
    global_link: LinkConfig = field(default_factory=LinkConfig)  # regional <-> global aggregator


@dataclass(frozen=True)
class GlobalAggregatorConfig:
    name: str
    host: str  # the regional aggregator it is co-located with


@dataclass(frozen=True)
class HierarchyConfig:
    """Regions, aggregators and nodes (CLAUDE.md section 4b/4c). Used by Phase H3 onward;
    Phase H1 loads, validates and partitions only."""

    name: str
    min_participants: int  # regional aggregation needs this many reporting nodes
    global_aggregator: GlobalAggregatorConfig
    regions: tuple[RegionConfig, ...]
    partition: str = "by_source"
    # by_source: Dirichlet concentration of a source's patient shares across its regions
    # (quantity skew; large = near-equal shares).
    quantity_alpha: float = 1.0
    # Per-node split. by_source: fractions of the node's patients (patient-disjoint).
    # one_subject_per_node: fractions of the subject's windows, validation = the tail;
    # test_frac must be 0 (the test set is the shared held-out subjects).
    train_frac: float = 0.7
    val_frac: float = 0.1
    test_frac: float = 0.2
    # one_subject_per_node: windows dropped between train and the validation tail, so
    # overlapping windows never straddle the boundary.
    val_gap_windows: int = 0
    min_patients_per_node: int = 3  # by_source: at least one patient per split

    @property
    def nodes(self) -> tuple[NodeConfig, ...]:
        return tuple(n for r in self.regions for n in r.nodes)

    def region_of(self, node: str) -> str:
        for r in self.regions:
            if any(n.name == node for n in r.nodes):
                return r.name
        raise KeyError(node)

    def validate(self) -> None:
        w = f"hierarchy {self.name!r}"
        if not self.regions:
            raise ConfigError(f"{w}: at least one region is required")
        if self.partition not in HIER_PARTITIONS:
            raise ConfigError(f"{w}: partition must be one of {HIER_PARTITIONS}")
        if self.min_participants < 1:
            raise ConfigError(f"{w}: min_participants must be >= 1")
        regions = [r.name for r in self.regions]
        aggregators = [r.aggregator for r in self.regions]
        nodes = [n.name for n in self.nodes]
        for what, names in (("region", regions), ("aggregator", aggregators), ("node", nodes)):
            if any(not x for x in names):
                raise ConfigError(f"{w}: {what} names must be non-empty")
            if len(set(names)) != len(names):
                raise ConfigError(f"{w}: duplicate {what} names: {sorted(names)}")
        if not self.global_aggregator.name or self.global_aggregator.name in aggregators:
            raise ConfigError(f"{w}: global_aggregator.name must be non-empty and differ from the regional aggregators")
        if self.global_aggregator.host not in aggregators:
            raise ConfigError(f"{w}: global_aggregator.host {self.global_aggregator.host!r} is not a "
                              f"regional aggregator {aggregators}")
        for r in self.regions:
            if len(r.nodes) < self.min_participants:
                raise ConfigError(f"{w}: region {r.name!r} has {len(r.nodes)} nodes, fewer than "
                                  f"min_participants = {self.min_participants}; it could never aggregate")
            r.node_link.validate(f"{w} region {r.name!r} node_link")
            r.global_link.validate(f"{w} region {r.name!r} global_link")
            if any(not n.agency for n in r.nodes):
                raise ConfigError(f"{w}: region {r.name!r}: every node needs an agency label")
            used = [s for n in r.nodes for s in n.sources]
            if len(set(used)) != len(used):
                raise ConfigError(f"{w}: region {r.name!r}: a source is listed by two nodes "
                                  f"(its patients could not be kept in one node)")
            for n in r.nodes:
                if self.partition == "by_source" and not n.sources:
                    raise ConfigError(f"{w}: node {n.name!r} needs sources (partition by_source)")
                if self.partition != "by_source" and n.sources:
                    raise ConfigError(f"{w}: node {n.name!r}: sources are only meaningful for partition by_source")
        fracs = (self.train_frac, self.val_frac, self.test_frac)
        if abs(sum(fracs) - 1.0) > 1e-9 or self.train_frac <= 0 or self.val_frac <= 0 or self.test_frac < 0:
            raise ConfigError(f"{w}: train_frac and val_frac must be > 0, test_frac >= 0, and the three must sum to 1")
        if self.partition == "by_source":
            if self.test_frac <= 0:
                raise ConfigError(f"{w}: by_source needs test_frac > 0 (each node has its own test set)")
            if self.quantity_alpha <= 0:
                raise ConfigError(f"{w}: quantity_alpha must be > 0")
            if self.min_patients_per_node < 3:
                raise ConfigError(f"{w}: min_patients_per_node must be >= 3 (one patient per split)")
            if self.val_gap_windows != 0:
                raise ConfigError(f"{w}: val_gap_windows is only meaningful for one_subject_per_node")
        else:
            if self.test_frac != 0:
                raise ConfigError(f"{w}: one_subject_per_node needs test_frac 0 (the test set is the "
                                  f"shared held-out subjects)")
            if self.val_gap_windows < 0:
                raise ConfigError(f"{w}: val_gap_windows must be >= 0")

    def as_silos(self) -> "TopologyConfig":
        """The nodes as a flat silo list, so the flat arms run over the same nodes. A
        node's flat latency is its path to the global aggregator (node link + regional
        link); its loss-of-signal profile is its node link's. data_share is unused by the
        hierarchical partition and set to 1."""
        silos = []
        for r in self.regions:
            base = r.node_link.one_way_latency_s
            for n in r.nodes:
                silos.append(SiloConfig(
                    name=n.name, data_share=1.0,
                    one_way_latency_s=base + r.global_link.one_way_latency_s,
                    los_profile=r.node_link.los_profile,
                    latency_presets_s={k: base + v for k, v in r.global_link.latency_presets_s.items()}))
        return TopologyConfig(name=self.name, silos=tuple(silos))


@dataclass(frozen=True)
class ShiftConfig:
    """SIMULATED location shift (data/shift.py): a region-dependent transformation of the
    inputs, so that regions differ. No dataset here has a measured location shift."""

    enabled: bool = False
    kind: str = "both"  # covariate (offset + scale), concept (class-conditional offset), both
    # In units of a feature's pooled training standard deviation. 0 = the control: the data
    # is bit-identical to no shift.
    magnitude: float = 0.0
    seed: int = 0  # directions and selected features; never the run seed
    feature_fraction: float = 0.5  # share of features (channels, for sequences) that shift
    features: tuple[int, ...] = ()  # explicit feature indices; overrides feature_fraction
    # region -> multiplier of the magnitude. Regions not named: the first region of the
    # topology is the unshifted reference (0), the others 1.
    region_strength: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if self.kind not in SHIFT_KINDS:
            raise ConfigError(f"shift.kind must be one of {SHIFT_KINDS}")
        if self.magnitude < 0:
            raise ConfigError("shift.magnitude must be >= 0")
        if self.seed < 0:
            raise ConfigError("shift.seed must be >= 0")
        if not 0 < self.feature_fraction <= 1:
            raise ConfigError("shift.feature_fraction must be in (0, 1]")
        if any(i < 0 for i in self.features) or len(set(self.features)) != len(self.features):
            raise ConfigError("shift.features must be distinct non-negative indices")
        if any(v < 0 for v in self.region_strength.values()):
            raise ConfigError("shift.region_strength values must be >= 0")


@dataclass(frozen=True)
class HierFLConfig:
    """The hierarchical arm (fl/hierarchy.py). Rounds, local epochs and the client
    strategy (fedavg / fedprox) come from ``fl``."""

    # The global aggregator merges the regional models every this many regional rounds.
    global_every: int = 5
    # Scripted loss of signal (deterministic, independent of the link model):
    # node -> rounds in which it cannot reach its regional aggregator, and
    # region -> rounds in which its aggregator cannot reach the global aggregator
    # (only matters in merge rounds).
    offline: dict[str, tuple[int, ...]] = field(default_factory=dict)
    region_offline: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # Simulated links (comms.enabled): a node whose update has not reached its regional
    # aggregator this many SIMULATED seconds after the round starts misses the round.
    # Required (> 0) with comms on: without it a node in loss of signal is waited for
    # and its region never falls back.
    region_deadline_s: float = 0.0
    # A region whose model has not reached the global aggregator this long after it was
    # sent is left out of that merge. 0 = wait (unless it can never arrive).
    global_deadline_s: float = 0.0
    # Solar-conjunction blackout of one region's link to the global aggregator, in
    # simulated seconds (scenario assumption; needs comms.enabled). 0 duration = none.
    conjunction_region: str = ""
    conjunction_from_s: float = 0.0
    conjunction_duration_s: float = 0.0

    def validate(self) -> None:
        if self.global_every < 1:
            raise ConfigError("hier_fl.global_every must be >= 1")
        for name in ("region_deadline_s", "global_deadline_s", "conjunction_from_s", "conjunction_duration_s"):
            if getattr(self, name) < 0:
                raise ConfigError(f"hier_fl.{name} must be >= 0")
        for what, sched in (("offline", self.offline), ("region_offline", self.region_offline)):
            for k, rounds in sched.items():
                if any(r < 1 for r in rounds) or len(set(rounds)) != len(rounds):
                    raise ConfigError(f"hier_fl.{what}.{k}: rounds must be distinct and >= 1")
        if bool(self.conjunction_region) != (self.conjunction_duration_s > 0):
            raise ConfigError("hier_fl.conjunction_region and conjunction_duration_s > 0 go together")

    @property
    def is_default(self) -> bool:
        return self == HierFLConfig()


SECAGG_LEVELS = ("regional", "global")


@dataclass(frozen=True)
class SecAggConfig:
    """Simulated pairwise-mask secure aggregation (fl/secagg.py) for the hierarchical arm.
    A protocol SIMULATION: masks from a seeded PRG, not cryptographically secure."""

    enabled: bool = False
    levels: tuple[str, ...] = SECAGG_LEVELS  # where it applies
    # Keys (and so masks) agreed afresh for every aggregation, as in the standard protocol.
    key_agreement: str = "per_round"
    # The global merge runs only if at least this many regions are in its mask group (with
    # one, the "sum" is that region's model). The regional minimum is min_participants.
    global_min_group: int = 2
    frac_bits: int = 24  # fixed-point fractional bits of the ring encoding
    # Scripted dropout after the masks are set: node (regional level) or region (global
    # level) -> rounds in which it joins the mask group, then its masked upload never
    # arrives, so the survivors run the recovery step.
    dropout: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.levels or any(lv not in SECAGG_LEVELS for lv in self.levels) \
                or len(set(self.levels)) != len(self.levels):
            raise ConfigError(f"secagg.levels must be distinct entries from {SECAGG_LEVELS}")
        if self.key_agreement != "per_round":
            raise ConfigError("secagg.key_agreement must be 'per_round'")
        if self.global_min_group < 2:
            raise ConfigError("secagg.global_min_group must be >= 2 (one member's sum is its own model)")
        if not 8 <= self.frac_bits <= 40:
            raise ConfigError("secagg.frac_bits must be in [8, 40]")
        for k, rounds in self.dropout.items():
            if any(r < 1 for r in rounds) or len(set(rounds)) != len(rounds):
                raise ConfigError(f"secagg.dropout.{k}: rounds must be distinct and >= 1")

    @property
    def is_default(self) -> bool:
        return self == SecAggConfig()


UPDATE_TRACKS = ("regional", "global")
INJECT_KINDS = ("noise", "zero")


@dataclass(frozen=True)
class UpdateInject:
    """A SCRIPTED fault on one region's update downlink (a test of the gate, not a model of
    any real failure): the copy of a new version delivered to the region's nodes in
    ``round`` is degraded. Training is unaffected."""

    round: int
    region: str
    track: str = "regional"  # which version: the region's new regional model, or the global
    kind: str = "noise"  # noise: w + scale * std(w) * N(0, 1) per array; zero: all weights 0
    scale: float = 5.0

    def validate(self) -> None:
        if self.round < 1:
            raise ConfigError("updates.inject.round must be >= 1")
        if self.track not in UPDATE_TRACKS:
            raise ConfigError(f"updates.inject.track must be one of {UPDATE_TRACKS}")
        if self.kind not in INJECT_KINDS:
            raise ConfigError(f"updates.inject.kind must be one of {INJECT_KINDS}")
        if self.scale <= 0:
            raise ConfigError("updates.inject.scale must be > 0")


@dataclass(frozen=True)
class UpdatesConfig:
    """On-the-fly model updates with a validation gate (fl/versioning.py), hierarchical arm
    only. The gate decides which model version each node deploys; training is unchanged.
    Tolerance rule frozen in PREREG.md section 4."""

    enabled: bool = False
    metric: str = "macro_f1"  # the only option (class imbalance; PREREG section 4)
    # tolerance = k x bootstrap standard error of the deployed version's validation score
    tolerance_rule: str = "bootstrap_se"
    k: float = 2.0
    n_bootstrap: int = 200
    inject: tuple[UpdateInject, ...] = ()

    def validate(self) -> None:
        if self.metric != "macro_f1":
            raise ConfigError("updates.metric must be 'macro_f1'")
        if self.tolerance_rule != "bootstrap_se":
            raise ConfigError("updates.tolerance_rule must be 'bootstrap_se'")
        if self.k <= 0:
            raise ConfigError("updates.k must be > 0")
        if self.n_bootstrap < 20:
            raise ConfigError("updates.n_bootstrap must be >= 20")
        for inj in self.inject:
            inj.validate()

    @property
    def is_default(self) -> bool:
        return self == UpdatesConfig()


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
    # Grouped data only (GroupedDatasetBundle). Both are left out of the run id when empty.
    # group_map: silo name -> the groups it holds, for by_group. Empty = one silo per group,
    #   with silo names equal to group names. Groups in no silo and not held out are unused.
    # holdout_groups: whole groups excluded from training (no samples, no scaler
    #   statistics); they join the global test set. Any partition method.
    group_map: dict[str, tuple[str, ...]] = field(default_factory=dict)
    holdout_groups: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.method not in PARTITION_METHODS:
            raise ConfigError(f"partition.method must be one of {PARTITION_METHODS}")
        if self.method in DIRICHLET_METHODS and self.alpha <= 0:
            raise ConfigError(f"partition.alpha must be > 0 for method {self.method!r}")
        if self.method not in DIRICHLET_METHODS and self.alpha != 0:
            raise ConfigError("partition.alpha is only meaningful for the dirichlet methods")
        if self.holdout_subjects < 0:
            raise ConfigError("partition.holdout_subjects must be >= 1 (>= 0 with holdout_groups)")
        if self.holdout_seed < 0:
            raise ConfigError("partition.holdout_seed must be >= 0")
        if len(set(self.holdout_groups)) != len(self.holdout_groups):
            raise ConfigError("partition.holdout_groups has duplicates")
        if self.group_map and self.method != "by_group":
            raise ConfigError("partition.group_map is only meaningful for method 'by_group'")
        mapped = [g for groups in self.group_map.values() for g in groups]
        if any(not groups for groups in self.group_map.values()):
            raise ConfigError("partition.group_map: every silo needs at least one group")
        if len(set(mapped)) != len(mapped):
            raise ConfigError("partition.group_map: a group is mapped to more than one silo")
        if set(mapped) & set(self.holdout_groups):
            raise ConfigError("partition.group_map: a held-out group cannot also train in a silo")


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
class EvaluationConfig:
    """Phase 12 evaluation options. At their defaults the whole block is left out of the
    run id, so every earlier run id is unchanged.

    holdout_draws: number of held-out subject sets. Draw 0 is the existing holdout (from
      partition.holdout_seed); draw d > 0 uses its own fixed rng stream. Each (seed, draw)
      is one run, with run id ``<name>-seed<s>-draw<d>-<hash>`` when holdout_draws > 1.
    primary: the pre-registered primary comparison, ``federated-<comparator>:<metric>``
      (CLAUDE2.md rule 12). Empty = every comparison is exploratory."""

    holdout_draws: int = 1
    primary: str = ""

    def validate(self) -> None:
        if self.holdout_draws < 1:
            raise ConfigError("evaluation.holdout_draws must be >= 1")
        if self.primary:
            pair, _, metric = self.primary.partition(":")
            a, _, b = pair.partition("-")
            if a != "federated" or b not in COMPARATORS or metric not in STAT_METRICS:
                raise ConfigError(f"evaluation.primary must be 'federated-<{'|'.join(COMPARATORS)}>:"
                                  f"<{'|'.join(STAT_METRICS)}>', got {self.primary!r}")

    @property
    def is_default(self) -> bool:
        return self.holdout_draws == 1 and not self.primary


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
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    notes: str = ""
    # Hierarchical topology (configs/topologies/), or None = flat. Set by load_config from
    # the ``hierarchy`` key; left out of the run id when None.
    hierarchy: HierarchyConfig | None = None
    shift: ShiftConfig = field(default_factory=ShiftConfig)  # needs a hierarchy; default off
    hier_fl: HierFLConfig = field(default_factory=HierFLConfig)  # the hierarchical arm only
    secagg: SecAggConfig = field(default_factory=SecAggConfig)  # the hierarchical arm only
    updates: UpdatesConfig = field(default_factory=UpdatesConfig)  # the hierarchical arm only
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
        self.evaluation.validate()
        self.resources.validate()
        self.shift.validate()
        self.hier_fl.validate()
        self.secagg.validate()
        self.updates.validate()
        hier = self.hierarchy
        self._validate_hierarchical_arm()
        if self.shift.enabled and hier is None:
            raise ConfigError("shift.enabled needs a hierarchy: the shift is defined per region")
        if hier is not None:
            unknown = sorted(set(self.shift.region_strength) - {r.name for r in hier.regions})
            if unknown:
                raise ConfigError(f"shift.region_strength names unknown regions {unknown}")
        by_source = hier is not None and hier.partition == "by_source"
        if self.partition.holdout_subjects < 1 and not self.partition.holdout_groups and not by_source:
            raise ConfigError("partition.holdout_subjects must be >= 1 (>= 0 with holdout_groups)")
        if hier is not None:
            hier.validate()
            if self.topology != hier.as_silos():
                raise ConfigError("with a hierarchy, the silo list must be the hierarchy's nodes")
            p = self.partition
            if p.method != "by_subject" or p.group_map or p.holdout_groups:
                raise ConfigError("with a hierarchy, the hierarchy partitions the data: leave "
                                  "partition.method, group_map and holdout_groups at their defaults")
            if self.evaluation.holdout_draws > 1:
                raise ConfigError("evaluation.holdout_draws > 1 is not supported with a hierarchy")
            if by_source and p.holdout_subjects != 0:
                raise ConfigError("hierarchy partition by_source has per-node test sets: set "
                                  "partition.holdout_subjects to 0")
            if self.data.is_synthetic:
                syn = self.data.synthetic
                if by_source:
                    names = set(syn.group_names or (f"group_{g}" for g in range(syn.n_groups)))
                    missing = sorted({s for n in hier.nodes for s in n.sources} - names)
                    if missing:
                        raise ConfigError(f"hierarchy sources {missing} are not synthetic groups "
                                          f"(data.synthetic.n_groups / group_names)")
                elif syn.n_subjects != len(hier.nodes) + p.holdout_subjects:
                    raise ConfigError(f"one_subject_per_node needs n_subjects = nodes + holdout_subjects "
                                      f"({len(hier.nodes) + p.holdout_subjects})")
        if self.evaluation.holdout_draws > 1 and self.partition.holdout_groups:
            # Which groups may rotate into the held-out set depends on the dataset and its
            # agency mapping, so group redraws are designed with it (Phase 13).
            raise ConfigError("evaluation.holdout_draws > 1 re-draws held-out subjects only; "
                              "it cannot be combined with partition.holdout_groups yet")
        if self.evaluation.primary:
            comparator = self.evaluation.primary.split(":")[0].split("-", 1)[1]
            if "federated" not in self.arms or comparator not in self.arms:
                raise ConfigError(f"evaluation.primary {self.evaluation.primary!r} needs arms "
                                  f"'federated' and {comparator!r}")
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
        if self.partition.group_map and set(self.partition.group_map) != {s.name for s in self.topology.silos}:
            raise ConfigError("partition.group_map must have exactly one entry per topology silo")
        if self.data.is_synthetic and (self.partition.method == "by_group" or self.partition.holdout_groups) \
                and self.data.synthetic.n_groups == 0:
            raise ConfigError("by_group and holdout_groups need grouped data (data.synthetic.n_groups > 0)")
        if self.data.is_synthetic:
            needed = n_silos + self.partition.holdout_subjects
            if self.data.synthetic.n_subjects < needed:
                raise ConfigError(
                    f"synthetic n_subjects must be >= silos + holdout_subjects ({needed})")

    def _validate_hierarchical_arm(self) -> None:
        h, hf = self.hierarchy, self.hier_fl
        if "hierarchical" not in self.arms:
            if not hf.is_default:
                raise ConfigError("hier_fl is only meaningful with the 'hierarchical' arm")
            if not self.secagg.is_default:
                raise ConfigError("secagg is only implemented for the 'hierarchical' arm")
            if not self.updates.is_default:
                raise ConfigError("updates (the version gate) is only implemented for the 'hierarchical' arm")
            return
        up = self.updates  # with enabled false its settings are ignored (and not in the run id)
        if up.enabled and h is not None:
            for inj in up.inject:
                if inj.region not in {r.name for r in h.regions}:
                    raise ConfigError(f"updates.inject.region {inj.region!r} is not a region")
                if inj.round > self.fl.rounds:
                    raise ConfigError("updates.inject.round must be <= fl.rounds")
                if inj.track == "global" and inj.round % hf.global_every:
                    raise ConfigError("updates.inject with track 'global' needs a merge round "
                                      "(a multiple of hier_fl.global_every)")
        sa = self.secagg  # with enabled false its settings are ignored (and not in the run id)
        if sa.enabled:
            known = {n.name for n in h.nodes} | {r.name for r in h.regions} if h is not None else set()
            unknown = sorted(set(sa.dropout) - known)
            if unknown:
                raise ConfigError(f"secagg.dropout names unknown nodes or regions {unknown}")
            if any(r > self.fl.rounds for rounds in sa.dropout.values() for r in rounds):
                raise ConfigError("secagg.dropout: rounds must be <= fl.rounds")
        if h is None:
            raise ConfigError("the 'hierarchical' arm needs a hierarchy")
        if "local_only" not in self.arms:
            raise ConfigError("the 'hierarchical' arm needs the 'local_only' arm (each node's local model "
                              "is one of the three models compared)")
        if self.fl.strategy not in SYNC_STRATEGIES:
            raise ConfigError(f"the 'hierarchical' arm supports fl.strategy {SYNC_STRATEGIES}")
        if self.privacy.enabled or self.privacy.attack.enabled:
            raise ConfigError("privacy (DP-SGD, attack) is not supported with the 'hierarchical' arm")
        if hf.global_every > self.fl.rounds:
            raise ConfigError("hier_fl.global_every must be <= fl.rounds (else the global model never forms)")
        nodes, regions = {n.name for n in h.nodes}, {r.name for r in h.regions}
        for what, sched, names in (("offline", hf.offline, nodes), ("region_offline", hf.region_offline, regions)):
            unknown = sorted(set(sched) - names)
            if unknown:
                raise ConfigError(f"hier_fl.{what} names unknown {'nodes' if what == 'offline' else 'regions'} {unknown}")
            if any(r > self.fl.rounds for rounds in sched.values() for r in rounds):
                raise ConfigError(f"hier_fl.{what}: rounds must be <= fl.rounds")
        if hf.conjunction_region:
            if hf.conjunction_region not in regions:
                raise ConfigError(f"hier_fl.conjunction_region {hf.conjunction_region!r} is not a region")
            if not self.comms.enabled:
                raise ConfigError("hier_fl.conjunction_region needs comms.enabled (it is a link outage)")
        if self.comms.enabled and hf.region_deadline_s <= 0:
            raise ConfigError("the 'hierarchical' arm with comms.enabled needs hier_fl.region_deadline_s > 0")
        if not self.comms.enabled and (hf.region_deadline_s or hf.global_deadline_s):
            raise ConfigError("hier_fl deadlines need comms.enabled")

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
        # Phase 11 fields: absent when unused, so earlier run ids are unchanged.
        if "synthetic" in d["data"] and not self.data.synthetic.n_groups:
            d["data"]["synthetic"].pop("n_groups")
        if "synthetic" in d["data"] and not self.data.synthetic.group_names:
            d["data"]["synthetic"].pop("group_names")
        if self.hierarchy is None:  # Phase H1 block, same rule
            d.pop("hierarchy")
        if not self.shift.enabled:  # Phase H2 block, same rule
            d.pop("shift")
        if "hierarchical" not in self.arms:  # Phase H3 block, same rule
            d.pop("hier_fl")
        if not self.secagg.enabled:  # Phase H4 block, same rule
            d.pop("secagg")
        if not self.updates.enabled:  # Phase H5 block, same rule
            d.pop("updates")
        if not self.partition.group_map:
            d["partition"].pop("group_map")
        if not self.partition.holdout_groups:
            d["partition"].pop("holdout_groups")
        if self.evaluation.is_default:  # Phase 12 block, same rule
            d.pop("evaluation")
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


def load_hierarchy(name: str, config_root: Path) -> HierarchyConfig:
    raw = _read_yaml(config_root / "topologies" / f"{name}.yaml")
    raw.setdefault("name", name)
    return _build(HierarchyConfig, raw, "hierarchy")


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
    experiment = _read_yaml(path)
    merged = deep_merge(_read_yaml(root / "base.yaml"), experiment)
    if overrides:
        merged = deep_merge(merged, dotted_to_nested(overrides))
    comms = merged.get("comms")
    if isinstance(comms, dict) and isinstance(comms.get("los_profiles"), dict):
        # Partial overrides change only the named profiles/fields; the rest keep defaults.
        defaults = {k: dataclasses.asdict(v) for k, v in _default_los_profiles().items()}
        comms["los_profiles"] = deep_merge(defaults, comms["los_profiles"])
    topo_name = merged.pop("silos", None)
    hier_name = merged.pop("hierarchy", None)
    hierarchy = None
    if hier_name is not None:
        if not isinstance(hier_name, str):
            raise ConfigError("'hierarchy' must name a topology file in configs/topologies/ (or be null)")
        if "silos" in experiment:
            raise ConfigError("set either 'silos' (flat) or 'hierarchy', not both: a hierarchy's "
                              "nodes are its silos")
        hierarchy = load_hierarchy(hier_name, root)
        hierarchy.validate()  # before its nodes are expanded into the silo list
        merged["topology"] = dataclasses.asdict(hierarchy.as_silos())
    else:
        if not isinstance(topo_name, str):
            raise ConfigError("'silos' must name a topology file in configs/silos/")
        merged["topology"] = dataclasses.asdict(load_topology(topo_name, root))
    seeds = merged.get("seeds")
    if seeds is None:
        merged["seeds"] = [merged.get("seed", 0)]
    cfg = _build(ExperimentConfig, merged, "")
    if hierarchy is not None:
        cfg = dataclasses.replace(cfg, hierarchy=hierarchy)
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
