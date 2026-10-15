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


DATASETS = ("synthetic",)  # Phase 1 adds "wesad"; Phase 9 adds "osd435", "eicu".
PARTITION_METHODS = ("by_subject",)  # Phase 1 adds "by_site", "dirichlet".
DEVICES = ("cpu", "cuda")
LOS_PROFILES = ("none", "rare", "short_occasional", "periodic", "long_blackout")


@dataclass(frozen=True)
class SyntheticParams:
    n_subjects: int = 20
    windows_per_subject: int = 30
    n_features: int = 8
    n_classes: int = 3
    class_sep: float = 1.0
    subject_shift: float = 0.5

    def validate(self) -> None:
        for name in ("n_subjects", "windows_per_subject", "n_features", "n_classes"):
            if getattr(self, name) < 1:
                raise ConfigError(f"data.synthetic.{name} must be >= 1")
        if self.n_classes < 2:
            raise ConfigError("data.synthetic.n_classes must be >= 2")
        if self.class_sep < 0 or self.subject_shift < 0:
            raise ConfigError("data.synthetic.class_sep and subject_shift must be >= 0")


@dataclass(frozen=True)
class DataConfig:
    name: str = "synthetic"
    synthetic: SyntheticParams = field(default_factory=SyntheticParams)

    def validate(self) -> None:
        if self.name not in DATASETS:
            raise ConfigError(f"data.name must be one of {DATASETS}, got {self.name!r}")
        self.synthetic.validate()

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

    def validate(self) -> None:
        if self.method not in PARTITION_METHODS:
            raise ConfigError(f"partition.method must be one of {PARTITION_METHODS}")


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
    notes: str = ""

    def validate(self) -> None:
        if not self.name or not all(c.isalnum() or c in "-_" for c in self.name):
            raise ConfigError("name must be non-empty and use only [A-Za-z0-9_-]")
        if self.seed < 0 or any(s < 0 for s in self.seeds):
            raise ConfigError("seeds must be non-negative integers")
        if self.device not in DEVICES:
            raise ConfigError(f"device must be one of {DEVICES}")
        self.data.validate()
        self.topology.validate()
        self.partition.validate()
        if self.data.is_synthetic and self.data.synthetic.n_subjects < len(self.topology.silos):
            raise ConfigError("synthetic n_subjects must be >= number of silos")

    # Execution details: where to write and which seeds to iterate. They do not change
    # what a single run computes, so they are excluded from the recorded experiment
    # definition and from the run id (they go into manifest.json instead).
    EXECUTION_KEYS = ("seed", "seeds", "output_dir")

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))

    def experiment_dict(self) -> dict[str, Any]:
        d = self.to_dict()
        for k in self.EXECUTION_KEYS:
            d.pop(k)
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


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    root = find_config_root(path)
    merged = deep_merge(_read_yaml(root / "base.yaml"), _read_yaml(path))
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
