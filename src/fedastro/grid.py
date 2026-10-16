"""Grid sweeps: one YAML expands into experiment variants x seeds.

    name: phase4_alpha
    base: ../experiments/wesad_dirichlet.yaml   # relative to this file
    seeds: [0, 1, 2, 3, 4]                       # optional; default = base seeds
    axes:                                        # cartesian product, dotted keys
      partition.alpha: [0.1, 1.0]
      fl.strategy: [fedavg, fedprox]
    fixed:                                       # optional overrides for every variant
      arms: [federated]

Each variant gets its own experiment name ``<sweep>-<axis values>`` and therefore its
own run ids and per-variant summary. ``--index`` runs a single variant so a cluster can
run one variant per job; ``--resume`` skips runs whose result.json already exists.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fedastro.config import ConfigError, ExperimentConfig, load_config


@dataclass(frozen=True)
class Variant:
    index: int
    name: str
    overrides: dict[str, Any]
    config: ExperimentConfig


def _slug(key: str, value: Any) -> str:
    leaf = key.split(".")[-1]
    text = re.sub(r"[^A-Za-z0-9]+", "", str(value).replace(".", "p"))
    short = {"alpha": "a", "strategy": "", "method": "", "proximal_mu": "mu"}.get(leaf, leaf)
    return f"{short}{text}"


def load_grid(path: str | Path) -> tuple[str, list[Variant]]:
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text()) or {}
    unknown = set(spec) - {"name", "base", "seeds", "axes", "fixed"}
    if unknown:
        raise ConfigError(f"{path}: unknown grid keys {sorted(unknown)}")
    for key in ("name", "base", "axes"):
        if key not in spec:
            raise ConfigError(f"{path}: grid needs '{key}'")
    axes: dict[str, list] = spec["axes"]
    if not axes or any(not isinstance(v, list) or not v for v in axes.values()):
        raise ConfigError(f"{path}: every axis must be a non-empty list")
    base = (path.parent / spec["base"]).resolve()
    variants = []
    for i, combo in enumerate(itertools.product(*axes.values())):
        overrides = dict(zip(axes, combo))
        name = "-".join([spec["name"], *(_slug(k, v) for k, v in overrides.items())])
        all_overrides = {**(spec.get("fixed") or {}), **overrides, "name": name}
        if "seeds" in spec:
            all_overrides["seeds"] = spec["seeds"]
        cfg = load_config(base, all_overrides)
        variants.append(Variant(i, name, overrides, cfg))
    return spec["name"], variants
