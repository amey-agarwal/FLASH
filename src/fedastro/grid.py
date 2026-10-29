"""Grid sweeps: one YAML expands into experiment variants x seeds.

    name: phase4_alpha
    base: ../experiments/wesad_dirichlet.yaml   # relative to this file
    seeds: [0, 1, 2, 3, 4]                       # optional; default = base seeds
    axes:                                        # cartesian product, dotted keys
      partition.alpha: [0.1, 1.0]
      fl.strategy: [fedavg, fedprox]
    fixed:                                       # optional overrides for every variant
      arms: [federated]
    include_base: true                           # optional: also run the base config as-is
                                                 # (variant 0, "<sweep>-base"; fixed not applied)
    common:                                      # optional overrides for EVERY variant,
      privacy.attack.trials: 10                  # the base variant included
    variants:                                    # optional, instead of (or after) axes: an
      - {fl.strategy: fedavg}                    # explicit list of override sets, for
      - {fl.strategy: fedavg, comms.sync_deadline_s: 3600}   # non-cartesian comparisons

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
    short = {"alpha": "a", "strategy": "", "method": "", "proximal_mu": "mu",
             "target_epsilon": "eps", "preset": "", "sync_deadline_s": "dl", "los": "los"}.get(leaf, leaf)
    return f"{short}{text}"


def load_grid(path: str | Path) -> tuple[str, list[Variant]]:
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text()) or {}
    unknown = set(spec) - {"name", "base", "seeds", "axes", "fixed", "include_base", "common", "variants"}
    if unknown:
        raise ConfigError(f"{path}: unknown grid keys {sorted(unknown)}")
    for key in ("name", "base"):
        if key not in spec:
            raise ConfigError(f"{path}: grid needs '{key}'")
    if not spec.get("axes") and not spec.get("variants"):
        raise ConfigError(f"{path}: grid needs 'axes' or 'variants'")
    axes: dict[str, list] = spec.get("axes") or {}
    if any(not isinstance(v, list) or not v for v in axes.values()):
        raise ConfigError(f"{path}: every axis must be a non-empty list")
    explicit = spec.get("variants") or []
    if not isinstance(explicit, list) or any(not isinstance(v, dict) or not v for v in explicit):
        raise ConfigError(f"{path}: 'variants' must be a list of non-empty override mappings")
    base = (path.parent / spec["base"]).resolve()
    seeds = {"seeds": spec["seeds"]} if "seeds" in spec else {}
    common = spec.get("common") or {}
    variants = []
    if spec.get("include_base", False):
        name = f"{spec['name']}-base"
        variants.append(Variant(0, name, {}, load_config(base, {**common, "name": name, **seeds})))
    combos = [dict(zip(axes, c)) for c in itertools.product(*axes.values())] if axes else []
    for overrides in combos + explicit:
        name = "-".join([spec["name"], *(_slug(k, v) for k, v in overrides.items())])
        cfg = load_config(base, {**common, **(spec.get("fixed") or {}), **overrides, "name": name, **seeds})
        variants.append(Variant(len(variants), name, overrides, cfg))
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise ConfigError(f"{path}: variants produce duplicate names {sorted(n for n in names if names.count(n) > 1)}")
    return spec["name"], variants
