"""Dataset registry. Adding a dataset means adding one ``DatasetSpec`` here (plus its
loader module and its ``DataConfig`` parameter block); ``pipeline.load_dataset`` and the
``data.name`` validation read from this table and need no edit.

A plain dict rather than a registration decorator: config validation needs the names
before any loader is imported, and loaders import ``fedastro.config``, so loader modules
are imported lazily on first use.

Each spec declares its provenance ``domain`` (CLAUDE2.md rule 10), and ``DatasetSpec.load``
writes it into every loaded bundle's ``Provenance``: the registry is the single source of a
dataset's domain. Artifacts written before Phase 11 have no domain field; readers resolve
those through ``domain_of``.
"""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fedastro.data.base import DOMAINS

if TYPE_CHECKING:
    from fedastro.config import DataConfig
    from fedastro.data.base import DatasetBundle


@dataclass(frozen=True)
class DatasetSpec:
    name: str  # value of data.name; also the DataConfig field holding the parameters
    label: str  # Provenance.dataset written by the loader
    domain: str  # one of base.DOMAINS
    module: str  # imported on first use
    loader: str = "load"  # function (data_config, seed) -> DatasetBundle

    def load(self, data: DataConfig, seed: int) -> DatasetBundle:
        """Load and stamp the registry's domain into the provenance. A loader that sets
        a different domain is a contradiction and is rejected."""
        bundle = getattr(importlib.import_module(self.module), self.loader)(data, seed)
        p = bundle.provenance
        if p.domain and p.domain != self.domain:
            raise ValueError(f"dataset {self.name!r}: loader domain {p.domain!r} != registry {self.domain!r}")
        if p.dataset != self.label:
            raise ValueError(f"dataset {self.name!r}: loader label {p.dataset!r} != registry {self.label!r}")
        return dataclasses.replace(bundle, provenance=dataclasses.replace(p, domain=self.domain))


REGISTRY: dict[str, DatasetSpec] = {
    spec.name: spec for spec in (
        DatasetSpec("synthetic", "SYNTHETIC", "synthetic", "fedastro.data.synthetic"),
        DatasetSpec("wesad", "WESAD", "terrestrial_proxy", "fedastro.data.wesad"),
    )
}

for _spec in REGISTRY.values():
    if _spec.domain not in DOMAINS:
        raise ValueError(f"dataset {_spec.name!r}: domain {_spec.domain!r} not in {DOMAINS}")


def dataset_names() -> tuple[str, ...]:
    return tuple(REGISTRY)


def get_dataset(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise KeyError(f"dataset {name!r} is not registered; known: {dataset_names()}")
    return REGISTRY[name]


def domain_of(provenance: dict[str, Any]) -> str | None:
    """The provenance's ``domain``, or for an artifact written before Phase 11, the
    registry's domain for its dataset label. None if neither is known."""
    if provenance.get("domain"):
        return provenance["domain"]
    for spec in REGISTRY.values():
        if spec.label == provenance.get("dataset"):
            return spec.domain
    return None
