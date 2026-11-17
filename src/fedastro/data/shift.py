"""SIMULATED location shift for hierarchical topologies (CLAUDE.md section 2 and Phase H2).

No dataset in this project contains real location-dependent physiology. This module
injects a region-dependent transformation so that regions have a reason to differ and
hypothesis H2 is testable. Every result that uses it is tagged ``shift.provenance:
SIMULATED``. The shift directions are arbitrary seeded vectors, not modelled physiology.

For a region with strength s and a config magnitude m (both >= 0), with a = m * s, each
sample (x, y) of that region becomes, per selected feature (or channel, for sequences):

    x' = mean + (x - mean) * gain + offset + class_offset[y]

- ``mean`` and ``std`` are those of the pooled, unshifted TRAINING data of all nodes, so
  the magnitude is in units of a feature's training standard deviation.
- Covariate shift (kinds "covariate", "both"): ``offset = a * std * u`` with u a unit-RMS
  direction over the selected features, and ``gain = exp(GAIN_RATE * a * g)``, g in [-1, 1].
- Concept shift (kinds "concept", "both"): ``class_offset[k] = a * std * v_k``, with the
  v_k unit-RMS and summing to zero over classes. The feature-label relationship then
  differs by region, which per-feature standardisation cannot undo.

The same transformation is applied to a region's train, validation and test data. All
draws come from ``shift.seed`` only (never the run seed) and are made for every region in
topology order, so changing one region's strength or the magnitude never changes another
region's directions. With a = 0 a region's data is returned untouched (bit-identical).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from fedastro.data.base import DatasetBundle
from fedastro.data.partition import HierFederation
from fedastro.data.preprocess import EPS, Standardizer

if TYPE_CHECKING:
    from fedastro.config import HierarchyConfig, ShiftConfig

PROVENANCE = "SIMULATED"
GAIN_RATE = 0.2  # gain = exp(GAIN_RATE * a * g): at a = 1 a selected feature's spread changes by up to ~22%
_SHIFT_STREAM = 0x53484946
_QUANTILES = np.linspace(0.005, 0.995, 199)


@dataclass(frozen=True)
class RegionShift:
    """One region's transformation; arrays have one entry per feature (or channel)."""

    amount: float  # a = magnitude * region strength; 0 = identity
    mean: np.ndarray
    gain: np.ndarray
    offset: np.ndarray
    class_offset: np.ndarray  # (n_classes, n_features)

    def apply(self, data: DatasetBundle) -> DatasetBundle:
        if self.amount == 0 or not len(data):
            return data
        shape = (1, -1) if data.X.ndim == 2 else (1, -1, 1)
        X = data.X.astype(np.float64)
        per_sample = self.class_offset[data.y]
        per_sample = per_sample if data.X.ndim == 2 else per_sample[:, :, None]
        out = (self.mean.reshape(shape) + (X - self.mean.reshape(shape)) * self.gain.reshape(shape)
               + self.offset.reshape(shape) + per_sample)
        return dataclasses.replace(data, X=out.astype(np.float32))


def strengths(shift: ShiftConfig, hierarchy: HierarchyConfig) -> dict[str, float]:
    """Per-region strength. Default: the first region is the unshifted reference (0), the
    others 1; ``shift.region_strength`` overrides the regions it names."""
    names = [r.name for r in hierarchy.regions]
    out = {name: (0.0 if i == 0 else 1.0) for i, name in enumerate(names)}
    out.update(shift.region_strength)
    return out


def selected_features(shift: ShiftConfig, n_features: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted indices of the shifted features. The rng draw is made even when an explicit
    list is given, so the directions do not depend on how the features were chosen."""
    k = max(1, int(np.ceil(shift.feature_fraction * n_features)))
    drawn = np.sort(rng.choice(n_features, size=k, replace=False))
    if not shift.features:
        return drawn
    if max(shift.features) >= n_features:
        raise ValueError(f"shift.features index {max(shift.features)} out of range for {n_features} features")
    return np.array(sorted(shift.features))


def _unit_rms(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = v * mask
    return v / (np.sqrt((v[..., mask > 0] ** 2).mean(axis=-1, keepdims=True)) + EPS)


def make_region_shifts(shift: ShiftConfig, hierarchy: HierarchyConfig, mean: np.ndarray, std: np.ndarray,
                       n_classes: int) -> tuple[dict[str, RegionShift], np.ndarray]:
    """The transformation of every region, and the selected feature indices."""
    n_features = len(mean)
    rng = np.random.default_rng([shift.seed, _SHIFT_STREAM])
    selected = selected_features(shift, n_features, rng)
    mask = np.zeros(n_features)
    mask[selected] = 1.0
    covariate = shift.kind in ("covariate", "both")
    concept = shift.kind in ("concept", "both")
    amount = strengths(shift, hierarchy)
    out: dict[str, RegionShift] = {}
    for region in hierarchy.regions:  # every draw is made for every region, in topology order
        u = _unit_rms(rng.normal(size=n_features), mask)
        g = rng.uniform(-1.0, 1.0, size=n_features) * mask
        v = rng.normal(size=(n_classes, n_features))
        v = _unit_rms(v - v.mean(axis=0, keepdims=True), mask)
        v = v - v.mean(axis=0, keepdims=True)  # exactly zero-sum over classes after rescaling
        a = shift.magnitude * amount[region.name]
        out[region.name] = RegionShift(
            amount=a, mean=mean,
            gain=np.exp(GAIN_RATE * a * g) if covariate else np.ones(n_features),
            offset=a * std * u if covariate else np.zeros(n_features),
            class_offset=a * std * v if concept else np.zeros((n_classes, n_features)))
    return out, selected


def apply_shift(fed: HierFederation, hierarchy: HierarchyConfig, shift: ShiftConfig) -> tuple[HierFederation, dict[str, Any]]:
    """Return the shifted federation and the result's ``shift`` section. With the shift
    disabled, the federation is returned as is."""
    if not shift.enabled:
        return fed, shift_section_off()
    first = next(iter(fed.nodes.values())).train
    pooled = np.concatenate([s.train.X for s in fed.nodes.values()])
    ref = Standardizer.fit(pooled)  # per feature, or per channel for sequences
    shifts, selected = make_region_shifts(shift, hierarchy, ref.mean, ref.std, len(first.class_names))
    nodes = {name: dataclasses.replace(
        s, train=shifts[s.region].apply(s.train), val=shifts[s.region].apply(s.val),
        test=None if s.test is None else shifts[s.region].apply(s.test)) for name, s in fed.nodes.items()}
    region_holdout = {}
    if fed.holdout is not None:  # shared test set: each region is tested on its own shifted copy
        region_holdout = {r.name: shifts[r.name].apply(fed.holdout) for r in hierarchy.regions}
    shifted = dataclasses.replace(fed, nodes=nodes, region_holdout=region_holdout)
    amount = strengths(shift, hierarchy)
    reference = next((r.name for r in hierarchy.regions if amount[r.name] == 0), hierarchy.regions[0].name)
    section = {
        "enabled": True, "kind": shift.kind, "magnitude": shift.magnitude, "provenance": PROVENANCE,
        "seed": shift.seed, "features": [int(i) for i in selected], "gain_rate": GAIN_RATE,
        "reference_region": reference,
        "note": "Injected by fedastro.data.shift; arbitrary seeded directions, not measured or modelled physiology.",
        "regions": {},
    }
    for region in hierarchy.regions:
        section["regions"][region.name] = {
            "strength": amount[region.name],
            # "before" is the natural difference between the regions' patients; "after" adds the shift.
            "distance_before": region_distance(fed, region.name, reference, ref),
            "distance_after": region_distance(shifted, region.name, reference, ref),
        }
    return shifted, section


def shift_section_off() -> dict[str, Any]:
    return {"enabled": False, "kind": None, "magnitude": 0.0, "provenance": PROVENANCE}


def _region_train(fed: HierFederation, region: str) -> tuple[np.ndarray, np.ndarray]:
    parts = [s.train for s in fed.nodes.values() if s.region == region]
    return np.concatenate([p.X for p in parts]), np.concatenate([p.y for p in parts])


def _columns(X: np.ndarray) -> np.ndarray:
    """(n_values, n_features): sequences are flattened over windows and time per channel."""
    return X if X.ndim == 2 else X.transpose(0, 2, 1).reshape(-1, X.shape[1])


def region_distance(fed: HierFederation, region: str, reference: str, ref: Standardizer) -> dict[str, float]:
    """Distance of a region's training data from the reference region's, in units of the
    pooled unshifted training standard deviation, averaged over features:
    - smd_mean / smd_max: |difference of feature means| (standardised mean difference).
    - wasserstein_mean: 1-D Wasserstein-1 distance per feature, from 199 quantiles.
    - smd_class_mean: the same mean difference within each class, averaged over the classes
      present in both regions. Concept shift shows here even when it cancels in the marginal."""
    Xa, ya = _region_train(fed, region)
    Xb, yb = _region_train(fed, reference)
    scale = ref.std + EPS
    A, B = _columns(Xa.astype(np.float64)), _columns(Xb.astype(np.float64))
    smd = np.abs(A.mean(axis=0) - B.mean(axis=0)) / scale
    w1 = np.abs(np.quantile(A, _QUANTILES, axis=0) - np.quantile(B, _QUANTILES, axis=0)).mean(axis=0) / scale
    per_class = []
    for k in np.intersect1d(ya, yb):
        Ak, Bk = _columns(Xa[ya == k].astype(np.float64)), _columns(Xb[yb == k].astype(np.float64))
        per_class.append((np.abs(Ak.mean(axis=0) - Bk.mean(axis=0)) / scale).mean())
    return {"smd_mean": round(float(smd.mean()), 6), "smd_max": round(float(smd.max()), 6),
            "wasserstein_mean": round(float(w1.mean()), 6),
            "smd_class_mean": round(float(np.mean(per_class)), 6) if per_class else None}


def format_shift_table(section: dict[str, Any]) -> str:
    """Plain-text distance table for ``fedastro inspect``."""
    if not section["enabled"]:
        return "shift: off"
    lines = [f"shift: {PROVENANCE} kind={section['kind']} magnitude={section['magnitude']} seed={section['seed']} "
             f"features={len(section['features'])} reference={section['reference_region']} "
             f"(distances from the reference region's training data, in training-std units)",
             f"{'region':<8}{'strength':>9}" + "".join(
                 f"{h:>25}" for h in ("smd_mean bef -> aft", "wasserstein bef -> aft", "class smd bef -> aft"))]
    for name, r in section["regions"].items():
        b, a = r["distance_before"], r["distance_after"]
        cells = "".join(f"{f'{b[k]:.3f} -> {a[k]:.3f}':>25}" for k in ("smd_mean", "wasserstein_mean", "smd_class_mean"))
        lines.append(f"{name:<8}{r['strength']:>9.2f}{cells}")
    return "\n".join(lines)
