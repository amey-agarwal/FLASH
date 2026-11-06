"""SYNTHETIC smoke-test data. Never a stand-in for real physiological data.

Generates Gaussian class clusters with a per-subject mean shift, so that a per-subject
split is mildly non-IID. Every bundle produced here carries provenance kind SYNTHETIC.

With ``n_groups > 0`` subjects are split into contiguous groups ``group_0..``, each with
its own mean shift (scale ``subject_shift``), and a GroupedDatasetBundle is returned, for
smoke-testing the by-group path. The group shifts are drawn after every other draw, so
the ungrouped data stays exactly as before.
"""

from __future__ import annotations

import numpy as np

from fedastro.config import DataConfig, SyntheticParams
from fedastro.data.base import SYNTHETIC, DatasetBundle, GroupedDatasetBundle, Provenance, array_checksum

GENERATOR_VERSION = "synthetic-v1"


def load(data: DataConfig, seed: int) -> DatasetBundle:
    """Registry entry point (fedastro.data.REGISTRY)."""
    return load_synthetic(data.synthetic, seed)


def load_synthetic(params: SyntheticParams, seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    class_means = rng.normal(0.0, params.class_sep, size=(params.n_classes, params.n_features))
    X_parts, y_parts, s_parts = [], [], []
    for subject in range(params.n_subjects):
        offset = rng.normal(0.0, params.subject_shift, size=params.n_features)
        y = rng.integers(0, params.n_classes, size=params.windows_per_subject)
        noise = rng.normal(0.0, 1.0, size=(params.windows_per_subject, params.n_features))
        X_parts.append(class_means[y] + offset + noise)
        y_parts.append(y)
        s_parts.append(np.full(params.windows_per_subject, subject))
    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.int64)
    subject_id = np.concatenate(s_parts).astype(np.int64)
    site_id = subject_id % params.n_sites if params.n_sites > 0 else None
    group_id = None
    if params.n_groups > 0:
        group_id = (subject_id * params.n_groups // params.n_subjects).astype(np.int64)
        group_shift = rng.normal(0.0, params.subject_shift, size=(params.n_groups, params.n_features))
        X = (X + group_shift[group_id]).astype(np.float32)
    checksum = array_checksum(X, y, subject_id)
    provenance = Provenance(
        dataset="SYNTHETIC",
        version=GENERATOR_VERSION,
        checksum=checksum,
        source="fedastro.data.synthetic (generated, not measured)",
        kind=SYNTHETIC,
        processed_checksum=checksum,
    )
    common = dict(X=X, y=y, subject_id=subject_id,
                  class_names=tuple(f"class_{i}" for i in range(params.n_classes)),
                  provenance=provenance, site_id=site_id)
    if group_id is None:
        return DatasetBundle(**common)
    return GroupedDatasetBundle(**common, group_id=group_id,
                                group_names=tuple(f"group_{g}" for g in range(params.n_groups)))
