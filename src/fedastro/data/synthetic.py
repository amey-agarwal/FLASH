"""SYNTHETIC smoke-test data. Never a stand-in for real physiological data.

Generates Gaussian class clusters with a per-subject mean shift, so that a per-subject
split is mildly non-IID. Every bundle produced here carries provenance kind SYNTHETIC.
"""

from __future__ import annotations

import numpy as np

from fedastro.config import SyntheticParams
from fedastro.data.base import SYNTHETIC, DatasetBundle, Provenance, array_checksum

GENERATOR_VERSION = "synthetic-v1"


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
    checksum = array_checksum(X, y, subject_id)
    provenance = Provenance(
        dataset="SYNTHETIC",
        version=GENERATOR_VERSION,
        checksum=checksum,
        source="fedastro.data.synthetic (generated, not measured)",
        kind=SYNTHETIC,
        processed_checksum=checksum,
    )
    return DatasetBundle(
        X=X,
        y=y,
        subject_id=subject_id,
        class_names=tuple(f"class_{i}" for i in range(params.n_classes)),
        provenance=provenance,
        site_id=site_id,
    )
