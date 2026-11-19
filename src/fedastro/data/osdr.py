"""OSD-435 loader (NASA Open Science Data Repository), Phase H9. Flat topologies only.

What the data is (OSDR study description): a GROUND-BASED study. Male C57BL/6J mice were
irradiated at the NASA Space Radiation Laboratory (protons, 16O ions, protons followed by 16O,
or 137Cs gamma rays as a reference) or sham-irradiated, and measured by ultrasonography
(echocardiography, Vevo 2100) after irradiation. Not spaceflight data, and not human data:
here it is a stated terrestrial analog of space-radiation exposure. FLUID (Casaletto et al.,
bioRxiv 10.1101/2025.01.14.633017) used this study for its biomedical run.

File: ``LSDS-22_Echocardiography_aorta_SUBMITTED.csv`` (abdominal aorta Doppler + heart rate),
downloaded by ``scripts/fetch_data.py osd435`` to ``data/raw/``. Its SHA-256 is pinned below;
a different file is refused (the repository may revise a file; re-inspect before accepting).

Task (user-approved 2026-11-19, the FLUID setup as reported in its paper):
- Label: radiation dose, 0 Gy (``sham``) vs >= 0.5 Gy (``irradiated``).
- Rows: low doses (0.05, 0.1, 0.25 Gy) removed; rows with a missing feature removed.
- Features: radiation type (one-hot, 4 types), month post-irradiation, heart rate, aorta peak
  velocity (``VELOCITY``, mm/s; the file also has ``PEAKVEL`` in cm, correlation 0.98: not
  used), velocity-time integral (``VTI``), mean velocity (``MEANVEL``).
- ``include_radtype: false`` drops the radiation-type columns and ``include_month0: false``
  drops month-0 rows: the pre-registered sensitivity variant. Radiation type partly gives the
  label away (type 4 has no sham mice), and month 0 is in every mouse's rows but not in the
  study description (meaning unconfirmed).
- ``subject_id`` = mouse (``SID``); every mouse has one dose. Mice have 1-5 rows, so splits
  are by mouse (no mouse in two splits).
No normalisation here; scalers are fit on training data only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fedastro.config import DataConfig, Osd435Params
from fedastro.data.base import REAL, DatasetBundle, Provenance, array_checksum
from fedastro.paths import raw_dir

FILENAME = "LSDS-22_Echocardiography_aorta_SUBMITTED.csv"
SHA256 = "df3286f1660c7adb5b33322f8c153767567524b714b33a86bcf010baa06cc4d0"
SOURCE_URL = ("https://osdr.nasa.gov/geode-py/ws/studies/OSD-435/download?source=datamanager&file="
              + FILENAME)
LANDING_PAGE = "https://osdr.nasa.gov/bio/repo/data/studies/OSD-435"
LOW_DOSES = (0.05, 0.1, 0.25)
RADTYPES = (1, 2, 3, 4)  # 1 protons, 2 16O, 3 gamma rays, 4 protons then 16O (data dictionary)
MEASURES = ("HR", "VELOCITY", "VTI", "MEANVEL")
CLASS_NAMES = ("sham", "irradiated")
PREPROC_VERSION = "osd435-v1"


def csv_path() -> Path:
    return raw_dir() / FILENAME


def feature_names(params: Osd435Params) -> list[str]:
    rad = [f"radtype_{t}" for t in RADTYPES] if params.include_radtype else []
    return rad + ["month", *(m.lower() for m in MEASURES)]


def rows(params: Osd435Params, path: Path | None = None) -> pd.DataFrame:
    """The filtered table (one row per measurement), before feature encoding."""
    path = path or csv_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/fetch_data.py osd435 --accept-license "
                                "(or set FEDASTRO_DATA_ROOT)")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != SHA256:
        raise ValueError(f"{path}: sha256 {sha} != pinned {SHA256}; the file changed, re-inspect it first")
    d = pd.read_csv(path, encoding="utf-8-sig")
    d = d[~d["DOSE"].isin(LOW_DOSES)].dropna(subset=list(MEASURES))
    if not params.include_month0:
        d = d[d["MONTH"] > 0]
    if (d.groupby("SID")["DOSE"].nunique() > 1).any():
        raise ValueError("a mouse has more than one dose")
    unknown = set(d["RADTYPE"]) - set(RADTYPES)
    if unknown:
        raise ValueError(f"unknown radiation types {sorted(unknown)}")
    return d.sort_values(["SID", "MONTH"], kind="stable").reset_index(drop=True)


def load_osd435(params: Osd435Params, path: Path | None = None) -> DatasetBundle:
    d = rows(params, path)
    cols = [d["MONTH"].to_numpy(np.float64)] + [d[m].to_numpy(np.float64) for m in MEASURES]
    if params.include_radtype:
        cols = [(d["RADTYPE"].to_numpy() == t).astype(np.float64) for t in RADTYPES] + cols
    X = np.stack(cols, axis=1).astype(np.float32)
    y = (d["DOSE"].to_numpy() >= 0.5).astype(np.int64)
    subject_id = d["SID"].to_numpy(np.int64)
    provenance = Provenance(dataset="OSD-435", version=f"{FILENAME}+{PREPROC_VERSION}",
                            checksum=f"sha256:{SHA256}", source=SOURCE_URL, kind=REAL,
                            processed_checksum=array_checksum(X, y, subject_id))
    return DatasetBundle(X=X, y=y, subject_id=subject_id, class_names=CLASS_NAMES, provenance=provenance)


def load(data: DataConfig, seed: int) -> DatasetBundle:
    """Registry entry point. Loading does not depend on the run seed."""
    return load_osd435(data.osd435)
