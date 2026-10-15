"""WESAD loader (Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable
Stress and Affect Detection", ICMI 2018). PRIMARY dataset.

WESAD is terrestrial lab data from 15 healthy adults. In this project it is a stated
proxy for crew physiological monitoring, never spaceflight data.

Source: ``data/raw/WESAD.zip`` from ``scripts/fetch_data.py wesad``. The archive is
never extracted; ``S<n>/S<n>.pkl`` members are read in place.

Pipeline per subject:
1. Load the synchronised pickle: chest RespiBAN at 700 Hz (ACC, ECG, EMG, EDA, Temp,
   Resp) and wrist Empatica E4 (ACC 32 Hz, BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz), plus the
   700 Hz protocol label.
2. Find contiguous label runs for baseline (1), stress (2), amusement (3). Labels 0
   and 4-7 (transient, meditation, ignore) are dropped.
3. Slide ``window_s`` windows with ``stride_s`` inside each run, so every window has a
   single pure label.
4. Per channel (3-axis ACC reduced to magnitude), either:
   - ``features``: mean, std, min, max, linear slope -> (n_windows, n_channels * 5)
   - ``sequence``: block-mean downsampling to ``sequence_hz`` -> (n_windows,
     n_channels, window_s * sequence_hz). High-rate content (ECG, EMG, BVP waveform
     morphology) is averaged away in this mode.
No normalisation happens here; scalers must be fit on training data only (Phase 2).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fedastro.config import WesadParams
from fedastro.data.base import REAL, DatasetBundle, Provenance, array_checksum
from fedastro.paths import cache_dir, raw_dir

PREPROC_VERSION = "wesad-preproc-v1"
SOURCE_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
ARCHIVE_NAME = "WESAD.zip"

# S1 and S12 were discarded by the dataset authors (sensor malfunction).
SUBJECTS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17)
LABELS = {1: "baseline", 2: "stress", 3: "amusement"}
CLASS_NAMES = tuple(LABELS.values())
LABEL_FS = 700

# (device, signal key, sample rate in Hz). ACC is reduced to its Euclidean magnitude.
CHANNELS = {
    "chest": (("ACC", 700), ("ECG", 700), ("EMG", 700), ("EDA", 700), ("Temp", 700), ("Resp", 700)),
    "wrist": (("ACC", 32), ("BVP", 64), ("EDA", 4), ("TEMP", 4)),
}
FEATURES = ("mean", "std", "min", "max", "slope")


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Pickles can run arbitrary code; only allow what a WESAD file needs."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("_codecs", "encode"),
    }

    def find_class(self, module: str, name: str):  # noqa: D401
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name} from WESAD pickle")
        return super().find_class(module, name)


def channel_names(devices: tuple[str, ...]) -> list[str]:
    return [f"{d}_{sig}" for d in devices for sig, _ in CHANNELS[d]]


def feature_names(params: WesadParams) -> list[str]:
    if params.representation != "features":
        raise ValueError("feature_names only applies to representation='features'")
    return [f"{c}_{f}" for c in channel_names(params.devices) for f in FEATURES]


def archive_path() -> Path:
    return raw_dir() / ARCHIVE_NAME


def archive_sha256(path: Path) -> str:
    """Read the hash recorded by fetch_data.py; compute and record it if missing."""
    meta_path = path.with_name(path.name + ".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())["sha256"]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    meta = {"dataset": "wesad", "url": SOURCE_URL, "bytes": path.stat().st_size,
            "sha256": h.hexdigest(), "note": "hash computed by loader; download not via fetch_data.py"}
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta["sha256"]


def read_subject(zf: zipfile.ZipFile, subject: int) -> dict:
    with zf.open(f"WESAD/S{subject}/S{subject}.pkl") as f:
        return _NumpyOnlyUnpickler(f, encoding="latin1").load()


def label_windows(labels: np.ndarray, window_s: float, stride_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Start times (s) and class indices of label-pure windows."""
    labels = np.asarray(labels).reshape(-1)
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    t_out, y_out = [], []
    for a, b in zip(starts, ends):
        lab = int(labels[a])
        if lab not in LABELS:
            continue
        t0, t_end = a / LABEL_FS, b / LABEL_FS
        n = int(np.floor((t_end - t0 - window_s) / stride_s + 1e-9)) + 1
        if n <= 0:
            continue
        t_out.append(t0 + stride_s * np.arange(n))
        y_out.append(np.full(n, CLASS_NAMES.index(LABELS[lab])))
    if not t_out:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    return np.concatenate(t_out), np.concatenate(y_out).astype(np.int64)


def _signal(raw: dict, device: str, key: str) -> np.ndarray:
    x = np.asarray(raw["signal"][device][key], dtype=np.float64)
    if key == "ACC":
        return np.linalg.norm(x.reshape(len(x), -1), axis=1)
    return x.reshape(-1)


def _gather(x: np.ndarray, t: np.ndarray, fs: int, window_s: float) -> np.ndarray | None:
    length = int(round(window_s * fs))
    idx0 = np.floor(t * fs + 1e-9).astype(np.int64)
    if len(idx0) and idx0[-1] + length > len(x):
        return None
    return x[idx0[:, None] + np.arange(length)[None, :]]


def _features(w: np.ndarray, fs: int) -> np.ndarray:
    tt = np.arange(w.shape[1]) / fs
    tt = tt - tt.mean()
    slope = (w - w.mean(axis=1, keepdims=True)) @ tt / (tt @ tt)
    return np.stack([w.mean(1), w.std(1), w.min(1), w.max(1), slope], axis=1)


def process_subject(raw: dict, params: WesadParams) -> tuple[np.ndarray, np.ndarray]:
    t, y = label_windows(raw["label"], params.window_s, params.stride_s)
    # Drop trailing windows that would overrun any channel (wrist streams can end early).
    for device in params.devices:
        for key, fs in CHANNELS[device]:
            n_avail = len(_signal(raw, device, key))
            ok = np.floor(t * fs + 1e-9) + round(params.window_s * fs) <= n_avail
            t, y = t[ok], y[ok]
    parts = []
    for device in params.devices:
        for key, fs in CHANNELS[device]:
            w = _gather(_signal(raw, device, key), t, fs, params.window_s)
            assert w is not None
            if params.representation == "features":
                parts.append(_features(w, fs))
            else:
                block = fs / params.sequence_hz
                if block % 1:
                    raise ValueError(f"{device}_{key}: {fs} Hz not divisible by sequence_hz")
                n_bins = int(round(params.window_s * params.sequence_hz))
                parts.append(w.reshape(len(w), n_bins, int(block)).mean(axis=2))
    if params.representation == "features":
        X = np.concatenate(parts, axis=1)
    else:
        X = np.stack(parts, axis=1)
    return X.astype(np.float32), y


def _cache_key(params: WesadParams, sha: str) -> str:
    d = asdict(params)
    d.pop("subjects")
    blob = json.dumps({"p": d, "sha": sha, "v": PREPROC_VERSION}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _process_to_cache(archive: Path, subject: int, params: WesadParams, cfile: Path) -> None:
    """Worker: read one subject from the archive, process it, write its cache file."""
    with zipfile.ZipFile(archive) as zf:
        X, y = process_subject(read_subject(zf, subject), params)
    tmp = cfile.with_name(cfile.stem + ".tmp.npz")
    np.savez(tmp, X=X, y=y)
    tmp.rename(cfile)


def _n_workers(n_jobs: int) -> int:
    """Each worker holds one ~1 GB decompressed subject; default 3 suits an 8 GB laptop."""
    requested = int(os.environ.get("FEDASTRO_WESAD_WORKERS", "3"))
    return max(1, min(requested, n_jobs, os.cpu_count() or 1))


def load_wesad(params: WesadParams, archive: Path | None = None) -> DatasetBundle:
    archive = archive or archive_path()
    if not archive.exists():
        raise FileNotFoundError(
            f"{archive} not found. Run: python scripts/fetch_data.py wesad --accept-license "
            "(or set FEDASTRO_DATA_ROOT)")
    sha = archive_sha256(archive)
    subjects = params.subjects or SUBJECTS
    unknown = set(subjects) - set(SUBJECTS)
    if unknown:
        raise ValueError(f"unknown WESAD subjects {sorted(unknown)}; valid: {SUBJECTS}")
    cdir = cache_dir() / "wesad" / _cache_key(params, sha)
    cdir.mkdir(parents=True, exist_ok=True)
    cfiles = {sub: cdir / f"S{sub}.npz" for sub in subjects}
    todo = [sub for sub in subjects if not cfiles[sub].exists()]
    workers = _n_workers(len(todo))
    if workers == 1:
        for sub in todo:
            _process_to_cache(archive, sub, params, cfiles[sub])
    elif todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_to_cache, archive, sub, params, cfiles[sub]) for sub in todo]
            for f in futures:
                f.result()
    X_parts, y_parts, s_parts = [], [], []
    for sub in subjects:  # fixed subject order, independent of worker completion order
        with np.load(cfiles[sub]) as npz:
            X_parts.append(npz["X"])
            y_parts.append(npz["y"])
        s_parts.append(np.full(len(y_parts[-1]), sub, dtype=np.int64))
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    subject_id = np.concatenate(s_parts)
    provenance = Provenance(
        dataset="WESAD",
        version=f"release-zip-sha256:{sha[:12]}+{PREPROC_VERSION}",
        checksum=f"sha256:{sha}",
        source=SOURCE_URL,
        kind=REAL,
        processed_checksum=array_checksum(X, y, subject_id),
    )
    return DatasetBundle(X=X, y=y, subject_id=subject_id, class_names=CLASS_NAMES,
                         provenance=provenance)
