"""PhysioNet/CinC Challenge 2021 loader (Phase H10). PRIMARY dataset, cinc15 topology.

What the data is: 12-lead clinical ECGs from eight public databases, as redistributed by the
Challenge (``challenge-2021/1.0.3``, open access, CC BY 4.0). Terrestrial hospital and Holter
recordings: a stated proxy for crew cardiac monitoring, never spaceflight data. Source, agency
and region names in the topology are scenario labels.

Files (``scripts/fetch_data.py cinc2021 --subset N --accept-license``) under
``<data root>/raw/cinc2021/``:
- ``RECORDS`` and ``lists/<group>RECORDS``: the record listing.
- ``training/<source>/<group>/<record>.hea`` (all ~88k headers: labels, age, sex) and
  ``.mat`` (MATLAB v4 ``val``, int16, leads x samples; only for the selected subset).
- ``dx_mapping_scored.csv`` / ``dx_mapping_unscored.csv``: the Challenge's label mapping.
- ``orig/``: metadata of the original databases, used only to group records by patient.
Every metadata file's SHA-256 is pinned in ``PINNED``; a different file is refused.

Task (user decisions, 2026-11-21; PROGRESS.md Phase H10 part a):
- Label: ``normal`` = normal sinus rhythm (SNOMED 426783006) and no other SCORED code;
  ``abnormal`` = any scored code other than NSR (sinus brady/tachycardia included). Records
  with only unscored codes are dropped; so are records shorter than ``window_s``.
- Leads I and II (``leads``), converted to mV with the header gain and baseline.
- Windows: the first ``window_s`` seconds of each record; INCART's 30 min Holter records give
  ``incart_segments`` evenly spaced windows each, all kept with their record (and patient) and
  all carrying the record's labels (noisy for a 10 s Holter segment: disclosed).
- ``features`` (logreg, MLP): per lead at ``sequence_hz``: mean, std, min, max, slope; then
  RR-interval statistics of lead II (R peaks on a 5-15 Hz band-pass at 500 Hz): beat count,
  heart rate, mean RR, SDNN, RMSSD, min RR, max RR (zeros below 3 beats). 17 values.
- ``sequence`` (1D-CNN): (n, len(leads), window_s * sequence_hz), polyphase resampled.
- Patients (CinC headers carry no patient id; mappings verified record by record in Phase
  H10a): PTB-XL ``HR<n>`` = PTB-XL 1.0.1 ``ecg_id`` n, grouped by its ``patient_id``; PTB
  ``S<n>`` = line n of ``ptbdb/1.0.0/RECORDS`` (``patientNNN/...``); INCART ``I00nn`` =
  ``Inn`` of ``incartdb``, grouped by ``files-patients-diagnoses.txt``. Every other source has
  no patient id: one record = one patient (a disclosed limitation).
- Subsets: ``subset`` = N records (0 = every usable record), a fixed, seeded, patient-complete
  sample per source in proportion to source size, with PTB and INCART always whole. Smaller
  subsets are prefixes of larger ones (the smoke records are inside the main set).
No normalisation here; scalers are fit on training data only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from fedastro.config import Cinc2021Params, DataConfig
from fedastro.data.base import REAL, GroupedDatasetBundle, Provenance, array_checksum
from fedastro.paths import cache_dir, raw_dir

VERSION = "1.0.3"
BASE_URL = f"https://physionet.org/files/challenge-2021/{VERSION}"
PREPROC_VERSION = "cinc2021-v1"
CLASS_NAMES = ("normal", "abnormal")
NSR = "426783006"
SUBSET_SEED = 2021  # fixed: the subset never depends on the run seed
RR_LEAD = "II"
RR_FS = 500
RR_BAND = (5.0, 15.0)

# Challenge directory -> source name (= the dataset's group names, used by cinc15.yaml).
SOURCES = {"chapman_shaoxing": "chapman", "cpsc_2018": "cpsc", "cpsc_2018_extra": "cpsc_extra",
           "georgia": "g12ec", "ningbo": "ningbo", "ptb": "ptb", "ptb-xl": "ptbxl",
           "st_petersburg_incart": "incart"}
GROUP_NAMES = tuple(SOURCES.values())
WHOLE_SOURCES = ("ptb", "incart")  # always included in full in every subset
SEGMENTED_SOURCES = ("incart",)

METADATA = {  # relative path -> URL
    "RECORDS": f"{BASE_URL}/RECORDS",
    "dx_mapping_scored.csv": "https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/dx_mapping_scored.csv",
    "dx_mapping_unscored.csv": "https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/dx_mapping_unscored.csv",
    "orig/ptbxl101_database.csv": "https://physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv",
    "orig/ptbdb_RECORDS": "https://physionet.org/files/ptbdb/1.0.0/RECORDS",
    "orig/incart_files-patients-diagnoses.txt": "https://physionet.org/files/incartdb/1.0.0/files-patients-diagnoses.txt",
}
PINNED = {
    "RECORDS": "b8c72c997b559aaedb1f9a64f2ca17e3806027007468065dd38c591fbe35df18",
    "dx_mapping_scored.csv": "fad13ad9f7ca230e7e6392ac8a264cb7cd157879525129f964c5f708eabb41d0",
    "dx_mapping_unscored.csv": "ce53c9e35406e922f1634a38bb13c31e8cf7603291641a16c201d8e679f4fe94",
    "orig/ptbxl101_database.csv": "cd9bf4e6445c5e022605857c4d6f90c90ff0e62df847da52efaebe94de786156",
    "orig/ptbdb_RECORDS": "1bdcd205432b66e98389e095c8ea9f2f642f79265a1874929a9cb6de270852fe",
    "orig/incart_files-patients-diagnoses.txt": "7710993dd08ac1ee6e3912756b78cc16e9df40d14e5fb768c3c42a5197c79c17",
}
# Listed in RECORDS but absent (HTTP 404 on PhysioNet and its S3 mirror, 2026-11-21). The fetch
# writes them to ``unavailable_records.txt``; the set is pinned by count and by the SHA-256 of
# its sorted names. Any other missing header is an incomplete download and is refused.
N_UNAVAILABLE = 85
UNAVAILABLE_SHA256 = "ff52e105dc8d6de778fa24c288fe0a0cb0b0ab11d53bf260ad2ea6db2783ebb8"


def root_dir() -> Path:
    return raw_dir() / "cinc2021"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_metadata(root: Path) -> None:
    for rel, sha in PINNED.items():
        p = root / rel
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run: python scripts/fetch_data.py cinc2021 --subset N "
                                    "--accept-license (or set FEDASTRO_DATA_ROOT)")
        got = _sha256(p)
        if got != sha:
            raise ValueError(f"{p}: sha256 {got} != pinned {sha}; the file changed, re-inspect it first")


# -- listing and headers ---------------------------------------------------------------------

def group_dirs(root: Path) -> list[str]:
    """``training/<source>/<group>/`` entries of the top-level RECORDS (blank lines dropped)."""
    return [ln.strip() for ln in (root / "RECORDS").read_text().splitlines() if ln.strip()]


def listed_records(root: Path) -> list[str]:
    """Every listed record as ``training/<source>/<group>/<record>`` (no extension)."""
    out = []
    for d in group_dirs(root):
        lst = root / "lists" / (d.replace("/", "_") + "RECORDS")
        out += [d + r.strip() for r in lst.read_text().splitlines() if r.strip()]
    return out


@dataclass(frozen=True)
class Header:
    record: str
    fs: float
    n_samples: int
    leads: tuple[str, ...]
    gains: tuple[float, ...]
    baselines: tuple[float, ...]
    age: str
    sex: str
    dx: tuple[str, ...]


_GAIN = re.compile(r"^([-+0-9.eE]+)(?:\((-?\d+)\))?")


def parse_header(text: str) -> Header:
    lines = text.splitlines()
    first = lines[0].split()
    nsig = int(first[1])
    leads, gains, bases = [], [], []
    for ln in lines[1 : 1 + nsig]:
        f = ln.split()
        m = _GAIN.match(f[2])
        gain = float(m.group(1)) if m else 200.0  # WFDB default gain
        leads.append(f[-1])
        gains.append(gain if gain else 200.0)
        # WFDB: "gain(baseline)/units"; the baseline defaults to the ADC zero (next field)
        bases.append(float(m.group(2)) if m and m.group(2) is not None else float(f[4]))
    meta = {}
    for ln in lines[1 + nsig :]:
        m = re.match(r"#\s*(\w+):\s*(.*)", ln)
        if m:
            meta[m.group(1).lower()] = m.group(2).strip()
    dx = tuple(d for d in meta.get("dx", "").split(",") if d)
    return Header(record=first[0], fs=float(first[2]), n_samples=int(first[3]), leads=tuple(leads),
                  gains=tuple(gains), baselines=tuple(bases), age=meta.get("age", ""), sex=meta.get("sex", ""), dx=dx)


def read_codes(path: Path) -> set[str]:
    with path.open(newline="") as f:
        return {row["SNOMEDCTCode"] for row in csv.DictReader(f)}


def label_of(dx: tuple[str, ...], scored: set[str]) -> int:
    """1 abnormal, 0 normal, -1 dropped (only unscored codes)."""
    sc = [d for d in dx if d in scored]
    if any(d != NSR for d in sc):
        return 1
    return 0 if NSR in sc else -1


# -- patients --------------------------------------------------------------------------------

def _patient_maps(root: Path) -> dict[str, dict[str, int]]:
    """Per source with known patients: record name -> patient number (Phase H10a mappings)."""
    xl: dict[str, int] = {}
    with (root / "orig/ptbxl101_database.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            xl[f"HR{int(row['ecg_id']):05d}"] = int(float(row["patient_id"]))
    ptb_lines = [ln.strip() for ln in (root / "orig/ptbdb_RECORDS").read_text().splitlines() if ln.strip()]
    ptb = {f"S{i:04d}": int(ln.split("/")[0].removeprefix("patient")) for i, ln in enumerate(ptb_lines, 1)}
    inc: dict[str, int] = {}
    txt = (root / "orig/incart_files-patients-diagnoses.txt").read_text().splitlines()
    for i, ln in enumerate(txt):
        m = re.match(r"patient (\d+)", ln.strip())
        if m:
            for rec in txt[i + 1].split():
                inc[f"I{int(rec[1:]):04d}"] = int(m.group(1))
    return {"ptbxl": xl, "ptb": ptb, "incart": inc}


# -- the index of usable records ---------------------------------------------------------------

INDEX_COLUMNS = ("record", "path", "source", "fs", "n_samples", "label", "patient")


def build_index(root: Path, window_s: float) -> list[dict]:
    """Every record with a usable label and at least ``window_s`` seconds, in listing order.

    Fails if a listed header is missing that is not one of the known-unavailable records."""
    scored = read_codes(root / "dx_mapping_scored.csv")
    pmaps = _patient_maps(root)
    unavailable = set(_read_unavailable(root))
    missing, rows = [], []
    next_id = {s: 0 for s in GROUP_NAMES}
    for rel in listed_records(root):
        hp = root / (rel + ".hea")
        if not hp.exists():
            if rel not in unavailable:
                missing.append(rel)
            continue
        source = SOURCES[rel.split("/")[1]]
        h = parse_header(hp.read_text())
        next_id[source] += 1
        y = label_of(h.dx, scored)
        if y < 0 or h.n_samples < window_s * h.fs:
            continue
        if source in pmaps:
            if h.record not in pmaps[source]:
                raise ValueError(f"{rel}: no patient mapping for {source} record {h.record}")
            local = pmaps[source][h.record]
        else:
            local = next_id[source]  # one record = one patient
        patient = (GROUP_NAMES.index(source) + 1) * 10_000_000 + local
        rows.append(dict(record=h.record, path=rel, source=source, fs=h.fs, n_samples=h.n_samples,
                         label=y, patient=patient))
    if missing:
        raise FileNotFoundError(f"{len(missing)} listed headers are missing (e.g. {missing[:3]}); the header "
                                "download is incomplete: re-run scripts/fetch_data.py cinc2021")
    return rows


def unavailable_digest(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def _read_unavailable(root: Path) -> list[str]:
    p = root / "unavailable_records.txt"
    names = [ln.strip().removesuffix(".hea") for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []
    if len(names) != N_UNAVAILABLE or unavailable_digest(names) != UNAVAILABLE_SHA256:
        raise ValueError(f"{p}: {len(names)} unavailable records, expected the pinned {N_UNAVAILABLE} "
                         f"(sha256 {UNAVAILABLE_SHA256[:12]}); PhysioNet's file set changed, re-inspect")
    return names


def index_checksum(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r[c] for c in INDEX_COLUMNS]).encode())
    return "sha256:" + h.hexdigest()


def select_subset(rows: list[dict], n: int) -> list[dict]:
    """Patient-complete, seeded sample of about n records (0 = all), listing order kept.

    WHOLE_SOURCES are taken in full; every other source gets a share of the remaining quota
    in proportion to its usable records, filled with whole patients in a fixed random order
    (so a smaller subset is a prefix of a larger one, per source)."""
    if n <= 0:
        return list(rows)
    by_src: dict[str, list[dict]] = {s: [] for s in GROUP_NAMES}
    for r in rows:
        by_src[r["source"]].append(r)
    whole = sum(len(by_src[s]) for s in WHOLE_SOURCES)
    rest = [s for s in GROUP_NAMES if s not in WHOLE_SOURCES]
    pool = sum(len(by_src[s]) for s in rest)
    budget = max(n - whole, 0)
    keep: set[str] = {r["record"] + "|" + r["source"] for s in WHOLE_SOURCES for r in by_src[s]}
    for k, s in enumerate(rest):
        quota = round(budget * len(by_src[s]) / pool) if pool else 0
        pats: dict[int, list[dict]] = {}
        for r in by_src[s]:
            pats.setdefault(r["patient"], []).append(r)
        order = np.random.default_rng([SUBSET_SEED, k]).permutation(np.array(sorted(pats), dtype=np.int64))
        taken = 0
        for p in order:
            if taken >= quota:
                break
            for r in pats[int(p)]:
                keep.add(r["record"] + "|" + s)
            taken += len(pats[int(p)])
    return [r for r in rows if r["record"] + "|" + r["source"] in keep]


# -- signals ---------------------------------------------------------------------------------

def _resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    from scipy.signal import resample_poly

    if fs_in == fs_out:
        return x
    fr = Fraction(fs_out / fs_in).limit_denominator(2000)
    return resample_poly(x, fr.numerator, fr.denominator, axis=-1)


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[-1] >= n:
        return x[..., :n]
    return np.concatenate([x, np.repeat(x[..., -1:], n - x.shape[-1], axis=-1)], axis=-1)


def _stats(w: np.ndarray, fs: float) -> np.ndarray:
    """Per lead: mean, std, min, max, linear slope (as WESAD's per-channel features)."""
    tt = np.arange(w.shape[1]) / fs
    tt = tt - tt.mean()
    slope = (w - w.mean(axis=1, keepdims=True)) @ tt / (tt @ tt)
    return np.stack([w.mean(1), w.std(1), w.min(1), w.max(1), slope], axis=1)


def rr_features(lead: np.ndarray, fs: float) -> np.ndarray:
    """Beat count, heart rate (bpm), mean RR, SDNN, RMSSD, min RR, max RR (seconds); zeros
    for the RR values below 3 detected beats."""
    from scipy.signal import butter, find_peaks, sosfiltfilt

    x = _resample(lead, fs, RR_FS)
    sos = butter(3, RR_BAND, btype="bandpass", fs=RR_FS, output="sos")
    e = np.abs(sosfiltfilt(sos, x))
    top = np.percentile(e, 99)
    peaks, _ = find_peaks(e, height=0.4 * top if top > 0 else np.inf, distance=int(0.25 * RR_FS))
    rr = np.diff(peaks) / RR_FS
    if len(peaks) < 3:
        return np.array([len(peaks), 0, 0, 0, 0, 0, 0], dtype=np.float64)
    return np.array([len(peaks), 60.0 / rr.mean(), rr.mean(), rr.std(), np.sqrt(np.mean(np.diff(rr) ** 2)),
                     rr.min(), rr.max()], dtype=np.float64)


N_RR = 7


def window_starts(n_samples: int, fs: float, params: Cinc2021Params, segmented: bool) -> list[int]:
    win = int(round(params.window_s * fs))
    if not segmented:
        return [0]
    return [int(s) for s in np.linspace(0, n_samples - win, params.incart_segments).round()]


def process_record(root: Path, row: dict, params: Cinc2021Params) -> tuple[np.ndarray, np.ndarray]:
    """(features (k, 5 * leads + 7), sequence (k, leads, window_s * sequence_hz)) for one record."""
    from scipy.io import loadmat

    h = parse_header((root / (row["path"] + ".hea")).read_text())
    val = loadmat(root / (row["path"] + ".mat"))["val"]
    if val.shape != (len(h.leads), h.n_samples):
        raise ValueError(f"{row['path']}: signal shape {val.shape} != header ({len(h.leads)}, {h.n_samples})")
    names = [ld.upper() for ld in h.leads]
    idx = [names.index(ld.upper()) for ld in params.leads]
    mv = np.stack([(val[i].astype(np.float64) - h.baselines[i]) / h.gains[i] for i in idx])
    rr_i = [ld.upper() for ld in params.leads].index(RR_LEAD)
    win = int(round(params.window_s * h.fs))
    n_seq = int(round(params.window_s * params.sequence_hz))
    feats, seqs = [], []
    for s in window_starts(h.n_samples, h.fs, params, row["source"] in SEGMENTED_SOURCES):
        seg = mv[:, s : s + win]
        seq = _fit_length(_resample(seg, h.fs, params.sequence_hz), n_seq)
        feats.append(np.concatenate([_stats(seq, params.sequence_hz).ravel(), rr_features(seg[rr_i], h.fs)]))
        seqs.append(seq)
    return np.asarray(feats, np.float32), np.asarray(seqs, np.float32)


def _cache_key(params: Cinc2021Params) -> str:
    d = asdict(params)
    for k in ("subset", "representation"):
        d.pop(k)
    blob = json.dumps({"p": d, "v": PREPROC_VERSION, "dx": PINNED["dx_mapping_scored.csv"]}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _process_chunk(root: str, rows: list[dict], params: Cinc2021Params, cdir: str) -> None:
    for row in rows:
        out = Path(cdir) / f"{row['source']}_{row['record']}.npz"
        if out.exists():
            continue
        f, s = process_record(Path(root), row, params)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, features=f, sequence=s)
        tmp.rename(out)


def _n_workers(n_jobs: int) -> int:
    requested = int(os.environ.get("FEDASTRO_CINC_WORKERS", str(os.cpu_count() or 1)))
    return max(1, min(requested, n_jobs, os.cpu_count() or 1))


def cached_index(root: Path, window_s: float) -> list[dict]:
    """``build_index``, cached. The cache key covers the pinned metadata, the window length and
    the name and size of every header file present, so a changed download rebuilds it."""
    h = hashlib.sha256(json.dumps([PINNED, UNAVAILABLE_SHA256, window_s, PREPROC_VERSION]).encode())
    for rel in listed_records(root):
        p = root / (rel + ".hea")
        h.update(f"{rel}:{p.stat().st_size if p.exists() else -1}\n".encode())
    f = cache_dir() / "cinc2021" / f"index-{h.hexdigest()[:16]}.json"
    if f.exists():
        return json.loads(f.read_text())
    rows = build_index(root, window_s)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(rows))
    tmp.rename(f)
    return rows


def load_cinc2021(params: Cinc2021Params, root: Path | None = None) -> GroupedDatasetBundle:
    root = root or root_dir()
    verify_metadata(root)
    rows = cached_index(root, params.window_s)
    sel = select_subset(rows, params.subset)
    missing = [r["path"] for r in sel if not (root / (r["path"] + ".mat")).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} signal files of the subset are missing (e.g. {missing[:3]}). Run: "
                                f"python scripts/fetch_data.py cinc2021 --subset {params.subset} --accept-license")
    cdir = cache_dir() / "cinc2021" / _cache_key(params)
    cdir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in sel if not (cdir / f"{r['source']}_{r['record']}.npz").exists()]
    workers = _n_workers(len(todo))
    if workers <= 1:
        _process_chunk(str(root), todo, params, str(cdir))
    elif todo:
        chunks = [todo[i::workers * 4] for i in range(workers * 4)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for f in [pool.submit(_process_chunk, str(root), c, params, str(cdir)) for c in chunks if c]:
                f.result()
    key = "features" if params.representation == "features" else "sequence"
    X, y, subj, grp = [], [], [], []
    for r in sel:  # listing order, independent of worker completion order
        with np.load(cdir / f"{r['source']}_{r['record']}.npz") as npz:
            x = npz[key]
        X.append(x)
        y.append(np.full(len(x), r["label"], np.int64))
        subj.append(np.full(len(x), r["patient"], np.int64))
        grp.append(np.full(len(x), GROUP_NAMES.index(r["source"]), np.int64))
    X, y, subj, grp = (np.concatenate(a) for a in (X, y, subj, grp))
    provenance = Provenance(
        dataset="CinC2021",
        version=f"challenge-2021/{VERSION}+subset{params.subset}+{PREPROC_VERSION}",
        checksum=index_checksum(sel),
        source=BASE_URL,
        kind=REAL,
        processed_checksum=array_checksum(X, y, subj),
    )
    return GroupedDatasetBundle(X=X, y=y, subject_id=subj, class_names=CLASS_NAMES, provenance=provenance,
                                group_id=grp, group_names=GROUP_NAMES)


def load(data: DataConfig, seed: int) -> GroupedDatasetBundle:
    """Registry entry point. Loading does not depend on the run seed."""
    return load_cinc2021(data.cinc2021)
