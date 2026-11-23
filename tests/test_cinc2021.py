"""Phase H10b: CinC 2021 loader (data/cinc2021.py).

Most tests build a tiny synthetic copy of the Challenge's file layout (headers, MATLAB
signals, label mappings, patient metadata) in a temporary data root, so they run without the
real data. The last tests check the real headers against the Phase H10a inspection and are
skipped when the headers have not been fetched."""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from fedastro.config import CINC_LEADS, ConfigError, Cinc2021Params, load_config
from fedastro.data import cinc2021 as c
from fedastro.data.partition import assert_hier_disjoint
from fedastro.pipeline import build_hier_federation

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "experiments" / "cinc15_hier.yaml"
NSR, AF, MI = "426783006", "164889003", "164865005"  # scored normal, scored abnormal, unscored
DIRS = {v: k for k, v in c.SOURCES.items()}
PREFIX = {"chapman": "JS", "cpsc": "A", "cpsc_extra": "Q", "g12ec": "E", "ningbo": "JN", "ptb": "S",
          "ptbxl": "HR", "incart": "I"}
WIDTH = {"ptb": 4, "incart": 4, "cpsc": 4, "cpsc_extra": 4}


def _ecg(fs: float, seconds: float, bpm: float = 60.0, seed: int = 0) -> np.ndarray:
    """12 leads in ADC units (1000 per mV): a 1 mV spike train at ``bpm`` plus small noise."""
    n = int(round(fs * seconds))
    t = np.arange(n) / fs
    beats = np.arange(0.3, seconds, 60.0 / bpm)
    x = sum(np.exp(-0.5 * ((t - b) / 0.01) ** 2) for b in beats)
    noise = np.random.default_rng(seed).normal(0, 0.01, (12, n))
    return np.round((x[None, :] + noise) * 1000).astype(np.int16)


def _write_record(root: Path, source: str, name: str, fs: float, seconds: float, dx: list[str], seed: int) -> str:
    rel = f"training/{DIRS[source]}/g1/{name}"
    sig = _ecg(fs, seconds, seed=seed)
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(root / f"{rel}.mat", {"val": sig})
    lines = [f"{name} 12 {fs:g} {sig.shape[1]}"]
    lines += [f"{name}.mat 16x1+24 1000.0(0)/mV 16 0 {sig[i, 0]} 0 0 {ld}" for i, ld in enumerate(CINC_LEADS)]
    lines +=["# Age: 60", "# Sex: Male", f"# Dx: {','.join(dx)}", "# Rx: Unknown", "# Hx: Unknown", "# Sx: Unknown"]
    (root / f"{rel}.hea").write_text("\n".join(lines) + "\n")
    return name


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A miniature CinC tree. Per source 12 patients (label alternates by patient); PTB-XL,
    PTB and INCART have some patients with two records; plus one MI-only record and one 5 s
    record per source (both dropped), and one listed-but-unavailable record."""
    monkeypatch.setenv("FEDASTRO_DATA_ROOT", str(tmp_path))
    root = tmp_path / "raw" / "cinc2021"
    root.mkdir(parents=True)
    records: dict[str, list[str]] = collections.defaultdict(list)
    xl_rows, ptb_lines, inc_patients = [], [], collections.defaultdict(list)
    for source in c.GROUP_NAMES:
        fs = {"ptb": 1000.0, "incart": 257.0}.get(source, 500.0)
        secs = {"ptb": 30.0, "incart": 120.0}.get(source, 10.0)
        k = 0
        for p in range(1, 13):
            for _ in range(2 if source in ("ptbxl", "ptb", "incart") and p <= 3 else 1):
                k += 1
                name = f"{PREFIX[source]}{k:0{WIDTH.get(source, 5)}d}"
                records[source].append(_write_record(root, source, name, fs, secs, [NSR if p % 2 else AF], k))
                if source == "ptbxl":
                    xl_rows.append(f"{k},{100 + p}.0")
                elif source == "ptb":
                    ptb_lines.append(f"patient{p:03d}/s{k:04d}_re")
                elif source == "incart":
                    inc_patients[p].append(f"I{k:02d}")
        for dx, s in (([MI], secs), ([AF], 5.0)):  # dropped: only unscored / too short
            k += 1
            name = f"{PREFIX[source]}{k:0{WIDTH.get(source, 5)}d}"
            records[source].append(_write_record(root, source, name, fs, s, dx, k))
            if source == "ptbxl":
                xl_rows.append(f"{k},999.0")
            elif source == "ptb":
                ptb_lines.append(f"patient{90 + k:03d}/s{k:04d}_re")
            elif source == "incart":
                inc_patients[50 + k].append(f"I{k:02d}")
    records["ningbo"].append("JN99999")  # listed, no header: the one "unavailable" record
    (root / "RECORDS").write_text("".join(f"training/{DIRS[s]}/g1/\n" for s in c.GROUP_NAMES) + "\n")
    (root / "lists").mkdir()
    for s in c.GROUP_NAMES:
        (root / "lists" / f"training_{DIRS[s]}_g1_RECORDS").write_text("\n".join(records[s]) + "\n")
    hdr = "Dx,SNOMEDCTCode,Abbreviation\n"
    (root / "dx_mapping_scored.csv").write_text(hdr + f"sinus rhythm,{NSR},NSR\natrial fibrillation,{AF},AF\n")
    (root / "dx_mapping_unscored.csv").write_text(hdr + f"myocardial infarction,{MI},MI\n")
    (root / "orig").mkdir()
    (root / "orig/ptbxl101_database.csv").write_text("ecg_id,patient_id\n" + "\n".join(xl_rows) + "\n")
    (root / "orig/ptbdb_RECORDS").write_text("\n".join(ptb_lines) + "\n")
    (root / "orig/incart_files-patients-diagnoses.txt").write_text(
        "".join(f"patient {p}\n{' '.join(rs)}\n\n" for p, rs in inc_patients.items()))
    unavailable = ["training/ningbo/g1/JN99999"]
    (root / "unavailable_records.txt").write_text(unavailable[0] + ".hea\n")
    monkeypatch.setattr(c, "PINNED", {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in c.PINNED})
    monkeypatch.setattr(c, "N_UNAVAILABLE", 1)
    monkeypatch.setattr(c, "UNAVAILABLE_SHA256", c.unavailable_digest(unavailable))
    return root


# -- index, labels, patients -----------------------------------------------------------------

def test_labels_and_drops(fake):
    rows = c.build_index(fake, 10.0)
    by = collections.Counter(r["source"] for r in rows)
    assert by["ptbxl"] == by["ptb"] == by["incart"] == 15 and by["chapman"] == 12  # MI-only and 5 s dropped
    for r in rows:
        assert r["label"] in (0, 1)
    assert c.label_of((NSR,), {NSR, AF}) == 0 and c.label_of((NSR, AF), {NSR, AF}) == 1
    assert c.label_of((MI,), {NSR, AF}) == -1 and c.label_of((), {NSR}) == -1


def test_patients_follow_the_original_databases(fake):
    rows = c.build_index(fake, 10.0)
    pat = {(r["source"], r["record"]): r["patient"] for r in rows}
    for source, (a, b) in {"ptbxl": ("HR00001", "HR00002"), "ptb": ("S0001", "S0002"),
                           "incart": ("I0001", "I0002")}.items():
        assert pat[(source, a)] == pat[(source, b)]  # patient 1 has two records
        assert pat[(source, a)] != pat[(source, "HR00003" if source == "ptbxl" else f"{a[0]}0003")]
        assert len({p for (s, _), p in pat.items() if s == source}) == 12
    chapman = [p for (s, _), p in pat.items() if s == "chapman"]
    assert len(set(chapman)) == len(chapman)  # no patient ids: one record = one patient
    assert len(set(pat.values())) == sum(1 for _ in set(pat.values()))  # ids never collide across sources
    assert len({p // 10_000_000 for p in pat.values()}) == len(c.GROUP_NAMES)


def test_a_missing_header_or_a_changed_unavailable_set_is_refused(fake, monkeypatch):
    (fake / "training/chapman_shaoxing/g1/JS00001.hea").unlink()
    with pytest.raises(FileNotFoundError, match="header download is incomplete"):
        c.build_index(fake, 10.0)
    monkeypatch.setattr(c, "UNAVAILABLE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="unavailable"):
        c.build_index(fake, 10.0)


def test_a_changed_metadata_file_is_refused(fake):
    (fake / "dx_mapping_scored.csv").write_text("Dx,SNOMEDCTCode,Abbreviation\n")
    with pytest.raises(ValueError, match="sha256"):
        c.verify_metadata(fake)


# -- subsets ---------------------------------------------------------------------------------

def test_subsets_are_patient_complete_nested_and_keep_whole_sources(fake):
    rows = c.build_index(fake, 10.0)
    small, large = c.select_subset(rows, 60), c.select_subset(rows, 90)
    assert c.select_subset(rows, 0) == rows
    assert {r["record"] + r["source"] for r in small} <= {r["record"] + r["source"] for r in large}
    for sub in (small, large):
        for s in c.WHOLE_SOURCES:
            assert sum(r["source"] == s for r in sub) == sum(r["source"] == s for r in rows)
        kept = {r["patient"] for r in sub}
        assert all(r in sub for r in rows if r["patient"] in kept)  # never half a patient
    assert c.select_subset(rows, 60) == small  # fixed seed


# -- signals ---------------------------------------------------------------------------------

def test_features_sequences_and_incart_segments(fake):
    f = c.load_cinc2021(Cinc2021Params(subset=0), fake)
    s = c.load_cinc2021(Cinc2021Params(subset=0, representation="sequence"), fake)
    n_rec = len(c.build_index(fake, 10.0))
    assert len(f) == len(s) == n_rec - 15 + 15 * 10  # INCART: 10 windows per record
    assert f.X.shape[1] == 5 * 2 + c.N_RR == 17 and s.X.shape[1:] == (2, 1000)
    assert np.isfinite(f.X).all() and np.isfinite(s.X).all()
    np.testing.assert_array_equal(f.y, s.y)
    np.testing.assert_array_equal(f.subject_id, s.subject_id)
    hr = f.X[:, 10 + 1]
    assert np.all(np.abs(hr - 60) < 3), hr  # the synthetic spike train is 60 bpm
    inc = f.in_groups(["incart"])
    for p in np.unique(f.subject_id[inc]):  # every INCART window keeps its record's patient and label
        assert len(np.unique(f.y[f.subject_id == p])) == 1
    assert f.provenance.kind == "REAL" and f.provenance.dataset == "CinC2021"
    assert f.group_names == c.GROUP_NAMES
    # cached: a second load gives the same arrays
    again = c.load_cinc2021(Cinc2021Params(subset=0), fake)
    assert again.provenance.processed_checksum == f.provenance.processed_checksum


def test_another_writers_temporary_files_do_not_break_the_cache(fake):
    """Parallel PACE jobs may build one cache at once: a half-written temporary file from
    another process (another pid) must neither be read nor block this writer."""
    params = Cinc2021Params(subset=0)
    rows = c.cached_index(fake, params.window_s)
    cdir = c.cache_dir() / "cinc2021" / c._cache_key(params)
    cdir.mkdir(parents=True)
    for r in rows[:5]:
        (cdir / f"{r['source']}_{r['record']}.tmp99999999.npz").write_bytes(b"half written")
    idx = next(iter(cdir.parent.glob("index-*.json")))
    idx.with_name(idx.name + ".tmp99999999").write_text("[")
    b = c.load_cinc2021(params, fake)
    assert np.isfinite(b.X).all() and len(b) == len(c.load_cinc2021(params, fake))
    assert not [p for p in cdir.glob("*.tmp*") if "99999999" not in p.name]  # ours were renamed into place


def test_missing_signals_are_refused(fake):
    (fake / "training/ningbo/g1/JN00001.mat").unlink()
    with pytest.raises(FileNotFoundError, match="signal files"):
        c.load_cinc2021(Cinc2021Params(subset=0), fake)


def test_cinc15_nodes_are_patient_disjoint(fake):
    for rep, arch in (("features", "mlp"), ("sequence", "cnn1d")):
        cfg = load_config(CONFIG, {"data.cinc2021.subset": 0, "data.cinc2021.representation": rep,
                                   "model.arch": arch})
        bundle, fed, _ = build_hier_federation(cfg, 0)
        assert_hier_disjoint(fed)
        assert len(fed.nodes) == 15 and fed.unused_sources == ()
        seen = [s for n in fed.nodes.values() for part in (n.train, n.val, n.test) for s in part.subjects]
        assert len(seen) == len(set(seen)) == len(bundle.subjects)
        assert {g for g in fed.nodes["leo_roscosmos"].train.groups} == {"incart"}


def test_config_rules():
    with pytest.raises(ConfigError, match="sequence input"):
        load_config(CONFIG, {"model.arch": "cnn1d"})
    with pytest.raises(ConfigError, match="include II"):
        load_config(CONFIG, {"data.cinc2021.leads": ["I", "V1"]})
    cfg = load_config(CONFIG)
    assert "cinc2021" in cfg.experiment_dict()["data"]
    other = load_config(REPO / "configs" / "experiments" / "wesad12_hier.yaml")
    assert "cinc2021" not in other.experiment_dict()["data"]  # earlier run ids unchanged


# -- the real headers (Phase H10a numbers) -----------------------------------------------------

needs_headers = pytest.mark.skipif(not (c.root_dir() / "RECORDS").exists(), reason="CinC headers not fetched")


@needs_headers
def test_real_index_matches_the_inspection():
    c.verify_metadata(c.root_dir())
    rows = c.cached_index(c.root_dir(), 10.0)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    assert len(rows) == 81_830 and sum(r["label"] for r in rows) == 63_373
    pats = {s: len({r["patient"] for r in v}) for s, v in by.items()}
    assert (pats["ptbxl"], pats["ptb"], pats["incart"]) == (18_691, 83, 21)
    assert all(pats[s] == len(by[s]) for s in ("chapman", "cpsc", "cpsc_extra", "g12ec", "ningbo"))
    sub = c.select_subset(rows, 2000)
    assert len(sub) == 2000 and {r["path"] for r in sub} <= {r["path"] for r in c.select_subset(rows, 20_000)}
