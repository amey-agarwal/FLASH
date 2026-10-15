"""WESAD parser tests against a tiny fake archive with the real pickle layout.

The real-data test at the bottom runs only if data/raw/WESAD.zip is present.
"""

import io
import pickle
import time
import zipfile

import numpy as np
import pytest

from fedastro.config import WesadParams
from fedastro.data import wesad
from fedastro.data.base import REAL

FS = 700


def _fake_subject(seconds_per_label=(130, 70, 0, 65), seed=0):
    """Label sequence: transient(0) 10 s, then baseline/stress/(gap)/amusement runs."""
    rng = np.random.default_rng(seed)
    labels = [np.zeros(10 * FS, dtype=np.int32)]
    for lab, sec in zip((1, 2, 4, 3), seconds_per_label):
        labels.append(np.full(sec * FS, lab, dtype=np.int32))
    label = np.concatenate(labels)
    T = len(label) / FS
    n = len(label)
    chest = {k: rng.normal(size=(n, 1)) for k in ("ECG", "EMG", "EDA", "Resp")}
    chest["ACC"] = rng.normal(size=(n, 3))
    chest["Temp"] = rng.normal(size=(n, 1)).astype(np.float32)
    wrist = {
        "ACC": rng.normal(size=(int(T * 32), 3)),
        "BVP": rng.normal(size=(int(T * 64), 1)),
        "EDA": rng.normal(size=(int(T * 4), 1)),
        "TEMP": rng.normal(size=(int(T * 4), 1)),
    }
    return {"signal": {"chest": chest, "wrist": wrist}, "label": label, "subject": "S2"}


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDASTRO_DATA_ROOT", str(tmp_path))
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "WESAD.zip", "w") as zf:
        for i, s in enumerate((2, 3)):
            zf.writestr(f"WESAD/S{s}/S{s}.pkl", pickle.dumps(_fake_subject(seed=i), protocol=2))
    return tmp_path


def test_label_windows_are_pure_and_counted():
    label = np.concatenate([np.full(130 * FS, 1), np.full(5 * FS, 0), np.full(70 * FS, 2)])
    t, y = wesad.label_windows(label, 60, 30)
    # baseline 130 s -> starts at 0,30,60 (3 windows); stress 70 s -> 1 window
    assert list(y) == [0, 0, 0, 1]
    for start, cls in zip(t, y):
        seg = label[int(start * FS): int(start * FS) + 60 * FS]
        assert len(np.unique(seg)) == 1 and wesad.CLASS_NAMES[cls] == wesad.LABELS[int(seg[0])]


def test_label_windows_drop_excluded_labels():
    label = np.concatenate([np.full(100 * FS, 4), np.full(100 * FS, 0), np.full(100 * FS, 7)])
    t, y = wesad.label_windows(label, 60, 30)
    assert len(t) == 0


def test_features_shape_and_provenance(fake_root):
    params = WesadParams(window_s=60, stride_s=30, subjects=(2, 3))
    b = wesad.load_wesad(params)
    # per subject: baseline 130 s -> 3, stress 70 s -> 1, amusement 65 s -> 1
    assert len(b) == 10
    assert b.X.shape == (10, 10 * len(wesad.FEATURES))
    assert b.X.shape[1] == len(wesad.feature_names(params))
    assert b.subjects == [2, 3]
    assert b.provenance.kind == REAL and b.provenance.dataset == "WESAD"
    assert b.provenance.checksum.startswith("sha256:")
    assert (fake_root / "raw" / "WESAD.zip.meta.json").exists()


def test_sequence_representation_shape(fake_root):
    params = WesadParams(devices=("wrist",), representation="sequence", sequence_hz=4.0, subjects=(2,))
    b = wesad.load_wesad(params)
    assert b.X.shape == (5, 4, 240)


def test_cache_is_used_and_identical(fake_root):
    params = WesadParams(subjects=(2, 3))
    a = wesad.load_wesad(params)
    (fake_root / "raw" / "WESAD.zip").unlink()
    (fake_root / "raw" / "WESAD.zip").write_bytes(b"")  # archive gone; must come from cache
    b = wesad.load_wesad(params)
    assert a.provenance.processed_checksum == b.provenance.processed_checksum
    assert np.array_equal(a.X, b.X)


def test_feature_values_match_direct_computation():
    raw = _fake_subject()
    params = WesadParams(devices=("chest",), window_s=60, stride_s=30)
    X, y = wesad.process_subject(raw, params)
    ecg = raw["signal"]["chest"]["ECG"].ravel()
    first = ecg[10 * FS: 70 * FS]  # first baseline window starts after 10 s transient
    ecg_cols = slice(5, 10)  # chest channel order: ACC, ECG, ...
    np.testing.assert_allclose(X[0, ecg_cols][:4], [first.mean(), first.std(), first.min(), first.max()], rtol=1e-5)
    slope = np.polyfit(np.arange(len(first)) / FS, first, 1)[0]
    np.testing.assert_allclose(X[0, 9], slope, rtol=1e-4, atol=1e-7)


def test_unpickler_refuses_arbitrary_objects():
    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    blob = pickle.dumps({"x": Evil()}, protocol=2)
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        wesad._NumpyOnlyUnpickler(io.BytesIO(blob), encoding="latin1").load()


def test_missing_archive_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDASTRO_DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fetch_data.py"):
        wesad.load_wesad(WesadParams())


REAL_ARCHIVE = wesad.archive_path()


@pytest.mark.skipif(not REAL_ARCHIVE.exists(), reason="real WESAD archive not downloaded")
def test_real_wesad_loads_all_subjects():
    b = wesad.load_wesad(WesadParams())
    assert b.subjects == list(wesad.SUBJECTS)
    assert set(np.unique(b.y)) == {0, 1, 2}
    assert np.isfinite(b.X).all()


def test_parallel_and_serial_loading_agree(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "WESAD.zip", "w") as zf:
        for i, s in enumerate((2, 3, 4)):
            zf.writestr(f"WESAD/S{s}/S{s}.pkl", pickle.dumps(_fake_subject(seed=i), protocol=2))
    checksums = []
    for workers in ("1", "3"):
        root = tmp_path / f"root{workers}"
        root.mkdir()
        (root / "raw").symlink_to(raw)
        monkeypatch.setenv("FEDASTRO_DATA_ROOT", str(root))
        monkeypatch.setenv("FEDASTRO_WESAD_WORKERS", workers)
        checksums.append(wesad.load_wesad(WesadParams(subjects=(2, 3, 4))).provenance.processed_checksum)
    assert checksums[0] == checksums[1]
