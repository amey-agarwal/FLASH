"""scripts/make_figures.py (Phase H11): input resolution, traceability of the sidecar numbers and
the exploratory fig 8.

Every input here is SYNTHETIC, built in ``tmp_path`` by the test itself, so the tests run from a
clean clone without ``runs/``. The numbers mean nothing; the tests check only that the figures
plot exactly what their input files contain and record where it came from."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_figures", REPO / "scripts" / "make_figures.py")
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

FAMILIES = ("logreg", "mlp", "cnn1d")


def _write(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def _row(comparison: str, diff: float, a: float = 0.9, b: float = 0.8) -> dict:
    return {"comparison": comparison, "a": a, "b": b, "diff_mean": diff,
            "ci95": [diff - 0.01, diff + 0.01], "verdict": "supported" if diff - 0.01 > 0 else "inconclusive"}


def _hypotheses(base: float) -> dict:
    fams = {}
    for i, fam in enumerate(FAMILIES):
        d = base + i / 100
        fams[fam] = {
            "H1": {"setting": "no shift (magnitude 0)",
                   "rows": [_row("federated - local_only (accuracy)", d), _row("federated - local_only (macro_f1)", d)],
                   "own_test_rows": [_row("federated - local_only, own test (accuracy)", -d)]},
            "H2": {"setting": "shift both magnitude 1.0",
                   "rows": [_row("(a) regional - local_only (accuracy)", d / 2), _row("(b) regional - global (accuracy)", d / 3)],
                   "control": _row("control: regional - global (accuracy)", -d / 4)},
            "H3": {"setting": "shift both magnitude 1.0",
                   "rows": [_row("agree - disagree", d, a=0.95, b=0.95 - d)]},
        }
    return {"families": fams}


# -- input resolution -------------------------------------------------------------------
def test_first_root_with_a_match_wins_and_later_roots_fill_the_gaps(tmp_path):
    new, old = tmp_path / "new", tmp_path / "old"
    _write(new / "a-summary.json", {"v": "new"})
    _write(old / "a-summary.json", {"v": "old"})
    _write(old / "b-summary.json", {"v": "old"})
    assert mf.one([new, old], "a-*.json") == new / "a-summary.json"
    assert mf.one([new, old], "b-*.json") == old / "b-summary.json"
    assert mf.many([new, old], "*.json") == [new / "a-summary.json"]  # the whole sweep from one root
    with pytest.raises(mf.FigureInputError):
        mf.one([new, old], "c-*.json")


def test_own_test_runs_skip_a_root_whose_runs_predate_h11a(tmp_path):
    pattern = "h9_cinc15_logreg_local_only-enabledTrue-kindboth-magnitude0p0-seed{}-x/result.json"
    old, new = tmp_path / "old", tmp_path / "new"
    for s in (0, 1):
        _write(old / pattern.format(s), {"arms": {"local_only": {"mean": {}}}})
    assert mf.own_test_runs([old], "cinc15", "logreg", "both", "0p0") == []
    assert mf.own_test_cell([old], "cinc15", "logreg", "both", "0p0", "accuracy") == "n/a"
    for s, acc in ((0, 0.8), (1, 0.6)):
        _write(new / pattern.format(s), {"arms": {"local_only": {"own_test": {"pooled": {"accuracy": acc}}}}})
    runs = mf.own_test_runs([old, new], "cinc15", "logreg", "both", "0p0")
    assert [p.parent.parent for p in runs] == [new, new]
    mean, _ = mf.own_test_ms(runs, "accuracy")
    assert mean == pytest.approx(0.7)


# -- traceability ------------------------------------------------------------------------
def _check_side(out: Path, name: str) -> dict:
    for ext in ("png", "pdf", "json"):
        assert (out / f"{name}.{ext}").is_file(), f"{name}.{ext} not written"
    side = json.loads((out / f"{name}.json").read_text())
    assert side["sources"], "a figure must list its source files"
    return side


def test_fig2_and_fig3_plot_exactly_the_hypotheses_numbers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hyps = {"cinc15": _write(tmp_path / "hypotheses_cinc15.json", _hypotheses(0.05)),
            "wesad12": _write(tmp_path / "hypotheses_wesad12.json", _hypotheses(0.20))}
    out = tmp_path / "figs"
    mf.fig_h3(hyps, out)
    mf.fig_h4(hyps, out)
    s2, s3 = _check_side(out, "fig2_h3_disagreement"), _check_side(out, "fig3_h4_model_comparison")
    for side in (s2, s3):
        for src in side["sources"]:
            assert (tmp_path / src).is_file()
    for ds, path in hyps.items():
        h = json.loads(path.read_text())
        for fam in FAMILIES:
            f = h["families"][fam]
            assert s2["data"][ds][fam]["diff_mean"] == f["H3"]["rows"][0]["diff_mean"]
            assert s3["data"][ds][fam]["H1"]["diff_mean"] == f["H1"]["rows"][0]["diff_mean"]
            assert s3["data"][ds][fam]["H1own"]["diff_mean"] == f["H1"]["own_test_rows"][0]["diff_mean"]
            assert s3["data"][ds][fam]["H2a"]["diff_mean"] == f["H2"]["rows"][0]["diff_mean"]
            assert s3["data"][ds][fam]["ctrl"]["ci95"] == f["H2"]["control"]["ci95"]


def test_fig8_recall_is_the_seed_mean_of_each_nodes_minority_recall(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "runs"
    nodes = {"leo_small": (100, {"normal": 0, "abnormal": 100}), "moon_esa": (300, {"normal": 140, "abnormal": 160})}
    recalls = {0: {"leo_small": (0.0, 0.0), "moon_esa": (0.8, 0.5)},
               1: {"leo_small": (0.0, 0.0), "moon_esa": (0.6, 0.7)}}
    for fam in FAMILIES:
        for seed, rec in recalls.items():
            _write(root / f"h9_cinc15_{fam}_hierarchical-enabledTrue-kindboth-magnitude0p0-seed{seed}-x/result.json", {
                "silos": [{"name": n, "n_train": nt, "label_counts": lc} for n, (nt, lc) in nodes.items()],
                "hierarchy": {"per_node": {n: {"local": {"recall_per_class": [loc, 1.0]},
                                                "global": {"recall_per_class": [glo, 1.0]}}
                                            for n, (loc, glo) in rec.items()}}})
    mf.fig_minority([root], tmp_path / "figs")
    side = _check_side(tmp_path / "figs", "fig8_minority_recall")
    assert side["caption"].startswith("Not pre-registered")
    for fam in FAMILIES:
        d = side["data"][fam]
        assert list(d) == ["leo_small", "moon_esa"]  # ordered by training size
        assert d["moon_esa"]["local"] == pytest.approx(0.7)
        assert d["moon_esa"]["global"] == pytest.approx(0.6)
        assert d["leo_small"]["n_normal_train"] == 0
        assert d["moon_esa"]["n_seeds"] == 2
