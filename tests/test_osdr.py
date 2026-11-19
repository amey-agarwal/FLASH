"""Phase H9: OSD-435 loader (data/osdr.py) in the FLUID setup. Needs the raw CSV
(``python scripts/fetch_data.py osd435 --accept-license``); skipped without it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fedastro.config import Osd435Params, load_config
from fedastro.data import osdr
from fedastro.grid import load_grid
from fedastro.pipeline import build_federation

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "experiments" / "osd435_fluid.yaml"
SENSITIVITY = Osd435Params(include_radtype=False, include_month0=False)

needs_csv = pytest.mark.skipif(not osdr.csv_path().exists(), reason="OSD-435 CSV not fetched")


@needs_csv
def test_filter_counts_match_the_inspection():
    b = osdr.load_osd435(Osd435Params())
    assert (len(b), len(b.subjects), int(b.y.sum())) == (809, 280, 528)
    assert b.X.shape == (809, len(osdr.feature_names(Osd435Params()))) == (809, 9)
    s = osdr.load_osd435(SENSITIVITY)
    assert (len(s), len(s.subjects), int(s.y.sum())) == (572, 215, 366)
    assert s.X.shape[1] == len(osdr.feature_names(SENSITIVITY)) == 5


@needs_csv
def test_rows_follow_the_fluid_filter():
    d = osdr.rows(Osd435Params())
    assert not set(d["DOSE"]) & set(osdr.LOW_DOSES)
    assert not d[list(osdr.MEASURES)].isna().any().any()
    assert set(d["DOSE"]) - {0.0} and min(set(d["DOSE"]) - {0.0}) >= 0.5
    assert (osdr.rows(SENSITIVITY)["MONTH"] > 0).all()


@needs_csv
def test_provenance_and_label():
    b = osdr.load_osd435(Osd435Params())
    assert b.provenance.kind == "REAL" and b.provenance.checksum == f"sha256:{osdr.SHA256}"
    assert b.class_names == ("sham", "irradiated")
    d = osdr.rows(Osd435Params())
    np.testing.assert_array_equal(b.y, (d["DOSE"] >= 0.5).astype(int))
    # one-hot radiation type: exactly one column set per row
    assert (b.X[:, :4].sum(axis=1) == 1).all()


def test_a_changed_file_is_refused(tmp_path):
    f = tmp_path / osdr.FILENAME
    f.write_text("SID,DOSE\n1,0\n")
    with pytest.raises(ValueError, match="sha256"):
        osdr.rows(Osd435Params(), f)


@needs_csv
@pytest.mark.parametrize("over", [{}, {"data.osd435.include_radtype": False, "data.osd435.include_month0": False,
                                       "partition.holdout_subjects": 65}])
def test_no_mouse_in_two_splits(over):
    cfg = load_config(CONFIG, over)
    for seed in range(3):
        bundle, fed = build_federation(cfg, seed, 0)
        sets = [set(fed.holdout.subjects)] + [set(s.data.subjects) for s in fed.silos.values()]
        assert sum(map(len, sets)) == len(set().union(*sets)) == len(bundle.subjects)
        assert abs(len(sets[0]) / len(bundle.subjects) - 0.3) < 0.01  # 30% of mice held out
    # the held-out mice are the same for every run seed
    assert build_federation(cfg, 0, 0)[1].holdout.subjects == build_federation(cfg, 4, 0)[1].holdout.subjects


def test_sweep_loads_and_unused_dataset_blocks_stay_out_of_run_ids():
    _, vs = load_grid(REPO / "configs" / "sweeps" / "h9_osd435.yaml")
    assert len(vs) == 8 and {v.config.data.name for v in vs} == {"osd435"}
    assert all("osd435" not in load_config(REPO / "configs" / "experiments" / "wesad_by_subject.yaml")
               .experiment_dict()["data"] for _ in [0])
