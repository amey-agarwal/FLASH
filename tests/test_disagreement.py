"""Phase H3: disagreement between the local, regional and global models (H3)."""

import numpy as np
import pytest

from fedastro.metrics.disagreement import disagreement_report


def test_counts_rates_and_subset_accuracy():
    y = {"a": np.array([0, 1, 2, 0]), "b": np.array([1, 1])}
    preds = {
        "a": {"local": np.array([0, 1, 0, 1]), "regional": np.array([0, 1, 2, 0]), "global": np.array([0, 1, 2, 2])},
        "b": {"local": np.array([1, 0]), "regional": np.array([1, 0]), "global": np.array([1, 0])},
    }
    r = disagreement_report(y, preds, {"a": "leo", "b": "moon"})
    a = r["per_node"]["a"]
    assert a["n"] == 4 and a["n_disagree"] == 2 and a["rate"] == 0.5
    assert a["pairwise"] == {"local-regional": 0.5, "local-global": 0.5, "regional-global": 0.25}
    assert a["acc_agree"] == {"local": 1.0, "regional": 1.0, "global": 1.0}
    assert a["acc_disagree"] == {"local": 0.0, "regional": 1.0, "global": 0.5}
    b = r["per_node"]["b"]
    assert b["rate"] == 0.0 and b["acc_disagree"] == {"local": None, "regional": None, "global": None}
    assert r["per_region"]["moon"] == b
    g = r["global"]
    assert g["n"] == 6 and g["n_disagree"] == 2
    # headline: the regional model on the pooled agreeing (4: a0, a1, b0, b1) / disagreeing examples
    assert r["headline_model"] == "regional"
    assert r["acc_agree"] == pytest.approx(3 / 4) and r["acc_disagree"] == 1.0
    assert "not an independent second opinion" in r["note"]


def test_regions_pool_their_nodes():
    y = {n: np.array([0, 1]) for n in ("a", "b", "c")}
    same = {m: np.array([0, 1]) for m in ("local", "regional", "global")}
    preds = {"a": same, "b": same, "c": {**same, "local": np.array([1, 1])}}
    r = disagreement_report(y, preds, {"a": "leo", "b": "leo", "c": "mars"})
    assert r["per_region"]["leo"]["n"] == 4 and r["per_region"]["leo"]["rate"] == 0.0
    assert r["per_region"]["mars"]["rate"] == 0.5
