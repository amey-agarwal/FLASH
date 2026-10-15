import json

from fedastro.cli import main
from fedastro.metrics.schema import validate_result


def _run(tmp, smoke_config, sub, monkeypatch, *extra):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    out = tmp / sub
    assert main(["run", str(smoke_config), "--seed", "0", "--output-dir", str(out), *extra]) == 0
    (result,) = out.glob("*/result.json")
    return result


def test_smoke_run_produces_valid_synthetic_result(tmp_path, smoke_config, monkeypatch):
    path = _run(tmp_path, smoke_config, "a", monkeypatch)
    result = json.loads(path.read_text())
    validate_result(result)
    assert result["provenance"][0]["kind"] == "SYNTHETIC"
    assert result["notes"].startswith("PROVENANCE: SYNTHETIC")
    assert result["seed"] == 0
    assert [s["name"] for s in result["silos"]][-1] == "mars_transit"
    assert sum(s["n_train"] for s in result["silos"]) + result["holdout"]["n_test"] == 20 * 30
    assert (path.parent / "manifest.json").is_file()


def test_smoke_run_is_byte_identical(tmp_path, smoke_config, monkeypatch):
    a = _run(tmp_path, smoke_config, "a", monkeypatch)
    b = _run(tmp_path, smoke_config, "b", monkeypatch)
    assert a.parent.name == b.parent.name
    assert a.read_bytes() == b.read_bytes()


def test_different_seed_changes_result(tmp_path, smoke_config, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    main(["run", str(smoke_config), "--seed", "0", "--output-dir", str(tmp_path)])
    main(["run", str(smoke_config), "--seed", "1", "--output-dir", str(tmp_path)])
    results = sorted(tmp_path.glob("*/result.json"))
    assert len(results) == 2
    assert results[0].read_bytes() != results[1].read_bytes()


def test_sweep_runs_every_seed(tmp_path, smoke_config, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    assert main(["sweep", str(smoke_config), "--output-dir", str(tmp_path)]) == 0
    seeds = sorted(json.loads(p.read_text())["seed"] for p in tmp_path.glob("*/result.json"))
    assert seeds == [0, 1, 2, 3, 4]


def test_bad_config_exits_2(tmp_path, capsys):
    bad = tmp_path / "nope.yaml"
    assert main(["run", str(bad)]) == 2
    assert "config error" in capsys.readouterr().err


def test_inspect_prints_table_and_writes_nothing(tmp_path, smoke_config, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["inspect", str(smoke_config)]) == 0
    out = capsys.readouterr().out
    assert "HOLDOUT (test)" in out and "mars_transit" in out and "kind=SYNTHETIC" in out
    assert not (tmp_path / "runs").exists()
