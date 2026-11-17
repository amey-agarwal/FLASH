"""Phase H1: hierarchical topology config (CLAUDE.md section 4b/4c). Loading, strict
validation, the nodes-as-silos rule, and run ids of flat configs left unchanged."""

import shutil

import pytest
import yaml

from fedastro.cli import main
from fedastro.config import ConfigError, load_config
from fedastro.metrics.schema import SCHEMA_VERSION, SchemaError, empty_result, validate_result
from fedastro.pipeline import topology_section


@pytest.fixture
def hier_cfg(repo):
    return load_config(repo / "configs" / "experiments" / "smoke_hier.yaml")


def test_cinc15_matches_claude_md_section_4b(hier_cfg):
    h = hier_cfg.hierarchy
    assert h.name == "cinc15" and h.partition == "by_source" and h.min_participants == 5
    assert [r.name for r in h.regions] == ["leo", "moon", "mars"]
    assert [r.aggregator for r in h.regions] == ["agg_leo", "agg_moon", "agg_mars"]
    assert (h.global_aggregator.name, h.global_aggregator.host) == ("agg_global", "agg_leo")
    agencies = {r.name: [n.agency for n in r.nodes] for r in h.regions}
    assert agencies == {"leo": ["nasa", "esa", "roscosmos", "cmsa1", "partner"],
                        "moon": ["nasa", "esa", "cmsa1", "cmsa2", "partner"],
                        "mars": ["nasa", "esa", "cmsa1", "cmsa2", "partner"]}
    sources = {n.agency: n.sources for n in h.nodes}
    assert sources == {"nasa": ("g12ec",), "esa": ("ptbxl", "ptb"), "roscosmos": ("incart",),
                       "cmsa1": ("ningbo",), "cmsa2": ("chapman",), "partner": ("cpsc", "cpsc_extra")}
    assert (h.train_frac, h.val_frac, h.test_frac) == (0.7, 0.1, 0.2)
    by_region = {r.name: r for r in h.regions}
    assert by_region["leo"].global_link.one_way_latency_s == 0.0  # agg_global is hosted at agg_leo
    assert by_region["moon"].global_link.one_way_latency_s == 1.3
    assert by_region["mars"].global_link.latency_presets_s == {"near": 182.0, "mean": 751.0, "far": 1337.0}


def test_wesad12_matches_claude_md_section_4c(repo):
    h = load_config(repo / "configs" / "experiments" / "smoke_hier_wesad12.yaml").hierarchy
    assert h.partition == "one_subject_per_node" and h.min_participants == 4
    assert [len(r.nodes) for r in h.regions] == [4, 4, 4]
    assert (h.val_frac, h.test_frac, h.val_gap_windows) == (0.2, 0.0, 1)


def test_nodes_are_the_silos(hier_cfg):
    h = hier_cfg.hierarchy
    assert [s.name for s in hier_cfg.topology.silos] == [n.name for n in h.nodes]
    assert len(hier_cfg.topology.silos) == 15
    mars = next(s for s in hier_cfg.topology.silos if s.name == "mars_nasa")
    assert mars.one_way_latency_s == pytest.approx(751.06)  # node link + regional link
    assert mars.latency_presets_s["far"] == pytest.approx(1337.06)


def test_flat_configs_still_load_without_a_hierarchy(repo, smoke_config):
    cfg = load_config(smoke_config)
    assert cfg.hierarchy is None
    assert [s.name for s in cfg.topology.silos][0] == "nasa_earth"
    assert "hierarchy" not in cfg.experiment_dict()  # so locked run ids are unchanged
    assert "group_names" not in cfg.experiment_dict()["data"]["synthetic"]
    assert "hierarchy" in load_config(repo / "configs" / "experiments" / "smoke_hier.yaml").experiment_dict()


def _root(repo, tmp_path, topology_edit=None, experiment_edit=None, name="cinc15", exp="smoke_hier"):
    """A temp config root with one edited topology and/or experiment."""
    shutil.copytree(repo / "configs", tmp_path / "configs")
    topo = tmp_path / "configs" / "topologies" / f"{name}.yaml"
    raw = yaml.safe_load(topo.read_text())
    if topology_edit:
        topology_edit(raw)
        topo.write_text(yaml.safe_dump(raw))
    path = tmp_path / "configs" / "experiments" / f"{exp}.yaml"
    if experiment_edit:
        e = yaml.safe_load(path.read_text())
        experiment_edit(e)
        path.write_text(yaml.safe_dump(e))
    return path


def _drop_node(raw):
    raw["regions"][0]["nodes"].pop()


def _dup_node(raw):
    raw["regions"][1]["nodes"][0]["name"] = "leo_nasa"


def _bad_host(raw):
    raw["global_aggregator"]["host"] = "agg_venus"


def _bad_los(raw):
    raw["regions"][2]["node_link"]["los_profile"] = "sometimes"


def _typo(raw):
    raw["regions"][0]["agregator"] = "x"


def _fracs(raw):
    raw["test_frac"] = 0.3


def _shared_source(raw):
    raw["regions"][0]["nodes"][1]["sources"] = ["g12ec"]


def _no_sources(raw):
    raw["regions"][0]["nodes"][0]["sources"] = []


def _unknown_source(raw):
    raw["regions"][0]["nodes"][0]["sources"] = ["mimic"]


def _same_aggregator(raw):
    raw["regions"][1]["aggregator"] = "agg_leo"


@pytest.mark.parametrize("edit, match", [
    (_drop_node, "fewer than min_participants"),
    (_dup_node, "duplicate node names"),
    (_bad_host, "is not a regional aggregator"),
    (_bad_los, "los_profile must be one of"),
    (_typo, "unknown keys"),
    (_fracs, "must sum to 1"),
    (_shared_source, "listed by two nodes"),
    (_no_sources, "needs sources"),
    (_unknown_source, "are not synthetic groups"),
    (_same_aggregator, "duplicate aggregator names"),
])
def test_bad_topologies_are_rejected(repo, tmp_path, edit, match):
    with pytest.raises(ConfigError, match=match):
        load_config(_root(repo, tmp_path, topology_edit=edit))


def _both(e):
    e["silos"] = "five_agency"


def _holdout(e):
    e["partition"]["holdout_subjects"] = 3


def _method(e):
    e["partition"]["method"] = "dirichlet"
    e["partition"]["alpha"] = 0.5


def _missing(e):
    e["hierarchy"] = "no_such_topology"


@pytest.mark.parametrize("edit, match", [
    (_both, "not both"),
    (_holdout, "holdout_subjects to 0"),
    (_method, "the hierarchy partitions the data"),
    (_missing, "config file not found"),
])
def test_bad_hierarchical_experiments_are_rejected(repo, tmp_path, edit, match):
    with pytest.raises(ConfigError, match=match):
        load_config(_root(repo, tmp_path, experiment_edit=edit))


def test_wesad12_rejects_a_test_fraction_and_wrong_subject_count(repo, tmp_path):
    def frac(raw):
        raw["train_frac"], raw["test_frac"] = 0.6, 0.2
    with pytest.raises(ConfigError, match="needs test_frac 0"):
        load_config(_root(repo, tmp_path / "a", topology_edit=frac, name="wesad12", exp="smoke_hier_wesad12"))

    def subjects(e):
        e["data"]["synthetic"]["n_subjects"] = 20
    with pytest.raises(ConfigError, match="n_subjects = nodes"):
        load_config(_root(repo, tmp_path / "b", experiment_edit=subjects, name="wesad12", exp="smoke_hier_wesad12"))


def test_flat_holdout_rule_is_unchanged(repo, tmp_path):
    def zero(e):
        e["partition"] = {"holdout_subjects": 0}
    with pytest.raises(ConfigError, match="holdout_subjects must be >= 1"):
        load_config(_root(repo, tmp_path, experiment_edit=zero, exp="smoke"))


def test_run_refuses_a_hierarchy_until_h3_and_inspect_prints_the_node_table(repo, tmp_path, capsys):
    cfg = str(repo / "configs" / "experiments" / "smoke_hier.yaml")
    assert main(["run", cfg, "--seed", "0", "--output-dir", str(tmp_path)]) == 2
    assert "Phase H3" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
    assert main(["inspect", cfg]) == 0
    out = capsys.readouterr().out
    assert "kind=SYNTHETIC" in out and "scenario labels" in out
    assert all(name in out for name in ("leo_roscosmos", "moon_cmsa2", "mars_partner"))
    assert main(["inspect", str(repo / "configs" / "experiments" / "smoke_hier_wesad12.yaml")]) == 0
    assert "shared held-out test set" in capsys.readouterr().out


# -- result schema "2" ---------------------------------------------------------------


def _valid():
    r = empty_result()
    r["provenance"] = [{"dataset": "SYNTHETIC", "version": "v", "checksum": "c"}]
    return r


def test_schema_2_sections_are_present_with_nulls():
    r = _valid()
    assert SCHEMA_VERSION == r["schema_version"] == "2"
    for section in ("topology", "shift", "hierarchy", "disagreement", "secagg", "updates", "stream", "calibration"):
        assert section in r
    assert r["shift"] == {"enabled": False, "kind": None, "magnitude": 0.0, "provenance": "SIMULATED"}
    assert r["secagg"]["simulated"] is True and r["secagg"]["enabled"] is False
    validate_result(r)


@pytest.mark.parametrize("path", [("topology",), ("shift", "magnitude"), ("hierarchy", "per_node"),
                                  ("disagreement", "acc_agree"), ("secagg", "overhead_bytes"),
                                  ("updates", "events"), ("stream", "label_fraction"), ("calibration", "ece")])
def test_dropping_a_schema_2_field_fails(path):
    r = _valid()
    d = r
    for p in path[:-1]:
        d = d[p]
    del d[path[-1]]
    with pytest.raises(SchemaError, match="missing field"):
        validate_result(r)


def test_schema_1_artifacts_stay_valid_and_labels_cannot_be_weakened():
    r = _valid()
    old = {k: v for k, v in r.items() if k not in (
        "schema_version", "topology", "shift", "hierarchy", "disagreement", "secagg", "updates", "stream", "calibration")}
    validate_result(old)  # an artifact written before the bump
    r["shift"]["provenance"] = "MEASURED"
    with pytest.raises(SchemaError, match="SIMULATED"):
        validate_result(r)


def test_topology_section(hier_cfg, smoke_config):
    flat = topology_section(load_config(smoke_config))
    assert flat == {"name": "five_agency", "regions": [], "aggregators": [], "min_participants": None}
    t = topology_section(hier_cfg)
    assert t["aggregators"] == ["agg_leo", "agg_moon", "agg_mars", "agg_global"] and t["min_participants"] == 5
    assert [len(r["nodes"]) for r in t["regions"]] == [5, 5, 5]
