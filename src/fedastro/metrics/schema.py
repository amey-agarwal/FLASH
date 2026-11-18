"""The run artifact contract (CLAUDE.md section 6).

``empty_result`` builds a result with every section present and unused values null.
``validate_result`` checks a result dict against the contract. The dashboard and
figure scripts depend on this shape; change it only together with them.
"""

from __future__ import annotations

from typing import Any

from fedastro.data.base import DOMAINS

# "2" (branch hier-fl, Phase H1) adds schema_version itself and the sections topology,
# shift, hierarchy, disagreement, secagg, updates, stream and calibration. Every "1" field
# is unchanged. Artifacts without schema_version are version "1" and stay valid.
SCHEMA_VERSION = "2"


class SchemaError(ValueError):
    pass


def empty_result() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "",
        "git_sha": "",
        "timestamp_utc": "",
        "config": {},
        "seed": 0,
        "provenance": [],
        "silos": [],
        "arms": {
            "centralized": {"final": {"accuracy": None, "auc": None}, "history": []},
            "local_only": {"per_silo": {}, "mean": {}},
            "federated": {"rounds": [], "final": {}},
        },
        "privacy": {
            "enabled": False,
            "epsilon": None,
            "delta": None,
            "max_grad_norm": None,
            "attack": {"mse": None},
        },
        "comms": {"preset": None, "los_events": []},
        "drift": {"enabled": False, "onset_round": None, "detected_round": None},
        "notes": "",
        # Schema "2" sections. Present in every run; null/empty until the phase that fills
        # them (hierarchy and disagreement H3, secagg H4, updates H5, calibration H6, stream H7).
        "topology": {"name": None, "regions": [], "aggregators": [], "min_participants": None},
        # provenance is always SIMULATED: no dataset here has a measured location shift.
        "shift": {"enabled": False, "kind": None, "magnitude": 0.0, "provenance": "SIMULATED"},
        "hierarchy": {"rounds": [], "per_node": {}, "per_region": {}, "global": {}},
        "disagreement": {"per_node": {}, "per_region": {}, "global": {},
                         "acc_agree": None, "acc_disagree": None},
        # simulated is always true: a protocol simulation, not a cryptographic implementation.
        "secagg": {"enabled": False, "simulated": True, "overhead_bytes": 0,
                   "overhead_sim_s": 0.0, "dropout_recoveries": 0},
        "updates": {"tolerance": None, "events": []},
        "stream": {"enabled": False, "label_fraction": None, "label_delay_rounds": None,
                   "test_then_train": {}},
        "calibration": {"ece": {}},
    }


# (path, allowed types). None in a type tuple means the value may be null.
_NUM = (int, float)
_REQUIRED: list[tuple[str, tuple]] = [
    ("run_id", (str,)),
    ("git_sha", (str,)),
    ("timestamp_utc", (str,)),
    ("config", (dict,)),
    ("seed", (int,)),
    ("provenance", (list,)),
    ("silos", (list,)),
    ("arms", (dict,)),
    ("arms.centralized", (dict,)),
    ("arms.centralized.final", (dict,)),
    ("arms.centralized.final.accuracy", (*_NUM, type(None))),
    ("arms.centralized.final.auc", (*_NUM, type(None))),
    ("arms.centralized.history", (list,)),
    ("arms.local_only", (dict,)),
    ("arms.local_only.per_silo", (dict,)),
    ("arms.local_only.mean", (dict,)),
    ("arms.federated", (dict,)),
    ("arms.federated.rounds", (list,)),
    ("arms.federated.final", (dict,)),
    ("privacy", (dict,)),
    ("privacy.enabled", (bool,)),
    ("privacy.epsilon", (*_NUM, type(None))),
    ("privacy.delta", (*_NUM, type(None))),
    ("privacy.max_grad_norm", (*_NUM, type(None))),
    ("privacy.attack", (dict,)),
    ("privacy.attack.mse", (*_NUM, type(None))),
    ("comms", (dict,)),
    ("comms.preset", (str, type(None))),
    ("comms.los_events", (list,)),
    ("drift", (dict,)),
    ("drift.enabled", (bool,)),
    ("drift.onset_round", (int, type(None))),
    ("drift.detected_round", (int, type(None))),
    ("notes", (str,)),
]

_NULL = type(None)
_REQUIRED_V2: list[tuple[str, tuple]] = [
    ("topology", (dict,)),
    ("topology.name", (str, _NULL)),
    ("topology.regions", (list,)),
    ("topology.aggregators", (list,)),
    ("topology.min_participants", (int, _NULL)),
    ("shift", (dict,)),
    ("shift.enabled", (bool,)),
    ("shift.kind", (str, _NULL)),
    ("shift.magnitude", _NUM),
    ("shift.provenance", (str,)),
    ("hierarchy", (dict,)),
    ("hierarchy.rounds", (list,)),
    ("hierarchy.per_node", (dict,)),
    ("hierarchy.per_region", (dict,)),
    ("hierarchy.global", (dict,)),
    ("disagreement", (dict,)),
    ("disagreement.per_node", (dict,)),
    ("disagreement.per_region", (dict,)),
    ("disagreement.global", (dict,)),
    ("disagreement.acc_agree", (*_NUM, _NULL)),
    ("disagreement.acc_disagree", (*_NUM, _NULL)),
    ("secagg", (dict,)),
    ("secagg.enabled", (bool,)),
    ("secagg.simulated", (bool,)),
    ("secagg.overhead_bytes", (int,)),
    ("secagg.overhead_sim_s", (*_NUM, _NULL)),  # null with comms off (rule 10)
    ("secagg.dropout_recoveries", (int,)),
    ("updates", (dict,)),
    # null = gate off; a number (fixed tolerance) or the rule object (Phase H5: k x bootstrap SE
    # per node and version, each event carries its tolerance_used)
    ("updates.tolerance", (*_NUM, dict, _NULL)),
    ("updates.events", (list,)),
    ("stream", (dict,)),
    ("stream.enabled", (bool,)),
    ("stream.label_fraction", (*_NUM, _NULL)),
    ("stream.label_delay_rounds", (int, _NULL)),
    ("stream.test_then_train", (dict,)),
    ("calibration", (dict,)),
    ("calibration.ece", (dict,)),
]

_HIER_ROUND_KEYS = {"round": (int,), "sim_wallclock_s": (*_NUM, _NULL), "regions": (dict,),
                    "global_aggregated": (bool,)}
_HIER_REGION_KEYS = {"participants": (list,), "fallback": (bool,), "bytes": (int,)}

_PROVENANCE_KEYS = ("dataset", "version", "checksum")
# Optional (CLAUDE2.md section 6): absent in artifacts that predate them, checked when present.
_GROUPS_KEYS = {"silo_to_groups": (dict,), "holdout_groups": (list,), "unused_groups": (list,)}
_SILO_KEYS = {"name": (str,), "n_train": (int,), "label_counts": (dict,)}
_ROUND_KEYS = {
    "round": (int,),
    "global": (dict,),
    "per_silo": (dict,),
    "bytes": (int,),
    # null until Phase 6 models link time; never filled with real wall-clock
    "sim_wallclock_s": (*_NUM, type(None)),
    "participants": (list,),
}


def _get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise SchemaError(f"missing field: {path}")
        cur = cur[part]
    return cur


def _check(value: Any, types: tuple, where: str) -> None:
    # bool is a subclass of int; only accept it where bool is listed explicitly.
    if isinstance(value, bool) and bool not in types:
        raise SchemaError(f"{where}: expected {types}, got bool")
    if not isinstance(value, types):
        raise SchemaError(f"{where}: expected {types}, got {type(value).__name__}")


def validate_result(result: dict[str, Any]) -> None:
    for path, types in _REQUIRED:
        _check(_get(result, path), types, path)
    version = result.get("schema_version", "1")
    if version not in ("1", "2"):
        raise SchemaError(f"schema_version: unknown version {version!r}")
    if version == "2":
        for path, types in _REQUIRED_V2:
            _check(_get(result, path), types, path)
        for i, rec in enumerate(result["hierarchy"]["rounds"]):
            for k, types in _HIER_ROUND_KEYS.items():
                _check(_get(rec, k), types, f"hierarchy.rounds[{i}].{k}")
            for name, reg in rec["regions"].items():
                for k, types in _HIER_REGION_KEYS.items():
                    _check(_get(reg, k), types, f"hierarchy.rounds[{i}].regions.{name}.{k}")
        if result["shift"]["provenance"] != "SIMULATED" or result["secagg"]["simulated"] is not True:
            raise SchemaError("shift.provenance must be 'SIMULATED' and secagg.simulated must be true")
        st = result["stream"]
        if st["enabled"]:  # Phase H7
            for k, types in (("label_fraction", _NUM), ("label_delay_rounds", (int,)), ("order", (str,)),
                             ("audit", (dict,)), ("test_then_train.rounds", (list,)),
                             ("test_then_train.models", (list,)), ("fixed_test", (dict,))):
                _check(_get(st, k), types, f"stream.{k}")
            if st["audit"].get("ok") is not True:
                raise SchemaError("stream.audit.ok must be true (an example was scored after training on it)")
    if "model_report" in result:  # Phase H6, optional
        for k, types in (("arch", (str,)), ("n_params", (int,)), ("payload_bytes", (int,)), ("rows", (dict,))):
            _check(_get(result, f"model_report.{k}"), types, f"model_report.{k}")
        for model, row in result["model_report"]["rows"].items():
            for k in ("accuracy", "macro_f1", "auc", "ece", "worst_node_accuracy"):
                _check(_get(row, k), (*_NUM, _NULL), f"model_report.rows.{model}.{k}")
    if not result["provenance"]:
        raise SchemaError("provenance must list at least one dataset")
    for i, p in enumerate(result["provenance"]):
        for k in _PROVENANCE_KEYS:
            _check(_get(p, k), (str,), f"provenance[{i}].{k}")
        if "domain" in p and p["domain"] not in DOMAINS:
            raise SchemaError(f"provenance[{i}].domain: {p['domain']!r} not in {DOMAINS}")
    if "groups" in result:
        for k, types in _GROUPS_KEYS.items():
            _check(_get(result, f"groups.{k}"), types, f"groups.{k}")
    for i, s in enumerate(result["silos"]):
        for k, types in _SILO_KEYS.items():
            _check(_get(s, k), types, f"silos[{i}].{k}")
    for i, r in enumerate(result["arms"]["federated"]["rounds"]):
        for k, types in _ROUND_KEYS.items():
            _check(_get(r, k), types, f"arms.federated.rounds[{i}].{k}")
