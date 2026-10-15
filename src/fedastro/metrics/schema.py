"""The run artifact contract (CLAUDE.md section 6).

``empty_result`` builds a result with every section present and unused values null.
``validate_result`` checks a result dict against the contract. The dashboard and
figure scripts depend on this shape; change it only together with them.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1"


class SchemaError(ValueError):
    pass


def empty_result() -> dict[str, Any]:
    return {
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

_PROVENANCE_KEYS = ("dataset", "version", "checksum")
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
    if not result["provenance"]:
        raise SchemaError("provenance must list at least one dataset")
    for i, p in enumerate(result["provenance"]):
        for k in _PROVENANCE_KEYS:
            _check(_get(p, k), (str,), f"provenance[{i}].{k}")
    for i, s in enumerate(result["silos"]):
        for k, types in _SILO_KEYS.items():
            _check(_get(s, k), types, f"silos[{i}].{k}")
    for i, r in enumerate(result["arms"]["federated"]["rounds"]):
        for k, types in _ROUND_KEYS.items():
            _check(_get(r, k), types, f"arms.federated.rounds[{i}].{k}")
