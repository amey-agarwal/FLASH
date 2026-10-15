import copy

import pytest

from fedastro.metrics.schema import SchemaError, empty_result, validate_result


def _valid():
    r = empty_result()
    r["provenance"] = [{"dataset": "SYNTHETIC", "version": "v", "checksum": "c"}]
    return r


def test_empty_result_with_provenance_is_valid():
    validate_result(_valid())


@pytest.mark.parametrize("path", [
    ("notes",), ("arms", "federated"), ("privacy", "attack", "mse"), ("drift", "onset_round"),
])
def test_dropping_a_field_fails(path):
    r = _valid()
    d = r
    for p in path[:-1]:
        d = d[p]
    del d[path[-1]]
    with pytest.raises(SchemaError, match="missing field"):
        validate_result(r)


def test_missing_provenance_fails():
    with pytest.raises(SchemaError):
        validate_result(empty_result())


def test_bad_round_entry_fails():
    r = _valid()
    r["arms"]["federated"]["rounds"] = [{"round": 1, "global": {}, "per_silo": {},
                                         "bytes": 0, "participants": []}]
    with pytest.raises(SchemaError, match="sim_wallclock_s"):
        validate_result(r)


def test_bool_is_not_a_number():
    r = _valid()
    r["privacy"]["epsilon"] = True
    with pytest.raises(SchemaError):
        validate_result(r)
