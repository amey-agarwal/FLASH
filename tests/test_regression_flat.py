"""Phase H0 reproducibility lock (CLAUDE.md rule 8): the flat five-silo WESAD results
recorded in PROGRESS.md (Phase 3, golden item G1) must still reproduce on ``hier-fl``.

The reference, ``tests/golden/flat_wesad_g1.json``, holds the full-precision final metrics
of the laptop runs behind the Phase 3 table. Seed 0 runs in the default suite (about 40 s);
seeds 1-4 carry the ``slow`` marker (``pytest -m slow``). All are skipped without the WESAD
archive. If one fails, a locked result moved: stop and report it, never edit the reference
without the user's approval in PROGRESS.md."""

import json
from pathlib import Path

import pytest

from fedastro.config import load_config
from fedastro.data import wesad
from fedastro.pipeline import run_experiment

REPO = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((REPO / "tests" / "golden" / "flat_wesad_g1.json").read_text())
TOLERANCE = REFERENCE["tolerance"]  # absolute; the laptop-vs-cluster tolerance
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "auc")
ARM_FIELDS = {"centralized": ("centralized", "final"), "local_only": ("local_only", "mean"),
              "federated": ("federated", "final")}

# The per-seed table in PROGRESS.md (Phase 3 acceptance results), as printed there.
PROGRESS_TABLE = {  # seed: (centralized acc, federated acc, local_only acc, fed macro-F1, local macro-F1)
    0: (0.849, 0.670, 0.631, 0.635, 0.512),
    1: (0.863, 0.722, 0.642, 0.688, 0.546),
    2: (0.901, 0.830, 0.705, 0.782, 0.593),
    3: (0.849, 0.679, 0.640, 0.621, 0.531),
    4: (0.877, 0.642, 0.652, 0.615, 0.576),
}

needs_wesad = pytest.mark.skipif(not wesad.archive_path().exists(), reason="real WESAD archive not downloaded")


def test_reference_matches_the_progress_table():
    assert TOLERANCE == 1e-3
    for seed, row in PROGRESS_TABLE.items():
        ref = REFERENCE["seeds"][str(seed)]
        got = (ref["centralized"]["accuracy"], ref["federated"]["accuracy"], ref["local_only"]["accuracy"],
               ref["federated"]["macro_f1"], ref["local_only"]["macro_f1"])
        assert tuple(round(v, 3) for v in got) == row


def test_reference_config_is_the_locked_one():
    cfg = load_config(REPO / REFERENCE["config"])
    assert cfg.config_hash().startswith(REFERENCE["config_hash_prefix"])
    assert cfg.privacy.enabled is False


@needs_wesad
@pytest.mark.parametrize("seed", [0] + [pytest.param(s, marks=pytest.mark.slow) for s in (1, 2, 3, 4)])
def test_flat_wesad_reproduces(seed):
    arms = run_experiment(load_config(REPO / REFERENCE["config"]), seed)["arms"]
    problems = []
    for name, (arm, field) in ARM_FIELDS.items():
        for metric in METRICS:
            want, got = REFERENCE["seeds"][str(seed)][name][metric], arms[arm][field][metric]
            if abs(got - want) > TOLERANCE:
                problems.append(f"{name}.{metric}: recorded {want:.4f}, got {got:.4f}")
    assert not problems, f"flat WESAD seed {seed} moved:\n" + "\n".join(problems)
