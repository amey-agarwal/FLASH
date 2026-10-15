"""One experiment run: config + seed -> result dict.

Phase 1 runs the data path only: load, hold out test subjects, partition into silos,
record. Training arms are added in Phases 2-3; until then their sections are present
but empty, as the schema requires.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fedastro.config import ExperimentConfig
from fedastro.data.base import SYNTHETIC, DatasetBundle
from fedastro.data.partition import Federation, make_federation
from fedastro.manifest import git_sha, make_run_id, timestamp_utc, write_run
from fedastro.metrics.schema import empty_result
from fedastro.seeding import seed_everything


def load_dataset(cfg: ExperimentConfig, seed: int) -> DatasetBundle:
    if cfg.data.name == "synthetic":
        from fedastro.data.synthetic import load_synthetic

        return load_synthetic(cfg.data.synthetic, seed)
    if cfg.data.name == "wesad":
        from fedastro.data.wesad import load_wesad

        return load_wesad(cfg.data.wesad)
    raise NotImplementedError(f"dataset {cfg.data.name!r} is not implemented yet")


def build_federation(cfg: ExperimentConfig, seed: int) -> tuple[DatasetBundle, Federation]:
    rng = seed_everything(seed)
    bundle = load_dataset(cfg, seed)
    shares = {s.name: s.data_share for s in cfg.topology.silos}
    fed = make_federation(
        bundle, shares, cfg.partition.method, cfg.partition.holdout_subjects, rng,
        alpha=cfg.partition.alpha,
    )
    return bundle, fed


def describe_federation(fed: Federation) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    silos = [
        {
            "name": name,
            "n_train": len(split.data),
            "n_subjects": len(split.data.subjects),
            "subjects": split.data.subjects,
            "label_counts": split.data.label_counts(),
            "n_dropped": split.n_dropped,
            "target_proportions": (
                None if split.target_proportions is None else list(split.target_proportions)
            ),
        }
        for name, split in fed.silos.items()
    ]
    holdout = {
        "n_test": len(fed.holdout),
        "subjects": fed.holdout.subjects,
        "label_counts": fed.holdout.label_counts(),
    }
    return silos, holdout


def run_experiment(cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    bundle, fed = build_federation(cfg, seed)
    silos, holdout = describe_federation(fed)

    result = empty_result()
    run_cfg = cfg.experiment_dict()
    run_cfg["seed"] = seed
    result.update(
        run_id=make_run_id(cfg, seed),
        git_sha=git_sha(),
        timestamp_utc=timestamp_utc(),
        config=run_cfg,
        seed=seed,
        provenance=[bundle.provenance.to_dict()],
        silos=silos,
        holdout=holdout,
    )
    notes = []
    if bundle.provenance.kind == SYNTHETIC:
        notes.append("PROVENANCE: SYNTHETIC. Smoke-test data, not physiological measurements.")
    notes.append("Data path only; training arms not implemented yet.")
    if cfg.notes:
        notes.append(cfg.notes)
    result["notes"] = " ".join(notes)
    return result


def run_and_write(cfg: ExperimentConfig, seed: int) -> Path:
    t0 = time.perf_counter()
    result = run_experiment(cfg, seed)
    return write_run(result, cfg.output_dir, time.perf_counter() - t0)


def format_distribution_table(fed: Federation, class_names: tuple[str, ...]) -> str:
    """Plain-text per-silo class distribution table (counts and row percentages)."""
    rows = [(name, split.data, split.n_dropped) for name, split in fed.silos.items()]
    rows.append(("HOLDOUT (test)", fed.holdout, 0))
    head = f"{'silo':<16}{'subj':>5}{'n':>7}" + "".join(f"{c:>18}" for c in class_names) + f"{'dropped':>9}"
    lines = [head, "-" * len(head)]
    for name, data, dropped in rows:
        counts = data.label_counts()
        n = max(len(data), 1)
        cells = "".join(f"{counts[c]:>8} ({100 * counts[c] / n:5.1f}%)" for c in class_names)
        lines.append(f"{name:<16}{len(data.subjects):>5}{len(data):>7}{cells}{dropped:>9}")
    return "\n".join(lines)
