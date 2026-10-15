"""One experiment run: config + seed -> result dict.

Phase 0 runs the data path only (load, partition, record). The training arms are
added in Phases 2-3; until then their sections are present but empty, as the
schema requires.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fedastro.config import ExperimentConfig
from fedastro.data import partition
from fedastro.data.base import SYNTHETIC, DatasetBundle
from fedastro.data.synthetic import load_synthetic
from fedastro.manifest import git_sha, make_run_id, timestamp_utc, write_run
from fedastro.metrics.schema import empty_result
from fedastro.seeding import seed_everything


def load_dataset(cfg: ExperimentConfig, seed: int) -> DatasetBundle:
    if cfg.data.name == "synthetic":
        return load_synthetic(cfg.data.synthetic, seed)
    raise NotImplementedError(f"dataset {cfg.data.name!r} is not implemented yet")


def run_experiment(cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    rng = seed_everything(seed)
    bundle = load_dataset(cfg, seed)
    shares = {s.name: s.data_share for s in cfg.topology.silos}
    if cfg.partition.method != "by_subject":
        raise NotImplementedError(cfg.partition.method)
    silo_data = partition.by_subject(bundle, shares, rng)

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
    )
    result["silos"] = [
        {
            "name": name,
            "n_train": len(d),
            "n_subjects": int(len(set(d.subject_id.tolist()))),
            "label_counts": d.label_counts(),
        }
        for name, d in silo_data.items()
    ]
    notes = []
    if bundle.provenance.kind == SYNTHETIC:
        notes.append("PROVENANCE: SYNTHETIC. Smoke-test data, not physiological measurements.")
    notes.append("Phase 0 scaffold: data path only; training arms not implemented.")
    if cfg.notes:
        notes.append(cfg.notes)
    result["notes"] = " ".join(notes)
    return result


def run_and_write(cfg: ExperimentConfig, seed: int) -> Path:
    t0 = time.perf_counter()
    result = run_experiment(cfg, seed)
    return write_run(result, cfg.output_dir, time.perf_counter() - t0)
