"""One experiment run: config + seed -> result dict.

Load, hold out test subjects, partition into silos, then train the configured arms.
Arms not run stay present in the result with empty/null values, as the schema requires.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fedastro.config import ExperimentConfig
from fedastro.data import get_dataset
from fedastro.data.base import SYNTHETIC, DatasetBundle
from fedastro.data.partition import Federation, make_federation
from fedastro.manifest import git_sha, make_run_id, timestamp_utc, write_run
from fedastro.metrics.schema import empty_result
from fedastro.seeding import seed_everything


def load_dataset(cfg: ExperimentConfig, seed: int) -> DatasetBundle:
    """Through the registry (fedastro.data.REGISTRY); a new dataset needs no edit here."""
    return get_dataset(cfg.data.name).load(cfg.data, seed)


def build_federation(cfg: ExperimentConfig, seed: int, draw: int = 0) -> tuple[DatasetBundle, Federation]:
    if not 0 <= draw < cfg.evaluation.holdout_draws:
        raise ValueError(f"holdout draw {draw} not in [0, {cfg.evaluation.holdout_draws})")
    rng = seed_everything(seed)
    bundle = load_dataset(cfg, seed)
    shares = {s.name: s.data_share for s in cfg.topology.silos}
    fed = make_federation(
        bundle, shares, cfg.partition.method, cfg.partition.holdout_subjects, rng,
        alpha=cfg.partition.alpha, holdout_seed=cfg.partition.holdout_seed,
        group_map=cfg.partition.group_map, holdout_groups=cfg.partition.holdout_groups,
        holdout_draw=draw,
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
    if fed.groups is not None:  # grouped data only, so earlier results are unchanged
        for entry in silos:
            entry["groups"] = fed.groups["silo_to_groups"][entry["name"]]
        holdout["groups"] = fed.holdout.groups
    return silos, holdout


def run_arms(cfg: ExperimentConfig, fed: Federation, seed: int, result: dict[str, Any]) -> dict[str, Any]:
    """Train the configured arms into ``result["arms"]``. Returns what the federated
    arm captured (global weights, scaler) for the privacy attack, or {}; with held-out
    groups or the ensemble arm, ``capture["models"]`` also holds the trained models."""
    from fedastro.baselines.centralized import run_centralized
    from fedastro.baselines.common import resolve_device
    from fedastro.baselines.local_only import run_local_only

    device = resolve_device(cfg.device)
    # Keep trained models only when something scores them after training (Phase 12).
    group_eval = bool(fed.groups and fed.groups["holdout_groups"])
    keep = group_eval or "ensemble_of_locals" in cfg.arms
    models: dict[str, Any] = {}
    cen_capture: dict[str, Any] | None = {} if group_eval else None
    local_models: list | None = [] if keep and "local_only" in cfg.arms else None
    if "centralized" in cfg.arms:
        result["arms"]["centralized"] = run_centralized(fed, cfg, seed, device, capture=cen_capture)
        if cen_capture:
            models["centralized"] = (cen_capture["model"], cen_capture["scaler"])
    if "local_only" in cfg.arms:
        result["arms"]["local_only"] = run_local_only(fed, cfg, seed, device, models=local_models)
    if "ensemble_of_locals" in cfg.arms:
        from fedastro.baselines.ensemble import run_ensemble_of_locals

        local_arm = result["arms"]["local_only"] if "local_only" in cfg.arms else None
        result["arms"]["ensemble_of_locals"], local_models = run_ensemble_of_locals(
            fed, cfg, seed, device, local_models, local_arm)
    if local_models is not None:
        models["local"] = local_models
    capture: dict[str, Any] = {}
    if "federated" in cfg.arms:
        from fedastro.fl.server import run_federated

        centralized_final = result["arms"]["centralized"]["final"] if "centralized" in cfg.arms else None
        result["arms"]["federated"] = run_federated(fed, cfg, seed, device, centralized_final, capture)
    if keep:
        capture["models"] = models
        capture["device"] = device
    return capture


def privacy_report(cfg: ExperimentConfig, fed: Federation, seed: int, result: dict[str, Any],
                   capture: dict[str, Any]) -> dict[str, Any]:
    """Collect per-model DP accounting from the arms, then run the attack if configured."""
    from fedastro.privacy.dp import privacy_section

    arms = result["arms"]
    accounting = {}
    if "privacy" in arms["centralized"]:
        accounting["centralized"] = arms["centralized"]["privacy"]
    if arms["local_only"]["per_silo"] and cfg.privacy.enabled:
        accounting["local_only"] = {k: v["privacy"] for k, v in arms["local_only"]["per_silo"].items()}
    if "privacy" in arms["federated"]:
        accounting["federated"] = arms["federated"]["privacy"]
    if "privacy" in arms.get("ensemble_of_locals", {}) and "local_only" not in accounting:
        accounting["ensemble_of_locals"] = arms["ensemble_of_locals"]["privacy"]  # its own local models
    section = privacy_section(cfg.privacy, accounting)
    if cfg.privacy.attack.enabled:
        from fedastro.baselines.common import init_model
        from fedastro.privacy.attack import run_attack

        names = [x.name for x in cfg.topology.silos
                 if not cfg.privacy.attack.silos or x.name in cfg.privacy.attack.silos]
        first = fed.silos[names[0]].data
        model = init_model(cfg, first.X.shape[1:], len(first.class_names), seed=0)  # weights replaced
        silos = {n: (capture["scaler"].transform(fed.silos[n].data.X), fed.silos[n].data.y) for n in names}
        section["attack"] = run_attack(cfg, seed, silos, capture["global_weights"], model)
    return section


def run_experiment(cfg: ExperimentConfig, seed: int, draw: int = 0) -> dict[str, Any]:
    bundle, fed = build_federation(cfg, seed, draw)
    silos, holdout = describe_federation(fed)

    result = empty_result()
    run_cfg = cfg.experiment_dict()
    run_cfg["seed"] = seed
    result.update(
        run_id=make_run_id(cfg, seed, draw),
        git_sha=git_sha(),
        timestamp_utc=timestamp_utc(),
        config=run_cfg,
        seed=seed,
        provenance=[bundle.provenance.to_dict()],
        silos=silos,
        holdout=holdout,
    )
    if fed.groups is not None:
        result["groups"] = fed.groups
    capture = run_arms(cfg, fed, seed, result)
    if "comms" in capture:
        result["comms"] = capture["comms"]
    result["privacy"] = privacy_report(cfg, fed, seed, result, capture)
    from fedastro.metrics.fairness import fairness_report

    result["fairness"] = fairness_report(result, list(bundle.class_names))
    # Phase 12 evaluation block: only with holdout redraws or held-out groups.
    group_eval = bool(fed.groups and fed.groups["holdout_groups"])
    if cfg.evaluation.holdout_draws > 1 or group_eval:
        result["evaluation"] = {"holdout_draw": draw, "holdout_draws": cfg.evaluation.holdout_draws,
                                "per_holdout_group": {}, "held_out_individuals": None}
    if group_eval:
        from fedastro.evaluation import arm_probabilities, holdout_group_report

        probas = arm_probabilities(cfg, fed, capture["models"], capture, capture["device"])
        result["evaluation"].update(holdout_group_report(fed, probas))
        if "ensemble_of_locals" in result["arms"]:
            result["arms"]["ensemble_of_locals"]["per_holdout_group"] = {
                g: row["ensemble_of_locals"] for g, row in result["evaluation"]["per_holdout_group"].items()}
    notes = []
    if bundle.provenance.kind == SYNTHETIC:
        notes.append("PROVENANCE: SYNTHETIC. Smoke-test data, not physiological measurements.")
    notes.append(f"Arms run: {', '.join(cfg.arms)}.")
    if "ensemble_of_locals" in cfg.arms:
        notes.append("ensemble_of_locals: the local-only models' predicted probabilities averaged "
                     "(unweighted) at test time; no weights exchanged.")
    if cfg.evaluation.holdout_draws > 1:
        notes.append(f"Held-out subject draw {draw} of {cfg.evaluation.holdout_draws}.")
    if fed.groups is not None:
        if fed.groups["holdout_groups"]:
            notes.append(f"Held-out groups (no training samples, no scaler statistics): "
                         f"{', '.join(fed.groups['holdout_groups'])}.")
        if fed.groups["unused_groups"]:
            notes.append(f"Unused groups (in no silo and not held out): {', '.join(fed.groups['unused_groups'])}.")
    if "federated" in cfg.arms and not cfg.comms.enabled:
        notes.append("Federated arm: synchronous lossless links; sim_wallclock_s is null (comms off).")
    if "federated" in cfg.arms and cfg.comms.enabled:
        notes.append(f"Federated arm: simulated links, preset {cfg.comms.preset}, loss of signal "
                     f"{'on' if cfg.comms.los else 'off'}; sim_wallclock_s is SIMULATED time, not real time.")
        stalled = result["arms"]["federated"].get("stalled")
        if stalled:
            notes.append(f"Federated arm STALLED at round {stalled['round']}: {stalled['reason']}.")
    if cfg.privacy.enabled:
        notes.append(f"DP-SGD on every trained model: record-level (epsilon, delta) with target "
                     f"epsilon {cfg.privacy.target_epsilon}; not subject-level, and the shared "
                     f"standardisation statistics are not covered.")
    if cfg.notes:
        notes.append(cfg.notes)
    result["notes"] = " ".join(notes)
    return result


def run_and_write(cfg: ExperimentConfig, seed: int, draw: int = 0) -> Path:
    t0 = time.perf_counter()
    result = run_experiment(cfg, seed, draw)
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
    if fed.groups is not None:
        lines.append("groups: " + "; ".join(f"{k}={','.join(v)}" for k, v in fed.groups["silo_to_groups"].items())
                     + f"; HOLDOUT={','.join(fed.holdout.groups)}"
                     + (f"; unused={','.join(fed.groups['unused_groups'])}" if fed.groups["unused_groups"] else ""))
    return "\n".join(lines)
