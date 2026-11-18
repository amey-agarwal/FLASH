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
from fedastro.data.partition import Federation, HierFederation, make_federation, make_hier_federation
from fedastro.manifest import git_sha, make_run_id, timestamp_utc, write_run
from fedastro.metrics.schema import empty_result
from fedastro.peek import PROGRESS_FILE
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


def build_hier_federation(cfg: ExperimentConfig, seed: int) -> tuple[DatasetBundle, HierFederation, dict[str, Any]]:
    """Per-node train / validation / test data for a hierarchical topology, with the
    SIMULATED location shift applied when configured. Returns (unshifted bundle, federation,
    the result's ``shift`` section)."""
    from fedastro.data.shift import apply_shift

    if cfg.hierarchy is None:
        raise ValueError("build_hier_federation needs a config with a hierarchy")
    rng = seed_everything(seed)
    bundle = load_dataset(cfg, seed)
    fed = make_hier_federation(bundle, cfg.hierarchy, rng, cfg.partition.holdout_subjects,
                               cfg.partition.holdout_seed)
    fed, shift = apply_shift(fed, cfg.hierarchy, cfg.shift)
    return bundle, fed, shift


def topology_section(cfg: ExperimentConfig) -> dict[str, Any]:
    """The result's ``topology`` block. Names are scenario labels, not deployments."""
    h = cfg.hierarchy
    if h is None:
        return {"name": cfg.topology.name, "regions": [], "aggregators": [], "min_participants": None}
    return {
        "name": h.name,
        "regions": [{"name": r.name, "aggregator": r.aggregator,
                     "nodes": [{"name": n.name, "agency": n.agency, "sources": list(n.sources)} for n in r.nodes]}
                    for r in h.regions],
        "aggregators": [r.aggregator for r in h.regions] + [h.global_aggregator.name],
        "min_participants": h.min_participants,
    }


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


def run_arms(cfg: ExperimentConfig, fed: Federation, seed: int, result: dict[str, Any],
             keep_models: bool = False) -> dict[str, Any]:
    """Train the configured arms into ``result["arms"]``. Returns what the federated
    arm captured (global weights, scaler) for the privacy attack, or {}; with held-out
    groups, the ensemble arm or ``keep_models``, ``capture["models"]`` also holds the
    trained models (keeping them changes no result)."""
    from fedastro.baselines.centralized import run_centralized
    from fedastro.baselines.common import resolve_device
    from fedastro.baselines.local_only import run_local_only

    device = resolve_device(cfg.device)
    # Keep trained models only when something scores them after training (Phase 12).
    group_eval = bool(fed.groups and fed.groups["holdout_groups"])
    keep = group_eval or keep_models or "ensemble_of_locals" in cfg.arms
    models: dict[str, Any] = {}
    cen_capture: dict[str, Any] | None = {} if group_eval or keep_models else None
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


def flat_federation(hfed: HierFederation, cfg: ExperimentConfig) -> Federation:
    """The flat arms' view of a hierarchical topology: one silo per node (its training
    data) and the pooled test sets of all regions as the holdout."""
    from fedastro.data.partition import SiloSplit
    from fedastro.fl.hierarchy import pooled_test_set

    return Federation(holdout=pooled_test_set(hfed, cfg.hierarchy),
                      silos={n: SiloSplit(s.train) for n, s in hfed.nodes.items()})


def run_hier_experiment(cfg: ExperimentConfig, seed: int, on_round=None) -> dict[str, Any]:
    """A run on a hierarchical topology: the flat arms over the nodes (scored on the pooled
    test sets), then the hierarchical arm, then every model scored on each node's own test
    set (``result.hierarchy``) and the disagreement analysis."""
    bundle, hfed, shift = build_hier_federation(cfg, seed)
    fed = flat_federation(hfed, cfg)
    silos, holdout = describe_federation(fed)
    for entry in silos:
        split = hfed.nodes[entry["name"]]
        entry.update(region=split.region, n_val=len(split.val), n_test=len(hfed.test_of(entry["name"])),
                     val_label_counts=split.val.label_counts())
    holdout["note"] = "flat arms: every region's test data pooled"
    result = empty_result()
    run_cfg = cfg.experiment_dict()
    run_cfg["seed"] = seed
    result.update(run_id=make_run_id(cfg, seed), git_sha=git_sha(), timestamp_utc=timestamp_utc(),
                  config=run_cfg, seed=seed, provenance=[bundle.provenance.to_dict()], silos=silos,
                  holdout=holdout, topology=topology_section(cfg), shift=shift)
    hier = "hierarchical" in cfg.arms
    capture = run_arms(cfg, fed, seed, result, keep_models=hier)
    if "comms" in capture:
        result["comms"] = capture["comms"]
    result["privacy"] = privacy_report(cfg, fed, seed, result, capture)
    from fedastro.metrics.fairness import fairness_report

    result["fairness"] = fairness_report(result, list(bundle.class_names))
    if hier:
        from fedastro.fl.hierarchy import (arm_summary, comms_section, evaluate_hierarchy, run_hierarchical,
                                           secagg_section)

        device = capture["device"]
        local = {name: (model, scaler) for name, model, scaler in capture["models"]["local"]}
        run = run_hierarchical(hfed, cfg, seed, device, local_models=local, on_round=on_round)
        extra = {}
        if "federated" in cfg.arms:
            extra["flat_federated"] = (capture["global_weights"]["final"], capture["scaler"])
        if "centralized" in capture["models"]:
            extra["centralized"] = capture["models"]["centralized"]
        metrics = evaluate_hierarchy(hfed, cfg, run, local, device, extra)
        result["arms"]["hierarchical"] = arm_summary(cfg, run, metrics)
        result["hierarchy"] = {"rounds": run["rounds"], "events": run["events"],
                               "per_node": metrics["per_node"], "per_region": metrics["per_region"],
                               "global": metrics["global"], "comms": comms_section(cfg, run)}
        result["disagreement"] = metrics["disagreement"]
        from fedastro.metrics.calibration import calibration_section
        from fedastro.metrics.model_report import hier_report

        result["calibration"] = calibration_section(metrics["pooled_proba"], cfg.calibration.n_bins,
                                                    "each node's own test set, pooled")
        result["model_report"] = hier_report(cfg.model.arch, run["n_params"], run["payload_bytes"], metrics,
                                             result["calibration"])
        if (sa := secagg_section(cfg, run)) is not None:
            result["secagg"] = sa
        if run["gate"] is not None:
            result["updates"] = run["gate"].section()
            result["arms"]["hierarchical"].update(n_updates_offered=result["updates"]["n_offered"],
                                                  n_updates_rejected=result["updates"]["n_rejected"])
        if "hierarchical_stream" in cfg.arms:
            result["stream"] = run_stream(cfg, hfed, seed, device, metrics)
    notes = []
    if bundle.provenance.kind == SYNTHETIC:
        notes.append("PROVENANCE: SYNTHETIC. Smoke-test data, not physiological measurements.")
    notes.append(f"Hierarchical topology {cfg.hierarchy.name} (region, agency and node names are scenario "
                 f"labels). Arms run: {', '.join(cfg.arms)}. Flat arms are scored on every region's test "
                 f"data pooled; result.hierarchy scores each model on each node's own test set.")
    if shift["enabled"]:
        notes.append(f"SIMULATED location shift ({shift['kind']}, magnitude {shift['magnitude']}): "
                     f"injected, not measured.")
    if hier:
        notes.append(f"Hierarchical arm: regional FedAvg every round when >= {cfg.hierarchy.min_participants} "
                     f"nodes report (else fallback to the latest global model), global merge every "
                     f"{cfg.hier_fl.global_every} rounds. Disagreement is a flag, not an independent opinion.")
        if cfg.secagg.enabled:
            notes.append("SIMULATED secure aggregation (seeded-PRG pairwise masks, not cryptographically "
                         "secure; no formal privacy guarantee).")
        if cfg.updates.enabled:
            notes.append(f"Update gate: a node deploys a new regional or global version only if its validation "
                         f"macro-F1 is >= the deployed version's minus {cfg.updates.k:g} x its bootstrap SE "
                         f"(version 0 = the local-only model); training is unchanged. "
                         f"hierarchy.per_node.deployed_* score the deployed versions.")
            if cfg.updates.inject:
                notes.append("Injected degraded versions are SCRIPTED faults for testing the gate.")
        if cfg.comms.enabled:
            notes.append("sim_wallclock_s is SIMULATED time, not real time.")
    if "hierarchical_stream" in cfg.arms:
        st = result["stream"]
        notes.append(f"Test-then-train (result.stream): each node's training split streamed in "
                     f"{st['order']} order ({st['order_provenance']}); {st['warmup_frac']:g} warm-up block "
                     f"labelled; {st['label_fraction']:g} of arrivals labelled, usable "
                     f"{st['label_delay_rounds']} round(s) after arrival; every example scored on arrival "
                     f"before any model trained on it. Fixed-split results are unchanged.")
    if cfg.notes:
        notes.append(cfg.notes)
    result["notes"] = " ".join(notes)
    return result


def order_provenance(order: str, kind: str) -> str:
    if order == "patient_shuffle":
        return "SIMULATED arrival order (seeded patient shuffle)"
    if kind == SYNTHETIC:
        return "index order (SYNTHETIC data has no real time structure)"
    return "recording time order"


def run_stream(cfg: ExperimentConfig, hfed: HierFederation, seed: int, device: str,
               fixed: dict[str, Any]) -> dict[str, Any]:
    """The hierarchical_stream arm (Phase H7): the hierarchy again, test-then-train on each
    node's streamed training split. ``fixed`` is the hierarchical arm's evaluation (the
    fixed-split comparison). Returns ``result.stream``."""
    from fedastro.data.stream import make_streams, resolve_order
    from fedastro.fl.hierarchy import evaluate_hierarchy, run_hierarchical
    from fedastro.metrics.stream import stream_report

    h = cfg.hierarchy
    names = [n.name for n in h.nodes]
    region_of = {n.name: r.name for r in h.regions for n in r.nodes}
    streams = make_streams(cfg, {n: hfed.nodes[n].train for n in names}, seed)
    run = run_hierarchical(hfed, cfg, seed, device, streams=streams)
    st = run["stream"]
    audit = st.audit()
    if not audit["ok"]:
        raise RuntimeError(f"test-then-train leakage check failed: {audit}")
    # Final stream models on the fixed test sets. Deployed versions are reported on the stream
    # only (their version 0 is the warm-up local model, not the final one).
    final = evaluate_hierarchy(hfed, cfg, {**run, "gate": None}, st.final_local_models(), device)
    order = resolve_order(cfg)
    three = ("local", "regional", "global")
    section = {
        "enabled": True,
        "label_fraction": cfg.stream.label_fraction,
        "label_delay_rounds": cfg.stream.label_delay_rounds,
        "warmup_frac": cfg.stream.warmup_frac,
        "order": order,
        "order_provenance": order_provenance(order, hfed.nodes[names[0]].train.provenance.kind),
        "schedule": {n: streams[n].summary() for n in names},
        "audit": audit,
        "test_then_train": stream_report(st.records, region_of, cfg.fl.rounds),
        "fixed_test": {"per_region": final["per_region"], "global": final["global"],
                       "scope": "the final stream-trained models on each node's own test set (pooled)"},
        "comparison": {"fixed_split": {m: fixed["global"][m] for m in three},
                       "stream_final": {m: final["global"][m] for m in three},
                       "note": "both on each node's own test set, pooled; fixed_split = the hierarchical arm"},
        "hierarchy": {"n_merges": run["n_merges"],
                      "n_fallbacks": sum(e["kind"] == "fallback" for e in run["events"]),
                      "events": run["events"],
                      "rounds": [{"round": r["round"], "sim_wallclock_s": r["sim_wallclock_s"],
                                  "global_aggregated": r["global_aggregated"],
                                  "fallback": [k for k, v in r["regions"].items() if v["fallback"]]}
                                 for r in run["rounds"]]},
        "updates": run["gate"].section() if run["gate"] is not None else None,
        "note": ("Every streamed example is predicted once, on arrival, before any model has trained on it "
                 "(audit). A model a node has not received yet scores nothing (n_scored). Fixed-split "
                 "results are elsewhere in this file and unchanged."),
    }
    return section


def flat_calibration(cfg: ExperimentConfig, fed: Federation, result: dict[str, Any], capture: dict[str, Any]) -> None:
    """Phase H6, flat runs with calibration.enabled: ECE per arm on the held-out set
    (local_only: the mean of the silos' ECE, as the arm's other metrics) and the model report."""
    from fedastro.baselines.common import init_model
    from fedastro.evaluation import arm_probabilities
    from fedastro.fl.client import get_weights
    from fedastro.metrics.calibration import calibration_section, ece
    from fedastro.metrics.comms import payload_bytes
    from fedastro.metrics.model_report import flat_report
    from fedastro.models.model import count_parameters

    y = fed.holdout.y
    probas = arm_probabilities(cfg, fed, capture["models"], capture, capture["device"])
    nb = cfg.calibration.n_bins
    section = calibration_section({a: (y, p) for a, p in probas.items() if a != "local_only"}, nb,
                                  "shared held-out set")
    if "local_only" in probas:
        per = {s: ece(y, p, nb) for s, p in probas["local_only"].items()}
        section["ece"]["local_only"] = float(sum(per.values()) / len(per))
        section["ece_local_only_per_silo"] = per
    result["calibration"] = section
    model = init_model(cfg, fed.holdout.X.shape[1:], len(fed.holdout.class_names), seed=0)
    result["model_report"] = flat_report(cfg.model.arch, count_parameters(model), payload_bytes(get_weights(model)),
                                         result["arms"], tuple(a for a in cfg.arms if a in probas), section)


def run_experiment(cfg: ExperimentConfig, seed: int, draw: int = 0, on_round=None) -> dict[str, Any]:
    """``on_round(rec, update_events)``: called after each hierarchical round (progress file)."""
    if cfg.hierarchy is not None:
        return run_hier_experiment(cfg, seed, on_round)
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
        topology=topology_section(cfg),
    )
    if fed.groups is not None:
        result["groups"] = fed.groups
    capture = run_arms(cfg, fed, seed, result, keep_models=cfg.calibration.enabled)
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
    if cfg.calibration.enabled:
        flat_calibration(cfg, fed, result, capture)
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




class ProgressLog:
    """``<run_dir>/progress.jsonl`` for hierarchical runs: one JSON line when the run starts,
    one per completed round, one when it ends (``done`` or ``failed``). It lets
    ``fedastro peek --follow`` watch a run in progress. Written only; nothing reads it back
    into the run, so it cannot change results. Rule-4 deviation (Phase H5 decision 6): during
    a run the terminal view reads this file, not result.json."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")  # a re-run starts a fresh log

    def write(self, record: dict[str, Any]) -> None:
        import json

        with self.path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def on_round(self, rec: dict[str, Any], updates: list[dict[str, Any]]) -> None:
        self.write({"kind": "round", "round": rec, "updates": updates})


def run_and_write(cfg: ExperimentConfig, seed: int, draw: int = 0) -> Path:
    t0 = time.perf_counter()
    log = None
    if cfg.hierarchy is not None and "hierarchical" in cfg.arms:
        run_id = make_run_id(cfg, seed, draw)
        log = ProgressLog(Path(cfg.output_dir) / run_id / PROGRESS_FILE)
        log.write({"kind": "start", "run_id": run_id, "rounds": cfg.fl.rounds, "arms": list(cfg.arms),
                   "comms": cfg.comms.enabled, "global_every": cfg.hier_fl.global_every,
                   "regions": {r.name: [n.name for n in r.nodes] for r in cfg.hierarchy.regions},
                   "data": cfg.data.name})
    try:
        result = run_experiment(cfg, seed, draw, on_round=log.on_round if log else None)
        run_dir = write_run(result, cfg.output_dir, time.perf_counter() - t0)
    except BaseException as e:
        if log is not None:
            log.write({"kind": "failed", "error": f"{type(e).__name__}: {e}"})
        raise
    if log is not None:
        log.write({"kind": "done"})
    return run_dir


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


def format_node_table(fed: HierFederation, class_names: tuple[str, ...]) -> str:
    """Plain-text per-node table for a hierarchical topology: patients and windows per
    split, and the train and test label mix (counts and row percentages)."""
    def mix(data: DatasetBundle) -> str:
        counts = data.label_counts()
        n = max(len(data), 1)
        return "".join(f"{counts[c]:>7} ({100 * counts[c] / n:5.1f}%)" for c in class_names)

    def labels(tag: str) -> str:
        return "".join(f"{f'{tag}:{c}'[:16]:>17}" for c in class_names)

    head = (f"{'node':<16}{'region':<8}{'pat':>5}{'n_train':>9}{'n_val':>7}{'n_test':>8}{'gap':>5}"
            + labels("train") + labels("test"))
    lines = [head, "-" * len(head)]
    for name, split in fed.nodes.items():
        test = fed.test_of(name)
        n_pat = len({*split.train.subjects, *split.val.subjects, *(split.test.subjects if split.test else [])})
        shared = "" if split.test is not None else "*"
        lines.append(f"{name:<16}{split.region:<8}{n_pat:>5}{len(split.train):>9}{len(split.val):>7}"
                     f"{f'{len(test)}{shared}':>8}{split.n_gap_dropped:>5}{mix(split.train)}{mix(test)}")
    if fed.holdout is not None:
        lines.append(f"* shared held-out test set: subjects {fed.holdout.subjects}; validation is the "
                     f"time-ordered tail of each node's own subject")
    if fed.unused_sources:
        lines.append(f"unused sources (in no node): {', '.join(fed.unused_sources)}")
    return "\n".join(lines)
