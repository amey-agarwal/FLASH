# PREREG — pre-registration for the `hier-fl` branch

**Status: IN PROGRESS. Frozen so far: §1 hypotheses (wording from CLAUDE.md), §3 location shift,
§4 update-gate tolerance, §5 gap-closing grid and selection rule (on the H8 commit).
The other sections are still placeholders.**
Each section is filled in and frozen (committed) by the phase named in it, **before** any
result-producing run that it covers (CLAUDE.md rule 9 and §7). After a section is frozen, any
change to it is a deviation and is listed in the Deviations section and in `PROGRESS.md`.

Scope notes that apply to everything here (CLAUDE.md §2):
- "Sparse" means sparse **data** (small, siloed astronaut populations), not sparse model updates.
- Every dataset is a terrestrial or animal **proxy** for astronaut data. Silo, region and agency
  names are scenario labels.
- The location shift is **simulated**. Secure aggregation is a **protocol simulation**.
- If federated or hierarchical FL loses to a baseline, that is a result and is reported.

The earlier pre-registration `prereg/p12-wesad-redraws.md` (previous plan) was never run and was
deleted on 2026-11-16 at the user's request; it remains in git history (`881045d`).

## 1. Hypotheses

- **H1 (carried over, already tested on WESAD):** federated training beats each silo's
  local-only model, at a quantifiable communication cost.
- **H2:** under a location-dependent data shift, regional models from hierarchical FL beat
  (a) local-only models and (b) the single global model on their own region's test sets. With
  the shift magnitude set to zero (control), the global model should be at least as good as
  the regional models.
- **H3 (descriptive):** test examples where the local, regional and global models disagree have
  lower accuracy than examples where they agree.
- **H4:** whether the federated advantage in H1 and H2 holds across model families (logistic
  regression, MLP, 1D-CNN).

What is already known (disclosed): H1 on flat five-silo WESAD, 5 seeds (the Phase 3 results in
`PROGRESS.md`, locked by `tests/test_regression_flat.py`).

## 2. Metrics and decision rules

PLACEHOLDER_ — frozen before the first result-producing run (Phase H9 at the latest).
- Primary metric per hypothesis, and the rule for supported / not supported / inconclusive.
- Reported alongside accuracy in every table: macro-F1, AUC, ECE, worst-node accuracy.
- Statistics: at least 5 seeds, mean ± std, hierarchical bootstrap 95% CI
  (`metrics/aggregate.py`).

## 3. Location shift (Phase H2) — FROZEN 2026-11-17

Chosen by the user at the H2 gate (design B). **The shift is SIMULATED:** no dataset here
contains location-dependent physiology, and the shift directions are arbitrary seeded vectors,
not modelled physiology. Every result that uses it carries `shift.provenance: SIMULATED`.

- **Design** (`src/fedastro/data/shift.py`): per region, on the selected features,
  `x' = mean + (x - mean) * gain + offset + class_offset[y]`.
  - Covariate part: a region offset and a region gain (`exp(0.2 * a * g)`, g in [-1, 1]).
  - Concept part: a class-conditional offset that sums to zero over classes, so the
    feature-label relationship differs by region.
  - `a` = magnitude x region strength, in units of a feature's pooled unshifted training
    standard deviation.
- **Kinds reported:** `covariate`, `concept`, `both`. **Primary kind: `both`.**
- **Magnitude grid: {0, 0.5, 1, 2}. Primary magnitude: 1.0.** Magnitude 0 is the control
  (data bit-identical to no shift), used for the control clause of H2.
- **Regions:** LEO is the unshifted reference (strength 0); Moon and Mars have strength 1 and
  different directions.
- **Shifted features:** a seeded random half of the features (channels, for sequences).
- **Shift seed: 0**, fixed across run seeds.
- **Test sets:** a region's train, validation and test data get the same transformation. Where
  nodes share one held-out test set (WESAD), each region is tested on its own shifted copy.
- The grid, kinds and primary were fixed before any model was trained on shifted data. They
  were not, and will not be, chosen from validation or test results.

## 4. Update-gate tolerance (Phase H5) — FROZEN 2026-11-17

Chosen by the user at the H5 plan ("approve as recommended"), before any result-producing run
that uses the gate (`src/fedastro/fl/versioning.py`, config block `updates`).

- **What the gate controls:** which model version each node deploys, on two tracks (regional,
  global). Training and aggregation are unchanged, so every H1–H4 result is unaffected.
- **Version 0 of both tracks:** the node's own local-only model.
- **Metric:** macro-F1 on the node's **own validation set** (never test data).
- **Rule:** an incoming version is deployed iff
  `val_after >= val_before - k * SE`, where `val_before` is the deployed version's validation
  macro-F1 and `SE` is the bootstrap standard error of that score: `n_bootstrap = 200` seeded
  resamples (with replacement) of the node's validation set, `std` with `ddof = 1`. **k = 2.**
  `SE` is recomputed each time a new version is deployed. Rejected versions are logged
  (`result.updates.events`) and never deployed.
- **Who receives a version:** a new regional version goes to the nodes whose update was
  aggregated that round; a new global version to the nodes in contact that round in every region
  that received the merge. A fallback round publishes no regional version.
- **Why a noise-based tolerance and not a fixed constant:** validation sets are small (e.g. a few
  tens of windows per node), so no fixed number has support on validation data; `k * SE` scales
  with how noisy each node's score is. k = 2 is a conventional choice, fixed here, not tuned.
- **Known consequence (disclosed before any real run):** when a deployed version scores 1.0 on a
  tiny validation set, its bootstrap SE is 0, so the tolerance is 0 and any imperfect version is
  rejected. This is reported, not patched.
- **Scripted faults** (`updates.inject`) are tests of the gate, labelled SCRIPTED wherever they
  appear; no headline result uses them.

## 4a. Test-then-train stream (Phase H7) — DRAFT, to be frozen before Phase H9

Implemented in `src/fedastro/data/stream.py`, `fl/stream.py`, `metrics/stream.py`
(arm `hierarchical_stream`, config block `stream`). The values below are the implementation
defaults; they are **not frozen yet**. They are frozen (this heading changes to FROZEN) before the
first result-producing stream run, and none of them is chosen from validation or test results.

- **What streams:** each node's training split. Validation (update gate) and test (fixed-split
  results) are unchanged.
- **Arrival order:** `auto` = recording time order for WESAD (one subject per node); a seeded
  patient shuffle (SIMULATED arrival order) for by_source topologies (CinC, synthetic).
- **Warm-up block:** first 10% of each node's stream (`warmup_frac: 0.1`), fully labelled, trained
  on from round 1 (pre-mission data). Never scored.
- **Chunks:** the rest in `fl.rounds - 1` equal chunks, arriving in rounds 2..`fl.rounds`.
- **Label fraction:** 10% (`label_fraction: 0.1`), cumulative rounding over the stream, seeded
  choice within a chunk. **Delay:** 1 round (`label_delay_rounds: 1`).
- **Scoring:** each arriving example is predicted once, at the start of its arrival round, by the
  local model and by the last regional and global models delivered to that node (and the deployed
  versions with the gate on); then it may join training once its label is usable. A model a node
  has not received yet scores nothing (counts reported).
- **Metrics:** test-then-train accuracy, balanced accuracy, macro-F1, AUC per model, pooled over
  the stream per node, region and overall; per-round accuracy; agreement / disagreement on the
  examples all three models scored (H3 on the stream).
- **Planned sensitivity grid (proposal, to confirm when freezing):** `label_fraction` {0.1, 1.0}
  x `label_delay_rounds` {0, 1}; primary = (0.1, 1).
- **Disclosed before any real run:** on WESAD each node holds roughly 45-60 training windows, so at
  10% labels a node trains on a handful of labels; and WESAD conditions arrive in blocks, so
  early rounds see only some classes. Both are reported, not adjusted.

## 5. Gap-closing grid and selection rule (Phase H8) — FROZEN on the Phase H8 commit (2026-11-18)

Chosen by the user at the H8 plan ("approve as recommended"). Frozen before any tuning run: no
run of the sweeps below existed when this was written (only a timing run, whose scores were not
looked at). Code: `src/fedastro/tuning.py`, `metrics/validation.py`, `metrics/gap.py`,
`fl/strategy.ServerOpt`.

- **Selection data:** `result.validation` only: every model on each node's own validation split
  (WESAD: the time-ordered tail of each node subject; CinC: the node's patient-disjoint 10%).
  Never a test set. The selector loads nothing else from a result (tested).
- **Seeds:** tuning seeds **100, 101, 102**. Evaluation seeds **0-4**, never used for tuning.
- **Per model family** (logreg, MLP, 1D-CNN) and **per arm**, independently:
  - **flat federated** (validation model `flat_federated`) and **hierarchical** (validation
    model `global`; regional reported alongside):
    - Stage A (`gapA_<dataset>_<family>`): FedAvg, no server optimizer; `train.lr` {3e-4, 1e-3,
      3e-3, 1e-2} x `fl.local_epochs` {1, 2, 5}. 12 variants. Run with arms local_only +
      federated + hierarchical; each arm's winner is selected separately.
    - Stage B (generated from `templates/gapB.yaml` on top of that arm's stage A winner):
      FedProx mu {0 (= FedAvg), 0.001, 0.01, 0.1} x server optimizer {none; momentum (FedAvgM,
      server lr 1.0, beta 0.9); FedAdam server lr 0.01; FedAdam server lr 0.1} (FedAdam beta1 0.9,
      beta2 0.99, tau 1e-3, Reddi et al. 2021). 16 variants. In the hierarchy the server optimizer
      runs at each regional aggregator (own state); the global merge stays a weighted average.
  - **centralized** (validation model `centralized`) and **local-only** (`local`, each node with
    its own model, pooled) (`gapC_<dataset>_<family>`): `train.lr` {3e-4, 1e-3, 3e-3, 1e-2} x
    `train.epochs` {30, 60}. 8 variants.
  - Fixed throughout: `fl.rounds` 30, `hier_fl.global_every` 5, batch size 32, Adam client
    optimizer, weight decay 1e-4, model architectures as in Phase H6.
- **Score:** mean over the tuning seeds of the pooled validation **macro-F1** of the arm's model.
- **Rule:** the winner is the **cheapest** variant whose score is within **0.005** of the best
  score. Cheaper = fewer local epochs, then FedAvg over FedProx, then no server optimizer, then
  fewer (centralized / local-only) epochs. Remaining ties: higher score, then lower variant index.
  Every candidate's scores and the winner are written to `<sweep>-selection.json`.
- **Evaluation:** each arm's winner, evaluation seeds 0-4, its own run (arms can have different
  learning rates). Gap table (`fedastro gap`): centralized - flat federated, centralized -
  hierarchical (global model), centralized - hierarchical (regional), per family, on accuracy,
  macro-F1 and AUC; paired by seed, mean +/- std, hierarchical bootstrap 95% CI. A negative gap
  is reported as it is.
- **Datasets:** WESAD (wesad12) in Phase H8; CinC (cinc15) in Phase H10 with the same grid and
  rule (sweep files added then).
- **SCAFFOLD:** not run (deferred stretch goal).
- **Known limitation, disclosed now:** the federated arms are selected on the hierarchical
  topology's node validation splits, which are small (WESAD: ~10-15 windows per node), so
  selection noise is real; the tolerance rule prefers cheap settings among near-ties.

## 6. Experiment configurations and seeds

PLACEHOLDER_ — one row per configuration: config file, config hash, topology, dataset, seeds.
- OSD-435 and WESAD (Phase H9).
- CinC 2021 (Phase H10).

## 7. Deviations

None yet.
