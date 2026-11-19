# PREREG — pre-registration for the `hier-fl` branch

**Status: IN PROGRESS. Frozen so far: §1 hypotheses (wording from CLAUDE.md), §3 location shift,
§4 update-gate tolerance, §4a test-then-train stream (on the H9 prep commit), §5 gap-closing grid
and selection rule (on the H8 commit), the §6 OSD-435 rows (on the OSD-435 commit). §2 and the §6
WESAD rows are DRAFTS, frozen after the H8 winners exist and before any WESAD H9 run.**
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

## 2. Metrics and decision rules (Phase H9) — DRAFT, frozen with §6 before any H9 run

Chosen by the user at the H9 plan ("approve as recommended", 2026-11-19). Code:
`src/fedastro/metrics/hypotheses.py`, `fedastro hypotheses configs/hypotheses/h9_<dataset>.yaml`.

- **Comparison:** a paired difference between two arms, per evaluation seed (0-4); each arm comes
  from its own runs at its own validation-selected configuration (§5), paired by seed.
  Reported: mean, std, and the hierarchical percentile bootstrap 95% CI (one value per seed,
  `metrics/aggregate.bootstrap_ci`).
- **Rule:** CI above 0 = **supported**; CI below 0 = **not supported**; CI contains 0 =
  **inconclusive**. No multiple-comparison correction. Disclosed: with 5 seeds the percentile
  bootstrap CI excludes 0 whenever all seeds agree in sign, so this rule is lenient.
- **Primary metric: test accuracy** (continuity with the locked flat H1); macro-F1 reported
  alongside every comparison. Every table also reports AUC, ECE and worst-node accuracy.
- **H1:** federated (final model) minus local-only (mean over nodes), no shift (magnitude 0).
- **H2:** at the primary shift (`both`, magnitude 1.0; §3), on the shifted regions (Moon, Mars)
  pooled, n_test-weighted: (a) regional minus local-only and (b) regional minus global, regional
  and global from the hierarchical arm's runs, local-only from its own runs. Supported iff (a)
  and (b) are both supported; not supported if either is not supported; else inconclusive.
  **Control clause** (magnitude 0, every region pooled): supported unless regional minus global
  has its CI above 0.
- **H3 (descriptive):** accuracy where local, regional and global agree minus where they disagree
  (`result.disagreement`, headline model regional), the hierarchical arm's runs at the primary
  shift. The three models in one run share one learning rate (the hierarchical arm's) and share
  training data, so agreement is correlated: a flag, not independent confirmation.
- **H4:** the H1 and H2 verdicts per family (logreg, MLP, 1D-CNN): "holds in every family" iff all
  three are supported; "does not hold" if any is not supported; else "mixed / inconclusive".
- **Trained, not deployed, models** answer H1-H4. The update gate's deployed versions are reported
  alongside, in the full-architecture runs only (§6), as the in-mission scenario.
- Other shift settings (§6) are reported descriptively with the same comparisons; they do not
  change the verdicts.

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

## 4a. Test-then-train stream (Phase H7) — FROZEN on the H9 prep commit (2026-11-19)

Implemented in `src/fedastro/data/stream.py`, `fl/stream.py`, `metrics/stream.py`
(arm `hierarchical_stream`, config block `stream`). The values below are the implementation
defaults, confirmed by the user at the H9 plan ("approve as recommended") and frozen before the
first result-producing stream run. None of them was chosen from validation or test results.

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
- **Sensitivity grid:** `label_fraction` {0.1, 1.0} x `label_delay_rounds` {0, 1}; primary =
  (0.1, 1). Run at the primary shift, links off (§6, `h9_<dataset>_<family>_stream`).
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

## 6. Experiment configurations and seeds (Phase H9) — DRAFT, frozen before any H9 run

Chosen by the user at the H9 plan ("approve as recommended", 2026-11-19). The sweep files are
**generated** by `fedastro h9-sweeps` from the Phase H8 evaluation sweeps (each arm's validation-
selected winner, §5), committed to `configs/sweeps/h9_<dataset>_*.yaml`, and listed here with
their config hashes when this section is frozen. Hyperparameters: `PLACEHOLDER_` until H8 ends.

**WESAD (wesad12), per family (logreg, MLP, 1D-CNN), evaluation seeds 0-4:**

| Sweep | Arms | Settings | Answers |
|---|---|---|---|
| `h9_wesad12_<family>_centralized` | centralized | shift grid | reference |
| `h9_wesad12_<family>_local_only` | local_only | shift grid | H1, H2 |
| `h9_wesad12_<family>_federated` | federated | shift grid | H1 |
| `h9_wesad12_<family>_hierarchical` | local_only + hierarchical | shift grid; simulated secure aggregation on | H2, H3 |
| `h9_wesad12_<family>_full` | local_only + hierarchical + hierarchical_stream | primary shift; links on (Mars `mean` preset, loss of signal, region deadline 120 simulated s), secure aggregation, update gate, test-then-train (§4a primary) | mission scenario, descriptive only |
| `h9_wesad12_<family>_stream` | local_only + hierarchical + hierarchical_stream | primary shift; links off; §4a sensitivity grid | test-then-train sensitivity |

- **Shift grid** (§3): `both` at magnitudes {0, 0.5, 1, 2}, plus `covariate` and `concept` at 1.0.
  Magnitude 0 is the control (no shift). Primary: `both` 1.0. SIMULATED.
- **Links off in the H1-H4 runs:** loss of signal changes which nodes take part, which would mix
  participation with the statistical comparison. The full-architecture runs report it.
- **Fixed** (as §5): `fl.rounds` 30, `hier_fl.global_every` 5, batch size 32, Adam, weight
  decay 1e-4.
- **Reference:** the locked flat five-silo WESAD results (§1), not re-run.
- Compute estimate (from the H8 timings, not measured): ~10-14 h sequential CPU; ~1-2 h wall as
  PACE array jobs.

**OSD-435 (flat, two silos) — FROZEN on the OSD-435 commit, approved 2026-11-19 ("approve as recommended"):** the FLUID setup
as its paper reports it. Code `src/fedastro/data/osdr.py`; config
`configs/experiments/osd435_fluid.yaml`; sweep `configs/sweeps/h9_osd435.yaml`; seeds 0-4.
- **Data:** `LSDS-22_Echocardiography_aorta_SUBMITTED.csv`, SHA-256
  `df3286f1660c7adb5b33322f8c153767567524b714b33a86bcf010baa06cc4d0` (pinned). A ground-based mouse
  space-radiation study (NASA Space Radiation Laboratory): a terrestrial analog, not spaceflight.
- **Label:** 0 Gy (sham) vs >= 0.5 Gy (irradiated). **Rows:** 0.05-0.25 Gy removed, rows with a
  missing feature removed: 809 rows, 280 mice, 65% irradiated.
- **Features (main):** radiation type (one-hot), month post-irradiation, heart rate, aorta peak
  velocity (`VELOCITY`, mm/s), VTI, mean velocity. **Sensitivity:** no radiation type and no
  month-0 rows (572 rows, 215 mice). Disclosed before any run: predicting the majority class per
  radiation type is right on 76% of the main rows (always "irradiated": 65%), so part of the main
  task's accuracy reflects the study design.
- **Split:** 30% of mice held out as the shared test set (84 / 280; sensitivity 65 / 215), fixed
  holdout seed; no mouse in two splits (FLUID's split method is not known to us; theirs may have
  been by row). No validation split: default settings, no tuning.
- **Silos:** `earth`, `iss` (scenario labels, `configs/silos/fluid_two.yaml`). (a) FLUID: whole
  mice, 50 / 50, IID. (b) non-IID: Dirichlet label skew, **alpha 5.0** (deviation 1, §7).
- **Models / arms:** logreg and MLP (FLUID's CRISP v2.0 is not available: this reproduces FLUID's
  data setup, not its model); centralized, local-only, federated (FedAvg); training defaults.
- **Reported alongside:** the radiation-type-only majority baseline.
- Compute: ~42 s per run on the laptop (one timing run, scores not read); 40 runs, ~30 min.

**CinC 2021:** Phase H10.

## 7. Deviations

1. **OSD-435 non-IID split, alpha 0.5 -> 5.0** (2026-11-19, before any OSD-435 result-producing
   run; user: "approve as recommended"). With two silos, alpha 0.5 left both silos without sham
   rows on seed 0 and one silo on seed 4; alpha 2.0 still left a silo without sham rows. Alpha
   5.0 is the smallest value checked (0.5, 1, 2, 5) with no single-class silo on seeds 0-4. Chosen
   from per-silo label counts only, never from a score. Counts in PROGRESS.md (Phase H9).
