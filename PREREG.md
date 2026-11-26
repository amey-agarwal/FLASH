# PREREG — pre-registration for the `hier-fl` branch

**Status: IN PROGRESS. Frozen so far: §1 hypotheses (wording from CLAUDE.md), §3 location shift,
§4 update-gate tolerance, §4a test-then-train stream (on the H9 prep commit), §5 gap-closing grid
and selection rule (on the H8 commit), the §6 OSD-435 rows (on the OSD-435 commit), §2 and the §6
WESAD rows (on the H9 WESAD sweeps commit, after the H8 winners existed and before any WESAD H9
run), the §5a CinC rows (on the H10c step 2 commit, before any CinC tuning run), the §6 CinC
rows (on the H10c step 3 commit, after the CinC winners existed and before any H9-style CinC
run). Every section is now frozen.**
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

## 2. Metrics and decision rules (Phase H9) — FROZEN with §6 on the H9 WESAD sweeps commit (2026-11-21), before any H9 run

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
- **Observed after selection (2026-11-20; an observation, not a deviation):** the WESAD winners
  sit on the grid edge: the top learning rate (0.01) for 5 of 6 federated / hierarchical arms, 60
  epochs for 5 of 6 baseline arms, and FedProx mu 0.1 (the largest mu) for all 6 stage B winners.
  Better settings may lie outside the grid; the grid was not extended. Winners and scores:
  PROGRESS.md, Phase H8.

### 5a. CinC 2021 rows (Phase H10c) — FROZEN on the H10c step 2 commit (2026-11-24), before any CinC tuning run

The grid, seeds, score, rule, evaluation and gap table above apply **unchanged**. Only the data
differs. No run of these sweeps existed when this was written. The only earlier CinC runs on
the 20k set were the step 1 timing run (`runs/h10t-e51b0ad`); its scores were not looked at.

- **Data:** PhysioNet/CinC Challenge 2021 v1.0.3, the fixed patient-complete **~20k main
  subset**: 20,001 records, 19,240 patients, index `sha256:b57091b794cd08d988a499405f36221b48e93899f193749195e6d32758bb90f4`
  (identical on the laptop and PACE). Target: binary normal sinus rhythm vs any scored
  abnormality (scored labels only). Leads I and II, 100 Hz, 10 s windows. INCART: 10 segments
  per record, kept with the patient. Topology `cinc15` (15 nodes, `min_participants` 5), 70/10/20
  patient-disjoint node splits.
- **Compute choice (Gate 1, user: option A):** tune and evaluate on the same 20k subset, 30
  rounds, CPU. It was chosen from run times only (step 1: MLP 145 s, 1D-CNN 527 s per stage-A run).
- **Sweeps** (base `configs/experiments/cinc15_main.yaml`, sha256 `feda69f44f3f6934efbae615049e1e43635317cfd5f46178b88ba76b358f0da9`). The 1D-CNN
  uses `representation: sequence`; logreg and MLP use the 17 features.

  | Sweep | Variants | sha256 |
  |---|---|---|
  | `gapA_cinc15_logreg.yaml` | 12 | `08ef611dfe9c0954fcacbf08a7c4046157a6060fbd0f9c9f591bf6f20e497997` |
  | `gapA_cinc15_mlp.yaml` | 12 | `e6ad0a470b3e685d1c4c5fc7163460cb6b9e91e1f42a553d072c098ccd039a0c` |
  | `gapA_cinc15_cnn1d.yaml` | 12 | `c63de51beaf5ed9527de2d962107a545f226d2b307fff379fd7cc32258ef9075` |
  | `gapC_cinc15_logreg.yaml` | 8 | `9c167a78f29643736f00c7b8fa5f79acb34b100a7ccb6c628568da943efb67f5` |
  | `gapC_cinc15_mlp.yaml` | 8 | `828fab1afac51fe0778f98cafed677f49ffbfcb6bb0f2e0d978749bca80abb92` |
  | `gapC_cinc15_cnn1d.yaml` | 8 | `78c1eee0a23f1fdb2063f3a003a8e8ec06791368eb9132dd0bf9836f28481acb` |

  180 tuning runs (seeds 100–102). The stage B sweeps (16 variants per federated or hierarchical
  arm, 288 runs) and the evaluation sweeps (seeds 0–4) are generated by `fedastro select`, as for
  WESAD.
- **Disclosures, fixed now:**
  - INCART (the Roscosmos analog) is a tiny, single-class LEO silo; it is kept.
  - Five of the six sources have one record per patient, so their patient grouping is trivial.
    PTB-XL, PTB and INCART are grouped by patient id.
  - On the 2k smoke subset, 4 nodes had a single-class validation split. The 20k count is
    taken with `fedastro inspect` before selection and reported in PROGRESS.md. It changes nothing
    here: selection pools validation over nodes, so single-class nodes still count.
  - The same known limitation as WESAD: selection noise from small node validation splits.
    It is expected to be smaller on CinC, where the 20k node validation splits are much larger.

## 6. Experiment configurations and seeds (Phase H9) — WESAD rows FROZEN on the H9 WESAD sweeps commit (2026-11-21), before any H9 run; CinC rows FROZEN on the H10c step 3 commit (2026-11-25)

Chosen by the user at the H9 plan ("approve as recommended", 2026-11-19). The sweep files are
**generated** by `fedastro h9-sweeps` from the Phase H8 evaluation sweeps (each arm's validation-
selected winner, §5), committed to `configs/sweeps/h9_<dataset>_*.yaml`, and listed below with
file hashes; every variant's config hash and run ids are in `tests/golden/run_ids.json`.

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
- **Selected settings (H8, validation only; Adam client optimizer throughout):**

| Family | centralized | local_only | federated | hierarchical (+ its local_only) |
|---|---|---|---|---|
| logreg | lr 0.003, 60 ep | lr 0.01, 60 ep | lr 0.01, E 2, FedProx mu 0.1, FedAdam server lr 0.1 | lr 0.01, E 1, FedProx mu 0.1, server momentum |
| mlp | lr 0.003, 60 ep | lr 0.001, 60 ep | lr 0.01, E 1, FedProx mu 0.1, server momentum | lr 0.01, E 1, FedProx mu 0.1 |
| cnn1d | lr 0.003, 60 ep | lr 0.001, 30 ep | lr 0.01, E 1, FedProx mu 0.1, server momentum | lr 0.003, E 2, FedProx mu 0.1, server momentum |

  The `_full` and `_stream` sweeps use the hierarchical arm's settings. In the hierarchical,
  `_full` and `_stream` runs the local-only model shares the hierarchical arm's learning rate
  (one `train.lr` per run, PROGRESS.md H8 decision 2); H1 and H2 use the `_local_only` sweep's
  own winner.
- **Files (SHA-256, first 12 hex)**, 435 runs in total (per family: 4 x 30 + 5 + 20):

| Family | centralized | local_only | federated | hierarchical | full | stream |
|---|---|---|---|---|---|---|
| logreg | `ad80739b30bc` | `8973b386cac5` | `a6b2f3bcbed2` | `7ccf596e62bd` | `61d85238336e` | `9089141015d1` |
| mlp | `c2ae544501d8` | `30ccc58ea734` | `497d892ab19c` | `af0aecd4027d` | `f3676caca8d9` | `7f4a477f707e` |
| cnn1d | `a954aef7b641` | `ede5cba7262f` | `b58427a7574d` | `0096e6c576fc` | `47821cb0799e` | `1d650657ab06` |

  Hypotheses spec: `configs/hypotheses/h9_wesad12.yaml`.
- Compute estimate (from the H8 stage A/B timings on PACE, not an H9 measurement): ~2-3 min per
  federated or hierarchical run; ~1-2 h wall as 18 parallel PACE jobs (one sweep each).

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

**CinC 2021 (cinc15) — FROZEN on the H10c step 3 commit (2026-11-25), before any H9-style CinC
run.** The same six sweep types, arms, shift grid, links-off rule and fixed settings as the WESAD
rows above. §2 (rules; primary metric test accuracy) applies unchanged. Generated by
`fedastro h9-sweeps runs/h10-f7db96a/generated/eval_*.yaml --dataset cinc15` from the 12 H10c
evaluation sweeps (each arm's validation-selected winner, §5 and §5a). Base
`configs/experiments/cinc15_main.yaml` (sha256 `feda69f44f3f…`, unchanged from §5a); the 20k
subset, topology `cinc15` and node splits of §5a. Evaluation seeds 0–4.

- **Selected settings (H10c, validation only; Adam client optimizer throughout):**

| Family | centralized | local_only | federated | hierarchical (+ its local_only) |
|---|---|---|---|---|
| logreg | lr 0.01, 60 ep | lr 0.01, 30 ep | lr 0.01, E 2, FedProx mu 0.01, server momentum | lr 0.01, E 2, FedProx mu 0.01, server momentum |
| mlp | lr 3e-4, 60 ep | lr 0.003, 30 ep | lr 0.001, E 2, FedAvg | lr 0.003, E 1, FedAvg |
| cnn1d | lr 0.001, 60 ep | lr 0.003, 60 ep | lr 0.01, E 5, FedAvg | lr 0.01, E 5, FedProx mu 0.001 |

  The 1D-CNN uses `representation: sequence` (leads I and II, 100 Hz); logreg and MLP use the 17
  features. As for WESAD, the `_full` and `_stream` sweeps use the hierarchical arm's settings,
  and H1 / H2 use the `_local_only` sweep's own winner.
- **Files (SHA-256, first 12 hex, computed with `shasum -a 256`)**, 435 runs in total (per family:
  4 x 30 + 5 + 20):

| Family | centralized | local_only | federated | hierarchical | full | stream |
|---|---|---|---|---|---|---|
| logreg | `c6d6e65fb2e0` | `39e1172e9b11` | `1bd2f4195edd` | `7a1563e478e4` | `6b151e98c57f` | `7e397052a0bf` |
| mlp | `501bea32ac24` | `21f0e404da31` | `1b8a19f75ea4` | `5621a103a1af` | `081cc746705b` | `c98da4136ea1` |
| cnn1d | `2c67f03f5dfb` | `51802d38104f` | `ed0cf4920ab6` | `243bf09981f0` | `efba5955d0ed` | `cab630178a50` |

  Hypotheses spec: `configs/hypotheses/h9_cinc15.yaml` (`5869f6f19921`). Run ids:
  `tests/golden/run_ids.json`.
- **Disclosures, fixed before any H9-style CinC run:**
  - **Test scores already seen.** Unlike WESAD, the CinC evaluation test scores (seeds 0–4) were
    seen before this freeze: the H10c gap table and a per-class recall check (PROGRESS.md, H10c).
    No setting here was chosen from them. Every setting is the validation winner, carried over
    mechanically by `h9-sweeps`; the rules and primary metric are §2's, frozen on 2026-11-21.
  - **Centralized logreg collapses toward "abnormal"** on the evaluation runs (normal recall
    0.10, AUC close to federated's). It is kept as selected. Class weighting and threshold tuning
    were not in the frozen grid and are not added.
  - **Accuracy stays the primary metric** although about 77% of usable CinC records are abnormal.
    Macro-F1, AUC, balanced accuracy, ECE and worst-node accuracy are reported alongside.
  - **Shift on CinC** (§3 rule, SIMULATED): `feature_fraction` 0.5, rounded up, so a seeded 9 of
    the 17 features (logreg, MLP) and one of the two leads (1D-CNN).
  - **Stream arrival order** on CinC is a seeded patient shuffle (SIMULATED; §4a `auto`).
  - **INCART (`leo_roscosmos`)** is single-class and LEO-only. LEO is the unshifted region, so
    it never enters the H2 regions, but it is in the H1 local-only mean and the control clause.
- **Compute estimate** (from the H10c evaluation runs on PACE, mean / max per run; not an H9
  measurement): 1D-CNN federated 16 / 27 min, hierarchical 14 / 25, local-only 10, centralized
  5; logreg and MLP about 5 min for federated or hierarchical. Stream runs are expected at
  about 1.5× and full runs at about 5–10× a hierarchical run (WESAD H9 ratios). About 55 task-hours;
  87 array tasks (one per variant, 5 seeds each) in two waves under the QOS cap.

## 7. Deviations

1. **OSD-435 non-IID split, alpha 0.5 -> 5.0** (2026-11-19, before any OSD-435 result-producing
   run; user: "approve as recommended"). With two silos, alpha 0.5 left both silos without sham
   rows on seed 0 and one silo on seed 4; alpha 2.0 still left a silo without sham rows. Alpha
   5.0 is the smallest value checked (0.5, 1, 2, 5) with no single-class silo on seeds 0-4. Chosen
   from per-silo label counts only, never from a score. Counts in PROGRESS.md (Phase H9).

2. **Phase H11a scoring audit: corrections, written 2026-11-26 before any rerun.** Found while
   building the H11 figures, after the H9 WESAD and H10 CinC verdicts had been recorded. The
   decision rules in §2 are unchanged. Every item below is disclosed whichever way its numbers
   fall.
   - **D1, H2(a) erratum (implementation bug).**
     - **What was wrong:** §2 defines H2 on the shifted regions' own test sets. The code
       compared the regional model on those sets with local-only models scored on **every
       region's test data pooled** (`baselines/local_only.py` scores on the flat arms' holdout;
       `metrics/hypotheses._local_region_pool` assumed own test data).
     - **Fix:** the local-only arm now also records each node's score on its own test set
       (`arms.local_only.own_test`), and H2(a) reads it. Runs without it are refused.
     - **Reruns:** the `h9_{wesad12,cinc15}_*_local_only` sweeps, with configs and run ids
       unchanged. The H2 verdicts in PROGRESS.md "Phase H9 (part 3)" and "Phase H10 (part d)"
       are superseded by the reruns.
     - **Indicative numbers, seen before this entry:** like-for-like H2(a) is about 0.000 on
       CinC and +0.20 to +0.31 on WESAD (descriptive, hierarchical runs' own local models).
   - **D2, local-only selection scope (disclosed).** Local-only hyperparameters were selected on
     each node's own validation set (§5), but the arm's test score in §2 H1 is on every region's
     test data pooled. The own-test score (D1) is in the selection's scope. The selection is not
     redone.
   - **D3, H1 own-test row (added after seeing indicative numbers).** §2 H1 says "local-only
     (mean over nodes)" without fixing the test scope.
     - **Primary stays as implemented:** every region's pooled test data, unweighted mean over
       nodes. Its verdict is unchanged.
     - **Added alongside:** flat federated (pooled) minus local-only on each node's own test set
       (`own_test.pooled`, n_test-weighted, the same examples), with its own verdict and an
       H4 line.
     - **Indicative numbers, seen before this entry:** own-test H1 on CinC is −0.011 to −0.036
       accuracy against the recorded +0.053 to +0.077; WESAD is unchanged. Both rows are
       reported; neither is hidden.
   - **D4 and D5, descriptive scope fixes:**
     - `fairness.local_only_own_test` is added. The pooled label-mix estimate assumed no
       per-node test data, which does not hold on hierarchical topologies. This was the source
       of the "local-only collapse" note in H10 part d.
     - Results tables gain a local own-test column next to the pooled local-only one.
   - **D6, simulated-clock fix in the full-architecture runs (implementation bug).**
     - **What was wrong:** when a region's secure-aggregation mask group fell below
       `min_participants`, the round ended at the group-formation time instead of the region
       deadline. With every node out of contact the clock stopped and the run livelocked; this
       hit 3 of 15 CinC `_full` runs. With no global deadline, a member that could never answer
       could also join key agreement.
     - **Fix:** both fixed (`fl/hierarchy._plan_secagg`, `_key_agreement`).
     - **Reruns:** all 30 `_full` runs (WESAD and CinC). They are descriptive only and do not
       enter H1–H4.
