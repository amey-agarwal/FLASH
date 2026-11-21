# PROGRESS

Handoff log between sessions. Newest phase first. Assume no memory of prior sessions.

**Current plan (2026-11-15): branch `hier-fl`, the new CLAUDE.md, Phases H0–H11. CLAUDE2.md is
removed; its Phases 13–18 are dropped. Phases H0 (`881045d`), H1 (`f3eac96`), H2 (`9590007`), H3
(`836bad2`), H4 (`b19f28b`), H5 (`e9738f9`, `3d5812a`), H6 (`11906a9`) and H7 (`7c556b0`) are
committed. Phase H8 is committed (`62cf33b`); its tuning (468 runs) and evaluation (60 runs, seeds
0-4) are DONE and the gap table is computed (2026-11-21, H8 "Evaluation"): **these are the numbers
for presentations**; the earlier tuning-seed page is superseded. H9 WESAD (435 runs, `84a40ec`) is
RUNNING on PACE.**
Phase H9 prep (part 1) is built locally (below): no H9 run exists. Next: finish the H8 grid and
gap table; the user picks the OSD-435 task; generate the H9 sweeps and freeze PREREG §2/§6.** Everything below the H0 entry is the record of the earlier plans; where it says
"CLAUDE2.md" or "not committed", read it as history (Phases 10–12 are committed, `4586d64`).

---

## Phase H9 (part 2) — WESAD sweeps generated, PREREG §2/§6 frozen — NOT COMMITTED, NO RUNS (2026-11-21)

Plan approved 2026-11-21 (user approved the plan file). No H9 WESAD run exists; nothing here uses
a test score (the H8 evaluation runs are not on the laptop yet).

### Built
- `fedastro h9-sweeps runs/h8-62cf33b/generated/eval_*.yaml --dataset wesad12` -> 18 sweeps
  `configs/sweeps/h9_wesad12_<family>_{centralized,local_only,federated,hierarchical,full,stream}.yaml`
  (generated, not hand-edited) and `configs/hypotheses/h9_wesad12.yaml`. **435 runs** (per family
  4 x 6 shifts x 5 seeds + full 5 + stream 4 x 5).
- **PREREG:** §2 and the §6 WESAD rows FROZEN (selected settings table, file hashes); §5 gains an
  observation (winners on the grid edge, not extended). Status line updated.
- **Golden lock:** the 18 sweeps' run ids added to `tests/golden/run_ids.json` (additions only,
  1080 lines, no existing entry changed; written by a targeted merge, not `ids --write --force`,
  so unlocked older sweeps stay unlocked).
- `.gitignore`: `PACE-run-evals2/`, `recentPACEruns/` (PACE logs carry the scratch path and user
  name). `PACE-run-evals2/` is still tracked (committed in `e506fcf`): the user runs
  `git rm -r --cached PACE-run-evals2`.
- Git-ignored: `runs/h8-62cf33b/generated/gap_wesad12.yaml` (gap spec; `fedastro gap` loads it and
  reports the 60 evaluation runs missing, as expected). Run with
  `fedastro gap runs/h8-62cf33b/generated/gap_wesad12.yaml --output-dir runs/h8-62cf33b --out runs/h8-62cf33b/gap_wesad12.json`.

### Tests
Full suite (2026-11-21, disk 98% full): 2 failed, the rest passed. Both failures were from the
OSD-435 commit (`2e8327d`, which ran targeted tests only): `test_registry_groups.py` still expected
the registry without `osd435`, and `scripts/run_report.LEGACY_DOMAINS` lacked `OSD-435`. Fixed (test
now probes the unregistered `cinc2021`; mapping added); `test_registry_groups.py` 21 passed. Golden,
hypotheses and OSD-435 tests: 25 passed.

### OSD-435 results (2026-11-21, laptop, commit `7174a1e`, 40/40 runs, REAL, seeds 0-4)
`fedastro grid configs/sweeps/h9_osd435.yaml --report` (saved: `runs/h9_osd435_report.txt`,
`runs/h9_osd435-grid.json`). Test accuracy / macro-F1, mean +/- std over seeds; held-out mice.

| Split | Features | Model | Federated acc | Fed. macro-F1 | Local-only acc | Centralized acc | Fed > local (seeds) |
|---|---|---|---|---|---|---|---|
| FLUID IID | main | logreg | 0.774 ± 0.017 | 0.683 | 0.765 ± 0.035 | 0.802 ± 0.002 | 3/5 |
| FLUID IID | main | mlp | 0.790 ± 0.012 | 0.696 | 0.763 ± 0.018 | 0.772 ± 0.014 | 4/5 |
| non-IID a 5 | main | logreg | 0.692 ± 0.043 | 0.610 | 0.716 ± 0.060 | 0.775 ± 0.056 | 1/5 |
| non-IID a 5 | main | mlp | 0.788 ± 0.038 | 0.693 | 0.742 ± 0.043 | 0.779 ± 0.048 | 4/5 |
| FLUID IID | sensitivity | logreg | 0.617 ± 0.076 | 0.503 | 0.620 ± 0.075 | 0.677 ± 0.047 | 1/5 |
| FLUID IID | sensitivity | mlp | 0.730 ± 0.019 | 0.474 | 0.697 ± 0.018 | 0.707 ± 0.009 | 5/5 |
| non-IID a 5 | sensitivity | logreg | 0.567 ± 0.172 | 0.474 | 0.617 ± 0.049 | 0.702 ± 0.090 | 2/5 |
| non-IID a 5 | sensitivity | mlp | 0.753 ± 0.018 | 0.449 | 0.667 ± 0.106 | 0.710 ± 0.081 | 4/5 |

- **Against the disclosed baselines** (whole dataset, not the test split): radiation-type majority
  76% of main rows, always "irradiated" 65% (main) / 64% (sensitivity). Main-task accuracy
  (0.69-0.80) is about what radiation type alone gives; sensitivity macro-F1 (0.45-0.50) is near
  the always-"irradiated" level (~0.39), i.e. the echo / heart-rate measures alone separate sham
  from irradiated only weakly.
- MLP federated beats local-only in 4-5 of 5 seeds in every setting; logreg does not (1-3 of 5).
- H1 verdicts by the PREREG §2 rule are not computed here (OSD-435 has no hypotheses spec; the
  `H1` column is the per-seed count from `--report`).

### Next
1. ~~User commits~~ done (`7174a1e`).
2. ~~OSD-435 locally~~ done (above).
3. Copy the H8 evaluation results from PACE, run `fedastro gap`, close the H8 gate, update the
   presentation page.
4. PACE pulls this commit (after step 3's files are collected), runs the 18 H9 sweeps
   (`OUT_PREFIX=h9`), then `fedastro hypotheses configs/hypotheses/h9_wesad12.yaml`.

---

## Phase H9 (part 1) — WESAD prep and OSD-435 inspection — BUILT, NO RUNS (2026-11-19)

**Not committed.** Plan approved "as recommended" (2026-11-19), OSDR request approved. Started
while H8's tuning grid runs on PACE; nothing here touches H8's frozen code or files (`tuning.py`,
`metrics/gap.py`, the gapA/gapC sweeps, PREREG §5, `scripts/slurm/gap.sbatch`), and PACE does not
pull it until H8's evaluation runs end. H9's WESAD runs need the H8 winners, so no H9 run exists.

### Built
- **`metrics/hypotheses.py`, `fedastro hypotheses <spec> [--out]`:** the PREREG §2 rules (paired by
  seed, bootstrap CI, supported / not supported / inconclusive) for H1, H2 (+ control), H3, H4.
- **`h9_sweeps.py`, `fedastro h9-sweeps <H8 eval sweeps> --dataset wesad12`:** writes
  `configs/sweeps/h9_<dataset>_<family>_{centralized,local_only,federated,hierarchical,full,stream}.yaml`
  (each arm's H8 winner + the PREREG §6 settings) and `configs/hypotheses/h9_<dataset>.yaml` (outside
  `configs/sweeps/`, which `scripts/golden.py` loads as runnable sweeps).
- **`metrics/aggregate.py`:** `result.stream` summarised across seeds (deferred from H7).
- **PREREG:** §2 and §6 DRAFTS; §4a FROZEN on this commit (settings unchanged from H7).
- **Tests:** `tests/test_hypotheses.py` (9, fabricated inputs).

### Acceptance so far
- Generated sweeps from 8 hand-written stand-in evaluation sweeps (mlp, cnn1d x 4 arms) all load,
  including `_full` (links + secure aggregation + gate + stream together) (tested).
- End-to-end mechanism check, **SYNTHETIC** wesad12 smoke data, 2 seeds, 4 rounds, magnitude 0 and
  1: `evaluate_hypotheses` ran on real result files (3.5 min laptop). Not a result.

### OSD-435 inspection (OSDR API, 2026-11-19; nothing stored in the repo)
- Files: ISA metadata zip (2.6 MB) and 5 echocardiography CSVs (LSDS-22, heart and aorta,
  SUBMITTED and TRANSFORMED, plus a data dictionary). SHA-256 of
  `LSDS-22_Echocardiography_heart_SUBMITTED.csv`:
  `5256a1a7f804e65755bd32e96ac4464cfec60da7149416dfdc38768a6bd89924`.
- **Study (from the OSDR description):** ground-based; male C57BL/6J mice irradiated at the NASA
  Space Radiation Laboratory (protons, 16O ions, protons then 16O, or 137Cs gamma rays as reference;
  sham controls); ultrasonography at 3, 5, 7 and 9 months after irradiation. Not a spaceflight
  study.
- **Heart table:** 1,152 rows, 385 mice (1-5 rows per mouse), 19 measures (heart rate, wall
  thicknesses, LV diameters/volumes/mass, SV, EF, FS, CO, mitral valve E/A). Mitral-valve columns
  are missing in 533 rows; the rest are missing in 2.
- Columns `RADTYPE` (4 types), `DOSE` (0-3 Gy, 8 levels), `COHORT`, `MONTH` {0, 2, 3, 5, 7, 9}.
  **Month 0 is in every mouse's rows but not in the study description** (which lists 3, 5, 7, 9
  months); its meaning (baseline?) is unconfirmed.
- Rows after month 0 with EF present: 765 rows, 309 mice; sham 223 rows / 90 mice (~29%).
- The study description reports EF and FS decreases only after 16O (published prior result, not
  from our data split).
- **Task (user-approved 2026-11-19, "approve as recommended"):** the FLUID setup as its paper
  reports it (user-supplied summary of the paper): 0 Gy vs >= 0.5 Gy; features radiation type,
  month, heart rate, peak velocity, VTI, mean velocity; low doses (0.05-0.25 Gy) and missing values
  removed; 50 / 50 Earth / ISS; 70 / 30 train / test. Plus a pre-registered sensitivity variant
  (no radiation type, no month-0 rows). Details in PREREG §6.
- **Findings disclosed before any run** (aorta table, counts only): 809 rows, 280 mice, 65%
  irradiated. Radiation type 4 has no sham mice; type 2 is 77% sham after the filter; majority
  class per radiation type is right on 76% of rows. 237 rows are month 0. Mice have 1-5 rows, so
  we split by mouse (FLUID's split method unknown). FLUID's "peak velocity" is ambiguous
  (`VELOCITY` mm/s vs `PEAKVEL` cm, r = 0.98): `VELOCITY` used.

### OSD-435 built (part 1b)
- `data/osdr.py` (loader, pinned SHA-256, filter, one-hot radiation type, mouse = subject),
  `Osd435Params` (`data.osd435.include_radtype`, `include_month0`; out of every other dataset's
  run id), registry entry (domain `terrestrial_analog`), `scripts/fetch_data.py osd435`,
  `configs/silos/fluid_two.yaml`, `configs/experiments/osd435_fluid.yaml`,
  `configs/sweeps/h9_osd435.yaml` (8 variants), `tests/test_osdr.py` (7).
- `fedastro inspect`: IID split earth 98 mice / 289 rows (35.6% sham), iss 98 / 255 (40.4%), test
  84 / 265 (28.3%). Sensitivity: 75 / 75 / 65 mice.
- One timing run (default MLP, FLUID IID, seed 0): 42 s laptop CPU; scores not read,
  output deleted.
- **Deviation found (needs the user):** Dirichlet alpha 0.5 with two silos gives degenerate splits:
  seed 0 leaves both silos with 0 sham rows; seed 4 one silo. Sham rows per silo, seeds 0-4
  (counts only): alpha 0.5 `0/186 0/152 | 94/250 101/113 | 103/108 103/122 | 98/221 108/110 |
  0/187 101/207`; alpha 2.0 still has seed 0 at 4/190 and 0/152; alpha 5.0 has no single-class
  silo (smallest 30/194). **User chose alpha 5.0 (2026-11-19, "approve as recommended"); PREREG
  §7 deviation 1.** Sweep updated.
- Plan deviation: no 90/10 train/validation split inside silos: the flat pipeline has none and
  nothing here is tuned.

### Decisions (user approved 2026-11-19, "approve as recommended")
1. §2 rule: paired CI vs 0, no correction; accuracy primary, macro-F1 alongside.
2. Trained models answer H1-H4; deployed versions alongside (full-architecture runs).
3. H1/H2 from each arm's own runs; H3 from the hierarchical run's models (one learning rate).
4. Main runs: links off; secure aggregation on (hierarchical arm only: the code allows it only
   there); shift grid {0, 0.5, 1, 2} `both` + `covariate`, `concept` at 1.0; 3 families, 5 seeds.
5. One full-architecture run set per family at the primary shift.
6. §4a sensitivity grid {0.1, 1.0} x {0, 1} at the primary shift.
7. OSD-435 settings after the inspection.

### Decisions made while building (review these)
1. H2's local-only score per region = the n_test-weighted mean of that region's nodes' local-only
   test scores (exact for accuracy; for macro-F1 a weighted mean of node scores).
2. The H2 control pools every region; H2 proper pools Moon and Mars only (LEO is unshifted).
3. Full-architecture region deadline 120 simulated s and global deadline 0 (wait), as in the
   H3/H4 smoke configs: scenario assumptions.
4. The PREREG §2 text discloses that the 5-seed bootstrap CI excludes 0 whenever all seeds agree
   in sign (a lenient rule).

### Open
- OSD-435 task choice; then `data/osdr.py`, fetch support, flat configs, PREREG §6 rows.
- After H8: run `fedastro h9-sweeps` on the H8 evaluation sweeps, commit the sweeps with PREREG
  §2/§6 frozen (config hashes listed), add their run ids to the golden lock, then run on PACE.

---

## Phase H8 — Closing the centralized vs federated gap — COMMITTED `62cf33b`, TUNING RUNNING (2026-11-18)

Committed by the user; PREREG §5 frozen on that commit, before any tuning run. Stage A +
baselines started on PACE 2026-11-19 (job 7167276; 1 node, 12 cores, 5 h).

### Built
- **Server optimizers** (`fl/strategy.ServerOpt`, `fl.server_opt` none | momentum | adam with
  `server_opt_lr`, `server_beta1`, `server_beta2`, `server_tau`): FedAvgM and FedAdam (Reddi et al.
  2021, no bias correction) on the round's aggregate. Flat sync arm, and each regional aggregator
  in the hierarchy (own state, kept across merges; the global merge stays a weighted average).
  Default `none`: keys out of the run id and out of the result, so every lock holds.
- **`result.validation`** (`metrics/validation.py`, `pipeline.hier_validation`): on every
  hierarchical-topology run, each trained model (centralized, local, flat_federated, regional,
  global) on each node's own validation split, per node / region / pooled. Hierarchical-topology
  runs now always keep their models (changes no result). Schema checks it when present.
- **`fedastro select <sweep> --arms ... [--emit-dir D [--template T | --eval-seeds ...]]`**
  (`tuning.py`): the PREREG §5 rule (mean validation macro-F1 over tuning seeds; cheapest within
  0.005). Loads only `result.validation` from each run. Writes `<sweep>-selection.json` (every
  candidate's scores, the winner) and generates the stage B sweep (from
  `configs/sweeps/templates/gapB.yaml`) or the evaluation sweep (winner, evaluation seeds) per arm.
  Hierarchical-only settings are dropped from flat-arm sweeps.
- **`fedastro gap <spec> [--out]`** (`metrics/gap.py`): centralized - flat federated, centralized -
  hierarchical (global), centralized - hierarchical (regional), per family, accuracy / macro-F1 /
  AUC, paired by seed, mean +/- std, hierarchical bootstrap 95% CI. Arms come from their own
  evaluation runs (each arm has its own selected learning rate).
- **Sweeps:** `gapA_wesad12_{logreg,mlp,cnn1d}` (12 variants, seeds 100-102),
  `gapC_wesad12_{...}` (8), `templates/gapB.yaml` (16), `gap_tuning_smoke` (SYNTHETIC).
- **`scripts/slurm/gap.sbatch`:** generic array job, one variant per task
  (`--export=ALL,SWEEP=...`).
- **PREREG §5** written; frozen on this commit.
- **Tests:** `tests/test_server_opt.py` (6), `tests/test_tuning.py` (10, 1 `slow`).

### Acceptance (so far)
| Check | Result |
|---|---|
| Grid and selection rule frozen in PREREG.md before running | §5 written; frozen on the H8 commit. No tuning run exists (one timing run, scores not read). |
| Selected configuration chosen by a rule applied to validation scores, recorded | `tuning.select` + `<sweep>-selection.json`; `test_selection_is_blind_to_test_scores` (random test scores, same winner and candidate scores), tie-break tests, missing-run error. **Pending on WESAD.** |
| Remaining gap reported honestly | Mechanism only (below). **WESAD gap pending the tuning runs.** |

End-to-end mechanism check, **SYNTHETIC** (`gap_tuning_smoke`, 2 learning rates x seeds 100-101,
then generated evaluation sweeps at seeds 0-1; CLI only): select picked lr 1e-2 for all three arms;
gap (MLP) centralized - flat federated accuracy +0.033, centralized - hierarchical +0.033,
centralized - hierarchical regional +0.040 (n = 2 seeds; not a result). Flat federated and
hierarchical global agree on almost every synthetic example (AUC 0.9544 vs 0.9545); expected with
full participation and a merge every 2 rounds, checked, not a bug.

### Timing (WESAD12, laptop CPU; scores not inspected)
One tuning run with local_only + federated + hierarchical, 30 rounds, 1 local epoch: **95.5 s**;
centralized + local_only: **7.4 s**. Estimated grid: stage A 108 runs (~4-6 h; 5 local epochs and
the CNN are slower), stage B 2 arms x 144 runs (~5-6 h), baselines 72 runs (~10 min), evaluation
~45 runs (~1 h): **about 11-14 h sequential on the laptop.** On PACE (one variant per array task)
each stage is ~1 h of wall time.

### Stage A + baselines on PACE (2026-11-19, copied to the laptop 2026-11-20)
- Jobs 7167276 (baselines + logreg, cancelled at 1h59 to avoid overlap), 7168448 (MLP), 7168433
  (CNN, 1h08, no errors, peak 12.5 GiB). 180/180 runs finished on PACE and copied (the last 2 CNN zips
  arrived 2026-11-20). All 180: `git_sha` 62cf33b, REAL/WESAD, `result.validation`
  present. Laptop copy: `runs/h8-62cf33b/` (git-ignored); raw zips in `recentPACEruns/`.
- **Selection (validation macro-F1, pooled, mean over seeds 100-102; `fedastro select`):**

| Family | Arm | Winner | Val. macro-F1 | Best (if not winner) |
|---|---|---|---|---|
| logreg | federated | lr 0.01, E 2 | 0.688 | |
| logreg | hierarchical | lr 0.01, E 1 | 0.680 | 0.683 (lr 0.003, E 5): cheaper within 0.005 |
| mlp | federated | lr 0.01, E 1 | 0.765 | |
| mlp | hierarchical | lr 0.01, E 1 | 0.723 | |
| cnn1d | federated | lr 0.01, E 1 | 0.725 | |
| cnn1d | hierarchical | lr 0.003, E 2 | 0.641 | |
| logreg | centralized | lr 0.003, 60 ep | 0.773 | |
| logreg | local_only | lr 0.01, 60 ep | 0.942 | |
| mlp | centralized | lr 0.003, 60 ep | 0.912 | |
| mlp | local_only | lr 0.001, 60 ep | 0.947 | |
| cnn1d | centralized | lr 0.003, 60 ep | 0.984 | |
| cnn1d | local_only | lr 0.001, 30 ep | 0.987 | 0.990 (lr 3e-4, 60 ep): cheaper within 0.005 |

- **Winners on the grid edge:** every logreg/MLP federated and hierarchical winner is at the top
  learning rate (0.01), and 5 of 6 baseline winners at the top epoch count (60). The best setting
  may lie outside the frozen grid; reported, not extended (extending would be a PREREG §5
  deviation).
- **Validation scores are not comparable across arms:** WESAD node validation is the tail of the
  node's own subject, so local-only (0.94-0.99) scores far above federated (0.69-0.77) there, while
  the test set is held-out subjects. Selection compares variants within one arm only, so it is
  unaffected; the gap is measured on test data in the evaluation runs.
- Generated (git-ignored): `runs/h8-62cf33b/generated/` stage B sweeps for every family x federated /
  hierarchical, evaluation sweeps for the baselines. Test scores of the tuning runs were not used
  and are not reported.

### Stage B on PACE (2026-11-20, copied to the laptop)
- Jobs 7201421-7201426 (one sweep each, 1 node, 12 cores, 32 GB; a first submission 7201162-67
  failed in seconds because the generated sweeps had not been uploaded). 287/288 runs; the missing
  one (cnn1d federated, FedAvg + Adam server lr 0.01, seed 102) was a Flower/Ray crash before any
  training (`query.stats: 0/12 replies within 600s`, node on `ice-bw-cpu`), rerun as job 7203022:
  288/288. All `git_sha` 62cf33b, REAL/WESAD.
- **Selection (validation macro-F1, mean over seeds 100-102):**

| Family | Arm | Winner (on stage A's lr / E) | Val. macro-F1 | Stage A |
|---|---|---|---|---|
| logreg | federated | FedProx mu 0.1 + Adam (server lr 0.1) | 0.734 | 0.688 |
| logreg | hierarchical | FedProx mu 0.1 + momentum | 0.715 | 0.680 |
| mlp | federated | FedProx mu 0.1 + momentum | 0.800 | 0.765 |
| mlp | hierarchical | FedProx mu 0.1 (best 0.749, cheaper within 0.005) | 0.747 | 0.723 |
| cnn1d | federated | FedProx mu 0.1 + momentum | 0.808 | 0.725 |
| cnn1d | hierarchical | FedProx mu 0.1 + momentum | 0.665 | 0.641 |

- **All six winners use mu 0.1, the largest mu in the frozen grid** (grid edge again; reported, not
  extended: a PREREG §5 deviation).
- Generated (git-ignored): `runs/h8-62cf33b/generated/eval_gapB_*` (6) + `eval_gapC_*` (6), seeds 0-4.

### Evaluation (seeds 0-4) — DONE 2026-11-21 (the numbers for presentations)
- 12 jobs `h8E-*` on PACE (2026-11-20), 60/60 runs, all `git_sha` 62cf33b, REAL/WESAD, 5 seeds per
  sweep. Copied from `PACE-run-evals3/` into `runs/h8-62cf33b/`. Test set = the 3 held-out WESAD
  subjects (13, 16, 17). Each arm at its own validation-selected settings (§5).
- Saved: `runs/h8-62cf33b/gap_wesad12.json` (`fedastro gap runs/h8-62cf33b/generated/gap_wesad12.yaml
  --output-dir runs/h8-62cf33b --out ...`), `runs/h8-62cf33b/eval_summary.txt` (per-arm table below).

**Test scores, mean ± std over seeds 0-4:**

| Family | Centralized acc | Flat FL (12 nodes) acc | Hier. global acc | Hier. regional acc | Local-only acc |
|---|---|---|---|---|---|
| logreg | 0.794 ± 0.010 | 0.619 ± 0.015 | **0.680 ± 0.047** | 0.663 ± 0.023 | 0.491 ± 0.015 |
| mlp | **0.861 ± 0.022** | 0.779 ± 0.035 | 0.676 ± 0.059 | 0.697 ± 0.037 | 0.528 ± 0.009 |
| cnn1d | 0.748 ± 0.082 | 0.704 ± 0.074 | 0.565 ± 0.102 | 0.530 ± 0.072 | 0.488 ± 0.024 |

(macro-F1, same order: logreg 0.729 / 0.563 / 0.640 / 0.613 / 0.417; mlp 0.798 / 0.726 / 0.639 /
0.654 / 0.420; cnn1d 0.673 / 0.660 / 0.511 / 0.473 / 0.377.)

**Gap table (centralized − other, accuracy, paired by seed, hierarchical bootstrap 95% CI, n = 5):**

| Family | − flat FL | − hier. global | − hier. regional |
|---|---|---|---|
| logreg | +0.175 [+0.157, +0.192] | +0.114 [+0.078, +0.157] | +0.132 [+0.119, +0.150] |
| mlp | +0.082 [+0.043, +0.125] | +0.185 [+0.141, +0.228] | +0.164 [+0.141, +0.192] |
| cnn1d | +0.044 [−0.056, +0.144] | +0.183 [+0.049, +0.281] | +0.218 [+0.106, +0.295] |

(macro-F1 and AUC rows in `gap_wesad12.json`.)

- **Remaining gap, honestly:** the centralized model beats every federated arm on every family; the
  only CI that includes 0 is CNN centralized − flat FL (CNN is noisy: std 0.07-0.10). Gap closing
  (stage B) raised validation scores 2-8 points but did not close the test gap.
- **Flat vs hierarchical depends on the family:** flat FL beats the hierarchy for MLP (0.779 vs
  0.676) and CNN (0.704 vs 0.565); for logreg the hierarchy's global model beats flat FL (0.680 vs
  0.619). This reverses the preliminary tuning-seed picture for logreg (flat 0.632 vs hier 0.604,
  different settings and seeds); the earlier presentation page is superseded.
- Every federated arm beats local-only on mean accuracy: flat FL by 13-25 points, the hierarchy's
  global model by 8-19 (CNN regional only +4). No CI computed here; H1 is answered in H9.
- **Disagreement (descriptive, H3-style, hierarchical runs):** accuracy where local, regional and
  global agree vs disagree: logreg 0.773 vs 0.583, mlp 0.789 vs 0.624, cnn1d 0.702 vs 0.441. The
  three models share training data, so agreement is a flag, not independent confirmation.
- No regression to report: these runs use no shift (H9 adds it).

### Decisions (user approved 2026-11-18, "approve as recommended")
1. WESAD12 tuning in H8; CinC in H10 with the same rule. 2. Tuning seeds 100-102, evaluation
seeds 0-4. 3. Centralized and local-only tuned on the same rule. 4. Per-arm selection.
5. Selection blind to test fields (tested), no validation-only run mode.

### Decisions made while building (review these)
1. **Hierarchical selection uses the global model's validation score**; regional is reported.
2. **Each arm is evaluated in its own runs**, because `train.lr` is shared by every arm of a run.
   Consequence for H9: a run holding local, regional and global together uses one learning rate
   (the hierarchical arm's winner); the local model in it is not local-only's own winner.
3. **Stage B template lives in `configs/sweeps/templates/`**: `scripts/golden.py` loads every
   `configs/sweeps/*.yaml` as a runnable sweep (the golden test failed while it sat there).
4. FedAdam has two server learning rates in the grid (0.01, 0.1) because its scale is not
   transferable from FedAvg; momentum keeps server lr 1.0.
5. Generated sweeps go under the run directory (untracked), so cluster jobs keep a clean tree.

### Tests
Full suite including slow tests (`pytest -m ""`, 377 collected): 371 passed, 1 skipped, 2 failed,
3 errors.
- `test_golden.py::test_run_ids_match_golden`: real; the stage B template sat in `configs/sweeps/`.
  Fixed (moved to `templates/`); passes.
- `test_models_h6.py[logreg]` and the 3 `test_hierarchy_fl.py` fixture errors: Ray "No heartbeat
  received from the run" (same flake as H4/H6/H7).
Re-run of the affected files (golden, tuning, models_h6, hierarchy_fl): 43 passed, 1 failed
(`test_one_region_hierarchy_reproduces_flat_fedavg_through_flower`). That test then passed alone,
and `test_hierarchy_fl.py` re-run whole: **15 passed**.

### Deferrals / open questions
- **Where to run the WESAD grid** (laptop ~12 h vs PACE ~3-4 h wall over the stages). Needs the
  H8 commit first (clean tree; PREREG §5 frozen).
- SCAFFOLD not built (deferred stretch goal, PREREG §5).
- CinC sweep files (`gapA_cinc15_*` etc.) come with H10.
- Ray flakiness now appears in most full runs; consider running the Ray test files in a separate
  pytest invocation at gates.

---

## Phase H7 — Test-then-train evaluation — COMMITTED `7c556b0` (2026-11-18)

### Built
- **`data/stream.py` (new):** each node's **training split** becomes a stream (validation and test
  untouched). Pure, seeded schedule per node (run seed + node index), computed by server and
  clients alike, so no index lists are sent:
  - order: `time` (index order = recording time order for WESAD) or `patient_shuffle` (seeded
    patient order, windows of a patient kept together; SIMULATED arrival order). `auto` = time for
    one_subject_per_node, patient_shuffle for by_source.
  - warm-up block = first `max(1, round(warmup_frac * n))` examples, labelled, trained on from
    round 1, never scored.
  - the rest in `fl.rounds - 1` equal chunks arriving in rounds 2..R (round 1 = warm-up only, so
    every arrival has a trained model to score it).
  - labels: cumulative rounding (`floor(p * arrived + 0.5)` labelled after each chunk; seeded
    choice within a chunk), usable from `arrival + label_delay_rounds`.
- **`fl/stream.py` (new):** `StreamRun`, the `stream` hook of `run_rounds`. Start of round r:
  predict round r's arrivals with the node's local model (trained through r - 1), the last
  regional and global model **delivered to that node**, and, with the gate on, the deployed
  versions (version 0 = the warm-up local model); then train each local stream model
  `fl.local_epochs` on its usable labels (warm-started, own seed stream `STREAM_ID = 9`, own
  scaler from the warm-up block). `deliver` uses the gate's recipients (H5 decision 4).
  `audit()` checks from the actual logs that no example was scored after any model trained on it.
- **`fl/hierarchy.py`:** `run_rounds(..., stream=None)`: per round, merge weights and simulated
  compute use the usable-label counts, `begin_round`, and `deliver` of each new version.
  `run_hierarchical(..., streams=None)` builds the `StreamRun` and its gate. With `stream=None`
  nothing changes (the byte-identical hierarchical run test and the golden tests pass).
- **`fl/client.py`:** with `streams`, a node trains in round r only on usable labels
  (`num-examples` = that count) and sends warm-up-only scaler statistics.
- **`metrics/stream.py` (new):** `stream_report`: per-round accuracy per model, pooled accuracy,
  balanced accuracy, macro-F1 and AUC per node / region / overall, `n_scored` per model, and
  the disagreement analysis on the examples all three models scored.
- **`pipeline.run_stream`:** fills `result.stream`: settings, order + order provenance, per-node
  schedule summary, audit (a failed audit raises), test_then_train, `fixed_test` (final stream
  models on each node's own test set), `comparison` (fixed-split hierarchical arm vs final stream
  models), the stream run's rounds / merges / fallbacks, gate section. Result note added.
- **Config:** arm `hierarchical_stream` (needs `hierarchical`; `fl.rounds >= 2`; `order: time`
  rejected for by_source); block `stream` (`label_fraction` 0.1, `label_delay_rounds` 1,
  `warmup_frac` 0.1, `order` auto), out of the run id unless the arm runs.
- **Schema:** with `stream.enabled`, required keys and `stream.audit.ok is true`.
- **Peek:** a test-then-train block (per-round accuracy and n per model) after the rounds.
- **Configs:** `smoke_hier_stream.yaml` (cinc15, SYNTHETIC, shift on, leo_roscosmos offline in
  round 3) and `smoke_hier_stream_wesad12.yaml` (wesad12 topology, SYNTHETIC).
- **PREREG §4a:** stream settings, DRAFT; to be frozen before H9.
- **Tests:** `tests/test_stream.py` (15; 1 `slow`).

### Acceptance
| Check | Result |
|---|---|
| No example is scored after the model has trained on it (tested) | `test_no_example_is_scored_after_a_model_trained_on_it`: audit 0 violations (local logs and client schedule), every stream example scored exactly once in its arrival round, emulated clients train on exactly the usable labels; `test_the_audit_is_not_vacuous` (an injected earlier training is caught). Smoke run: 1494 scored, 0 violations. |
| Label fraction 1.0, delay 0 vs fixed-split (reported) | Below. |
| Fixed-split results unchanged | `test_stream_arm_end_to_end_leaves_the_fixed_split_results_unchanged` (slow): arms, hierarchy, disagreement, calibration, model_report identical with and without the stream arm; golden + flat WESAD lock pass. |

Label fraction 1.0, delay 0, `smoke_hier_stream` (cinc15, **SYNTHETIC**, shift on), seed 0, 6
rounds (a mechanism check, not a result). Each node's own test set, pooled:

| model | fixed split acc / macro-F1 | final stream models acc / macro-F1 | whole-stream test-then-train acc |
|---|---|---|---|
| local | 0.723 / 0.718 | 0.706 / 0.704 | 0.440 |
| regional | 0.756 / 0.746 | 0.688 / 0.657 | 0.432 |
| global | 0.656 / 0.637 | 0.577 / 0.525 | 0.378 |

Sensible: with every label usable at once, the final stream models end close to the fixed-split
ones but below them (early chunks get fewer passes, since the pool grows round by round), and
test-then-train accuracy rises by round (local 0.33 -> 0.62, regional 0.35 -> 0.58, global
0.32 -> 0.43, rounds 2 -> 6). At the defaults (0.1, delay 1) a node trains on ~9 labels by the last
round and test-then-train accuracy stays at 0.33-0.38 (3 classes): the models barely learn.
wesad12 smoke (SYNTHETIC, defaults): 4 warm-up + 4 stream labels per node; stream acc local 0.353,
regional 0.312, global 0.318. Real time ~2 min per smoke run (laptop CPU; two Ray runs).

### Decisions (user approved 2026-11-18, "approve as recommended")
1. Separate `hierarchical_stream` arm (second Ray run); fixed-split arms unchanged.
2. Only the training split streams. 3. Time order for WESAD; seeded patient shuffle (SIMULATED)
for CinC / synthetic. 4. Equal chunks; seeded label choice. 5. Labelled warm-up block (10%).
6. Delay d: usable from round arrival + d (d = 0: after scoring in the same round); default 1.
7. Cumulative training pool. 8. Scaler from the warm-up block. 9. Local stream model trained
in-process. 10. Predict with the models last delivered to the node (+ deployed with the gate).
11. Merge weights by usable labels.

### Decisions made while building (review these)
1. **Round 1 = warm-up only; the stream arrives in rounds 2..R.** If chunk 1 arrived in round 1,
   no model would have been trained to score it. Needs `fl.rounds >= 2`.
2. **No `stream.enabled` key:** the arm is the switch (as `hier_fl` / `hierarchical`); `stream`
   settings without the arm are rejected. `result.stream.enabled` is still filled.
3. **Cumulative label rounding**, so 10% of 3-window chunks (WESAD at 20 rounds) still gives
   labels; per-chunk rounding would give none.
4. **Models a node has not received score nothing** (global before the first merge reaches it,
   regional before its first successful regional round); `n_scored` per model is reported and
   the disagreement analysis uses only examples all three scored.
5. **Deployed versions are scored on the stream only**; `fixed_test` covers local / regional /
   global (deployed version 0 there would be the warm-up model, not the final local).
6. The runner lives in a new `fl/stream.py` (not in the CLAUDE.md layout; hierarchy.py is
   already ~850 lines).
7. Stream clients reuse the federated client seeds (common random numbers with the fixed-split arm).

### Tests
Full suite including slow tests (`pytest -m ""`, 363 collected): 358 passed, 1 skipped,
1 failed + 3 errors, all in `tests/test_hierarchy_fl.py` and all from one Ray infrastructure crash
("node marked dead ... missed too many heartbeats", ActorDiedError; the 3 errors share the failed
smoke fixture). Re-run of that file alone: **15 passed**. Same flake class as H4/H6; no code those
tests use changed except additive optional hooks (default off).

### Deferrals / open questions
- **PREREG §4a is a draft.** Confirm the settings and the sensitivity grid (label fraction
  {0.1, 1.0} x delay {0, 1}; primary 0.1 / 1) and freeze before H9.
- **WESAD label starvation.** ~45-60 training windows per node; at 10% labels a node trains on
  a handful of labels, and conditions arrive in blocks. Reported, not adjusted.
- **Overlapping WESAD windows** (60 s, 30 s stride): a new window overlaps the last labelled one.
  Realistic for past data; disclosed.
- `metrics/aggregate.py` (`summarize`, `grid --report`) does not yet summarise `result.stream`
  across seeds; add it with the H9 runs.
- Runtime: the stream arm doubles a hierarchical run's Ray time.

---

## Phase H6 — Model comparison — COMMITTED `11906a9` (2026-11-17)

### Built
- **`models/model.py`:** `LogReg` (`arch: logreg`), one linear layer, softmax in the loss. With 8
  features and 3 classes it has 27 parameters.
- **Input rule (config):** logreg and MLP need feature input, the 1D-CNN needs sequences, for
  WESAD and synthetic data alike.
- **`data/synthetic.py`:** `representation: sequence` (`seq_length`, default 32): each window's
  feature vector becomes `n_features` channels at that level plus seeded AR(1) noise (phi 0.8,
  unit variance, own rng stream). SYNTHETIC; the default feature data is byte-identical.
- **`metrics/calibration.py`:** top-label ECE and reliability bins (10 equal-width bins; counts
  reported). `result.calibration` = ece and reliability per model, n_bins, scope.
- **`metrics/model_report.py`:** `result.model_report` = arch, n_params, payload_bytes, and one row
  per model: accuracy, macro-F1, AUC, ECE, worst-node accuracy (+ node).
  - Hierarchical runs (always): every model on each node's own test set, pooled (local, regional,
    global, flat_federated, centralized, deployed_* with the gate on).
  - Flat runs (only with `calibration.enabled: true`): each arm on the shared holdout; worst node
    only for local_only.
- **`metrics/aggregate.py`:** `summarize` and `grid --report` show the model report as mean ± std
  over seeds, with the worst-node counts.
- **Schema:** optional `model_report` checked when present.
- **Config:** `calibration` block (`enabled`, `n_bins`); out of the run id unless enabled; non-default
  `n_bins` needs `enabled`. Synthetic `representation` / `seq_length` out of the run id at defaults.
- **Configs:** `experiments/smoke_hier_models.yaml` (cinc15, SYNTHETIC, shift on, every arm, 2
  rounds); `sweeps/model_families_smoke.yaml`; `experiments/wesad12_hier.yaml` and
  `sweeps/model_families_wesad12.yaml` (real WESAD; **not run**; to be frozen in PREREG §6 before H9).
- **Tests:** `tests/test_models_h6.py` (12).

### Acceptance
| Check | Result |
|---|---|
| All three architectures run through every arm and the hierarchy, and appear in the result schema | `test_every_family_runs_through_every_arm_and_the_hierarchy[logreg|mlp|cnn1d]`: centralized, local_only, federated, ensemble_of_locals and hierarchical on cinc15 synthetic; `result.model_report` and `result.calibration` filled and schema-valid. Real time per run 37-42 s (laptop CPU). |
| Flat locks unchanged | Golden smoke result byte-identical and every golden run id unchanged (test_golden); flat WESAD regression passes; with `calibration.enabled` the smoke arms equal the golden arms exactly. |

Model report, `model_families_smoke`, **SYNTHETIC**, seed 0, 2 rounds x 2 epochs (a mechanism
check, not a result; each node's own test set pooled):

| arch | params | payload B | model | acc | macro-F1 | AUC | ECE | worst node acc |
|---|---|---|---|---|---|---|---|---|
| logreg | 27 | 108 | centralized | 0.623 | 0.610 | 0.797 | 0.075 | 0.300 |
| | | | local | 0.371 | 0.362 | 0.533 | 0.161 | 0.125 |
| | | | regional | 0.256 | 0.226 | 0.402 | 0.242 | 0.025 |
| | | | global | 0.252 | 0.221 | 0.390 | 0.247 | 0.025 |
| mlp | 2755 | 11020 | centralized | 0.798 | 0.798 | 0.934 | 0.082 | 0.600 |
| | | | local | 0.654 | 0.649 | 0.813 | 0.271 | 0.200 |
| | | | regional | 0.642 | 0.596 | 0.851 | 0.266 | 0.300 |
| | | | global | 0.504 | 0.431 | 0.729 | 0.134 | 0.300 |
| cnn1d | 3443 | 13772 | centralized | 0.821 | 0.821 | 0.944 | 0.068 | 0.700 |
| | | | local | 0.746 | 0.744 | 0.895 | 0.261 | 0.000 |
| | | | regional | 0.740 | 0.739 | 0.920 | 0.264 | 0.400 |
| | | | global | 0.592 | 0.583 | 0.791 | 0.153 | 0.400 |

**logreg is undertrained in this smoke config, not broken:** at lr 1e-3, 2 epochs on ~150 windows
is about 10 optimizer steps, and its weights move ~0.016 (mean |dw|) from the initialisation, so its
federated models stay near the (arbitrary) initial model: AUC below 0.5, and `global` and
`flat_federated` give identical predictions on every node. On one node, local logreg reaches test
accuracy 0.675 after 2 epochs and 0.975 after 30 (checked). The CNN numbers come from synthetic
sequences, which are not the same data as the synthetic features.

### Decisions (user approved 2026-11-17, "approve as recommended")
1. Calibration always in hierarchical runs; flat runs opt in (`calibration.enabled`), so the locked
   flat golden result and every run id are unchanged.
2. Top-label ECE, 10 equal-width bins.
3. CNN smoke input = synthetic sequences.
4. Worst-node accuracy from per-node test sets (hierarchical); flat runs: local_only only.
5. One architecture per run; the comparison is a sweep.
6. Architecture settings unchanged (MLP 64-32, CNN 16-32 k5); tuning belongs to H8 (validation only).

### Decisions made while building (review these)
1. **No new sweep key.** The grid already had `variants:` (explicit override lists), so the planned
   `cases:` key was not added.
2. **Hierarchical ECE uses the per-node pooled test sets only**; the flat arms' shared-holdout ECE is
   not repeated there (local / flat_federated / centralized are in the per-node table).
3. **local_only ECE in flat runs** = the mean of the silos' ECE (as the arm's other metrics).
4. ECE is computed from the models already kept for evaluation; keeping them changes no arm result
   (tested against the golden smoke result).

### Tests
Full suite including slow tests (`pytest -m ""`): **347 passed, 1 skipped, 0 failed** (343 in
the main run, 19 min 42 s; 4 tests deselected there by a mistaken `-k` filter, run separately:
4 passed).

Flaky (Ray infrastructure, as in H4): `test_secagg.py::test_one_region_flat_equivalence_holds_with_secagg`
failed twice (once in an earlier `-x` suite run, once standalone, both with a
PytestUnhandledThreadExceptionWarning from a Flower/Ray thread, one at 58 s vs a usual 36 s), then
passed 4 times in a row, including in the full run. No code it uses changed except additive return
values.

### Deferrals / open questions
- **Equal training budget across families (H4 fairness).** One `train` / `fl` budget for all three
  families favours the faster-converging ones; the smoke run shows logreg barely moving at lr 1e-3.
  Recommendation: the H8 grid tunes lr and local epochs **per family** on validation, and H9 uses
  each family's selected setting (to be frozen in PREREG §5).
- `sweeps/model_families_wesad12.yaml` is not run; its rounds and seeds go into PREREG §6 before H9.

---

## Phase H5 — On-the-fly model updates and terminal peek — COMMITTED `e9738f9` (2026-11-17)

### Built
- **`fl/versioning.py`:** `UpdateGate`. Every node holds a deployed **regional** and **global**
  version; version 0 of both is its local-only model. An incoming version is scored on the node's
  own validation set (macro-F1) and deployed iff `val_after >= val_before - k * SE` (SE = bootstrap
  standard error of the deployed version's score, 200 seeded resamples, k = 2; PREREG §4, frozen).
  Rejected versions are logged and never deployed. `degrade()` is the SCRIPTED fault for testing.
- **`fl/hierarchy.py`:** `run_rounds` gains three optional hooks that change no training or
  aggregation: `gate` (new regional version -> the round's participants; new global version -> the
  in-contact nodes of every region that received it), `diag` (per-round test-set accuracy of the
  local / regional / global models and their disagreement rate, pooled over nodes; diagnostic
  only) and `on_round` (progress file). Every round record now also has `node_state` per region
  (training / fallback / in_transit / loss_of_signal; LOS = an exchange that could never complete,
  in transit = still under way at the deadline) and `global_version` on merges.
  `evaluate_hierarchy` adds `deployed_regional` / `deployed_global` per node when the gate is on.
- **`peek.py` + `fedastro peek <run_dir> [--follow] [--interval s] [--timeout s]`:** plain-text
  view per round: simulated time (labelled SIMULATED, or "comms off"), node states, regional and
  global accuracy, the per-round diagnostic, update offers / accepts / every rejection with its
  scores. Opens files for reading only. A finished run is read from `result.json`; `--follow`
  tails `progress.jsonl`.
- **`pipeline.ProgressLog`:** hierarchical runs written by `fedastro run` get
  `runs/<run_id>/progress.jsonl` (start line, one line per completed round, `done` / `failed`).
  Write-only from the run's side.
- **Result:** `result.updates` = `enabled`, `tolerance` (the rule object), `events` (round, node,
  region, model, version, version_before, accepted, val_before, val_after, tolerance_used,
  injected), counts, final `deployed` versions per node, `inject`, note. `arms.hierarchical` gains
  `n_updates_offered` / `n_updates_rejected`. Schema: `updates.tolerance` may be a number, the rule
  object or null.
- **Config:** `updates` block (hierarchical arm only; out of the run id unless enabled, so every
  earlier run id is unchanged): `enabled`, `metric` (`macro_f1` only), `tolerance_rule`
  (`bootstrap_se` only), `k`, `n_bootstrap`, `inject` (list of {round, region, track, kind
  noise|zero, scale}; a global-track fault must fall on a merge round).
- `experiments/smoke_hier_updates.yaml` (cinc15, SYNTHETIC, shift on, leo_roscosmos offline in
  round 2, two scripted faults: Moon regional round 3 noise x5, Mars global round 4 zeroed).
- **Tests:** `tests/test_versioning.py` (14, one `slow`).

### Acceptance
| Check | Result |
|---|---|
| Results identical with and without peek | `test_results_are_identical_with_and_without_peek` (slow): a second run with `peek --follow` tailing it from start to finish writes a byte-identical `result.json`; the follower saw all 6 rounds and `done`. `test_peek_shows_the_run_and_writes_nothing`: file hashes and mtimes in the run dir unchanged after `peek`, `--follow` and the CLI. |
| At least one rejected update in a scripted run | Smoke run seed 0: 10 injected offers, **9 rejected** (all 5 Moon nodes in round 3; 4 of 5 Mars nodes in round 4). Schedule test: every injected offer rejected, every clean one accepted. |
| Gate changes no training | Schedule test: rounds, regional and global models identical with and without the gate. Through Flower: the H3 smoke result (`smoke_hier_fl`, seed 0) at `b19f28b` and now are identical after removing only the new additive fields (`node_state`, `diagnostic`, `global_version`); run id unchanged. |

Smoke run (`smoke_hier_updates`, **SYNTHETIC**, seed 0; numbers are a mechanism check, not a
result): 129 versions offered, 47 rejected. Pooled per-node test accuracy: local 0.723, regional
0.769, global 0.6625, deployed regional 0.773, deployed global 0.731.

### Decisions (user approved 2026-11-17, "approve as recommended")
1. The gate controls deployment only; training is H3/H4 unchanged.
2. Version 0 = the node's local-only model.
3. Tolerance = k x bootstrap SE (k = 2, 200 resamples), macro-F1 on validation; frozen in PREREG §4.
4. Recipients: regional -> the round's participants; global -> in-contact nodes of regions that
   received the merge; fallback rounds publish no regional version.
5. Injection degrades the delivered copy only.
6. `--follow` tails `progress.jsonl`; a finished run is read from `result.json` only (a documented
   deviation from rule 4 during a run).
7. Plain text, no curses, 1 s polling.

### Decisions made while building (review these)
1. **Many genuine federated versions are rejected.** On the smoke data the local-only baselines
   are strong on their own validation sets, so 38 of 119 clean offers fail the gate (e.g.
   leo_roscosmos, the tiny INCART node, never leaves v0). That is the gate doing what PREREG §4
   says, not a bug; it will matter when reading H9/H10 results.
2. **Zero tolerance at a perfect score.** A deployed version with validation macro-F1 1.0 has
   bootstrap SE 0, so any imperfect version is rejected (seen: moon_esa). Disclosed in PREREG §4.
3. **The gate is relative.** One injected fault was accepted (mars_partner, global round 4:
   0.154 vs a deployed 0.222 minus tolerance): a node that already deploys a poor version is only
   protected against something worse than that.
4. **Additive round fields are always written** (`node_state`, `diagnostic`, `global_version`),
   gate on or off, so `peek` works on every hierarchical run. Existing fields and run ids are
   unchanged (checked against `b19f28b` above).
5. `progress.jsonl` is written only through `fedastro run` / `sweep` / `grid` (`run_and_write`);
   `run_experiment` called directly writes nothing.
6. A deployed version that was injected and accepted stays degraded in `deployed_*` (it is what
   the node received).

### Tests
Full suite including slow tests (`pytest -m ""`): **336 passed, 1 skipped, 0 failed** (exit 0;
337 collected). Warnings are sklearn's "y_pred contains classes not in y_true" from existing
metric code on tiny test sets.

### Deferrals / open questions
- Whether the H9/H10 headline uses trained (`regional`, `global`) or deployed (`deployed_*`)
  models: to be fixed in PREREG §2 before those runs. Recommendation: trained models for H1–H4,
  deployed models reported alongside as the on-the-fly scenario.

---

## Phase H4 — Simulated secure aggregation — COMMITTED `b19f28b` (2026-11-17)

**A protocol SIMULATION, not cryptography.** Masks come from a seeded PRG (`derive_seed`); there
is no key agreement, encryption or secret sharing, and anyone with the run seed can unmask.
What is modelled is the data flow (the aggregator receives only masked vectors) and the cost
(extra messages in bytes and simulated time). No formal privacy guarantee; the final model can
still leak through membership inference (CLAUDE.md §2).

### Built
- **`fl/secagg.py`:** pairwise masks (Bonawitz et al. 2017 without the self-mask) in the ring
  Z_(2^64). Each member uploads `encode(n * w)` plus its count `n`, both masked; masks cancel
  exactly in the sum; the only difference from FedAvg is fixed-point quantisation (24 fractional
  bits). Recovery rebuilds a dropped member's masks with the survivors. Below the group minimum,
  nothing is unmasked. An overflow bound check raises instead of wrapping silently.
- **Regional level:** the Flower client masks its own update (`secagg-group` in the train
  config), so the aggregator process never receives an unmasked node update; `_Ctx.train`
  rejects any unmasked reply. Group minimum = `min_participants`.
- **Global level:** each regional aggregator masks its regional model (weighted by region
  examples) before the merge. Group minimum = `secagg.global_min_group` (2).
- **Per aggregation, modelled** (decision 3): advertise keys (round trip) -> share keys (round
  trip) -> masked upload (the regional model download carries the forwarded shares; at the
  global level the shares need one extra down leg) -> recovery round trip only if a member
  dropped after masking. Message sizes are named scenario constants (64 B request, 64 B keys,
  96 B share per peer, 8 B per masked parameter).
- **Result:** `result.secagg` gains levels, key_agreement, frac_bits, min_group, message_bytes,
  `per_round` (per region and global: group, dropped, recovered, overhead bytes and simulated
  seconds) and a note. `overhead_sim_s` is null with comms off (schema accepts null: rule 10).
  Region `bytes` and `global_bytes` include the overhead. Events `secagg_recovery` and
  `secagg_group_too_small`.
- **Config:** `secagg` block (hierarchical arm only; out of the run id unless enabled, so every
  earlier run id is unchanged): `enabled`, `levels`, `key_agreement` (`per_round` only),
  `global_min_group`, `frac_bits`, scripted `dropout` (node or region -> rounds).
- `experiments/smoke_hier_secagg.yaml` (synthetic, comms on, LOS off, two scripted dropouts);
  `sweeps/secagg_mars_presets.yaml` (Mars preset x secagg off/on, no dropouts).
- **Tests:** `tests/test_secagg.py` (17).

### Acceptance
| Check | Result |
|---|---|
| Aggregates with secagg on equal off within float tolerance | Primitive: masks cancel exactly; aggregate equals FedAvg at 1e-6 for groups of 2-12 and after recovery. Schedule: every regional and global aggregate equal at 1e-6 with comms off and on (about 1e-10 seen). Through Flower: the one-region hierarchy with secagg still matches flat FedAvg (final accuracy and macro-F1 within 1e-3). |
| No unmasked update stored or logged | The aggregator only receives uint64 masked vectors (spy on `unmask_sum`; `_Ctx.train` refuses unmasked replies); no array in the schedule output equals any node's update. |
| Overhead per round and per Mars preset | `result.secagg.per_round`; preset sweep below. |

Preset sweep (`fedastro grid configs/sweeps/secagg_mars_presets.yaml`, **SYNTHETIC** smoke data,
seed 0, 6 rounds, 3 merges, no dropouts, LOS off; simulated seconds):

| Mars preset | sim end, off | sim end, on | overhead_sim_s | bytes off | bytes on |
|---|---|---|---|---|---|
| near (182 s) | 1111.1 | 3844.7 | 2734.8 | 2,185,800 | 3,395,076 |
| mean (751 s) | 4525.1 | 15793.7 | 11269.8 | 2,185,800 | 3,395,076 |
| far (1337 s) | 8041.1 | 28099.7 | 20059.8 | 2,185,800 | 3,395,076 |

Final global and regional macro-F1 are identical on and off (0.6452 / 0.75) at every preset.
The time overhead is almost all the Mars merge: each merge adds five one-way light-times (two key
round trips plus the share leg), so secagg roughly triples to quadruples the simulated run time
at Mars distance. Bytes rise 55%, mostly from the 8-byte masked words (vs 4-byte float32).

### Decisions (user approved 2026-11-17, "approve as recommended")
1. Fixed-point ring encoding (uint64, 24 fractional bits), not float masks.
2. Members mask `n * w` and `n`, so the aggregator learns only the totals.
3. Per-round key agreement (two round trips), recovery as one extra round trip.
4. Message sizes as named scenario constants; masked upload is 8 B per parameter.
5. Regional group minimum = `min_participants`; global = `global_min_group` (2); below it the
   merge is skipped and logged, which can change training compared with secagg off.
6. Group = members reached at the start of the aggregation; a dropout is a member that joined
   and then failed to upload.
7. Hierarchical arm only; the flat federated arm is untouched.
8. Overhead = extra protocol steps on the critical path, per round; per preset via the sweep.

### Decisions made while building (review these)
1. **No regional recovery is possible on the CLAUDE.md topologies.** Every region has exactly
   `min_participants` nodes (cinc15: 5 of 5; wesad12: 4 of 4), so any dropout after masking
   leaves too few survivors and the region falls back without unmasking. Recovery only happens at
   the global level (3 regions, minimum 2). Tested with `min_participants` lowered to 4.
2. **The key exchange happens inside `region_deadline_s`** (the deadline counts from round
   start). On the short node links it costs under 1 simulated second.
3. **The aggregator stops waiting for keys once the last member that will answer has answered**
   (an oracle timeout). A real deployment needs a timeout there, which would only add time.
4. **Scripted secagg dropouts skip the dropped node's training** (as H3 does for discarded
   updates); the result is the same.
5. `secagg` settings with `enabled: false` are ignored, not rejected, so a sweep can switch it
   off without editing the base config.

### Open questions / deferrals
- The self-mask and real Shamir thresholds are not modelled (the pairwise-only variant is what
  needs recovery on dropout). Stated in the module docstring.
- The flat federated arm has no secagg; add it only if a flat comparison needs it.

### Tests
`pytest -m "slow or not slow"`: **321 passed, 1 failed, 1 skipped (CUDA), 10 warnings, 20 min 23 s**.
The failure was `test_hierarchical_run_is_byte_identical`. The second run of the H3 smoke config
crashed inside the unchanged flat federated arm with Flower's "No heartbeat received from the run"
(a Ray simulation infrastructure error, not a result difference). Rerun on its own, together with
the H3 smoke tests: 3 passed, and the hierarchical run is byte-identical. The warnings are the
known sklearn notices from tiny synthetic node test sets.

---

## Phase H3 — Hierarchical FL — COMMITTED `836bad2` (2026-11-17)

Committed as `836bad2` (`feat(fl): phase H3 hierarchical FL, threshold fallback, disagreement metrics`).

### Built
- **WESAD validation tail fix (first, user approved 2026-11-17).** Checked on real WESAD at the
  start of H3: the H1 whole-session tail left **every** node's validation set without baseline
  windows, 6 of 12 with stress only, and the other 6 with no amusement in training. Now the
  validation set is the time-ordered last 20% **of each condition** of the subject, with one
  window per condition dropped at the boundary (`partition.tail_split`). On real WESAD every
  node now has all three classes in train and validation (e.g. `leo_nasa` train 29 / 16 / 8,
  validation 8 / 4 / 2; tested, skipped without the archive).
- **`fl/hierarchy.py`**, the `hierarchical` arm, through Flower (same client code as the flat arm):
  - Each round, every region's reachable nodes train from the regional model; the region does
    FedAvg if >= `min_participants` reported, else **falls back** to the latest global model it
    holds (updates discarded, event logged).
  - Every `hier_fl.global_every` rounds (default 5) the global aggregator merges the regional
    models it receives, weighted by region training examples; receiving regions reset to it.
  - Regions cut off at merge time are left out and keep their regional model.
  - **Common random numbers:** the arm uses the flat federated arm's initial model and client
    seed stream, so with one region and `global_every: 1` it is flat FedAvg (tested through
    Flower: identical final metrics on every node).
  - Simulated time (comms on): node links from each region's `node_link`, regional-to-global
    links from its `global_link` (Mars presets via `comms.preset`), optional solar-conjunction
    blackout (`hier_fl.conjunction_*`). Requires `hier_fl.region_deadline_s > 0`; optional
    `global_deadline_s`.
  - Scripted loss of signal: `hier_fl.offline` (node -> rounds) and `hier_fl.region_offline`
    (region -> merge rounds), deterministic, for tests and the fallback-timeline figure.
- **Evaluation** (`evaluate_hierarchy`): on every node's own test set, the node's `local` model
  (its own scaler), its region's `regional` model (before the last reset) and the `global`
  model, plus `flat_federated` and `centralized` for reference. Pooled per region and overall.
  Regional and global share the federated scaler.
- **`metrics/disagreement.py`:** per node / region / overall: disagreement rate, pairwise rates,
  and each model's accuracy on agreeing vs disagreeing examples. Headline = the regional model
  (decision 6). The section says disagreement is a flag, not an independent opinion.
- **Pipeline:** `fedastro run/sweep/grid` now run hierarchical configs. The flat arms run over
  the nodes and are scored on every region's test data pooled. The result fills `topology`,
  `shift`, `hierarchy` (`rounds`, `events`, `per_node`, `per_region`, `global`, `comms`),
  `disagreement` and `arms.hierarchical`; each `silos[]` entry adds region, n_val, n_test and the
  validation label mix. The sweep summary and printout add the hierarchy per region.
- **Refactor (no behaviour change):** the flat server's scaler setup moved into
  `fl/server.federated_scaler`, shared by both arms. Locked run ids, the smoke lock and all five
  flat WESAD seeds are unchanged.
- **Config:** `hier_fl` block (left out of the run id without the hierarchical arm), arm
  `hierarchical` (needs a hierarchy and the local_only arm; rejects fedbuff and privacy).
  `experiments/smoke_hier_fl.yaml`: cinc15, synthetic, shift on, 6 rounds, merge every 2, one
  LEO node scripted offline in rounds 2-3.
- **Tests:** `tests/test_hierarchy_fl.py` (17), `tests/test_disagreement.py` (2), plus updated
  partition and config tests.

### Decisions (user approved 2026-11-17, "approve as recommended")
1. Regional models reset to the global at each merge; the evaluated regional model is the one
   before the last reset.
2. One federated scaler for the regional and global models; local models keep their own.
3. `global_every` default 5 (frozen in PREREG before H9).
4. Scripted loss of signal alongside the stochastic link model.
5. A region cut off at merge time is skipped and keeps its regional model.
6. Headline disagreement accuracy = the regional model; all three are reported.
7. The hierarchical arm rejects DP and FedBuff.

### Decisions made while building (review these)
1. **A region with no new regional round since its last global model sits out the merge**
   (event `merge_skipped`), but still receives the new global. Without this, a region that fell
   back carries the old global into the merge and drags the new global back. Found on the smoke
   run: LEO fell back in round 2, when its "latest global" was still the untrained initial model.
2. **Fallback skips the discarded training.** A client's seed depends only on its index and the
   round, so not computing discarded updates changes nothing else. Bytes are still charged as if
   the round was attempted (downloads to reachable nodes and uploads from participants).
3. **Regions run synchronised rounds** in simulated time: a round ends when the slowest region
   finishes. Regions could run asynchronously; that is not modelled.
4. **Per-round regional and global scores in the log are on the test sets.** They are
   diagnostics for the log and the H5 terminal view, never used to select anything.
5. **Common random numbers** between the flat and hierarchical arms (same initial model and
   client seeds), so their difference is the aggregation scheme, not the draw.

### Acceptance results
| Check | Result |
|---|---|
| One command produces all arms plus the hierarchy into one result file | **Yes.** `fedastro run configs/experiments/smoke_hier_fl.yaml` writes centralized, local_only, federated and hierarchical plus `hierarchy`, `disagreement`, `topology`, `shift` (`test_one_command_produces_every_arm_and_the_hierarchy`) |
| One region, `global_every: 1` reduces to flat FedAvg | **Yes**, exactly: through Flower, the hierarchical final metrics equal the flat federated ones, and per node `global == flat_federated == regional` (`test_one_region_hierarchy_reproduces_flat_fedavg_through_flower`); also on the Flower-free schedule |
| A node in loss of signal triggers the fallback without crashing | **Yes**, scripted (`leo_nasa` offline -> LEO falls back in exactly those rounds) and through the link model (a node dark for the whole run misses every regional deadline; its region falls back every round and sits out every merge). Plus: a region cut off at merge time, a solar-conjunction blackout of Mars, and no region reachable at all |
| Laptop CPU, synthetic data, < 10 min | **Yes: 55 s** for the smoke run (15 nodes, 6 rounds, all four arms), real wall-clock on the laptop |
| `pytest` | `caffeinate -i uv run --frozen pytest -m "slow or not slow"` (laptop, 2026-11-17): **305 passed, 1 skipped (CUDA), 11 warnings**, 12 min 17 s. That is 286 earlier + 19 new. The warnings: the known sklearn notice, plus 10 sklearn "y_pred contains classes not in y_true" notices from the hierarchical smoke runs, where a tiny synthetic node test set lacks a class (balanced accuracy is still computed correctly). Golden run ids unchanged; smoke result byte-identical; all 5 flat WESAD regression seeds reproduce |

Smoke run (`smoke_hier_fl`, SYNTHETIC data, SIMULATED shift, seed 0), macro-F1 per region on
that region's own test data. This checks the mechanism only; it is not a result.
| region | local | regional | global | flat_federated | centralized |
|---|---|---|---|---|---|
| leo (unshifted) | 0.724 | 0.716 | 0.730 | 0.716 | 0.803 |
| moon | 0.818 | 0.887 | 0.713 | 0.680 | 0.818 |
| mars | 0.606 | 0.646 | 0.429 | 0.443 | 0.743 |

Regional model accuracy on agreeing examples 0.909, on disagreeing 0.604.

### For Phase H4
- Secure aggregation will mask node updates within a region (group size `min_participants`) and
  regional models at the global merge. The fallback rule already guarantees no regional
  aggregate is formed from fewer than `min_participants` nodes.

---

## Phase H2 — Location-shift simulation — COMPLETE (2026-11-17), committed `9590007`

Committed as
`feat(data): phase H2 simulated location shift (covariate and concept), PREREG section 3`.

**The shift is SIMULATED.** WESAD, CinC and OSD-435 contain no location-dependent physiology.
The shift directions are arbitrary seeded vectors, not measured and not modelled physiology.
Every result that uses the shift carries `shift.provenance: SIMULATED` (enforced by the schema).

### Built
- **`data/shift.py`:** per region, on the selected features (channels, for sequences):
  `x' = mean + (x - mean) * gain + offset + class_offset[y]`.
  - **Covariate** part: region offset (`a * std * u`, u unit-RMS) and gain (`exp(0.2 * a * g)`).
  - **Concept** part: class-conditional offset (`a * std * v_k`, zero-sum over classes), which
    changes the feature-label relationship and which standardisation cannot undo.
  - `a` = `shift.magnitude` x region strength; `mean`, `std` are those of the pooled unshifted
    training data of all nodes, so magnitudes are in training-std units.
  - The same transformation is applied to a region's train, validation and test data.
  - All draws come from `shift.seed`, never the run seed, and are made for every region in
    topology order, so changing a strength or the magnitude never moves another region's
    directions.
  - `a = 0` returns the data untouched.
- **Config:** `shift` block (`enabled`, `kind` covariate / concept / both, `magnitude`, `seed`,
  `feature_fraction` 0.5, optional `features`, optional `region_strength`). Default off; left
  out of the run id when off. It needs a hierarchy. By default the first region of the topology
  (LEO) is the unshifted reference and the others have strength 1.
- **Shared test set (WESAD):** `HierFederation.region_holdout` holds each region's shifted copy
  of the held-out set; `test_of(node)` returns the node's region's copy.
- **Reporting:** `build_hier_federation` now returns `(bundle, federation, shift_section)`. The
  section has the kind, magnitude, seed, selected features, reference region and, per region,
  the distance from the reference region's training data **before and after** the shift
  (standardised mean difference, mean per-feature Wasserstein-1, and the within-class mean
  difference), in training-std units. `fedastro inspect` prints it. It is written into
  `result.shift` when hierarchical runs exist (H3).
- **Configs:** `experiments/smoke_hier_shift.yaml` (cinc15), `smoke_hier_shift_wesad12.yaml`.
- **`PREREG.md` §3 frozen:** design, kinds, grid {0, 0.5, 1, 2}, primary = `both` at 1.0.
- **Tests:** `tests/test_shift.py`, 15 tests.

### Decisions (user approved 2026-11-17, "approve as recommended")
1. Design B: covariate + class-conditional concept shift, separately switchable.
2. LEO unshifted; Moon and Mars shifted in different seeded directions.
3. Magnitudes in pooled training-std units; grid {0, 0.5, 1, 2}, primary 1.0, frozen in PREREG §3.
4. A seeded random half of the features shifts (optional explicit list).
5. Distances: standardised mean difference and mean per-feature Wasserstein, against LEO.
6. WESAD: each region is tested on its own shifted copy of the shared held-out set.

### Decisions made while building (review these)
1. **Gain rate 0.2** (`GAIN_RATE`): at magnitude 1 a selected feature's spread changes by up to
   about 22%. The gain acts around the pooled training mean, not around zero, so features with
   a large raw mean (e.g. temperature) do not get an outsized offset from it.
2. **The magnitude-0 control is its own run id** (`shift.enabled: true, magnitude: 0`), distinct
   from a config with no shift block, though its data is bit-identical.
3. **A third distance was added, the within-class mean difference.** The concept part sums to
   zero over classes, so with balanced classes it barely shows in the marginal distances (see
   the table); the within-class distance shows it.
4. `Provenance` of the dataset is unchanged by the shift (it still describes the source data).
   The shift is recorded in the result's `shift` section.

### Acceptance results
| Check | Result |
|---|---|
| Magnitude 0 is bit-identical to no shift | **Yes**, for all three kinds, every node and split, and the shared-holdout topology (`test_magnitude_zero_is_bit_identical_*`, byte comparison of X, y and subject ids) |
| Shifted regions show a measurable distance from unshifted ones | **Yes** (table below; tested with a margin of 0.1 on all three distances, and monotone in the magnitude) |
| Documented as simulated in code and here | **Yes**: module docstring, config docstring, the `shift` section's `provenance` and `note`, the configs, PREREG §3 and this entry |
| `pytest` | `caffeinate -i uv run --frozen pytest -m "slow or not slow"` (laptop, 2026-11-17): **286 passed, 1 skipped (CUDA), 1 warning** (the known sklearn notice), 14 min 00 s. That is 271 earlier + 15 new. Golden run ids unchanged; the smoke result is byte-identical; all 5 flat WESAD regression seeds reproduce |

Distances from LEO's training data on `smoke_hier_shift.yaml` (SYNTHETIC data, seed 0), in
training-std units, as `Wasserstein mean / within-class mean difference`. Magnitude 0 is the
natural difference between the regions' (synthetic) patients.
| kind | magnitude | Moon | Mars |
|---|---|---|---|
| any | 0 | 0.172 / 0.171 | 0.248 / 0.241 |
| covariate | 0.5 | 0.306 / 0.299 | 0.431 / 0.425 |
| covariate | 1 | 0.528 / 0.524 | 0.616 / 0.610 |
| covariate | 2 | 0.978 / 0.974 | 0.987 / 0.981 |
| concept | 0.5 | 0.180 / 0.306 | 0.248 / 0.348 |
| concept | 1 | 0.285 / 0.509 | 0.269 / 0.534 |
| concept | 2 | 0.659 / 0.915 | 0.539 / 0.936 |
| both | 0.5 | 0.308 / 0.369 | 0.430 / 0.483 |
| both | 1 | 0.532 / 0.641 | 0.629 / 0.767 |
| both | 2 | 0.987 / 1.190 | 1.128 / 1.345 |

### For Phase H3 (open)
- **Which scaler do the regional and global models use?** Every arm standardises its inputs,
  and a per-region scaler would undo most of the covariate part for the regional model. This
  decides how much of H2 rests on the concept part; it is an H3 design decision for the user.
- The WESAD validation-tail class mix is still to be checked on real data (H1 caveat).

---

## Phase H1 — Hierarchical topology config — COMPLETE (2026-11-16), committed `f3eac96`

Committed as
`feat(config): phase H1 hierarchical topologies, per-node splits, result schema 2`,
together with the `prereg/` deletion (user decision 2026-11-16).

### Built
- **Config (`config.py`):** `HierarchyConfig` with regions (`RegionConfig`), nodes (`NodeConfig`:
  name, agency, sources), links (`LinkConfig`: latency, loss-of-signal profile, presets), the
  global aggregator and its host, `min_participants`, split fractions.
  - An experiment opts in with the top-level key `hierarchy: <file in configs/topologies/>`.
    Default `null` (flat). It is left out of the run id when unset.
  - **A hierarchy's nodes are also the silo list** (`HierarchyConfig.as_silos`), so the flat arms
    will run over the same nodes. Setting both `silos` and `hierarchy` in an experiment is an error.
  - Strict validation: unknown keys; duplicate region / aggregator / node names; a region with
    fewer nodes than `min_participants`; a global aggregator hosted on an unknown aggregator;
    unknown loss-of-signal profiles; fractions not summing to 1; a source listed by two nodes of
    one region; sources missing (by_source) or present (one_subject_per_node).
- **Topologies:** `configs/topologies/cinc15.yaml` (3 regions x 5 nodes, `min_participants: 5`,
  Roscosmos/INCART LEO-only) and `wesad12.yaml` (3 x 4, `min_participants: 4`), per CLAUDE.md §4.
  Link values are scenario assumptions, labelled in the files; links are consumed from H3.
- **Partitioning (`data/partition.py`):** `make_hier_federation` -> `HierFederation` of
  `NodeSplit(train, val, test)`, checked by `assert_hier_disjoint` on every call.
  - `by_source` (CinC): a source's patients are divided across the regions that use it with
    Dirichlet quantity shares (`quantity_alpha`), with a floor of `min_patients_per_node` (3)
    per source per node; each node then splits its patients 70 / 10 / 20.
  - `one_subject_per_node` (WESAD): the shared held-out subjects come from the same stream as
    the flat path (so 13, 16, 17 on WESAD); each remaining subject is one node; validation is
    the last 20% of the subject's windows, with 1 window dropped at the boundary.
- **`fedastro inspect`** prints the per-node table (patients, n per split, train and test label
  mix) for a hierarchical config. **`run` / `sweep` / `grid` refuse a hierarchy until H3.**
- **Synthetic data:** optional `data.synthetic.group_names`, so synthetic groups can carry the
  CinC source names. Left out of the run id when empty.
- **Result schema "2"** (`metrics/schema.py`): `schema_version` plus the sections `topology`,
  `shift`, `hierarchy`, `disagreement`, `secagg`, `updates`, `stream`, `calibration`, present in
  every run with nulls. `topology.name` is filled now; the rest by their phases. Artifacts
  without `schema_version` still validate as version "1". `shift.provenance` must be `SIMULATED`
  and `secagg.simulated` must be true.
- **Configs:** `experiments/smoke_hier.yaml` (synthetic on cinc15), `smoke_hier_wesad12.yaml`.
- **Tests:** `tests/test_hierarchy_config.py` and `tests/test_node_partition.py` (46 new tests
  in total).

### Decisions (user approved 2026-11-16, "approve as recommended")
1. Nodes act as the silo list for the existing arms.
2. Schema bumped to "2" with every new section added at once. **`tests/golden/smoke_result.json`
   was regenerated with this approval.** The diff is additions only (the new sections and
   `schema_version`); no existing line changed. `golden.py ids`: every locked run id unchanged.
3. WESAD validation = last 20% of each subject's windows, 1 boundary window dropped.
4. At least 3 patients per node, else an error.
5. Link schema reuses the existing loss-of-signal profile names and Mars presets. The
   solar-conjunction blackout is added in H3.
6. `fedastro run` refuses a hierarchical config until H3.

### Decisions made while building (review these)
1. **The by_source assignment does not depend on the run seed.** Which patient sits in which
   node and split comes from `partition.holdout_seed` only, so validation and test membership
   are frozen across the seeds of an experiment (CLAUDE.md §4b: "validation is frozen"). Seeds
   then vary model initialisation and training only. Alternative: redraw the node assignment per
   seed, which also moves every test set.
2. **Under one_subject_per_node the subject -> node assignment follows the run seed** (as silo
   membership does in the flat path); the held-out test set does not.
3. **The 70 / 10 / 20 split counts patients, not windows** (test = round(0.2 n), val =
   round(0.1 n), each at least 1).
4. **A node's flat-arm latency is its path to the global aggregator** (node link + regional
   link). Only relevant once comms is on (H3).
5. `partition.holdout_subjects` must be set to 0 explicitly with a by_source hierarchy. The
   ">= 1" rule for flat configs is unchanged (now checked at the experiment level).

### Acceptance results
| Check | Result |
|---|---|
| No patient in two nodes or two splits (tested) | **Yes.** `test_no_patient_in_two_nodes_or_two_splits`, `test_one_subject_per_node_with_shared_holdout`, and `test_disjointness_assertion_catches_a_leak` (a leak across nodes, across splits, and into the holdout each raise) |
| The flat topology still loads | **Yes.** `test_flat_configs_still_load_without_a_hierarchy`; golden run ids unchanged; all 5 flat WESAD regression seeds reproduce |
| Per-node size and label tables print on synthetic data | **Yes.** `fedastro inspect configs/experiments/smoke_hier.yaml` (15 nodes; 4 to 30 patients per node from the Dirichlet shares) and `smoke_hier_wesad12.yaml` (12 nodes, 31 train / 8 val windows, shared test set of 120) |
| `pytest` | `caffeinate -i uv run --frozen pytest -m "slow or not slow"` (laptop, 2026-11-16): **271 passed, 1 skipped (CUDA), 1 warning** (the known sklearn notice), 21 min 02 s. That is 225 earlier + 46 new |

### Caveats and open questions
- **WESAD validation tails may be single-class.** The tail of a WESAD session is whatever
  condition came last, so a node's validation set can lack a class. To check on real data at the
  start of H3/H5 (`fedastro inspect` shows train and test mixes only; add the validation mix
  then if needed).
- **Synthetic groups are equal-sized,** so the smoke table does not show INCART as the tiny
  silo. Real quantity skew appears with the real sources (H10).
- **CinC source names** (`g12ec`, `ptbxl`, `ptb`, `incart`, `ningbo`, `chapman`, `cpsc`,
  `cpsc_extra`) are provisional until the H10 loader defines the group names.
- The full suite took 21 min this time against 11 min for H0, with only seconds of new tests.
  Not investigated.

---

## Phase H0 — Branch setup and reproducibility lock — COMPLETE (2026-11-15), committed `881045d`

Committed as
`test(regression): phase H0 flat WESAD reproducibility lock and PREREG skeleton`.

### Session start
- The tree had `CLAUDE.md` modified and `CLAUDE2.md` deleted. The user committed that
  (`c20c455`) and tagged it `pre-hier-fl`. Work started from a clean tree.

### Built (no file under `src/` or `configs/` changed)
- **`tests/test_regression_flat.py`:** runs `configs/experiments/wesad_by_subject.yaml` and
  compares centralized `final`, local_only `mean` and federated `final` (accuracy, balanced
  accuracy, macro-F1, AUC) with the reference, absolute tolerance **1e-3** (the
  laptop-vs-cluster tolerance, as in `scripts/golden.py`).
  - Seed 0 runs in the default suite; seeds 1–4 carry the `slow` marker.
  - Skipped without the WESAD archive.
  - It also checks that the reference rounds to the Phase 3 per-seed table in this file, that
    the config hash is still `ce31e25dda`, and that `privacy.enabled` is false.
- **`tests/golden/flat_wesad_g1.json`:** the reference. Full-precision final metrics copied from
  the laptop runs `runs/wesad_by_subject-seed{0..4}-ce31e25dda` (git `494e164-dirty`,
  2026-10-15), which are the runs behind the Phase 3 table.
- **`pyproject.toml`:** registers the `slow` marker; `addopts` is now `-q -m 'not slow'`. Run
  everything with `uv run --frozen pytest -m "slow or not slow"`.
- **`PREREG.md`:** skeleton with H1–H4 and PLACEHOLDER_ sections for metrics, shift, update
  tolerance, the H8 grid and the configuration list. Nothing frozen yet.

### Decisions (user approved 2026-11-15)
1. **Phases 10–12 code stays** as the baseline of this branch (additive, default off).
2. **The lock is golden Tier A plus the new regression test.** It does not wait for the cluster
   Tier B freeze.
3. **The Phase 12 redraw run and the Tier B freeze are dropped as superseded.**
   `prereg/p12-wesad-redraws.md` was never run (deleted 2026-11-16, in git history).
   `tests/golden/metrics.json` stays `PENDING`.
4. **The schema bump to "2" is deferred to Phase H1.** It changes the byte-locked
   `tests/golden/smoke_result.json`, so regenerating that file needs the user's approval then.

### Acceptance results
| Check | Result |
|---|---|
| `pytest` passes, including the regression test | **Yes.** `caffeinate -i uv run --frozen pytest -m "slow or not slow"` (laptop, 2026-11-15): **225 passed, 1 skipped (CUDA), 1 warning** (the known sklearn notice), 11 min 17 s. That is the 218 earlier tests plus 7 new (2 reference checks + 5 seeds). All 5 seeds reproduce within 1e-3 |
| Existing configs run unchanged | **Yes.** Golden Tier A passes inside the suite: every locked run id is unchanged and the smoke result is byte-identical |
| `privacy.enabled` false by default | **Yes.** `base.yaml` has no `privacy` key; the default is `PrivacyConfig.enabled = False` (`config.py`), asserted in the regression test |
| Branch point tagged | **Yes:** `pre-hier-fl` on `c20c455` (by the user) |

### Open questions
- ~~Keep or delete the old `prereg/` directory?~~ **Deleted 2026-11-16 (user decision).** It
  stays in git history (`881045d`). Mentions of CLAUDE2.md anywhere are history, not instructions.
- `Makefile` and `scripts/golden.py` comments still cite CLAUDE2.md. Left alone in H0 (not in
  the approved plan); a one-line cleanup for a later phase.

---

## Earlier plans (history)

**Phase 12 (evaluation and statistics): code built, STOPPED before the real-data run (2026-11-05).
The pre-registration `prereg/p12-wesad-redraws.md` must be committed before the WESAD redraw run
(CLAUDE2.md rule 9). Not committed.**
**Phase 11 (dataset registry + provenance domain): committed `d3cb42a` (branch `agency-build`, not
yet pushed). Its gate waits for Tier B, which waits for the Phase 10 freeze (R10).**
**Phase 10 (reproducibility lock, CLAUDE2.md): COMPLETE (2026-11-05). Committed on branch
`agency-build` (`121838e`), tagged `v0.6-golden`. The G4b amendment (below) is not committed yet
and needs the new tag `v0.6.1-golden`. Tier A passes; Tier B is pending clean-commit cluster runs
(job script ready).** The plan now follows CLAUDE2.md: Phases 10–18 replace CLAUDE.md Phases 7–9.
Phases 0–6 are committed (Phase 6 in `b99ef93`). Long commands run under `caffeinate -i`.
**GPU cluster (PACE ICE):** shakedown DONE 2026-10-29. **Clean-commit batch started
2026-11-04 at `9bd7753`:** attack settings frozen, prep job passed; `p5epsc` and the poster
configs are next (below).

---

## Phase 12 — Evaluation and statistics upgrade — code built; real run pending the pre-registration commit (2026-11-05)

**Not committed.** Suggested message:
`feat(eval): phase 12 ensemble_of_locals arm, holdout redraws, paired bootstrap statistics, per-group evaluation`.
Commit `prereg/p12-wesad-redraws.md` with it (or before it). Then run the experiment from that
clean commit.

### Session start
Golden Tier A before any edit: `tests/test_golden.py` 9 passed; `ids` and `smoke` match the lock.

### Built
- **`ensemble_of_locals` arm** (`baselines/ensemble.py`): the unweighted mean of the local models'
  predicted probabilities on the held-out set, with no weight exchange.
  - It reuses the local_only models when that arm runs, and otherwise trains the very same models
    (same seeds and scalers).
  - Tested: adding it leaves the centralized and local_only arms unchanged.
  - Under DP each local model keeps its own budget; averaging is post-processing, and the
    accounting is reported.
  - It is also scored in silo fairness.
- **Holdout redraws:** `evaluation.holdout_draws` (default 1).
  - Draw 0 is exactly the existing holdout stream. Draw d > 0 has its own fixed stream,
    independent of the run seed.
  - Each (seed, draw) is one run, with run id `<name>-seed<s>-draw<d>-<hash>` when draws > 1.
  - `fedastro run/inspect --draw d`; `sweep` and `grid` iterate seeds × draws.
  - `manifest.run_keys`.
- **Paired statistics** (`metrics/stats.py`): federated − {local_only (silo mean),
  ensemble_of_locals, centralized} × 4 metrics, paired per run. Each has a mean difference, a
  hierarchical bootstrap CI (`aggregate.bootstrap_ci`, seeds then draws), `n_pairs`, `n_seeds`
  and `n_a_better`.
  - Each comparison's role is `primary` (from `evaluation.primary`) or `exploratory`, and the
    number of comparisons is reported.
  - Written as the `stats` block of the **sweep summary** (`<name>-<hash>-summary.json`) and
    printed by `fedastro sweep/summarize`.
- **Fairness, first-class:** for every arm the summary adds `worst_silo` (value, plus which silo
  was worst in how many runs) and `gap` (best − worst = the existing spread). `run_report.py` has
  a Silo fairness section.
- **Held-out-group evaluation** (`evaluation.py`): with `holdout_groups`, every arm's final model
  is scored on each whole held-out group and on the held-out individuals of the training groups.
  - local_only is the mean over silos.
  - Written to `result.evaluation.per_holdout_group` and `held_out_individuals`, and the
    ensemble's to `arms.ensemble_of_locals.per_holdout_group`.
  - Tested: the per-group numbers recombine to the arm's overall accuracy.
- **`result.evaluation`** = {holdout_draw, holdout_draws, per_holdout_group,
  held_out_individuals}. It is present only with redraws or held-out groups.
- **`run_report.py`:**
  - runs labelled by seed and draw;
  - integrity checks distinct (seed, draw) and the same held-out subjects within each draw;
  - the ensemble as a fourth series (`--series-4`);
  - sections for paired differences (read from the sweep summary next to the runs), silo
    fairness and held-out groups.
  - Checked on the redraw smoke, the grouped smoke and the old `ce31e25dda` runs.
- **Configs:**
  - `experiments/smoke_eval.yaml` (synthetic; 4 arms; 2 seeds × 2 draws; primary set).
  - `experiments/wesad_by_subject_redraws.yaml` (the pre-registered run; hash `362a9b9ba3`).
- **Pre-registration:** `prereg/p12-wesad-redraws.md` (below).
- **Tests:** `tests/test_evaluation.py`, 16 tests, plus one assertion added to the grouped
  end-to-end test.

### Decisions (review these)
1. **The `stats` block lives in the sweep summary, not in `result.json`.** A pair is one
   (seed, draw) run, so a CI over seeds × draws exists only across runs. The summary is the
   multi-run artifact; `run_report.py` reads it from next to the runs.
2. **Worst silo and gap are summary/report metrics, not new `result.json` fields.** Each result
   already stores per-silo fairness values, min and spread, so nothing is lost, and the smoke
   lock stays unchanged. Alternative: copy them into `result.fairness` (needs a smoke-lock
   update with approval).
3. **Ensemble is unweighted**, like the local_only mean: each silo is one stakeholder.
   Alternative: weight by n_train.
4. **Group redraws are deferred to Phase 13.** `holdout_draws > 1` with `holdout_groups` is
   rejected by validation. Which groups may rotate into the held-out set depends on the dataset
   and its agency mapping, so it is designed with them. This is a scope flag: CLAUDE2.md says
   "subjects or groups".
5. **`h1.n_seeds` keeps its name but counts runs.** That is equal to seeds without redraws, and
   the printouts now say "runs".
6. **Proposed primary comparison: federated − local_only, macro-F1** (class imbalance), with
   success = CI lower bound > 0. Key secondary: federated − ensemble_of_locals, macro-F1. **The
   user may change these before committing the pre-registration.** After the commit they are
   fixed.

### Pre-registered WESAD run (not yet run)
- `prereg/p12-wesad-redraws.md`: 5 seeds × 3 draws = 15 runs.
- Held-out subjects by draw (computed from subject ids and the fixed holdout stream before any
  run): **0: 13, 16, 17** (the G1 holdout, 212 windows); **1: 2, 14, 15** (207); **2: 6, 7, 9**
  (207).
- Disclosed in the pre-registration: draw 0's federated, local-only and centralized outcomes are
  already known (G1); draws 1–2 and the ensemble are not.
- Run on the laptop from the clean commit:
  `caffeinate -i uv run --frozen fedastro sweep configs/experiments/wesad_by_subject_redraws.yaml`
  (about 15 × 40 s).

### Acceptance results
| Check | Result |
|---|---|
| Golden Tier A unchanged (new blocks default off) | **Yes.** `make test` (laptop, 2026-11-05): **218 passed, 1 skipped (CUDA), 1 warning**, 8 min 09 s. `golden.py ids`: every locked run id unchanged (additions: `smoke_eval`, `wesad_by_subject_redraws`); the smoke result is byte-identical. The first full run had **1 failure and 3 errors**, both recorded here: (a) `test_grid_continues_after_a_failed_run` stubbed `run_and_write(cfg, seed)`, which now takes `draw`; the stub was updated (the behaviour under test is unchanged). (b) The 3 `test_privacy.py` DP fixtures hit a Ray `ActorDiedError` in the very first Flower exchange (`query.stats`), before any changed code ran. Re-run alone: all 31 pass; the full re-run is clean. Treated as a transient Ray worker crash, like the 2026-10-26 startup hang |
| Run once on WESAD by-subject, 5 seeds × 3 draws | **Pending:** waits for the pre-registration commit |
| H1 vs the ensemble and the redraws, reported honestly | **Pending:** after the run, against the pre-registration |
| Tier B (`src/` changed) | **Blocked**, as for Phase 11 (R10) |

---

## Phase 11 — Dataset registry and provenance domain — AT THE GATE, blocked on Tier B (2026-11-05)

**Not committed.** Suggested message:
`feat(data): phase 11 dataset registry, provenance domain, grouped bundles, by_group and holdout_groups`.

### Session start
Golden Tier A before any edit: `tests/test_golden.py` 9 passed; `golden.py ids` and `golden.py smoke`
both match the lock.

### Built
- **Registry** (`src/fedastro/data/__init__.py`): `REGISTRY` maps `data.name` to a
  `DatasetSpec(name, label, domain, module, loader)`.
  - `pipeline.load_dataset` is now `get_dataset(name).load(cfg.data, seed)`.
  - `config.DATASETS = dataset_names()`.
  - `synthetic` and `wesad` are registered. Each has a `load(data_config, seed)` entry point that
    wraps the unchanged `load_synthetic` / `load_wesad`.
  - Also `domain_of(provenance)`.
- **Provenance domain** (`data/base.py`): `DOMAINS`, plus an optional `Provenance.domain`.
  - It is validated when set and omitted from `to_dict()` when empty (only for a bundle built
    outside the registry).
  - **`DatasetSpec.load` stamps the registry's domain into every loaded bundle**, so new WESAD
    results carry `terrestrial_proxy` and synthetic ones `synthetic`. It rejects a loader whose
    own domain or dataset label contradicts the registry.
  - The schema checks the domain when present.
  - `run_report.py` shows a Domain row. Artifacts written before Phase 11 have no domain field,
    so it falls back to `LEGACY_DOMAINS`, a stdlib copy of the registry that a test keeps in step.
- **`GroupedDatasetBundle(DatasetBundle)`**: adds `group_id` (per sample) and `group_names`.
  - Every subject must belong to exactly one group, enforced on construction.
  - `subset` keeps the type; also `groups` and `in_groups(names)`.
  - `DatasetBundle` and `site_id` are unchanged.
- **Synthetic groups:** `data.synthetic.n_groups` (default 0 = ungrouped). Groups are contiguous
  blocks of subjects `group_0..`, each with its own mean shift. That shift is drawn after every
  other draw, so ungrouped data is unchanged.
- **Partitioning** (`data/partition.py`):
  - `by_group`: one silo per group, or the groups listed per silo in `partition.group_map`.
    Deterministic, and data shares are not used: natural silos keep their natural size.
  - `partition.holdout_groups`: whole groups removed before anything else; they join the test set.
  - Under `by_group`, **unused groups** (in no silo, not held out) are removed before held-out
    individuals are drawn, so they appear neither in training nor in the test set.
  - `assert_disjoint` also fails if a held-out group has samples in a silo.
  - `Federation.groups = {silo_to_groups, holdout_groups, unused_groups}`.
- **Result** (grouped data only, so earlier results are unchanged):
  - top-level `groups` block;
  - `silos[].groups` and `holdout.groups`;
  - notes naming the held-out and unused groups;
  - `inspect` prints a groups line;
  - `run_report.py` gains a groups table and a "Held-out groups contribute no training data"
    integrity check.
- **Config:** `partition.group_map` and `partition.holdout_groups`, both excluded from the run id
  when empty, as is `data.synthetic.n_groups` when 0. Validation:
  - `group_map` only with `by_group`, with one entry per topology silo;
  - a group in at most one silo, and never in both a silo and `holdout_groups`;
  - grouped data required for `by_group` and `holdout_groups`;
  - `holdout_subjects` may be 0 only with `holdout_groups`.
- **`configs/experiments/smoke_grouped.yaml`**: 24 synthetic subjects in 8 groups, `by_group` over
  `five_agency`, `group_6` held out, 2 held-out individuals, and `group_7` deliberately unused.
- **Tests:** `tests/test_registry_groups.py`, 20 tests.

### Decisions (review these)
1. **Every artifact records its domain (user decision 2026-11-05: "write the domain, the final
   repo should be as per the planned goal").**
   - The first pass left the domain out of synthetic and WESAD artifacts to keep the smoke lock
     byte-identical. The user chose the planned behaviour instead (CLAUDE2.md rule 10).
   - The registry is the single source of a dataset's domain; `DatasetSpec.load` writes it.
   - **Golden file change, approved by the user (2026-11-05):** `tests/golden/smoke_result.json`
     regenerated with `golden.py smoke --write --force`. Before writing, the diff was checked:
     **the only change is the added line `"domain": "synthetic"`** in the provenance record. Every
     metric and every other byte is identical.
   - **Unaffected:**
     - every run id (provenance is not in the config hash; `golden.py ids` matches the lock);
     - `tests/golden/metrics.json` and Tier B (they compare metrics, not provenance);
     - the PACE golden runs at `v0.6.1-golden`: that checkout tests against its own, older smoke
       file, which is consistent with its code.
   - Old artifacts (the 360 laptop runs and the `9bd7753` batch) have no domain field. They
     resolve through `domain_of` / `LEGACY_DOMAINS`.
2. **The registry is a dict with lazily imported loaders, not a decorator.** Config validation
   needs the names before any loader is imported, and loaders import `fedastro.config`, so a
   decorator would create an import cycle. A new dataset still needs its own `DataConfig`
   parameter block in `config.py` (strict validation); `pipeline.py` needs no edit.
3. **`GroupedDatasetBundle` is a subclass, not new optional fields on `DatasetBundle`.** That
   leaves the base class and every existing code path untouched. `site_id` stays as it was:
   `by_site` assigns sites randomly by share, while `by_group` maps natural groups deterministically.
4. **Under `by_group`, `holdout_subjects` draws held-out individuals from the training groups**
   (with the fixed `holdout_seed`). Per-silo held-out individuals and per-group metrics are
   Phase 12/13 work.
5. **Unused groups are excluded from the test set too.** Found while testing:
   - The first version drew held-out individuals from a pool that still contained the unused
     `group_7`, so an excluded cohort would have been scored (the `inspect` output showed
     `HOLDOUT=…,group_7`).
   - Fixed before any result was produced.
   - Regression test: `test_holdout_groups_contribute_no_samples`.

### Acceptance results
| Check | Result |
|---|---|
| Golden Tier A unchanged | **Yes, with the approved smoke-file change** (decision 1). `make test` (laptop, 2026-11-05, after the domain change): **203 passed, 1 skipped (CUDA), 1 warning** (the known sklearn notice), 6 min 59 s. That is 182 earlier + 21 new. `golden.py ids`: every locked run id unchanged; the only addition is `smoke_grouped.yaml` (allowed). The smoke result equals the regenerated lock |
| Registry path gives byte-identical smoke results | **Yes**: `test_registry_path_gives_identical_data_and_byte_identical_smoke`. The arrays through the registry are identical to the direct loader's; the provenance is identical apart from the added domain; `fedastro run smoke` equals the smoke lock (regenerated with the approved domain line, decision 1) |
| Held-out groups contribute no samples and no scaler statistics | **Yes**: `test_holdout_groups_contribute_no_samples` (subjects and groups) and `test_holdout_groups_contribute_no_scaler_statistics`. In the latter, the held-out group's features are set to 1e6. Silo data stays below 1e3, so the centralized and local scalers are clean, and the federated scaler from the real Flower stats exchange equals the scaler fit on silo data (rtol 1e-6) |
| Tier B (required: `src/` changed) | **Blocked**: `metrics.json` is still PENDING (Phase 10 freeze, R10). The gate closes after `golden.sbatch` at `v0.6.1-golden` + `freeze`, then a Tier B run on this phase's commit |

---

## Phase 10 — Reproducibility lock — COMPLETE (2026-11-05)

Committed by the user on branch `agency-build` as `121838e`
(`test(golden): phase 10 reproducibility lock ...`), tagged `v0.6-golden`.

### Amendment 2026-11-05: G4b added (user decision, approval recorded here)
- **Decision:** the user asked for `p5eps-base` as a 6th pinned item, so Tier B covers both
  halves of G4: the DP utility collapse (G4, ε 1) and the no-DP attack success against batch size
  (G4b, B=1 vs B=32). This closes risk R11.
- **G4b compares** the base fields, `privacy.attack.mse` (the headline), and **every attack
  cell**: 5 silos × 2 model states × 3 batch sizes × 2 label modes, the trial-mean `mse` and
  `nn_identified` (120 values per run).
- **Files changed (no `src/` change):**
  - `scripts/golden.py`: `PINNED` gains G4b, plus an `attack_cells` flag and `ATTACK_CELL_SCORES`.
  - `scripts/slurm/golden.sbatch`: array 0–5, `ITEMS=(G1 G2 G3 G4 G5 G4b)`. Indices 0–4 are
    unchanged; G4b is index 5.
  - `tests/test_golden.py`: 9 tests (new `test_g4b_compares_every_attack_cell`).
  - `Makefile` comment: about 2.5 h of CPU.
- **Golden file change, approved by the user (2026-11-05):** `tests/golden/metrics.json`
  regenerated with `golden.py pending --force`. It is still PENDING and now lists 30 run ids
  (G4b: `p5eps-base-seed{0..4}-eeaeb768e6`). `run_ids.json` and `smoke_result.json` are
  unchanged: G4b is an existing variant.
- **Verified:**
  - `golden.py ids`: run ids match the lock.
  - `pytest tests/test_golden.py`: 9 passed.
  - Cell extraction on a current-format attack result (`smoke_dp`, synthetic): 80 cell fields for
    its grid. The local `p5eps-base` runs use the older single-silo layout, so they cannot be used.
- **Tag:** `v0.6-golden` (`121838e`) predates G4b. Golden runs must use the amended commit.
  Tag it `v0.6.1-golden` rather than moving the pushed tag:
  `git tag -a v0.6.1-golden -m "Phase 10 lock + G4b" && git push origin v0.6.1-golden`.
  Suggested commit message:
  `test(golden): pin p5eps-base as G4b (no-DP attack vs batch size)`.

### Session start
- Golden Tier A did not exist yet, so the full existing suite was the clean-start check, run
  before any edit to an existing file: **173 passed, 1 skipped (CUDA), exit 0** (laptop).

### Built (no file under `src/` changed)
- **`scripts/golden.py`**, the lock tool. Subcommands:
  - `ids [--write]`: run ids of every experiment config and sweep variant.
  - `smoke [--write]`: the smoke result, compared byte for byte.
  - `pending`: writes the placeholder `metrics.json`.
  - `run --item G1..G5 G4b`: re-runs the pinned subset through the normal CLI (`fedastro sweep`
    for G1, `fedastro grid --index i` for the others).
  - `check --runs-dir`: compares final metrics with `metrics.json` at 1e-3.
  - `freeze --runs-dir`: writes `metrics.json` from clean runs.
- **`tests/golden/run_ids.json`**: 10 experiment configs and 53 sweep variants, 306 run ids, with
  full 64-hex config hashes. G1 = `wesad_by_subject`, hash `ce31e25dda…` (the laptop-vs-cluster hash).
- **`tests/golden/smoke_result.json`**: `fedastro run configs/experiments/smoke.yaml --seed 0` with
  `SOURCE_DATE_EPOCH=0`, and `git_sha` masked as `GOLDEN-MASKED`.
- **`tests/golden/metrics.json`: status `PENDING`.** It lists the pinned run ids that need
  clean-commit runs: 25 at `v0.6-golden`, 30 after the G4b amendment (below).
- **`tests/test_golden.py`** (Tier A, 8 tests; 9 after the amendment):
  - run ids against the lock;
  - G1 hash check;
  - smoke byte identity;
  - `metrics.json` covers exactly the pinned subset;
  - the comparison and freeze rules: tolerance, changed or new fields, dirty, mixed-commit and
    synthetic runs rejected, git_sha masking.
- **`make golden`** (Tier B): `rm -rf runs/_golden`, then `golden.py run`, then `golden.py check`.
  It uses `caffeinate -i` only where it exists, so the target also runs on Linux.
- **`scripts/slurm/`**: account `bios`, QOS `coc-ice`, no `-p`, 1 node × 4 cores, 16 GB.
  - `common.sh`: sources `~/.bashrc` safely (the PACE `/etc/bashrc` issue), refuses a dirty tree,
    sets `OUT=runs/<prefix>-<sha>` and a per-job `RAY_TMPDIR`.
  - `p5epsc.sbatch`: array 0–7, one variant per job, 8 h.
  - `golden.sbatch`: array 0–4 = G1–G5 (0–5 with G4b after the amendment), one item per job, 3 h.
  - Submit from the repo root after `mkdir -p runs/slurm`.
- **Frozen `privacy.attack` block and commit sha**: recorded 2026-11-04 in the clean-commit batch
  section below (commit `9bd7753`, `trials: 10`, every silo, both label modes). This was Phase 5
  deferred item 7; nothing new was needed.

### Pinned Tier-B subset (decisions, review these)
| ID | Config | Variant | Fields compared (per seed, 5 seeds) |
|---|---|---|---|
| G1 | `experiments/wesad_by_subject.yaml` | (experiment) | centralized `final`, local_only `mean`, federated `final` (accuracy, balanced accuracy, macro-F1, AUC) |
| G2 | `sweeps/phase4_alpha.yaml` | `p4alpha-a0p1-dirichlet-fedavg` | the same |
| G3 | `sweeps/phase4_mu.yaml` | `p4mu-mu1p0` | the same |
| G4 | `sweeps/phase5_eps.yaml` | `p5eps-eps1p0` | the same + `privacy.epsilon`, `privacy.attack.mse` |
| G5 | `sweeps/phase6_presets.yaml` | `p6pre-marsfar-fedbuff` | the same + `sim_wallclock_s_total`, per-silo `dispatches` |
| G4b | `sweeps/phase5_eps.yaml` | `p5eps-base` (no DP) | the same + `privacy.attack.mse` + every attack cell's `mse` and `nn_identified` (amendment, 2026-11-05) |

1. **G2 = α 0.1**, the strongest skew and the end point of the degradation curve.
2. **G3 = μ 1**, the only μ with a measurable effect, so it is the most sensitive canary for
   the FedProx path.
3. **G4 = `p5eps` ε 1 (3 attack trials), not `p5epsc`.**
   - It covers Opacus and the attack under DP, at ~1 h per Tier B run instead of ~3 h.
   - Trade-off: the "attack succeeds at B=1, modest at B=32" half of G4 is a no-DP result, so it
     is not in the pinned variant. **Resolved 2026-11-05:** `p5eps-base` added as G4b, alongside
     G4 rather than replacing it, so Opacus coverage stays.
   - The local `p5eps` runs carry an older hash (`4a68779d84`, before `chance_draws`), so G4
     needs new runs anyway (`ea373fd066`).
4. **G5 = Mars-far FedBuff**: comms, the outage model and FedBuff in one variant.
5. **Tolerance: absolute 1e-3 on every number**, including simulated seconds and dispatch counts
   (counts are integers, so effectively exact). `None` must stay `None`, and the set of fields
   must not change.
6. **Freeze rules:** all pinned runs (30 with G4b) must come from **one clean commit** with REAL provenance.
   Overwriting a frozen golden file needs `--force` and the user's approval recorded here.
   Replacing the PENDING placeholder is the intended first freeze.
7. **Smoke byte identity masks `git_sha`.** It is the one field that names the commit, so an
   unmasked golden file would fail at every commit. The timestamp is fixed by
   `SOURCE_DATE_EPOCH=0`. No paths or hostnames appear in `result.json`; they live in
   `manifest.json`.
8. **Additions allowed:** configs or variants added after the lock pass Tier A and are listed
   as additions. Changed or deleted golden ones fail.

### Acceptance results
| Check | Result |
|---|---|
| Tier A passes | **Yes.** `make test` (laptop, 2026-11-05): **181 passed, 1 skipped (CUDA), 1 warning** (the known sklearn notice), 7 min 46 s. That is the 173 earlier tests plus the 8 new golden tests. The smoke byte check also holds when run after the whole suite in one process |
| Tier B passing, or pending with job scripts ready | **Pending.** `metrics.json` is PENDING; `scripts/slurm/golden.sbatch` is ready. Verified: `check` exits 2 on PENDING; `freeze` refuses missing runs and the local `-dirty` runs (494e164-dirty for G1) and leaves `metrics.json` unchanged |
| No file under `src/` changed | **Yes**: `git diff --stat -- src` and `git status -- src` are empty |
| Tag `v0.6-golden` | **Created by the user** on `121838e` (branch `agency-build`). The G4b amendment needs `v0.6.1-golden` |

### Clean-commit runs needed for Tier B (the gate's stop)
Only `-dirty` runs exist on the laptop. **Run `scripts/slurm/golden.sbatch` (6 jobs) on the
commit tagged `v0.6.1-golden`.** Freezing requires one commit, so G1–G3 from the running `9bd7753`
batch cannot be combined with new G4/G5 runs. Run ids:
- G1: `wesad_by_subject-seed{0..4}-ce31e25dda`
- G2: `p4alpha-a0p1-dirichlet-fedavg-seed{0..4}-51351f4a1a`
- G3: `p4mu-mu1p0-seed{0..4}-f22cc0ff09`
- G4: `p5eps-eps1p0-seed{0..4}-ea373fd066`
- G5: `p6pre-marsfar-fedbuff-seed{0..4}-0d31114ef8`
- G4b: `p5eps-base-seed{0..4}-eeaeb768e6`

Steps:
1. Commit the G4b amendment and tag it `v0.6.1-golden`.
2. Let the `9bd7753` batch finish, or use a second clone. Pulling into `$SCR/FLASH` mid-batch
   would give later jobs a different `git_sha`.
3. `mkdir -p runs/slurm && sbatch scripts/slurm/golden.sbatch`.
4. On the login node: `uv run --frozen python scripts/golden.py freeze --runs-dir runs/golden-<sha>`,
   then `... check --runs-dir runs/golden-<sha>` (it must report `Tier B PASSED`).
5. Commit `tests/golden/metrics.json` (it records its source sha).

### Risk statement for PACE runs (read before every Tier B or golden run)
Written 2026-11-05. The Phase 10 commit is on branch `agency-build`, tagged `v0.6-golden`. Golden
runs use the G4b amendment, tagged `v0.6.1-golden`. The `9bd7753` batch (`p5epsc` + poster
configs) is still running in `$SCR/FLASH`. Repo edits will not be pulled on PACE until it finishes.

| # | Risk | What goes wrong | Mitigation |
|---|---|---|---|
| R1 | Changing `$SCR/FLASH` while its jobs run | Jobs read source files live: later seeds could run different code, and results record another `git_sha` | Do not `checkout`, `pull` or edit in `$SCR/FLASH` until the `9bd7753` batch has finished and been copied back |
| R2 | Golden runs from the wrong commit | `metrics.json` would record a commit that is not the tag | Run golden from a separate clone of the tag: `git clone --branch v0.6.1-golden <repo> $SCR/FLASH-golden`, `make install`, `export FEDASTRO_DATA_ROOT=$SCR/FLASH/data` before `sbatch` (it reuses the WESAD zip and cache) |
| R3 | Mixing batches | `freeze` refuses runs from more than one commit | Freeze only `runs/golden-<sha>` from `golden.sbatch`. Never add `9bd7753` batch runs to it |
| R4 | Nodes not giving identical results | Different nodes (CPU model, math library path) could give slightly different floats; a later Tier B on another node could then "move" with no code change | Cross-check: golden G1–G3 must equal the `9bd7753` batch results for the same run ids (same code). Record each job's node (`hostname` in the log, `manifest.json`) and `lscpu \| grep "Model name"`. If they differ, report it before freezing |
| R5 | Tier B across platforms | Freeze on PACE, check on the laptop: G2 (α 0.1) and G4 (1000-step attack) are the most likely to exceed 1e-3 | Treat the PACE `check` as the gate result; a laptop `make golden` is informative only |
| R6 | Smoke byte test on Linux | The smoke golden file was written on the Intel Mac; the first PACE `make test` may fail only `test_smoke_result_is_byte_identical_to_golden` | That is a platform difference, not a regression. Do not regenerate the file; report it and decide (per-platform file or 1e-3 tolerance) |
| R7 | Time limits | `golden.sbatch` allows 3 h. G4 and G4b (attack, all silos, both label modes) are estimated at ~1 h each on 4 cores, unverified. G1 runs through `fedastro sweep`, which has no `--resume`, so a resubmit re-runs all 5 seeds | If a task times out: `sbatch --array=<i> scripts/slurm/golden.sbatch` (all but G1 skip finished seeds), or raise `-t` on the command line |
| R8 | Parallel Ray jobs on one node | Untested before the 2026-11-04 batch; a startup hang would time the task out | Per-job `RAY_TMPDIR` (in `common.sh`). If a task dies at Ray startup, resubmit that index |
| R9 | Merge strategy | A squash or rebase merge gives main new commit ids; the tag and `metrics.json` would then name a commit that is not on main | Merge the branch with a merge commit or a fast-forward only |
| R10 | Gate ordering | Phase 11+ change `src/`, so their gates need Tier B, which cannot pass while `metrics.json` is PENDING | Freeze before closing the Phase 11 gate. Development of Phase 11 can go ahead on the branch meanwhile |
| R11 | G4 coverage gap | Tier B did not cover the no-DP attack numbers (B=1 success vs B=32) | **Resolved 2026-11-05:** G4b (`p5eps-base`) added before the first freeze, with every attack cell compared. It needs the `v0.6.1-golden` commit (R2) |
| R12 | Golden file updates | Regenerating a golden file to make a check pass hides a moved finding | The first `freeze` replaces PENDING. Any later change needs `--force` and the user's approval recorded here (CLAUDE2.md §3) |
| R13 | Scratch purge | PACE scratch is purged after inactivity | Copy `runs/golden-<sha>` and `$SCR/results/` back to the laptop as soon as each batch finishes |

Order on PACE once the `9bd7753` batch is done and copied back:
1. Clone the tag `v0.6.1-golden` (R2).
2. `mkdir -p runs/slurm && sbatch scripts/slurm/golden.sbatch` (6 jobs).
3. Cross-check G1–G3 against the `9bd7753` batch (R4).
4. `golden.py freeze`, then `golden.py check` (must print `Tier B PASSED`).
5. Commit `metrics.json` on the branch.

### Caveats of freezing `metrics.json` (read before relying on Tier B)
1. **Golden means "unchanged", not "correct".** The freeze locks whatever the tagged code
   produces, including any bug it contains. A later bug fix that changes a number fails Tier B
   by design. Re-freezing then needs the user's approval recorded here, and every poster number
   from the old code has to be re-run.
2. **At 1e-3, accuracy is checked exactly.** One holdout window is 1/212 ≈ 0.0047 of accuracy,
   more than the tolerance, so a single flipped prediction fails. The same holds for integer
   fields (dispatch counts). Tier B tolerates float noise only where it does not change a
   prediction.
3. **Only the listed fields are locked.** These are final metrics per arm, plus each item's
   extras. Round-by-round curves, per-silo results, fairness, rounds-to-target, bytes and the
   attack trials beyond the cell means are not compared. Other configs are locked only by run id
   (Tier A), not by numbers.
4. **The freeze is tied to the platform and the environment.** Node type (R4), and any change to
   `uv.lock`, Python, torch, Opacus, Flower or Ray, can move floats. A dependency upgrade is
   therefore a re-freeze event, with approval.
5. **Tied to the data version.** `freeze` checks REAL provenance but does not compare the WESAD
   checksum across runs. A different archive or preprocessing version would show up only as
   moved metrics. The checksum is in each run's `provenance`, so keep the source runs.
6. **Keep the evidence.** `metrics.json` stores values, the source sha and the run-folder name,
   not the runs. Copy `runs/golden-<sha>` back (R13) and keep it with the tag.
7. **The metrics commit comes after the tag.** `metrics.json` is committed after
   `v0.6.1-golden`, and it records the tagged commit as its source. Tier A checks only that
   `metrics.json` covers the pinned run ids, not which commit produced them.
8. **A Tier B pass is evidence for the pinned subset only.** It does not re-validate the full
   sweeps behind G2–G5. Those numbers come from their own clean-commit runs.

### Open questions and risks
- **Cross-platform smoke identity is unverified.** The smoke golden file was written on the
  Intel Mac. The cluster shakedown matched to 3 decimals, not byte for byte. If the first cluster
  `make test` fails only `test_smoke_result_is_byte_identical_to_golden`, that is a platform
  difference, not a regression. Options then: per-platform smoke golden files, or a 1e-3
  comparison for the smoke file. This is the user's call; the file stays unchanged until then.
- `metrics.json` will come from the cluster. A laptop `make golden` then compares across
  platforms at 1e-3, which G1 met in the shakedown.
- The ad-hoc `$SCR/jobs/p5epsc4.sbatch` currently in use runs two variants per task. The repo's
  `scripts/slurm/p5epsc.sbatch` follows CLAUDE2.md (one variant per job). Both write the same
  run ids.

---

## Clean-commit cluster batch on PACE ICE — IN PROGRESS (2026-11-04)

Purpose: publishable numbers. Every earlier result in `runs/` was produced from a `-dirty`
tree (360 result files across 7 commits), so none passes `run_report.py`'s clean-commit check.

### Setup
- Commit `9bd7753a8b3dc092e596b2fbeefd400eddab60c6`, tree verified clean in every job
  (`common.sh` aborts on a dirty tree).
- Output: `$SCR/results/clean-9bd7753/` (outside the repo). Job scripts: `$SCR/jobs/` (outside
  the repo).
- Requests are split into parts:
  1. prep: 1 node × 4 cores, 16 GB, 2 h.
  2. `p5epsc`: 4-task array, 1 node × 4 cores, 16 GB, 8 h each; task i runs variants i and i+4.
  3. Poster configs: 1 node × 4 cores, 16 GB, 4 h. Runs `wesad_by_subject`, `wesad_by_subject_cnn`,
     `phase4_by_subject`, `phase4_alpha`, `phase4_mu`. Phase 6 is left out (level two).
- Fix needed on PACE: `/etc/bashrc` reads an unset variable, so sourcing `~/.bashrc` under
  `set -u` killed the first prep job (6816077) in 7 s. `common.sh` now disables `-e`/`-u` while
  sourcing and restores them afterwards.
- Each job gets its own `RAY_TMPDIR`. Several Ray jobs running in parallel on one node had not
  been tested before this batch.

### Attack settings frozen (pre-registration, Phase 5 deferred item 7) — DONE
Recorded 2026-11-04T23:35:39 UTC, before any `p5epsc` job was submitted. File:
`$SCR/results/p5epsc_prereg.txt`.
```
commit 9bd7753a8b3dc092e596b2fbeefd400eddab60c6
privacy.attack = {enabled: true, silos: [] (all), label_modes: [known, inferred],
  model_states: [init, final], batch_sizes: [1, 4, 32], trials: 10, iterations: 1000,
  lr: 0.1, prior_weight: 0.01, chance_draws: 50}
```
These settings must not change for the `p5epsc` results.

### Prep job 6816173 — PASSED
- `make test`: **173 passed, 1 skipped (the CUDA test, CPU node), 1 warning**, 12 min 33 s.
- `make repro-check`: **BYTE-IDENTICAL**.
- `fedastro inspect wesad_privacy.yaml --seed 0`: per-silo class table identical to the
  laptop (345 / 137 / 138 / 142 / 68; holdout 212).
- 13 min 14 s wall, 4 cores, peak memory 5.2 GB.

### Still to do in this batch
1. `p5epsc` (request 2), then `--report`.
2. Poster configs (request 3), then `--report` per sweep and `run_report.py` on
   `wesad_by_subject`; its clean-commit check must pass.
3. Copy `$SCR/results/` back to the laptop (scratch is purged after inactivity).

---

## Cluster shakedown on PACE ICE (Georgia Tech) — DONE (2026-10-29)

### Environment
- **Home quota is 30 GB and was full.** Everything lives in scratch
  `/storage/ice1/4/7/aagarwal710` (300 GB): repo at `$SCR/FLASH`, uv at `$SCR/uv/bin`,
  `UV_CACHE_DIR` / `UV_PYTHON_INSTALL_DIR` under `$SCR/uv/`, set in `~/.bashrc`. Scratch is
  purged after inactivity: copy results back to the laptop promptly.
- Setup on the login node: `make install`, `make fetch-wesad`. Compute only in SLURM jobs.
- **SLURM:** account `bios`, QOS `coc-ice`. Don't pass `-p`: jobs are routed automatically
  (CPU → `coc-cpu`, `--gres=gpu:...` → `coc-gpu`). An explicit `-p coc-gpu` was refused.
  Time limits: CPU 18 h, GPU 16 h.
- **GPUs:** L40S and V100 work with torch 2.2.2+cu121. **Blackwell nodes (`ice-bw-gpu`, RTX PRO
  6000 Blackwell, MIG) do not**: they need a newer CUDA build of torch than the Intel-Mac pin
  allows. Request `--gres=gpu:l40s:1` (or use a V100). Not upgrading torch: the GPU only speeds
  up execution and is not needed for any poster run.
- An OnDemand interactive session (`sys/dash`) holds its GPU while idle: close it after use.

### Results
- **`make test`: 173 passed, 1 warning, 9 min 51 s** (CPU job 6665444, 8 cores). The warning
  is sklearn's single-label notice from the deliberately degenerate input in
  `tests/test_models.py:53`; production metrics pass the full label list. No fix needed.
- **`fedastro sweep configs/experiments/wesad_by_subject.yaml`: identical to the laptop** to
  every printed decimal, all arms, per silo, fairness and H1, same run id hash `ce31e25dda`
  (byte identity was not expected across platforms; it held at 3 decimals). 13.5 min wall,
  7.7 GB peak memory, 8 cores busy.
- **CUDA check: bug found and fixed.** On a GPU the federated arm crashed: the server's global
  model ([server.py:148](src/fedastro/fl/server.py#L148)) and the client's evaluation model
  ([client.py:64](src/fedastro/fl/client.py#L64)) were never moved to the device; training was
  fine because `train_local` moves its model. Fixed with `.to(device)` (commit `35e28d4`); CPU
  results unchanged. New test `test_federated_smoke_on_cuda` (skipped without CUDA).
  **After the fix, on a V100: smoke run completes with `"device": "cuda"`, the CUDA test
  passes.** L40S also reported `2.2.2+cu121 True 12.1`.
- **`scripts/run_report.py`** (commit `6b91ccc`): self-contained HTML report from run
  artifacts: integrity checks (one config, distinct seeds, ≥ 5 seeds, same clean commit, REAL
  data, same checksum, subject disjointness, manifests), three-arm tables and charts, per-silo,
  per-seed, data, comms, provenance. Stdlib only. On the laptop copy of `ce31e25dda` the clean
  commit check correctly fails (those runs were `-dirty`, 2026-10-15).

### Next on the cluster
1. Freeze the attack settings: record the `privacy.attack` block and the commit sha checked out
   on the cluster here, before the run (Phase 5 deferred item 7).
2. `p5epsc`: 8 variants × 5 seeds, CPU only, ~3 h per variant. Size the job to `coc-cpu`'s
   cores per node (`sinfo -p coc-cpu -o "%c %m"`).

---

## Phase 6 — Deep-space comms realism — COMPLETE at the gate (2026-10-22, updated 2026-10-26, sweeps re-run 2026-10-29)

**Not committed.** The user commits. Suggested message:
`feat(fl): phase 6 simulated deep-space links, FedBuff async aggregation, dark-silo handling`.

### Built
- **`fl/comms.py`: per-silo link model in SIMULATED seconds.**
  - Transfer = wait for the link to be up + transmit (bytes × 8 / bandwidth, which must
    fit in one up-window, else it restarts after the outage: store-and-forward, as in
    delay-tolerant deep-space networking) + one-way light time + |N(0, jitter)|.
  - Outages come from each silo's `los_profile`: poisson or periodic. They are
    generated lazily from their own seeded RNG stream, so the outage schedule does
    not depend on traffic.
  - `comms.dark` forces a silo's link down from a time, for a duration or forever.
  - Nothing sleeps. `fedastro run ... --real-sleep [SCALE]` (live demo only) sleeps
    simulated time × SCALE; it is an execution key and never changes results.
- **Presets** `earth | moon | mars_near | mars_mean | mars_far` move only silos that
  define `latency_presets_s` (mars_transit): 0 / 1.3 / 182 / 751 / 1337 s. The
  other silos keep their topology latency. The topology file is unchanged.
- **Link parameters** (`CommsConfig`, defaults; scenario assumptions unless sourced):
  - Bandwidth 256 kbps, jitter 0.05 s, compute 0.01 s per example per epoch.
  - Outage profiles:

    | Profile | Silo | Model | Basis |
    |---|---|---|---|
    | `rare` | esa_analog | poisson, 1 per day, 450 s | 450 s = FLUID ~7.5 min LOS, the calibration point |
    | `short_occasional` | iss_commercial | poisson, 1 per 92 min, 450 s | ISS orbital period |
    | `periodic` | lunar_habitat | 48 min of every 2 h | Apollo-era far-side LOS; confirm before poster |
    | `long_blackout` | mars_transit | poisson, 1 per 6 h, 2 h | scenario |

  - Partial YAML overrides of a profile keep the other defaults.
- **Server, simulated time, both paths deterministic.**
  - **Setup:** the stats and scaler exchanges are timed. The federation starts when
    every reachable silo has the scaler (`comms.setup_sim_s`).
  - **Sync (FedAvg/FedProx):** a round lasts until the last update arrives. With
    `comms.sync_deadline_s` > 0, silos that miss it are dropped for that round.
    Without a deadline, a silo that can never be reached **stalls** the federation:
    the run stops cleanly with `arms.federated.stalled = {round, unreachable, reason}`
    and the notes say so. Even a stall before round 1 is a result (final = the
    untrained model), not a crash.
  - **FedBuff (`fl.strategy: fedbuff`), Nguyen et al. 2022.** Updates are queued by
    simulated arrival. Every `buffer_k` (default 3) updates, the server applies
    w += server_lr · Σ (n_i/Σn) · (1/√(1+staleness_i)) · Δ_i, and each arriving silo
    is immediately re-sent the current model.
    - It runs ceil(rounds × n_silos / K) aggregations, which matches the total client
      updates of the sync run.
    - A silo that can never be reached simply stops contributing (`lost_dispatches`).
- **Flower still carries every real training message.** Sync runs with comms on
  learn exactly as with comms off: identical weights, only time is added. This is
  tested.
- **Result fields:**
  - `rounds[i].sim_wallclock_s` (cumulative simulated time).
  - `comms = {preset, los_events[], links, setup_sim_s, sim_end_s, ...}`.
  - `rounds_to_target.*.sim_wallclock_s`.
  - Federated `sim_wallclock_s_total`, `stalled`; FedBuff `dispatches`,
    `target_aggregations`.
  - Summaries and `grid --report` show simulated time-to-target and stall counts.
  - Run ids of every comms-off config are unchanged: the `comms`, `buffer_k` and
    `server_lr` keys are left out of the hash when unused.
- **Grid:** explicit `variants:` lists for non-cartesian comparisons, and a
  duplicate-name check.
- **Configs:**
  - `experiments/wesad_comms.yaml`: arms centralized + federated, so the target is 95%
    of centralized accuracy. local_only has no communication.
  - `experiments/smoke_comms.yaml`.
  - `sweeps/phase6_presets.yaml` (`p6pre`): 4 presets × {fedavg, fedbuff} × 5 seeds,
    LOS on.
  - `sweeps/phase6_dark.yaml` (`p6dark`): Mars-far, LOS on, mars_transit dark forever
    from round 16; {fedavg no deadline, fedavg 1 h deadline, fedbuff} × 5 seeds.
- **Tests:** `tests/test_comms.py`, 26 tests (link model, outage independence, presets,
  validation, FedBuff math, sync equivalence, dark-silo behaviour for all three
  strategies, byte-identical comms run).

### Decisions made (review these)
1. **Store-and-forward on outage, never drop.** Real deep-space links queue (DTN).
   Losses come only from permanent darkness or a sync deadline.
2. **Late sync updates are discarded**, and the late silo is re-sent the next round's
   model as if idle. A simplification: in reality it would still be uploading.
3. **Bytes per sync round** = delivered downloads + participating uploads, × payload.
   **Per FedBuff aggregation** = 2 × K × payload.
4. **The per-silo "evaluate" exchange is diagnostic and not timed**, and only
   participants are evaluated.
5. **FedBuff is not combined with DP** (a fast silo trains more often, so per-silo
   budgets cannot be fixed in advance) and is **not attacked** (the attack is wired for
   synchronous updates). Both are rejected by config validation.
6. **The sync deadline of 1 h in `p6dark` was fixed a priori:** above the ~45 min
   Mars-far round trip, below one 2 h outage.
7. **Setup waits for every reachable silo**, including Mars, so it costs the same for
   sync and FedBuff.
8. **The Mars `mean` preset** (751 s) = 225 × 10⁶ km ÷ c ≈ 750.5 s (12.5 min). 225 million km
   is the widely quoted "average Earth–Mars distance" (secondary sources, e.g. Space.com,
   Universe Today); no primary NASA/JPL page was found for it (2026-10-29 search). It may
   not be a true time average of the separation. For the poster, compute it from a JPL
   Horizons ephemeris or label it "commonly quoted average". The near/far presets
   (182 s / 1337 s) match the CLAUDE.md figures (3 min 2 s, 22 min 17 s).

### Acceptance results (re-run 2026-10-29; 55 runs, no failures, ~55 min laptop CPU incl. one pause)
Run dirs: `runs/p6pre-*` and `runs/p6dark-*` with the current hashes; tables in
`runs/p6pre-grid.json` and `runs/p6dark-grid.json`. The first-pass dirs (2026-10-22/23,
other hashes) are still in `runs/` and are superseded.
WESAD by_subject, MLP, 30 rounds × 1 local epoch, 5 seeds, loss of signal on.
Simulated seconds are mean ± std over seeds. `fed acc` is final holdout accuracy.
Target = `absolute_target` 0.673. The centralized-relative target (95% of 0.868 ≈ 0.82)
is reached in 0/5 runs of every variant, as in the first pass.

**`p6pre` is identical to the first pass**, in every simulated total and every accuracy.
This is consistent with the buffer rules never firing here: **0 drops in all 40 runs**
(no supersede, no expiry).

| Variant | Simulated total (s) | Time to 0.673 (s) | fed acc | fed F1 | Updates per silo (150 total) |
|---|---|---|---|---|---|
| earth, FedAvg | 257 ± 200 | 155 ± 221 | 0.708 ± 0.074 | 0.668 ± 0.070 | all 5, every round |
| earth, FedBuff | 108 ± 9 | 29 ± 32 | 0.738 ± 0.048 | 0.682 ± 0.051 | all 5; mars_transit most (46–56); iss only 6 in seed 1 |
| moon, FedAvg | 257 ± 200 | 155 ± 221 | 0.708 ± 0.074 | 0.668 ± 0.070 | all 5, every round |
| moon, FedBuff | 131 ± 14 | 33 ± 33 | 0.732 ± 0.068 | 0.681 ± 0.063 | all 5; mars_transit 25–32 |
| mars_near, FedAvg | 23,284 ± 5,045 (6.5 h) | 8,059 ± 7,406 (2.2 h) | 0.708 ± 0.074 | 0.668 ± 0.070 | all 5, every round |
| mars_near, FedBuff | 692 ± 0.3 (546 is setup) | 570 ± 20 | 0.745 ± 0.072 | 0.695 ± 0.066 | **mars_transit 0 in every seed** |
| **mars_far, FedAvg** | **111,020 ± 7,788 (30.8 h)** | **44,627 ± 27,761 (12.4 h)** | 0.708 ± 0.074 | 0.668 ± 0.070 | all 5, every round |
| **mars_far, FedBuff** | **4,850 ± 31 (4,681 is setup)** | **4,716 ± 42 (1.31 h)** | 0.728 ± 0.076 | 0.679 ± 0.072 | **mars_transit 0 in every seed; iss_commercial 0 in seeds 0–1** |

- The target is reached in 5/5 runs of every `p6pre` variant.
- Sync FedAvg reaches it at the same round under every preset (rounds 3, 17, 2, 19, 16
  for seeds 0–4), because comms does not change learning. Only the time differs.
- **The time-to-target spread is bimodal, not noise around a mean.** 0.673 sits close to
  the FedAvg plateau (0.708), so seeds cross it either early (rounds 2–3) or late
  (16–19). Report per-seed values or a time-to-accuracy curve over thresholds, not just
  mean ± std (Phase 8).
- Mars-far, sync vs FedBuff: **9.5× sooner to target, 23× sooner to finish** in
  simulated time.

Dark silo (`p6dark`: Mars far, LOS on, mars_transit dark forever from round 16):

| Variant | Outcome |
|---|---|
| FedAvg, no deadline | **stalled 5/5, exactly at round 16** (rounds 1–15 done, onset 51,963–70,496 s). No crash. Simulated total 60,421 ± 7,536 s (16.8 h). Final (round-15) acc 0.680 ± 0.040, F1 0.543 ± 0.049. Target reached in 2/5 before the stall. 0 drops |
| FedAvg, 1 h deadline | all 30 rounds, 103,388 ± 1,618 s (28.7 h). acc 0.708 ± 0.069, F1 0.668 ± 0.060. Target 5/5, 38,282 ± 24,287 s. **Mars was in 8–12 of the 15 rounds before onset**: outages made it miss rounds before it went dark. Rounds 16–30 each cost the full 1 h deadline |
| FedBuff | **identical to the Mars-far FedBuff run in `p6pre`.** The onset resolved correctly (4,755–4,785 s, after aggregation 25), but mars_transit had already contributed 0 updates: its only dispatch, at setup, is recorded as lost. **This test is still uninformative for FedBuff,** now because latency had already excluded the dark silo, not because of timing |

**Buffer drops (expiry check):** all drops in the 55 runs are mars_transit messages in the
1 h-deadline dark runs. Per seed, 10 downlink models expired (plus 2 uplinks in seed 2),
with 1–3 superseded downlinks and 0–2 superseded uplinks.
- **Supersede** fired before onset, when Mars missed a round and the next model replaced
  the one still waiting behind an outage.
- **Expiry** fired only for messages handed to the dark link. A message whose arrival is
  already known to be > 6 h away is marked expired at hand-over, so the next model does
  not supersede it. The log records it at hand-over + 6 h, and drops dated after the end
  of the run are not listed. That is why there are 10, not 15.
- **Expiry never fired under the defaults without permanent darkness.** This confirms
  the Oct 23 expectation.

| Check | Result |
|---|---|
| Time-to-target in simulated wall-clock per preset | **Met** with the absolute target 0.673 (5/5 in every variant; table above). The centralized-relative target is never reached (Phase 3 H1 gap). Caveat: the target is derived from the Phase 3 FedAvg mean, so sync reaching it is expected; the measured quantity is time. |
| Sync vs buffered-async under Mars-far with LOS | **Done.** FedBuff reaches the target 9.5× sooner, **because it leaves out the slow silos**: mars_transit contributes 0 updates in every seed, and iss_commercial 0 in 2/5 seeds. This is not a free speed-up. |
| A silo going dark mid-run does not crash | **Met for all three strategies**, 15/15 runs, plus tests. Sync without a deadline stops cleanly at the onset round with `stalled` recorded. |

### Findings and caveats (recorded, not tuned)
- **FedBuff's speed comes from exclusion.** Under Mars-far, FedBuff completes its
  matched number of updates (50 aggregations × 3) in ~170 s after setup. That is less
  than one Mars round trip (~45 min), so Mars data can never enter the model. The same
  holds under Mars-near (146 s of training vs a ~6 min round trip). Accuracy is similar
  here only because by_subject labels are near-IID; under label skew this would be a
  fairness problem for the far silo.
- **Short FedBuff runs make participation depend on the outage schedule.** The whole run
  after setup is shorter than one ISS outage (450 s). So whether iss_commercial
  contributes at all depends on whether its link happens to be down at the start: 0 in
  2/5 Mars-far seeds, 6 in one earth seed.
- **A deadline without failure detection wastes time on a dead silo.** After onset,
  every round of the 1 h-deadline run waits the full hour for mars_transit: rounds 16–30
  are ~15 h of the 28.7 h run. A rule like "exclude after N consecutive misses" would
  remove that cost. It is not implemented; noted for Phase 7 or the poster discussion.
- **The compute model shapes participation.** At 0.01 s per example, the smallest
  silo (mars_transit) is the fastest under the earth preset and contributes the most
  updates.
- **Setup dominates FedBuff's time.** The federation waits for the Mars scaler
  exchange (~4,680 s) before training. An async system could start without Mars.
- **Caveats of the dark-silo sweep:**
  - mars_transit holds only 8% of the data, so losing it barely moves accuracy;
    losing nasa_earth (40%) would be the harsh test.
  - Darkness is permanent and single; rejoining is not tested.
  - The no-deadline stall shows a failure mode, not an operating mode.
  - The 1 h deadline was fixed a priori.
  - Dark onset is now by round (16), which is mid-run for both sync and FedBuff. The
    FedBuff case is still uninformative, because the dark silo is mars_transit, which
    FedBuff never hears from under Mars-far. An informative FedBuff dark test would darken
    a silo that contributes, e.g. nasa_earth (40% of the data). That needs a new variant
    fixed a priori; awaiting the user.
  - If a silo were dark from t = 0, its standardisation stats would still reach the
    scaler, because the real Flower exchange is not gated.
- **Caveats of FedBuff:**
  - Fast silos dominate, and staleness weighting further down-weights slow ones.
  - Updates are matched in total, not per silo.
  - K = 3, the staleness function and server_lr = 1 are fixed and untuned.
  - No DP and no attack.
  - Its "rounds" are aggregations, so compare on simulated time.
  - No retry when a silo goes silent.
  - Server compute and busy clients are not modelled.
- **Sync deadline trade-off:** discarding late updates bounds round time but
  systematically under-represents the outage-prone silo and wastes its compute.
  Applying them late would need staleness weighting (semi-async); waiting means
  2 h+ rounds and a stall when a silo goes dark.

### Store-and-forward buffer rules (user decision 2026-10-23, implemented)
- **One sender buffer per link direction** (`comms.supersede`, `comms.ttl_s`,
  `comms.cache_bytes`).
- **Supersede (on):** a newer message replaces one still waiting in the buffer.
- **Expiry (`ttl_s` = 6 h):** a message that would arrive more than 6 h after
  hand-over is dropped (DTN bundle lifetime). 6 h exceeds the longest modelled outage
  (2 h) plus Mars-far light time; it is a scenario assumption.
- **Cache cap (off; `cache_bytes: 0`):** a message that must wait and does not fit is
  dropped (drop-tail).
- **Every drop is logged** in `comms.drops` (silo, direction, reason, handed/dropped
  time), and the settings in `comms.buffer`.
- **A dropped message counts as never arriving.** Sync with a deadline drops that silo
  for the round; sync without one stalls (reason now names expiry/drops); FedBuff
  records a lost dispatch.
- **Why no cache-full rule alone:** with supersede, a buffer holds at most one ~22 KB
  message per direction, so a realistic cap never fills. Expiry is the rule that
  actually loses data in DTN.
- **Scope:** runs with comms on only. Comms-off run ids and results are unchanged
  (Phase 4 reports verified). Later phases only see additive `comms` fields.
- **Tests:** supersede, no supersede once sent or when disabled, expiry (outage and pure
  light time), cache cap, validation, and end-to-end with a TTL below the Mars-far light
  time (every Mars message expires, the federation continues). 32 comms tests pass.
- **Confirmed by the 2026-10-29 re-run:** under the defaults, expiry fires only for
  messages sent to a permanently dark silo (see the acceptance results).

### Dark onset by round and a reachable target (proposals 2 and 3, implemented 2026-10-26)
- **`comms.dark.from_round`** (exclusive with `from_s`; needs `dark.silo`; <= `fl.rounds`).
  - Sync: the silo goes dark when round `from_round` is sent.
  - FedBuff: it goes dark once (from_round − 1) × n_silos client updates have been
    aggregated, rounded down to whole aggregations of K, i.e. the same share of the
    matched updates.
  - **How the time is found.** The server runs a timing-only pilot of the same run,
    without the outage (`_dark_onset_s`, `_TimingCtx`: no Flower traffic, no training).
    Simulated time depends only on links, payload sizes and n_train, never on what is
    learned, and nothing before the onset can depend on it. So the pilot's schedule up
    to the onset is the real one. The window is then set on the link before any
    simulated transfer, and messages in the buffer at onset are blocked correctly.
  - If the pilot stops before the onset, the silo never goes dark (`from_s: null`).
  - **Recorded:** `comms.dark = {silo, from_round, from_s (resolved), duration_s}`.
- **`p6dark` now uses `comms.dark.from_round: 16`** (of 30, the midpoint, fixed a
  priori). For FedBuff this is after 25 of 50 aggregations. The 12 h absolute onset is
  gone.
- **`wesad_comms.yaml`: `fl.absolute_target: 0.673`** = round(0.95 × 0.708, 3). 0.708 is
  the Phase 3 by-subject FedAvg mean final accuracy. It sits alongside the
  centralized-relative target.
  - Caveat to state on the poster: sync FedAvg with comms on learns exactly as in
    Phase 3 (tested). Its reaching this target is expected; the measured quantity is
    the simulated time to get there.
- **Run ids:** every comms-on config gets a new hash (the new `dark.from_round` key, and
  `absolute_target` for the WESAD comms configs). Comms-off ids are unchanged.
- Tests added: validation (3 cases), the link builder leaves the window to the server,
  sync onset = start of round `from_round` with a schedule identical to the no-dark run
  before it, FedBuff onset = time of aggregation (r−1)·n//K with an identical schedule
  before it.
- **Full suite: 173 passed, 0 failed** (pytest exit 0, JUnit counts; 2026-10-26, ~8.5 min on
  the laptop). One earlier comms run hit a Ray startup hang (`query.stats: 0/5 replies`)
  while a second pytest was being killed alongside it; the test passed on a clean re-run.

### Still open
- **Optional proposal 4:** let FedBuff start training before slow silos finish setup.
  Today setup waits for Mars (~4,680 s of FedBuff's 4,850 s under Mars-far), and it
  costs sync and FedBuff the same. **User decision 2026-10-29: skip for now.**
  - Recommendation (2026-10-29): don't implement. It would not change participation (Mars
    contributes 0 either way under the matched budget), only shrink the headline time. It
    would also mix a startup-protocol difference into the sync vs FedBuff comparison.
    Instead, show setup (`comms.setup_sim_s`) and training as separate segments in the
    Phase 8 figure.
  - If wanted anyway: keep the global scaler (wait for Mars's stats; start the near silos
    once they hold the scaler), as an opt-in key. It invalidates the 25 FedBuff comms runs.
- **Optional:** an informative FedBuff dark-silo variant (darken a contributing silo, e.g.
  nasa_earth, from round 16). 5 runs, a few minutes. Awaiting the user.
- The Mars `mean` preset source: see decision 8.
- Full suite last run 2026-10-26 (173 passed). No src/tests/configs changes since, so it
  was not re-run for the gate.

---

## Phase 5 — Privacy — COMPLETE, attack fairness accepted (2026-10-16/17)

### User decisions applied (2026-10-17)
- delta = 0.1/N_train is accepted.
- Attack fairness is accepted as is.
- **The federated DP utility collapse is reported as is.** No DP-specific retuning.
- **Chance-level attack scores are now in the code.** They will be produced by the GPU
  cluster runs. The local `p5eps` results predate them and are not re-run on the laptop.
- The current focus is completing the code. Publishable runs happen on the cluster.

### Added after the review (2026-10-17)
- **Chance baseline per attack cell and per trial** (`privacy.attack.chance_draws`,
  default 50). Random N(0, 1) guesses, which are the attack's own starting point, are
  scored with the same matching. They appear in summaries and `grid --report`. Cost:
  under 1 s per run. The new key enters the config hash, so cluster attack runs get
  new run ids and cannot be confused with the local pre-chance `p5eps` runs.
  - WESAD check (seed 0, no DP): chance re-identification is 0.013 / 0.005–0.008 / 0.06
    at B = 1 / 4 / 32; the attack's is 1.0 / 0.42–0.5 / 0.13–0.18.
- **Subject-level bound reported next to every record-level epsilon** (`subject_level`
  per model, and `privacy.subject_level` per run). It uses group privacy, (k·eps,
  k·e^((k−1)·eps)·delta), with k = the most windows any one subject has in that
  training set. It is a valid but loose upper bound. `vacuous: true` means its delta
  reaches 1. For WESAD (k ≈ 70) it is vacuous at every epsilon in the sweep.
- Tests: 128 passed.

### Robustness items 2, 4 and 6 added (user request, 2026-10-17)
- **Item 2, unknown labels.** `privacy.attack.label_modes`, default `[known, inferred]`.
  - `inferred` recovers the batch's label counts from the released update's
    last-layer bias gradient: n_c = B·(p̄_c − g_c), with p̄ estimated from the model's
    mean prediction on random inputs (Geng et al. 2021 style), rounded by largest
    remainder.
  - Each trial's single released update is attacked under both modes, a paired
    comparison. `label_accuracy` (multiset overlap) is reported per trial and cell.
  - Under DP the attacker uses the same formula; it cannot undo clipping or noise.
- **Item 4, every silo.** `privacy.attack.silos`, where empty = all silos (the default).
  - Result layout changed: `privacy.attack.silos.<name>.cells[]`, each cell keyed by
    (state, batch_size, labels).
  - Headline `privacy.attack.mse` = last model state, smallest batch, known labels, on
    the silo where the attack did best. That is the worst case for any client;
    `headline` names it.
  - Batch draws are seeded per (seed, silo, batch size, trial).
- **Item 6, more trials and confidence intervals.**
  - Summaries now give, per attack cell, a hierarchical percentile bootstrap 95% CI
    (`ci95`, `chance_ci95`; resample seeds, then trials within a seed; 2000 draws;
    fixed rng). This is next to mean ± std over seeds. `metrics/aggregate.bootstrap_ci`
    is reusable for the Phase 8 figures.
  - New sweep `configs/sweeps/phase5_eps_cluster.yaml` (`p5epsc`): the same 8 variants
    with `privacy.attack.trials: 10`, i.e. 50 trials per cell across 5 seeds.
  - New grid key `common:` applies overrides to every variant, the `include_base` one
    included (`fixed:` does not touch the base).
- **Attack batches are seeded by topology position** (seed, silo position, batch size,
  trial). A silo's attacked records therefore do not depend on which other silos are
  selected. Regression test: `test_attack_batches_do_not_depend_on_silo_selection`.
- **WESAD check, seed 0.** Untrained global weights were used for both "states", so
  this is a timing and sanity check, not a result. Values are means of 3 trials.

  | | B=1 re-identified | B=4 re-identified | B=32 re-identified | B=32 chance |
  |---|---|---|---|---|
  | No DP, known labels (5 silos) | 0.67–1.0 | 0.25–0.67 | 0.19–0.24 | 0.04–0.11 |
  | No DP, inferred labels | 0.67–1.0 | 0.08–0.67 | 0.17–0.24 | same |
  | epsilon = 1, known labels | 0 (one silo 0.33, 1 of 3 trials) | 0 | 0.06–0.13 | 0.04–0.11 |

  - **Without DP, label inference was 100% correct at every batch size and silo.** The
    leakage with inferred labels matches known labels, so the "labels known"
    assumption does not drive the no-DP result.
  - Under DP, label recovery falls to 0–0.79, and re-identification is at or near
    chance.
  - The one 0.33 (mars_transit, B=1, epsilon = 1) is a single trial out of 3. It is
    the kind of outlier the 10-trial cluster run and the CIs exist to resolve. It is
    not an argument either way.
- **Cost.**
  - All 5 silos x 2 label modes x 18 cells take about 9 min per run without DP and
    about 11 min with DP (133 s for one silo at epsilon = 1, measured as CPU time ≈
    wall time).
  - One silo with one label mode stays near 1 min, so the < 2 min acceptance check
    still holds per attack.
  - A first epsilon = 1 timing of 88 min was the laptop sleeping: the check ran without
    `caffeinate`. Wall time was far above CPU time, and rerunning under `caffeinate`
    reproduced the normal cost.
  - **Cluster estimate** for `p5epsc` (10 trials): ~30–40 min of attack + ~1 min of
    training per run, i.e. ~3 h per variant job of 5 seeds; 8 jobs in parallel.
    - The attack is single-threaded per silo. Parallelising silos across processes is
      a possible later speed-up, not done.
    - To shorten it, lower `privacy.attack.trials` in `common:`.
- **Superseded:** the local `p5eps` result files use the earlier single-silo,
  known-labels layout without chance scores. Their numbers in the tables below remain
  valid as recorded, but the new summary code no longer reads their attack section.
  The cluster `p5epsc` run replaces them.
- Tests: 135 passed.

### Per-window guarantee vs the alternatives
| | Record-level (implemented) | Subject-level DP | Silo-level DP (DP-FedAvg) |
|---|---|---|---|
| Unit protected | one 60 s window | all of one crew member's data | a whole silo's data |
| Question it answers | was this window used in training? | was this person in the training data? | did this agency take part at all? |
| Mechanism | DP-SGD, per-window clipping (Opacus) | per-subject clipping: sample subjects, clip each subject's summed gradient, add noise | the server clips each silo's update and adds noise once per round |
| Units available per silo | 70–346 windows | 1–5 subjects | 5 silos in total |
| Expected utility here | already collapses for the federated arm | worse: noise must hide one person among 1–5 | not workable: needs hundreds or thousands of clients |
| Support | Opacus, implemented | custom code; Opacus has no per-subject clipping | custom code; the server must be trusted to add noise |
| Honest claim | strong per window; for a subject only via group privacy, which is vacuous at k ≈ 70 | the claim that matters medically | the claim that matters to an agency |

Recommendation: keep record-level DP, which is what is implemented. On the poster,
state the unit, and quote the reported group-privacy figure for subject-level
protection. Name subject-level DP as future work that needs far more subjects per silo.

### Who the simulated attacker is
- **An honest-but-curious (semi-honest) aggregation server, or anyone who can read
  the update on the link.** It follows the protocol exactly: it sends the real global
  model, aggregates correctly, and does not tamper with or disrupt anything.
- **Its goal is confidentiality.** It reconstructs a silo's private training records
  (crew physiological windows) from an update it legitimately receives.
- **What the attacker is not doing:**
  - It is not hacking into a silo.
  - It is not degrading or invalidating the service (no availability or integrity
    attack).
  - It is not poisoning or backdooring the model.
  - It is not modifying the global model to make it leak more (a malicious server, as
    in "Robbing the Fed").
- **It targets the data, not the training.** Training proceeds normally; the leak is a
  side effect of what an update reveals.

### Making the attack more trustworthy to a reviewer
Code-only items can be added before the cluster run; each is a configurable option.
1. **Chance baselines in every result.** Done (above).
2. **Unknown-label variant.** Infer labels (iDLG for B=1, bias-gradient counts for B>1)
   instead of assuming them. This removes the "labels known" objection.
3. **Attack the real multi-step FedAvg update.** Invert w_before − w_after by unrolling
   the client's local steps on the dummy batch. This removes the "single step"
   objection. It is the biggest change and the slowest attack.
4. **Attack every silo, not only the largest.** Small silos have more DP noise but fewer
   records to hide among.
5. **Random restarts, keeping the restart with the lowest gradient-matching loss.**
   Selection uses only what the attacker sees, never ground truth.
6. **More trials on the cluster** (`privacy.attack.trials: 10`) and bootstrap
   confidence intervals in the figures.
7. **Freeze the attack settings before the cluster run**, as a pre-registration. Publish
   the code, seeds, every trial and the record indices. Most of this is already
   stored.
8. **Complementary membership-inference test** (loss-threshold attack). It measures the
   threat DP's epsilon is actually defined against.
9. **Implementation sanity check.** Reproduce full recovery in the textbook setting
   (B=1, untrained network). This exists as `test_attack_recovers_single_record_from_untrained_mlp`.

Recommendation: 2, 4 and 5 are cheap and close the obvious objections; do them before
the cluster run if wanted. Treat 3 and 8 as stretch.

**Not committed.** The user commits. Suggested message:
`feat(privacy): phase 5 DP-SGD (Opacus) and gradient-inversion attack, epsilon sweep`.

### Built
- **`privacy/dp.py`**: record-level DP-SGD through Opacus 1.5.2, RDP accountant.
  - Applied through the shared `train_local(dp=...)`, so all three arms run DP when
    `privacy.enabled` is set. Each trained model has its own budget over its own data:
    the centralized model, each local-only model, each federated client.
  - **delta = 0.1 / N_train** per model (`privacy.delta_factor`, validated < 1), which
    keeps delta strictly below 1/N_train (CLAUDE.md section 2).
  - **Calibration:** `make_private_with_epsilon` is `get_noise_multiplier` +
    `make_private`. They are split because a federated client trains once per round with
    a fresh optimizer, while its budget must cover all rounds. Sigma is calibrated once
    for rounds x local epochs x steps per epoch, and epsilon spent is the closed-form RDP
    composition. A test checks equality with Opacus's `make_private_with_epsilon` +
    `get_epsilon` (relative error < 1e-9).
  - The per-silo epsilon spent is recorded every round
    (`rounds[i].per_silo[silo].epsilon`). `result.privacy` holds the per-model
    accounting plus `epsilon`/`delta` of the weakest model in the run.
  - `result.privacy.not_covered` states the gaps: no subject-level guarantee; the
    shared count/sum/sum-of-squares statistics; per-silo record counts (FedAvg weights).
  - Models are already Opacus-valid (GroupNorm, no BatchNorm); `ModuleValidator` reports
    no errors for the MLP or the CNN, so no `ModuleValidator.fix` is needed.
- **`privacy/attack.py`**: gradient inversion against one client's single-step update.
  - Cosine gradient matching (Geiping et al. 2020) with a weak Gaussian prior on
    standardised inputs, Adam, 1000 iterations, step-decayed learning rate.
  - The victim gradient is computed in training mode (dropout on, mask unknown to the
    attacker). With DP it is exactly Opacus `DPOptimizer.pre_step()` output: clip,
    noise, average.
  - Grid: model state {init, final} x batch size {1, 4, 32} x 3 seeded batches. Every
    trial is stored, with its record indices. Batches depend only on the seed, so
    every epsilon variant attacks the same records.
  - Scores: matched MSE, rel_mse (vs guessing the mean), correlation (per channel for
    sequences), and nn_identified (the reconstruction's nearest silo record is the true
    one).
  - Attack target: the first silo (`nasa_earth`), the largest, so the least DP noise.
- **Config:** `privacy:` block (`enabled`, `target_epsilon`, `delta_factor`,
  `max_grad_norm`, `attack:`). Bools are now supported by the strict loader. When
  unused the block is left out of the experiment hash, so every pre-Phase-5 run id is
  unchanged (verified: Phase 4 `--report` still finds all summaries).
- **Rejected combinations:** DP + FedProx (Opacus would silently drop the proximal
  gradient); DP + balanced class weighting (it breaks the per-sample sensitivity bound).
- **Grid:** `include_base: true` adds the base experiment unchanged as variant 0.
  Summaries and `--report` now carry epsilon spent and per-cell attack scores.
- **Configs:** `experiments/wesad_privacy.yaml` (no DP + attack), `experiments/smoke_dp.yaml`,
  `sweeps/phase5_eps.yaml` (base = epsilon inf, plus epsilon in {0.5, 1, 2, 3, 5, 10, 13}).
- **Tests:** `tests/test_privacy.py`, 22 tests. Full suite: **127 passed**.

### Bugs found and fixed during Phase 5
- **Denormal floats.** Matching a noise-dominated DP gradient drove the attack into
  float32 denormals: 333 s instead of 0.16 s for one trial. Flush-to-zero is turned on
  for the attack only and switched off afterwards.
- **Thread-count leak.** The attack's `torch.set_num_threads(1)` persisted into the next
  run in the same process and changed float reduction order, so two identical runs were
  not byte-identical. The thread count is now restored. Regression test:
  `test_dp_run_is_byte_identical`.

### Acceptance results
`p5eps` sweep: WESAD by_subject, MLP, 30 rounds x 1 local epoch, holdout subjects
13/16/17, 5 seeds per variant, 40 runs, no failures. Values are mean ± std over seeds.

**Attack: trained (`final`) global model, silo nasa_earth (~345 records).**
Each seed's value is the mean of 3 trials.

| epsilon | B=1 rel_mse | B=1 re-identified | B=4 rel_mse | B=4 re-identified | B=32 rel_mse | B=32 re-identified |
|---|---|---|---|---|---|---|
| inf (no DP) | **0.079 ± 0.042** | **0.93 ± 0.15** | 0.62 ± 0.09 | 0.45 ± 0.08 | 0.97 ± 0.07 | 0.15 ± 0.04 |
| 13 | 2.04 ± 0.60 | 0 | 2.08 ± 0.30 | 0 | 3.42 ± 0.86 | 0.07 ± 0.02 |
| 10 | 2.10 ± 0.68 | 0 | 2.22 ± 0.39 | 0 | 3.49 ± 0.86 | 0.05 ± 0.04 |
| 5 | 2.39 ± 0.72 | 0 | 2.52 ± 0.39 | 0 | 3.90 ± 1.09 | 0.05 ± 0.01 |
| 3 | 2.48 ± 0.81 | 0 | 2.75 ± 0.45 | 0 | 4.17 ± 1.19 | 0.06 ± 0.04 |
| 2 | 2.58 ± 0.79 | 0 | 2.82 ± 0.48 | 0 | 4.42 ± 1.32 | 0.05 ± 0.01 |
| 1 | 2.82 ± 0.80 | 0 | 3.00 ± 0.56 | 0 | 4.59 ± 1.30 | 0.05 ± 0.02 |
| 0.5 | 2.92 ± 0.93 | 0 | 3.09 ± 0.66 | 0 | 4.63 ± 1.32 | 0.05 ± 0.02 |

On the untrained (`init`) model without DP: B=1 rel_mse 0.23 ± 0.11, re-identified
0.87 ± 0.18; B=32 0.70 ± 0.03 and 0.18 ± 0.04. Full grid: `runs/p5eps-grid.json`.

**Chance levels**, measured separately (random N(0,1) guesses through the same matching
and scoring, 200 draws x 5 seeds, scratchpad script, not in result files):
- rel_mse ≈ 2.8 at B=1, ≈ 2.1 at B=4, ≈ 1.7 at B=32.
- re-identified ≈ 0 at B=1, ≈ 0.007 at B=4, **≈ 0.04–0.05 at B=32.** Hungarian
  matching inflates this well above 1/345.

| Check | Result |
|---|---|
| Attack visibly succeeds at epsilon = inf, small batch | **Yes.** Trained model, B=1: rel_mse 0.079, window re-identified in 93% of trials. B=4: 45%. Even B=32 (the real training batch): 15%, about 3x chance. |
| Reconstruction error rises monotonically as epsilon falls | **Yes, in the means**, for every batch size on the trained model (table above). **But** every DP setting already puts the attack at or below chance: re-identification is 0 at B=1 and B=4 and at chance at B=32. The differences between epsilon values sit inside the seed std and do not show meaningful protection differences. |
| Utility-vs-epsilon curve over {0.5, 1, 2, 3, 5, 10, 13} | Produced (table below). |
| Attack < 2 min on laptop CPU | **Yes.** ~60 s for the full 18-trial grid (59 s without DP, 64 s with DP), measured directly. |

**Utility vs epsilon** (holdout accuracy; F1 = macro-F1). Majority-class accuracy on the
holdout is 114/212 = **0.538**.

| epsilon | federated acc | federated F1 | local_only acc (silo mean) | centralized acc |
|---|---|---|---|---|
| inf | 0.708 ± 0.074 | 0.668 ± 0.070 | 0.654 ± 0.029 | 0.868 ± 0.022 |
| 13 | 0.579 ± 0.037 | 0.313 ± 0.070 | 0.613 ± 0.056 | 0.693 ± 0.022 |
| 10 | 0.565 ± 0.028 | 0.288 ± 0.056 | 0.597 ± 0.049 | 0.694 ± 0.023 |
| 5 | 0.546 ± 0.016 | 0.251 ± 0.034 | 0.570 ± 0.039 | 0.699 ± 0.027 |
| 3 | 0.546 ± 0.016 | 0.251 ± 0.034 | 0.532 ± 0.032 | 0.681 ± 0.046 |
| 2 | 0.548 ± 0.018 | 0.256 ± 0.040 | 0.505 ± 0.034 | 0.634 ± 0.022 |
| 1 | 0.540 ± 0.017 | 0.274 ± 0.058 | 0.470 ± 0.051 | 0.575 ± 0.043 |
| 0.5 | 0.514 ± 0.037 | 0.299 ± 0.051 | 0.426 ± 0.063 | 0.556 ± 0.040 |

### Key findings (recorded, not tuned)
- **Under DP the federated model collapses to the majority class.** At epsilon ≤ 10 it
  predicts "baseline" for nearly every window (per-class recall [1.0, 0–0.12, 0.0]).
  That is why epsilon 3 and 5 give identical numbers: same collapse, same holdout.
  Centralized DP keeps 0.63–0.70 accuracy for epsilon ≥ 2.
- **H1 under DP:** federated beats the local-only silo mean at epsilon ≤ 3 (4–5/5 seeds),
  but only because local-only degrades faster. It loses at epsilon 5–13 (2/5 seeds).
  Neither is a useful model at epsilon ≤ 3.
- **Likely causes, not investigated or tuned:**
  - Every silo adds its own noise for its own budget. Silos are small (70–346 records),
    so sigma is 1.5–2.5 even at epsilon 5.
  - Adam is reset every round and only 3–11 noisy steps run per round. Adam normalises
    noise into full-size steps.
  - Training hyperparameters were set for non-DP training (batch 32, lr 1e-3, clip 1.0,
    30 epochs).
- **Leakage without DP is real but batch-dependent.** It is near-total at B=1, partial
  at B=4, and modest at B=32. The B=1 headline must always be shown next to B=32.

### Run cost
40 runs, 96 min of compute on the laptop CPU. That is 86–95 s per run without DP and
98–178 s with DP; the attack accounts for ~60 s. The wall-clock span was longer
(19:19–23:26), which suggests sleep or throttling despite `caffeinate -i`.

### Decisions made (review these)
1. **Unit of privacy: one record (a 60 s window), per trained model.** This is not
   subject-level privacy. A subject contributes ~70 windows that overlap by 50%, so by
   group privacy the subject-level epsilon is far larger. Say so on the poster.
2. **delta = 0.1/N_train**, to satisfy CLAUDE.md section 2's strict delta < 1/N. Phase 5's
   text says delta = 1/N_train, which is not strictly below 1/N.
3. **RDP accountant**, not Opacus's default PRV. RDP is the standard, slightly
   conservative choice. Opacus sometimes warns that "optimal order is the largest
   alpha", which means the bound may be loose: the reported epsilon over-states the
   spend, the safe direction.
4. **DP is applied to all three arms** at the same target epsilon, so the three-way
   comparison stays like-for-like.
5. **Attack hyperparameters** (cosine loss, prior weight 0.01, 1000 iterations, lr 0.1)
   were chosen on seed 0 without DP, by comparing cosine vs L2 matching and prior
   weights {0, 0.001, 0.01}. They were picked to make the attacker strong; this was
   never tuned against DP.
6. **Labels are known to the attacker** (best case for the attacker). For B=1 they are
   recoverable from the last-layer bias gradient anyway.
7. **Sigma for the attacked step is recalibrated per batch size** for the same epsilon
   and epochs, so B=1 is not judged against batch-32 noise.
8. **Headline `privacy.attack.mse`** = trained model at the smallest batch, which is
   the worst case for the client. The full grid is always stored next to it.
9. **The attack is on a single-step (FedSGD-style) update**, not the multi-step FedAvg
   update the federated arm really sends. This is stated in `threat_model`.

### For the attack-fairness review (CLAUDE.md gate)
CLAUDE.md lists three objections; each is handled as follows.
- **"Batch size 1":** B=4 and B=32 are reported beside it, and B=32 is the real
  training batch. Leakage at B=32 is modest (15% re-identified vs ~5% chance).
- **"Untrained net":** the trained global model is attacked too, and leakage is not lower.
- **"Cherry-picked sample":** batches are seeded draws, all trials are stored, and the
  record indices are in the result.

Remaining weaknesses a judge could raise:
- The attack uses a single step, not a multi-step FedAvg update.
- Labels are assumed known.
- Chance baselines live only in this log. Recommendation: add them to the attack
  output, and re-run the sweep before the figures.

### Cluster run for Phase 5 (publishable numbers)
- `fedastro grid configs/sweeps/phase5_eps_cluster.yaml --index i --output-dir <dir>`,
  i = 0..7, one job per variant, 5 seeds each. Cost per run: see the timing below.
- Then `fedastro grid configs/sweeps/phase5_eps_cluster.yaml --report --output-dir <dir>`.
- Chance baselines are included automatically.
- Use a fresh output dir. With the new config hash, old runs would not be resumed
  anyway.
- The DP runs are CPU-bound (small MLP, Opacus per-sample gradients). A GPU does not
  speed them up; many CPU cores in parallel jobs do.

### Deferred robustness items (user decision 2026-10-22: keep for later)
- **3, attack the real multi-step FedAvg update.** No later phase depends on it.
- **5, random restarts, keeping the best attacker-side loss.** No later phase depends
  on it.
- **7, freeze attack settings before the publishable cluster run (pre-registration).**
  Not code. **It must be done before the Phase 5 cluster run**: record the frozen
  `privacy.attack` block and the git sha here.
- **8, membership-inference test.** No later phase depends on it. It could feed an
  optional Phase 8 dashboard panel.
- **Dependency found in the check, and done now:** the Phase 8 "attack before/after"
  panel needs stored reconstructions, and results held only scores. Each attack cell
  now stores one `example` (true record plus its matched reconstruction, rounded to 5
  decimals). It is chosen by a fixed rule, the lowest record index in the trial-0
  batch, never by quality. Cluster results will therefore feed Phase 8 without a
  re-run.

### Plan for Phase 6 (deep-space comms), not started
1. **`fl/comms.py`: a per-silo link model in simulated time.**
   - One-way delay from the silo config, with Mars `near`/`mean`/`far` presets.
   - Jitter.
   - A bandwidth cap, so payload bytes become transfer time.
   - Loss-of-signal windows with configurable probability and duration, calibrated to
     the ~7.5 min FLUID pause.
   - Seeded and deterministic. It never sleeps (`--real-sleep` exists only for the
     demo).
2. **Fill `sim_wallclock_s` per round and `comms.los_events`.** Simulated time stays a
   separate field from the real `real_elapsed_s` in the manifest.
3. **Buffered-async strategy (FedBuff-style).** The server aggregates once K updates
   arrive, with staleness-aware weighting. Late or dark silos contribute stale updates
   instead of blocking.
4. **Sync vs buffered-async under Mars-far with LOS.** Report time-to-target accuracy
   in simulated wall-clock per preset (Earth, Moon, Mars-near, Mars-far).
5. **A silo going dark mid-run does not crash the federation.** Covered by a test.
6. **Implementation choice to confirm in Phase 6.** The event-driven schedule (who
   arrives when) is computed in simulated time on the server. Flower still delivers
   messages synchronously underneath, so results stay deterministic and fast.

---

## Phase 4 — Non-IID robustness — COMPLETE, awaiting review (2026-10-15)

**Not committed.** The user commits. Suggested message:
`feat(fl): phase 4 FedProx, Dirichlet alpha sweep, silo fairness, grid sweeps`.

### Built
- **FedProx.** `fl.strategy: fedprox` with `fl.proximal_mu`, default 0.01, fixed
  before any result was seen.
  - The client-side term (mu/2)·||w − w_global||² lives in the shared
    `train_local(proximal_mu=...)`. Server aggregation is FedAvg's.
  - `proximal_mu` is recorded and hashed only when the strategy is fedprox, so FedAvg
    run ids are unchanged.
- **Silo fairness metric** in `metrics/fairness.py`.
  - Definition: holdout per-class recall, re-weighted to each silo's training label
    mix. Spread is max − min across silos.
  - Applied to all three arms. Written to `result["fairness"]` and the summaries.
  - Limitation: it captures label-mix heterogeneity only, not inter-subject feature
    shift. Subject-disjoint per-silo test sets are impossible with 1 to 5 subjects
    per silo.
- **`evaluate()`** now also returns `recall_per_class`.
- **Grid sweeps** in `fedastro/grid.py`, run with `fedastro grid <sweep.yaml>`:
  - `--list`.
  - `--index i ...`, one variant per cluster job.
  - `--resume`, which skips runs that already have a result.
  - `--report`, a cross-variant table plus `<sweep>-grid.json`.
  - A failed run is logged and the sweep continues. The exit code is 1 with a list of
    failures.
  - `load_config(path, overrides)` takes dotted-key overrides, validated as strictly
    as YAML.
- **Sweep files:**
  - `configs/sweeps/phase4_alpha.yaml`: alpha in {0.1, 0.3, 1, 3, 10} x
    {dirichlet, quantity control} x {fedavg, fedprox} x 5 seeds = 100 runs.
  - `phase4_by_subject.yaml`: the no-skew reference.
  - `phase4_mu.yaml`: mu in {0.001, 0.01, 0.1, 1} at alpha = 0.3.
- **`make grid SWEEP=...`** wraps the sweep in `caffeinate -i`.

### Bug found and fixed during Phase 4: Dirichlet prior
- **The first pass drew silo class proportions from a symmetric Dir(alpha·1).** As
  alpha grows, that pulls every silo towards uniform 1/3 proportions, not WESAD's
  natural 54/30/16 mix.
- **This caused a label-prior shift against the holdout even at "low skew."**
  Federated accuracy at alpha = 10 was 0.548 versus 0.681 for the quantity control,
  which exposed it.
- **Fix:** q ~ Dir(alpha·p), with p the training pool's class distribution, as in Hsu
  et al. 2019. alpha → ∞ now recovers the natural mix. A regression test was added.
- **Results archived and re-run.** All symmetric-prior results (143 run directories)
  were moved out of `runs/` into the session scratchpad. Every number below comes from
  the re-run.
- **Superseded tables:** the Phase 1 Dirichlet class tables and the Phase 2
  `wesad_dirichlet` and control tables used the old prior. Re-run those configs if
  needed.

### Acceptance results
5 seeds per variant. Holdout subjects 13, 16, 17. MLP, 30 rounds x 1 local epoch.
Accuracy and macro-F1 are mean ± std across seeds. "Spread" is the fairness spread,
lower is fairer.

**Federated performance versus heterogeneity, FedAvg** (alpha = ∞ is `p4subj-fedavg`):
| alpha | fed acc (Dirichlet) | fed macro-F1 (Dirichlet) | fed acc (quantity control) | fed macro-F1 (control) | central acc (Dirichlet) | local acc (Dirichlet) | spread fed | spread local |
|---|---|---|---|---|---|---|---|---|
| ∞ (by_subject) | 0.708 ± 0.074 | 0.668 ± 0.070 | — | — | 0.868 ± 0.022 | 0.654 ± 0.029 | 0.002 | 0.300 |
| 10 | 0.652 ± 0.096 | 0.524 ± 0.062 | 0.722 ± 0.077 | 0.633 ± 0.106 | 0.838 ± 0.047 | 0.644 ± 0.034 | 0.160 | 0.225 |
| 3 | 0.628 ± 0.118 | 0.489 ± 0.126 | 0.696 ± 0.066 | 0.593 ± 0.104 | 0.750 ± 0.064 | 0.573 ± 0.057 | 0.471 | 0.335 |
| 1 | 0.575 ± 0.055 | 0.384 ± 0.166 | 0.671 ± 0.046 | 0.576 ± 0.077 | 0.647 ± 0.104 | 0.476 ± 0.075 | 0.641 | 0.468 |
| 0.3 | 0.571 ± 0.035 | 0.348 ± 0.106 | 0.685 ± 0.049 | 0.573 ± 0.063 | 0.688 ± 0.105 | 0.509 ± 0.077 | 0.657 | 0.176 |
| 0.1 | 0.492 ± 0.103 | 0.218 ± 0.034 | 0.700 ± 0.077 | 0.547 ± 0.054 | 0.536 ± 0.156 | 0.414 ± 0.070 | 0.995 | 0.176 |

- **Degradation with heterogeneity: PASS.**
  - Federated macro-F1 falls strictly as alpha falls: 0.668, 0.524, 0.489, 0.384,
    0.348, 0.218.
  - Accuracy falls monotonically within noise. 0.3 and 1 are tied at 0.571 and 0.575.
- **The quantity-matched control stays flat**, at accuracy 0.67 to 0.72 and macro-F1
  0.55 to 0.63. The degradation is caused by label skew, not by the windows that
  subsampling removes.
- **Federated beats the local-only silo mean at every alpha**, by +0.008 to +0.099
  accuracy.
  - H1 holds in 3/5 to 5/5 seeds per variant.
  - The smallest margin is at alpha = 10 (+0.008, 3/5 seeds).
- **Centralized also degrades under skew**, from 0.868 at ∞ to 0.536 at alpha 0.1.
  The pooled training mix itself shifts away from the holdout mix.
- **Fairness:**
  - The single global model serves silos evenly at no skew (spread 0.002), but very
    unevenly at strong skew (0.995 at alpha 0.1). It collapses towards the dominant
    pooled class.
  - Local-only models are less even at mild skew. At strong skew they are more even,
    because each silo's own model predicts that silo's own majority class.
  - Federation trades per-silo fit for global accuracy. Report this honestly.

**FedAvg versus FedProx (mu = 0.01) at matched rounds, seeds and partitions:**
- Paired per-seed differences in final federated accuracy (FedProx − FedAvg), mean:
  - alpha 0.1: +0.0000
  - alpha 0.3: −0.0009
  - alpha 1: +0.0000
  - alpha 3: +0.0019
  - alpha 10: −0.0019
  - by_subject: −0.001
- The largest single-seed difference is 0.009.

**mu sensitivity at alpha = 0.3.** Every value is reported. The default 0.01 was not
re-selected.
| mu | fed acc | fed macro-F1 | paired Δacc vs FedAvg (mean [min, max]) | paired ΔF1 |
|---|---|---|---|---|
| 0.001 | 0.571 ± 0.035 | 0.348 ± 0.106 | +0.000 [+0.000, +0.000] | +0.000 |
| 0.01 | 0.570 ± 0.036 | 0.347 ± 0.105 | −0.001 [−0.005, +0.000] | −0.001 |
| 0.1 | 0.569 ± 0.036 | 0.347 ± 0.105 | −0.002 [−0.009, +0.000] | −0.001 |
| 1.0 | 0.587 ± 0.031 | 0.375 ± 0.087 | +0.016 [+0.000, +0.061] | +0.027 |

**Verdict:** with 1 local epoch per round, FedProx gives **no measurable benefit** over
FedAvg for mu ≤ 0.1. At mu = 1 there is a small, inconsistent gain: 0 in some seeds,
+0.061 in one.
- This matches the mechanism. FedProx limits client drift, and with 1 local epoch of
  2 to 11 Adam steps the clients barely drift.
- Re-test once `fl.local_epochs` > 1 is used on the GPU cluster, as the user plans.
- Do not claim FedProx "fixes" non-IID here.

### Run cost
On the laptop CPU, 130 runs took about 75 min, roughly 27 s per run for all three
arms. The first mu sweep attempt failed when **the laptop went to sleep at 18:18**:
Ray stalled and the round hit its 600 s timeout. That failure led to the fault-tolerant
grid and the `caffeinate` target.

### Decisions made (review these)
1. **mu = 0.01** was fixed a priori. The sensitivity sweep is reported in full.
   Selecting mu from it would be tuning on the test set.
2. **Alpha grid** {0.1, 0.3, 1, 3, 10} plus by_subject as ∞.
3. **Fairness definition** is label-mix re-weighted holdout recall, with spread as
   max − min. Its limitation is stated above.
4. **Dirichlet prior** is centred on the pool distribution, Hsu et al. 2019. This is
   a bug fix, not a choice.

---

## Phase 3 — Federated core (FedAvg) — COMPLETE, reviewed (2026-10-15)

**User decisions after review (2026-10-15):**
- 30 rounds x 1 local epoch is accepted for laptop development. It will be revisited
  for the GPU-cluster training runs.
- Adam state is reset every round, following standard FedAvg practice. Keep it that
  way.

**Not committed.** The user commits. Suggested message:
`feat(fl): phase 3 federated core (FedAvg) on Flower 1.23 Message API`.

### User decisions applied (2026-10-15)
- The MLP on 50 features is the Phase 3 model.
- **Federated standardisation:** each silo shares (count, sum, sum of squares) once.
  This is used only to standardise inputs for training the shared model. The user
  will cover it in the privacy section.

### How the federation is simulated
- **One process starts the run.** `fedastro run` calls
  `flwr.simulation.run_simulation`, which runs a Flower ServerApp in a thread. It
  hosts 5 virtual SuperNodes, one per silo, on the Ray backend.
- **Each node is one silo.** Its Flower `partition-id` is the silo index, and its
  client code reads only that silo's partition.
  - Simulation shortcut: the ClientApp closure, and therefore every Ray actor, holds
    all 5 silos' arrays in memory.
  - Isolation is enforced by code, not by process boundaries.
- **Only Flower messages cross between server and clients**, all through
  `grid.send_and_receive`:
  1. `query.stats`: each silo sends its count, sum and sum of squares.
  2. `query.set_scaler`: the server returns the combined scaler, stored in node state.
  3. Each round sends `train`. Clients run the shared `train_local()` for 1 local
     epoch and return weights and example counts.
  4. FedAvg aggregates in silo order and float64, which is deterministic.
  5. The server scores the global model on the 3-subject holdout.
  6. `evaluate` scores the global model on each silo's own data.
- **The server holds the held-out test subjects** in its role as experimenter. No
  client ever sees them.
- **Links are synchronous and lossless.** Every silo participates every round, with
  no latency, LOS or stragglers, and a failed reply raises. Phase 6 adds link time,
  so `sim_wallclock_s` is null until then.
- **Seeding:**
  - Client training uses `derive_seed(seed, federated, silo, round)`.
  - The initial model uses `derive_seed(seed, federated)`.
  - Torch runs 1 thread per client.
  - The result is byte-identical across executions.

### Built
- `fl/strategy.py`: `FedAvg`, an example-weighted mean. It is order-independent,
  tested bitwise over all 120 orderings of 5 silos.
- `fl/client.py`: `make_client_app()` with the stats, set_scaler, train and evaluate
  handlers.
  - Flower 1.23's Message API replaces the older NumPyClient
    `get_parameters/fit/evaluate`.
  - `train` and `evaluate` are the equivalents. Initial weights come from the server.
- `fl/server.py`: `make_server_app()` with a custom `@app.main()` loop, and
  `run_federated()`.
- `metrics/comms.py`: `payload_bytes` and `rounds_to_target`.
- The summary now has the federated final metrics, a per-round mean ± std curve,
  rounds-to-target, total bytes, and an H1 per-seed tally.
- New config sections `fl` and `resources`. `resources` is an execution detail
  excluded from the run id.
- `configs/experiments/smoke_fl.yaml`. `smoke.yaml` now runs only the two
  baselines, to keep it fast.
- Default `arms` is now all three.

### Acceptance results
**`wesad_by_subject`** uses the MLP, 5 seeds, holdout subjects 13, 16 and 17
(n = 212), and 30 rounds x 1 local epoch.
| arm | accuracy | balanced acc | macro-F1 | AUC |
|---|---|---|---|---|
| centralized | 0.868 ± 0.022 | 0.793 ± 0.025 | 0.803 ± 0.027 | 0.881 ± 0.008 |
| **federated (FedAvg)** | **0.708 ± 0.074** | **0.677 ± 0.065** | **0.668 ± 0.070** | **0.807 ± 0.036** |
| local_only (silo mean) | 0.654 ± 0.029 | 0.571 ± 0.018 | 0.551 ± 0.033 | 0.778 ± 0.011 |

Per seed:
| seed | centralized | federated | local_only mean | fed macro-F1 | local macro-F1 |
|---|---|---|---|---|---|
| 0 | 0.849 | 0.670 | 0.631 | 0.635 | 0.512 |
| 1 | 0.863 | 0.722 | 0.642 | 0.688 | 0.546 |
| 2 | 0.901 | 0.830 | 0.705 | 0.782 | 0.593 |
| 3 | 0.849 | 0.679 | 0.640 | 0.621 | 0.531 |
| 4 | 0.877 | **0.642** | **0.652** | 0.615 | 0.576 |

Federated global holdout accuracy over rounds, mean over 5 seeds:
| round | 1 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| accuracy | 0.562 | 0.683 | 0.674 | 0.680 | 0.708 | 0.710 | 0.708 |
| macro-F1 | 0.406 | 0.488 | 0.485 | 0.543 | 0.631 | 0.663 | 0.668 |

- **H1 is testable from one command.** `fedastro sweep` writes all three arms into
  each `result.json`.
- **Federated lands between the bounds in 4/5 seeds** on accuracy.
  - **Seed 4 is recorded as is:** federated 0.642 is below local-only 0.652 on
    accuracy, but above it on macro-F1 (0.615 versus 0.576).
  - On macro-F1 and balanced accuracy, federated beats the local-only mean in 5/5
    seeds.
- **Rounds to 95 % of centralized accuracy: not reached** in 0/5 runs within 30
  rounds.
- **Communication:**
  - 21,772-byte payload (5,443 float32 parameters).
  - 217,720 bytes per round (payload x 5 silos x 2).
  - 6,539,600 bytes total, including 8,000 setup bytes.
- **Timing:** 38 to 41 s per seed for all three arms, with the 30-round FL arm about
  30 s of that. Under the 10 min budget for 5 silos x 20 rounds.
- **Tests:** `pytest` 92 passed in 64 s, including two real Flower/Ray runs that are
  byte-identical.

### Observations (recorded, not tuned)
- FedAvg accuracy plateaus around 0.71 from about round 20, while macro-F1 is still
  rising at round 30. The gap to centralized is large even though the
  `by_subject` labels are near-IID.
- Likely contributors, for Phase 4 to examine rather than tune away:
  - Inter-subject feature heterogeneity.
  - Adam state is re-initialised every round on every client.
  - Tiny silos (mars_transit has 70 windows, about 3 batches per round).
- Phase 2 numbers shifted slightly (centralized 0.857 to 0.868) because
  `train_local` now seeds the global torch RNG, which dropout uses. The Phase 3
  table supersedes the Phase 2 by-subject table.

### Decisions made (review these)
1. **Custom ServerApp loop on the Flower Message API instead of
   `flwr.serverapp.strategy.FedAvg.start()`.** Read from the installed source:
   - Built-in FedAvg aggregates replies in arrival order, which is non-deterministic
     in floating point.
   - It has no hook for the stats exchange, per-silo holdout scoring, or byte
     accounting.
   - All traffic still goes through Flower messages and the Flower simulation
     engine. Phase 4 FedProx and the Phase 6 buffered-async strategy go in
     `fl/strategy.py`.
2. **30 rounds x 1 local epoch**, compute-matched to the baselines' 30 epochs. The
   acceptance check names 20 rounds, and 30 is a superset. Not tuned after seeing
   results.
3. **Optimizer state** (Adam moments) is not carried across rounds. This is the
   standard FedAvg simulation, with a fresh optimizer per `train` call.
4. **Per-silo metrics each round:**
   - `update_on_holdout`: the silo's local update, scored on the global test set.
   - `global_on_local`: the global model on that silo's own training data, which is
     optimistic.
   - Phase 4 fairness should use `update_on_holdout`, or a new per-silo test split
     if subjects allow.
5. **Bytes count float32 tensor payload only**, with no protocol framing. The
   one-time stats and scaler exchange is reported as `setup_bytes`, separately.
6. **Schema change:** `rounds[].sim_wallclock_s` may be null. It stays null until
   Phase 6 and is never filled with real time.
   - Additive federated fields: `strategy`, `local_epochs`, `n_params`,
     `payload_bytes`, `setup_bytes`, `total_bytes`, `rounds_to_target`, and the
     per-silo sub-dicts.
7. **Ray resources:** 1 CPU and 0 GPU per client by default, set via `resources.*`.
   - A CUDA run of the federated arm requires `client_num_gpus > 0`.
   - With 8 logical CPUs, Ray starts 8 actors and runs the 5 clients concurrently.

### GPU cluster plan
- **Now (end of Phase 3): shakedown run.** Its purpose is to prove the environment,
  not to produce numbers.
  - `uv.lock` resolves Linux x86_64 wheels for the same pins. torch 2.2.2 from PyPI on
    Linux is the CUDA 12.1 build, which needs an NVIDIA driver >= 530.
  - Steps, all on CPU:
    1. `git clone` then `make install`.
    2. `make fetch-wesad`.
    3. `make test`.
    4. `fedastro sweep configs/experiments/wesad_by_subject.yaml`.
  - Compare against the laptop numbers above. Expect agreement within noise, not
    byte identity, because a different platform and BLAS can change float results.
  - Then one run with `--device cuda` and `resources.client_num_gpus: 0.2`. This
    checks CUDA and Ray GPU placement only. The model is too small for a GPU to beat
    CPU.
- **First substantive cluster batch: Phase 4.** The alpha sweep x {FedAvg, FedProx}
  x {Dirichlet, quantity control} x 5 seeds is the first work that is slow on the
  laptop. Run each config as a separate CPU job.
- **Where GPUs actually pay off: Phase 5.** Opacus per-sample gradients, the epsilon
  sweep, and gradient-inversion sweeps.
- **Publishable final runs** come after all required phases. Phase 9's `sweep.py`
  adds resumability and per-run logs for that batch.

---

## Phase 2 — Model + non-FL baselines — COMPLETE (2026-10-15)

**Not committed.** The user commits. Suggested message:
`feat(baselines): phase 2 MLP/1D-CNN, centralized + local-only arms, fixed holdout, quantity control`.

### Phase 1 amendments (user decisions, 2026-10-15)
- **Fixed test set.**
  - `partition.holdout_seed` (default 0) draws the held-out subjects from its own
    RNG stream, independent of the run seed.
  - Every seed and arm of an experiment is evaluated on the same subjects. With the
    default seed they are **13, 16 and 17**, giving 212 windows.
  - This replaces Phase 1 decision 5. Phase 1 tables and run ids are superseded.
- **Quantity-matched control.** `partition.method: dirichlet_quantity_control` with
  the same `alpha`:
  - Reproduces the Dirichlet split's subjects per silo and window counts per silo
    exactly, for the same seed.
  - Keeps windows uniformly at random, so each silo has its natural label mix.
  - The Dirichlet RNG draws were reordered so both modes consume identical draws.
  - Config: `configs/experiments/wesad_dirichlet_control.yaml`. Use it in Phase 4
    wherever a Dirichlet result is reported.
- **Default sensors:** chest and wrist. The user confirmed this is the default for
  the crew scenario.

### Built
- `models/model.py`: `MLP` (hidden 64-32, 5,443 parameters on 50 features) and
  `CNN1D` (Conv-GroupNorm-ReLU-MaxPool x2, global average pool, linear).
  - No BatchNorm anywhere. Opacus `ModuleValidator` passes both models (tested).
  - `cnn1d` requires `representation: sequence`, and `mlp` requires features. The
    config validator enforces this.
- `data/preprocess.py`: `Standardizer`, fit on training data only.
  - Stored as sufficient statistics (count, sum, sum of squares), so per-silo
    statistics combine exactly to the pooled fit.
  - Phase 3 can use this for federated standardisation.
- `training.py`: shared `train_local()`.
  - Fixed epochs, seeded shuffling, last-epoch weights, no early stopping.
  - The optional per-epoch holdout evaluation is for curves only and never selects
    a checkpoint.
  - Also `predict_proba()` and `derive_seed()`.
- `metrics/classification.py`: accuracy, balanced accuracy, macro-F1, and macro
  one-vs-rest AUC.
- `baselines/centralized.py`: pooled silo data, one scaler and one model.
- `baselines/local_only.py`: one scaler and model per silo, each evaluated on the
  shared holdout. Reports mean, std, min and max across silos.
- `metrics/aggregate.py`, `fedastro summarize <result.json...>`, and an automatic
  `<output_dir>/<name>-<hash>-summary.json` after `fedastro sweep`. The std is the
  sample std (ddof=1) across seeds.
- New config sections `model`, `train` and `arms`, plus
  `configs/experiments/wesad_by_subject_cnn.yaml`.

### Acceptance results
All runs use 5 seeds (0 to 4), the fixed holdout (subjects 13, 16, 17; n = 212), and
training defaults fixed before any result was seen.

**`wesad_by_subject`, MLP on 50 features.** 830 training windows. The full 5-seed
sweep with both arms took **31.6 s**.
| arm | accuracy | balanced acc | macro-F1 | AUC |
|---|---|---|---|---|
| centralized | 0.857 ± 0.016 | 0.782 ± 0.012 | 0.791 ± 0.012 | 0.879 ± 0.003 |
| local_only (silo mean) | 0.656 ± 0.034 | 0.574 ± 0.017 | 0.554 ± 0.031 | 0.778 ± 0.011 |

**`wesad_by_subject_cnn`, 1D-CNN on 10 x 240 sequences.** The full sweep took
**106.6 s**. The sequence cache cold build took 114 s.
| arm | accuracy | balanced acc | macro-F1 | AUC |
|---|---|---|---|---|
| centralized | 0.770 ± 0.072 | 0.698 ± 0.067 | 0.689 ± 0.073 | 0.854 ± 0.079 |
| local_only (silo mean) | 0.521 ± 0.050 | 0.445 ± 0.029 | 0.413 ± 0.035 | 0.693 ± 0.031 |

**`wesad_dirichlet` versus `wesad_dirichlet_control`, MLP, alpha 0.5.**
| arm | dirichlet acc | control acc | dirichlet macro-F1 | control macro-F1 |
|---|---|---|---|---|
| centralized | 0.552 ± 0.167 | 0.754 ± 0.045 | 0.509 ± 0.131 | 0.691 ± 0.040 |
| local_only (silo mean) | 0.478 ± 0.109 | 0.662 ± 0.038 | 0.362 ± 0.102 | 0.528 ± 0.045 |

Per-silo numbers are in each `runs/*-summary.json`.

- **centralized >= local_only on average across silos: PASS** in all four
  configurations and on every metric.
- **5-seed mean ± std reported: PASS.**
- **One arm < 5 min on laptop CPU: PASS.** The slowest full sweep, CNN with 5 seeds
  and 2 arms, took 107 s.
- `pytest`: 80 passed. A WESAD run with training is byte-identical across two
  executions.

### Observations to carry forward (not tuned, just recorded)
- The MLP on hand-crafted features beats the CNN on 4 Hz sequences on every metric.
  This is expected, because the downsampling discards ECG, EMG and BVP morphology.
  **Recommend the MLP as the primary model for Phase 3.** Keep the CNN as a secondary
  configuration and for the Phase 5 per-channel time-series attack.
- Under Dirichlet skew, **even the centralized arm drops** from 0.754 to 0.552
  accuracy against the control. The pooled training label mix then differs from the
  holdout's natural mix, a label prior shift.
  - Phase 4 must not attribute the whole Dirichlet penalty to federation.
  - The fair comparison is federated versus centralized and local-only within the
    same partition, with the control alongside.
- Per-silo local-only spread across seeds is large (std 0.06 to 0.26), because the
  silos are tiny (1 to 5 subjects). Expect noisy per-silo fairness numbers in Phase 4.

### Decisions made (review these)
1. **Training defaults, fixed before seeing results:** Adam, lr 1e-3, weight decay
   1e-4, 30 epochs, batch 32, dropout 0.2, unweighted cross-entropy, no early
   stopping.
   - `class_weighting: balanced` is available but off.
   - These must not be retuned per arm.
2. **Local-only uses the same epoch count for every silo**, so small silos take fewer
   gradient steps. That is part of what "small silo" means. The alternative is
   matching gradient steps.
3. **Local-only "mean" is unweighted across silos.** Each silo is one stakeholder.
   Min, max and std across silos are also stored.
4. **Federated standardisation for Phase 3.** Approved by the user and implemented
   in Phase 3. The original recommendation follows. Clients share
   (count, sum, sum of squares) once, which gives the pooled scaler exactly.
   - This leaks per-silo feature means and variances. Note that in the privacy
     discussion.
   - The alternative is per-client scalers, but then a single global model sees
     inconsistently scaled inputs.
5. **Additive schema fields:**
   - `arms.centralized` gains `n_train` and `n_params`.
   - `arms.local_only` gains `std_across_silos`, `min` and `max`.
   - Each `per_silo` entry holds `final`, `history` and `n_train`.

### Not done / deferred
- Federated arm: Phase 3.

---

## Phase 1 — Data layer — COMPLETE (2026-10-15)

**Not committed.** The user makes all commits and pushes; this was decided 2026-10-15.
Suggested message: `feat(data): phase 1 WESAD loader, subject-disjoint partitioners, holdout`.

### Built
- `scripts/fetch_data.py wesad --accept-license` (or `make fetch-wesad`).
  - Downloads the 2.25 GB `WESAD.zip` into `$FEDASTRO_DATA_ROOT/raw/`.
  - Resumes from a `.part` file and records the sha256 in `WESAD.zip.meta.json`.
  - Archive sha256: `5e15d2606adf16d819ba535c1786d92e94bae5ff5392f7ae01fb63f9938fd71c`.
- `src/fedastro/data/wesad.py` reads `S<n>/S<n>.pkl` straight from the zip.
  - The pickles total 13.8 GB uncompressed, and the dev laptop had 8 GB free.
  - Uses a restricted unpickler that only reconstructs NumPy arrays.
  - Cuts label-pure windows and extracts features or sequences.
  - Caches per subject in `$FEDASTRO_DATA_ROOT/cache/wesad/<params+archive hash>/`.
  - Uncached subjects are read in parallel processes. `FEDASTRO_WESAD_WORKERS`
    sets the count, default 3.
- `src/fedastro/paths.py`: data locations come only from `FEDASTRO_DATA_ROOT`
  (default `./data`), so cluster paths never change a run id.
- `data/partition.py`:
  - `holdout_split`: whole subjects go to the global test set before any partitioning.
  - `by_subject`, `by_site`, and `dirichlet`.
  - `make_federation` calls `assert_disjoint` on every build.
- `DatasetBundle.site_id` is optional. `Provenance.processed_checksum` is new.
- `fedastro inspect <config>` prints the per-silo class table and writes nothing.
- New configs: `configs/experiments/wesad_by_subject.yaml` and `wesad_dirichlet.yaml`.
- `--synthetic` now swaps any config to the SYNTHETIC dataset, and results are tagged.

### Acceptance results
| Check | Result |
|---|---|
| `pytest` | 54 passed, including the real-WESAD test and a fake-archive parser suite |
| No subject in two silos | asserted in `make_federation` and tested for all 3 methods |
| Holdout disjoint from all silo training data | tested, including a direct check that no training sample has a holdout subject id |
| Per-silo label distributions visibly non-uniform | **yes under `dirichlet`; no under `by_subject`**, see the tables below |
| Loader < 60 s for one silo, laptop CPU, cold cache | largest silo `nasa_earth` (5 subjects, 346 windows): **34.5 s** with 3 workers; 69.2 s serial; 0.0 s warm |
| Full WESAD cold build (15 subjects) | 124 s serial (before parallelism) |
| End-to-end `fedastro run wesad_dirichlet.yaml` twice | byte-identical `result.json` |

WESAD after windowing has 1,042 windows from 15 subjects at the 60 s window and 30 s
stride, with feature dimension 50.

**Per-silo class distribution, `wesad_by_subject`, seed 0.**
Holdout subjects are 9, 10 and 14.
```
silo             subj      n          baseline            stress         amusement  dropped
nasa_earth          5    346     188 ( 54.3%)     103 ( 29.8%)      55 ( 15.9%)        0
esa_analog          2    139      76 ( 54.7%)      41 ( 29.5%)      22 ( 15.8%)        0
iss_commercial      2    137      75 ( 54.7%)      40 ( 29.2%)      22 ( 16.1%)        0
lunar_habitat       2    139      75 ( 54.0%)      42 ( 30.2%)      22 ( 15.8%)        0
mars_transit        1     70      38 ( 54.3%)      21 ( 30.0%)      11 ( 15.7%)        0
HOLDOUT (test)      3    211     114 ( 54.0%)      64 ( 30.3%)      33 ( 15.6%)        0
```
**Per-silo class distribution, `wesad_dirichlet` with alpha = 0.5, seed 0.**
```
silo             subj      n          baseline            stress         amusement  dropped
nasa_earth          5     60       0 (  0.0%)       5 (  8.3%)      55 ( 91.7%)      286
esa_analog          2    103      76 ( 73.8%)      20 ( 19.4%)       7 (  6.8%)       36
iss_commercial      2     47       0 (  0.0%)      40 ( 85.1%)       7 ( 14.9%)       90
lunar_habitat       2     53       6 ( 11.3%)      42 ( 79.2%)       5 (  9.4%)       86
mars_transit        1     12       0 (  0.0%)       1 (  8.3%)      11 ( 91.7%)       58
HOLDOUT (test)      3    211     114 ( 54.0%)      64 ( 30.3%)      33 ( 15.6%)        0
```

### Key finding
Every WESAD subject followed the same protocol, so subject-level splits are
**non-IID in X but nearly IID in labels.** Label skew exists only if it is imposed,
by the `dirichlet` method here. Report this honestly on the poster.

### Decisions made (review these)
1. **Window 60 s, stride 30 s** (50 % overlap). Pure-label windows only; labels
   0 and 4 to 7 are dropped.
   - A shorter stride gives more but more strongly correlated windows.
   - Both values are configurable as `data.wesad.window_s` and `stride_s`.
2. **Signals: chest and wrist** by default, 10 channels. 3-axis ACC is reduced to
   its magnitude.
   - Wrist-only, the more realistic wearable story for crew, is
     `data.wesad.devices: [wrist]`.
3. **Features: 5 generic statistics per channel** (mean, std, min, max, linear
   slope), giving 50 features.
   - No domain features such as HRV from ECG/BVP peaks or EDA phasic/tonic split.
   - Those would raise accuracy but add scope. Consider them only if Phase 2 numbers
     are uninformative.
4. **`sequence` representation for the Phase 2 1D-CNN:** every channel is block-mean
   downsampled to 4 Hz, giving shape (10, 240).
   - Waveform morphology of ECG, EMG and BVP is averaged away.
   - The alternative is per-bin std for high-rate channels as an activity envelope.
     Revisit in Phase 2.
5. **Holdout = 3 subjects, drawn from the run seed.** The test set therefore changes
   across seeds, and seed variance includes test-set choice.
   - Within one seed, all three arms share the same holdout.
   - The alternative is a fixed holdout seed. Decide before Phase 2 numbers go on the
     poster.
6. **Dirichlet stays subject-disjoint by subsampling.** It drops 36 to 286 windows
   per silo (40 to 83 %) at alpha = 0.5.
   - This confounds label skew with data quantity in the Phase 4 alpha sweep.
   - It also stops `nasa_earth` from being "most representative".
   - Options for Phase 4:
     - (a) Keep it and add a quantity-matched `by_subject` control. **Recommended.**
     - (b) Exempt `nasa_earth` from skew.
     - (c) Classic sample-level Dirichlet. This violates the no-subject-in-two-silos
       rule, so it is not recommended.
7. **The subject allocator changed** to plain largest-remainder proportional
   allocation with a floor of 1.
   - On the 12 training subjects it gives 5/2/2/2/1. The Phase 0 rule gave 4/2/2/2/2
     and ignored the "very small" Mars share.
   - With 12 subjects, ESA, ISS and Lunar cannot be distinguished by subject count.
8. **`by_site`** is implemented generically for future eICU data. WESAD has no sites,
   so it raises an error.
9. **Additive schema fields**, with no section 6 field removed:
   - top-level `holdout` with `n_test`, `subjects` and `label_counts`
   - `silos[]` gains `subjects`, `n_dropped` and `target_proportions`
   - `provenance[]` gains `processed_checksum`
   - Unused dataset parameter blocks are omitted from the recorded `config`.
   - Run ids changed versus Phase 0, because the config gained fields.

### Not done / deferred
- No feature scaling. Scalers must be fit on training data only, so this is Phase 2
  work, and per-silo locally for FL.
- `osdr.py` and `eicu.py` are Phase 9.

### Resolved by the user (2026-10-15)
- Decision 5: the holdout is now **fixed** via `partition.holdout_seed`. See Phase 2
  amendments.
- Decision 6: option (a). A `dirichlet_quantity_control` method was added.
- Default sensors stay chest and wrist.

---

## Phase 0 — Scaffold — COMPLETE (2026-10-15)

### Built
- `pyproject.toml` with exact pins on every direct dependency; full transitive lock in
  `uv.lock`. Install with `make install` (`uv sync --frozen`).
- `src/fedastro/` package skeleton matching CLAUDE.md section 3. Empty subpackages
  (`models`, `fl`, `privacy`, `baselines`) exist for later phases.
- `config.py`: YAML loader. Experiment file is deep-merged over `configs/base.yaml`;
  `silos: <name>` expands `configs/silos/<name>.yaml` inline. Strict frozen-dataclass
  validation, and unknown keys are rejected at every nesting level.
- `seeding.py`: seeds Python, NumPy, Torch and CUDA. Turns on
  `torch.use_deterministic_algorithms(True)` and sets `CUBLAS_WORKSPACE_CONFIG`.
- `metrics/schema.py`: `empty_result()` with every section 6 field present (nulls
  when unused) and `validate_result()`.
- `manifest.py`: deterministic `run_id`, `git_sha` with a `-dirty` suffix, and the
  writer for `result.json` plus `manifest.json`.
- `data/base.py` holds `DatasetBundle` and `Provenance`. `Provenance.kind` is
  `REAL | SYNTHETIC`.
- `data/synthetic.py` and a minimal `data/partition.by_subject`.
- `pipeline.py` runs the data path only. `cli.py` provides `fedastro run` and
  `fedastro sweep`.
- `configs/base.yaml`, `configs/silos/five_agency.yaml`, `configs/experiments/smoke.yaml`.
- `Makefile` targets: `install`, `test`, `smoke`, `repro-check`, `clean`.

### Acceptance results
| Check | Result |
|---|---|
| `pytest` | 28 passed |
| `fedastro run configs/experiments/smoke.yaml` | valid `runs/smoke-seed0-9477b66386/result.json`, provenance kind SYNTHETIC |
| Fresh `git clone` + `uv sync --frozen` + run with `SOURCE_DATE_EPOCH=0` vs original checkout | byte-identical; sha256 `f82a5ddabd7c8319027d1b40e13992763e88f1431090426701a7c42fbe4296ee` at commit `5aff626` |
| Test suite in the fresh clone | 28 passed |

### Decisions made (review these)
1. **Dependency pins are forced by the Intel-Mac dev laptop (x86_64).**
   - torch 2.2.2 is the last release with macOS x86_64 wheels, which forces numpy 1.26.4.
   - opacus 1.5.4 fails to import on torch 2.2, because it references `nn.RMSNorm`.
     Opacus is pinned to 1.5.2.
   - ray >= 2.32 has no macOS x86_64 wheels. Unpinned, uv picks flwr 1.40 on Linux but
     flwr 1.23 on the Mac. **flwr 1.23.0 is pinned on every platform** so the laptop
     and the cluster share one Flower API. It resolves ray 2.31.0. Ray startup was
     verified on the laptop.
   - Cost: the cluster runs older torch and Flower than it could. That is acceptable for
     a small model. Revisit only if a needed feature is missing. Moving to Apple
     Silicon or Linux-only development would unlock newer pins.
   - Python is pinned to 3.11.
2. **Byte-identical reproducibility uses `SOURCE_DATE_EPOCH`.** `timestamp_utc` is
   wall-clock time unless that env var is set, following the reproducible-builds
   convention. Everything else in `result.json` is deterministic. Environment details
   such as hostname, versions and real elapsed time go to `manifest.json`, which is
   never byte-compared.
3. **Execution keys stay out of the experiment identity.** `seed`, `seeds` and
   `output_dir` are excluded from the recorded `config` block and from the run-id
   hash. The run's `seed` is still recorded both at top level and in `config.seed`.
   `run_id` is `<name>-seed<seed>-<sha256(experiment)[:10]>`. Re-running an
   identical experiment overwrites the same directory.
4. **Additive schema fields.**
   - `provenance[]` carries extra `source` and `kind` keys.
   - `silos[]` carries an extra `n_subjects` key.
   - The validator checks the section 6 fields and permits additions. No section 6
     field was removed or renamed.
5. **Silo data shares** are 0.40 / 0.20 / 0.20 / 0.12 / 0.08. This is a scenario
   design choice for the large / medium / medium / small / very small labels, not a
   measured quantity. Subjects are allocated by largest remainder, with at least one
   subject per silo.
6. **Mars latency presets.**
   - `near` = 182 s and `far` = 1337 s come from CLAUDE.md section 4.
   - `mean` = 751 s is derived from a ~225e6 km mean Earth-Mars distance divided by c.
     **Confirm a citable source before this number appears on the poster.**

### Not done / deferred
- No training arms yet. `arms.*` sections are present and empty, and `notes` says so.
  This is Phase 2-3 work.
- `--synthetic` CLI flag exists but is a no-op until a real dataset exists (Phase 1).
- `partition.by_site`, `dirichlet`, and the global held-out set are Phase 1.
- `scripts/` is empty. `fetch_data.py` arrives in Phase 1, `sweep.py` in Phase 9
  (`fedastro sweep` covers sequential multi-seed runs until then), and
  `make_figures.py` in Phase 8.
- Flower orchestration API not yet chosen. Phase 3 must inspect flwr 1.23.0 itself.
  Verified so far: `flwr.simulation` exports `run_simulation(server_app, client_app,
  num_supernodes, backend_name='ray', backend_config=...)`, and `start_simulation`
  still exists as legacy.

### Open questions for the user
- Is the dev laptop staying Intel? If not, the pins in decision 1 can move forward.
- Are the silo data shares in decision 5 acceptable for the scenario narrative?

