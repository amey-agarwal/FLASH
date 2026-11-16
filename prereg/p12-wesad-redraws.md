# Pre-registration: Phase 12, WESAD holdout redraws with the ensemble baseline

**Status:** written 2026-11-05, before any run of this experiment. It must be committed
before the first run (CLAUDE2.md rule 9). Results are reported against this file in
`PROGRESS.md`, and any deviation is listed there.

## Question
Does H1 (federated beats each silo training alone) survive two stricter tests?
1. **A stronger baseline:** `ensemble_of_locals`, the local models' predicted probabilities
   averaged at test time, with no federation.
2. **Other "new crew members":** three different held-out subject sets instead of one.

## What is already known (disclosed, not hidden)
- **Draw 0 is the Phase 3 holdout** (subjects 13, 16, 17), with the same training code and
  defaults. Its federated, local-only and centralized results are known: they are the G1
  numbers in `PROGRESS.md`. Adding the ensemble arm does not change the other arms
  (`test_adding_the_ensemble_does_not_change_the_other_arms`).
- **Not yet seen:** draws 1 and 2, and the ensemble arm on any draw.

## Experiment
- **Config:** `configs/experiments/wesad_by_subject_redraws.yaml` (config hash `362a9b9ba3` at the
  time of writing; any edit changes it and is a deviation).
- **Data:** WESAD (`domain: terrestrial_proxy`), 15 subjects, 60 s windows with a 30 s stride,
  50 features, 3 classes (baseline / stress / amusement).
- **Partition:** `by_subject`. Silos get 5/2/2/2/1 of the 12 training subjects; silo membership
  is drawn from the run seed.
- **Holdout draws** (3 whole subjects each, from the fixed holdout stream, independent of the run
  seed; computed before any run, from subject ids only):

  | Draw | Held-out subjects | Held-out windows |
  |---|---|---|
  | 0 | 13, 16, 17 | 212 |
  | 1 | 2, 14, 15 | 207 |
  | 2 | 6, 7, 9 | 207 |

- **Seeds:** 0, 1, 2, 3, 4. With 3 draws that is **15 runs**; each run is one (seed, draw) pair.
- **Arms:** centralized, local_only, federated (FedAvg, 30 rounds × 1 local epoch),
  ensemble_of_locals (unweighted mean of the 5 local models' probabilities).
- **Training:** the unchanged defaults. MLP 64-32, Adam, lr 1e-3, weight decay 1e-4, batch 32,
  dropout 0.2, 30 epochs for the baselines. No tuning.

## Metrics and statistics
- **Metrics:** accuracy, balanced accuracy, macro-F1 and macro one-vs-rest AUC on the held-out
  windows.
- **Pairing:** within each run, federated − comparator.
- **Summary:** the mean paired difference over the 15 runs, with a 95% hierarchical percentile
  bootstrap CI. It resamples seeds, then draws within a seed, 2000 times, with a fixed rng
  (`metrics/stats.py`). No p-values.

### Primary comparison
**federated − local_only (silo mean), macro-F1.**
- Why macro-F1: the classes are imbalanced (about 54/30/16); always predicting "baseline" scores
  about 0.54 accuracy (CLAUDE.md §8).
- **Success criterion:** the 95% CI lower bound is above 0.
- Otherwise H1 is **not supported** by this test, and that is reported as is.

### Key secondary comparison (reported whatever the outcome)
**federated − ensemble_of_locals, macro-F1.**
- **H1 survives the ensemble** if the CI lower bound is above 0.
- **It does not** if the CI upper bound is below 0.
- **Inconclusive** if the CI contains 0.

### Exploratory
- The other 10 comparisons: the same comparators on the other metrics, plus federated −
  centralized.
- In total **12 comparisons are made**, and all are reported.
- Per draw: the primary difference as a mean over the 5 seeds, as a descriptive check that no
  single held-out set drives the result.
- Silo fairness: worst silo and best-to-worst gap per arm.

## Thresholds
None beyond the criteria above. No early stopping, no checkpoint selection, no re-runs to
change an outcome.

## Execution
- **Where:** laptop CPU, from a clean commit that includes this file. The run must not
  produce `-dirty` results.
- **Command:**
  `caffeinate -i uv run --frozen fedastro sweep configs/experiments/wesad_by_subject_redraws.yaml`
  It writes 15 `runs/wesad_by_subject_redraws-seed<s>-draw<d>-<hash>/` folders and the summary.
- **Report:**
  `python3 scripts/run_report.py runs/wesad_by_subject_redraws-seed*-<hash> --out runs/reports/p12.html`
  (inside `runs/`, which is gitignored, so the tree stays clean)
- **Expected cost:** about 15 × 40 s on the laptop (the Phase 3 per-seed time for three arms); the
  ensemble adds only predictions.

## Deviations
Any change to the config, the seeds, the draws, the metrics or the criteria after this file
is committed is listed here and in `PROGRESS.md`, with the reason.
