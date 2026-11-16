# PREREG — pre-registration for the `hier-fl` branch

**Status: SKELETON (Phase H0, 2026-11-15). Nothing below is frozen yet.**
Each section is filled in and frozen (committed) by the phase named in it, **before** any
result-producing run that it covers (CLAUDE.md rule 9 and §7). After a section is frozen, any
change to it is a deviation and is listed in the Deviations section and in `PROGRESS.md`.

Scope notes that apply to everything here (CLAUDE.md §2):
- "Sparse" means sparse **data** (small, siloed astronaut populations), not sparse model updates.
- Every dataset is a terrestrial or animal **proxy** for astronaut data. Silo, region and agency
  names are scenario labels.
- The location shift is **simulated**. Secure aggregation is a **protocol simulation**.
- If federated or hierarchical FL loses to a baseline, that is a result and is reported.

Earlier pre-registration (previous plan, kept for the record, **not run, superseded**):
[`prereg/p12-wesad-redraws.md`](prereg/p12-wesad-redraws.md).

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

## 3. Location shift (Phase H2)

PLACEHOLDER_ — shift design chosen by the user at the H2 gate; kinds, channels and magnitudes
(including the 0 control).

## 4. Update-gate tolerance (Phase H5)

PLACEHOLDER_ — validation tolerance for accepting an incoming model version, set on validation
data only.

## 5. Gap-closing grid and selection rule (Phase H8)

PLACEHOLDER_ — grid (FedProx μ, local epochs, server optimizer, learning rate) and the selection
rule, applied to validation scores only.

## 6. Experiment configurations and seeds

PLACEHOLDER_ — one row per configuration: config file, config hash, topology, dataset, seeds.
- OSD-435 and WESAD (Phase H9).
- CinC 2021 (Phase H10).

## 7. Deviations

None yet.
