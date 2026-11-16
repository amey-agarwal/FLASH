# CLAUDE.md — FLASH: Cross-Silo Sparse Federated Learning for Autonomous Astronaut Healthcare

> Project instruction file for Claude Code. Read this fully before touching any code.
> This file replaces the original CLAUDE.md. Phases 0–6 of the original plan are
> complete; this file carries their rules forward and adds the `hier-fl` work plan.
> **Work one phase at a time. Every phase runs in plan mode first, then code mode (§7).**

---

## 0. What this repo is

A reproducible simulation backing an ASGSR 2026 **presentation** (42nd Annual Meeting,
2–5 Dec 2026, Arlington VA). Abstract: "A Cross-Silo Sparse Federated Learning System
Architecture for Autonomous Astronaut Healthcare Systems."

"Sparse" in the title means **sparse data** (small, siloed astronaut populations), not
sparse model updates. Say so wherever the title appears in docs.

### What the project claims (and only these)
1. Astronaut health data is siloed (silos = agencies **and** missions) and sparse.
   Silos interact only indirectly, by sharing model updates through FL.
2. The data in question is astronaut health data. Every dataset here is a **proxy**.
3. Planetary missions need privacy-preserving human-biology research and continuous
   health monitoring: a shared global model plus regional models across agencies, with
   the goal of **reduced** (not eliminated) dependence on Earth-based medical support.
4. The architecture includes security and **allows** scalability. Do not claim a
   scalability demonstration unless an experiment adds silos and measures it.
5. Agencies differ in label mix (label skew, Dirichlet) and in data volume (quantity
   skew, data partitioning). The two are separate mechanisms in code and in reporting.
6. Physiological data only. Environmental sensor data is not integrated.

### Hypotheses (pre-registered in `PREREG.md` before any result-producing run)
- **H1 (carried over, already tested on WESAD):** federated training beats each silo's
  local-only model, at a quantifiable communication cost.
- **H2 (new):** under a location-dependent data shift, regional models from hierarchical
  FL beat (a) local-only models and (b) the single global model on their own region's
  test sets. With the shift magnitude set to zero (control), the global model should be
  at least as good as the regional models.
- **H3 (new, descriptive):** test examples where the local, regional and global models
  disagree have lower accuracy than examples where they agree. This is what makes
  disagreement worth flagging for clinician review.
- **H4 (new):** whether the federated advantage in H1 and H2 holds across model families
  (logistic regression, MLP, 1D-CNN).

If federated or hierarchical loses to a baseline under some condition, **that is a
result, not a bug.** Report it. Never tune to make FL look good.

### Execution environment
- **Development:** laptop (Intel Mac), CPU, small subsets, fast iteration.
- **GPU runs:** Georgia Tech PACE ICE cluster, headless, from a git clone (§8).
- No notebooks in the critical path, no interactive prompts, no hardcoded paths.
  Everything is driven by config files and the CLI.

---

## 1. Non-negotiable rules

1. **Verify library APIs against the installed version.** Flower is pinned at 1.23.0
   (Ray-backed simulation). Check installed docs and docstrings before writing
   orchestration code. Do not write from memory of an older tutorial.
2. **No synthetic data standing in for real data without a loud label.** Every loader
   records provenance (source, version, checksum) in the run manifest. Synthetic runs
   are tagged `provenance: SYNTHETIC`. Simulated location shifts are tagged too (§2).
3. **Seeded and deterministic.** Every run takes `--seed`. Any number that goes into
   the presentation uses at least 5 seeds and is reported as mean ± std with the
   hierarchical bootstrap 95% CI already in `metrics/aggregate.py`.
4. **Results are files, not stdout.** Every run writes `runs/<run_id>/result.json`
   conforming to §6. Figure scripts and the terminal peek read only those files.
5. **Never fabricate numbers, citations or results** in code, comments, docs or figure
   captions. Placeholders are named `PLACEHOLDER_`.
6. **Keep models small.** Logistic regression, a 2–3 layer MLP, or a small 1D-CNN.
   Model capacity is not the contribution.
7. **Stop at phase gates.** At the end of each phase: run the acceptance checks,
   update `PROGRESS.md`, suggest a commit message, and stop. **The user commits.**
   Do not roll into the next phase unprompted.
8. **Reproducibility lock.** The flat five-silo WESAD results recorded in PROGRESS.md
   (Phases 3–6) must still reproduce on this branch. A regression test guards them
   (Phase H0). New features are additive and default to off.
9. **Tune on validation only.** No hyperparameter, threshold or design choice may be
   selected using test-set results. Validation sets exist for this.
10. **Simulated wall-clock is never real wall-clock.** Separate fields, labelled axes.

---

## 2. Scientific framing to preserve in code and docs

These are defensibility constraints. Violating them costs the presentation.

- **FLUID (Casaletto et al., bioRxiv 10.1101/2025.01.14.633017) is prior art.** It is a
  real ISS-deployed FL framework (OpenFL on HPE Spaceborne Computer-2), but its
  biomedical run used IID mouse cardiac data (NASA OSDR OSD-435) split 50/50
  Earth/ISS. It did not demonstrate non-IID, multi-agency human FL.
- **Frontier Development Lab 2021** (NASA, Mayo Clinic, GeneLab via OpenFL) already did
  cross-institution FL for astronaut-health research. Our contribution is the
  **measured behaviour** of non-IID, quantity-skewed, multi-agency FL under deep-space
  delay, with hierarchical regional models, not "FL for astronaut health" as such.
- **Astronaut N is tiny.** Terrestrial and animal data are stated proxies. Silo,
  region and agency names are **scenario labels**, never real deployments.
- **Location shift is simulated.** WESAD, CinC and OSD-435 contain no real
  location-dependent physiology. Every result that uses an injected shift says so.
- **Not deployable today.** The architecture applies once crewed lunar and Mars
  missions run. An Earth–ISS demonstration is the near-term option (FLUID precedent).
- **Privacy in this branch rests on two things:** an assumed **neutral aggregator**
  (operated by no single agency) and **secure aggregation**. Neither gives a formal
  guarantee. The final global model can still leak through membership inference. Say
  so wherever privacy is mentioned.
- **DP-SGD is set aside in this branch.** Its code stays and its Phase 5 results
  (attack driven to chance; federated model collapsing toward the majority class at
  ε ≤ 10) are a discussion point, not a headline result. Do not delete DP code.
- **Secure aggregation here is a simulation of the protocol**, not a cryptographic
  implementation. Masks come from a seeded PRG, not real key agreement. Label it.
- **Disagreement between models is a flag, not a second opinion.** Local, regional and
  global models share training data, so their agreement is correlated. Never describe
  agreement as independent confirmation.
- Claims take the form "we demonstrate the mechanism under simulated conditions," never
  "we validate an operational or clinical system." Clinical validation is a written
  protocol in this project, not a result.

---

## 3. Repo layout (target state on `hier-fl`)

Existing files keep their paths. New files are marked `NEW`.

```
.
├── CLAUDE.md
├── PROGRESS.md               # handoff log, newest phase first
├── PREREG.md                 # NEW: hypotheses, metrics, configs, seeds, frozen before runs
├── docs/
│   └── clinical_validation_protocol.md   # NEW (Phase H11)
├── configs/
│   ├── base.yaml
│   ├── silos/                # five_agency.yaml (flat, existing)
│   ├── topologies/           # NEW: hierarchical topologies (regions, aggregators, nodes)
│   ├── experiments/
│   └── sweeps/
├── src/fedastro/
│   ├── data/
│   │   ├── wesad.py          # existing
│   │   ├── osdr.py           # NEW: OSD-435
│   │   ├── cinc2021.py       # NEW: PhysioNet/CinC 2021 ECG
│   │   ├── shift.py          # NEW: location-shift injection
│   │   ├── stream.py         # NEW: time-ordered streams for test-then-train
│   │   ├── partition.py      # extended: node/region partitioning, per-node splits
│   │   └── synthetic.py
│   ├── models/model.py       # extended: logistic regression arch
│   ├── fl/
│   │   ├── hierarchy.py      # NEW: regional + global aggregation
│   │   ├── secagg.py         # NEW: simulated pairwise-mask secure aggregation
│   │   ├── versioning.py     # NEW: on-the-fly model updates with validation gate
│   │   ├── strategy.py       # extended: server optimizers for gap closing
│   │   ├── comms.py          # extended: aggregator-relative latency
│   │   └── client.py / server.py
│   ├── privacy/              # unchanged; DP off by default
│   ├── metrics/
│   │   ├── disagreement.py   # NEW
│   │   ├── calibration.py    # NEW: ECE, reliability bins
│   │   └── stream.py         # NEW: test-then-train metrics
│   ├── peek.py               # NEW: terminal view, read-only, on request
│   └── cli.py
├── scripts/
│   ├── fetch_data.py         # extended: OSD-435, CinC subset/full
│   ├── run_report.py
│   └── make_figures.py       # NEW: slide-ready figures (16:9 PNG + PDF)
└── tests/
```

---

## 4. Topologies

### 4a. Flat (existing, locked)
`configs/silos/five_agency.yaml`: `nasa_earth`, `esa_analog`, `iss_commercial`,
`lunar_habitat`, `mars_transit`. Used for the reproducibility lock and for OSD-435.

### 4b. Hierarchical — CinC 2021 (primary)
Three regional aggregators plus one global aggregator:
- `agg_leo`: neutral orbital data center in LEO; also hosts `agg_global`.
- `agg_moon`: lunar-orbit data center.
- `agg_mars`: Mars-orbit data center.
- `agg_global`: aggregates the three regional models.

**15 nodes, 5 per region.** Agency analogs map to CinC 2021 sources (scenario labels):

| Agency analog | CinC source(s) |
|---|---|
| NASA | Georgia G12EC (USA) |
| ESA | PTB-XL + PTB (Germany / Europe) |
| Roscosmos | St Petersburg INCART (Russia) |
| CMSA-1 | Ningbo First Hospital |
| CMSA-2 | Chapman-Shaoxing |
| Partner/commercial | CPSC + CPSC-Extra |

| Region | Nodes |
|---|---|
| LEO | NASA, ESA, Roscosmos, CMSA-1, Partner |
| Moon | NASA, ESA, CMSA-1, CMSA-2, Partner |
| Mars | NASA, ESA, CMSA-1, CMSA-2, Partner |

Rules:
- Roscosmos is LEO-only (INCART is too small to split). It is the deliberately tiny silo.
- A source's patients are divided across its regions with Dirichlet quantity shares
  (configurable, seeded). **No patient appears in two nodes.**
- Each node splits its own patients 70 / 10 / 20 into train / validation / test,
  patient-disjoint. Validation is frozen and used only for tuning and the update gate.
- **Regional aggregation runs only when ≥ `min_participants` (5) nodes report.** Below
  that, the region falls back to the global model for that round, and the fallback is
  logged. A region drops below threshold whenever one node is in loss of signal.

### 4c. Hierarchical — WESAD
WESAD has 12 training subjects after holdout (13, 16, 17), so: **12 nodes, 4 per region,
`min_participants: 4`**, one subject per node. The test set is the existing global
held-out subjects. Each node's validation set is a time-ordered tail split of its own
subject, used only for the update gate and tuning. Record this asymmetry in PROGRESS.md.

### 4d. OSD-435
Too few samples for the hierarchy. Run on a flat topology only (Phase H9).

### 4e. Latency model (scenario assumptions unless sourced)
- Node ↔ regional aggregator: short light-time with **contact windows** (orbital passes).
- `agg_leo` ↔ `agg_global`: co-located, zero.
- `agg_moon` ↔ `agg_global`: ~1.2–1.3 s one-way, periodic loss of signal.
- `agg_mars` ↔ `agg_global`: existing Mars presets (`near` 182 s, `mean` 751 s, `far`
  1337 s) plus a solar-conjunction blackout window.
- Reuse `fl/comms.py`; latency becomes relative to the receiving aggregator.

---

## 5. Phased work plan (branch `hier-fl`)

Phases 0–6 of the original plan are complete (see PROGRESS.md). Original Phases 7–9
(drift, Streamlit dashboard, cluster hardening) are superseded by the phases below.
Drift detection is future work (§9).

Each phase: **Goal → Deliverables → Acceptance checks → Gate.**
**Build order:** H0–H8 build the complete architecture, tested on synthetic data and
WESAD smoke configs. Then the data runs go in this order: **OSD-435 and WESAD first
(H9), then CinC (H10).** H11 produces figures and docs.

### Phase H0 — Branch setup and reproducibility lock
**Goal:** a clean starting point on `hier-fl` that provably reproduces existing results.
**Deliverables:**
- Check that the working tree is clean and Phase 6 is committed. **If not, stop and ask
  the user.** Do not commit on the user's behalf.
- Ask the user to tag the branch point (suggested tag: `pre-hier-fl`).
- A regression test reproducing the flat WESAD FedAvg / local-only / centralized
  results recorded in PROGRESS.md, within a stated tolerance.
- `PREREG.md` skeleton containing H1–H4 from §0.
- `privacy.enabled` confirmed false in `base.yaml`.
**Acceptance:** `pytest` passes, including the regression test. Existing configs run unchanged.
**Gate.**

### Phase H1 — Hierarchical topology config
**Goal:** regions, aggregators and nodes expressible in config.
**Deliverables:**
- `configs/topologies/` schema: regions, aggregators, node → region mapping,
  `min_participants`, per-link latency and loss-of-signal profiles.
- `cinc15.yaml` and `wesad12.yaml` topologies per §4.
- Partitioning: per-node train/val/test with patient disjointness asserted.
- Validation in the strict config loader.
**Acceptance:**
- No patient in two nodes or two splits (tested).
- The flat topology still loads.
- Per-node size and label tables print on synthetic data.
**Gate.**

### Phase H2 — Location-shift simulation
**Goal:** give regions a reason to differ, so H2 is testable.
**Deliverables:** `data/shift.py`, which applies a region-dependent transformation to
inputs:
- **Covariate shift:** region-specific baseline offset and scale on selected channels
  or features.
- **Concept shift:** a configurable change in the feature–label relationship.
- **Magnitude** parameter; 0 = control.
- Applied identically to train, val and test within a region.
- Recorded in the manifest with `shift: SIMULATED`.
**Acceptance:**
- Magnitude 0 is bit-identical to no shift.
- Shifted regions show a measurable distribution distance from unshifted ones (reported).
- Documented as simulated in code and in PROGRESS.md.
**Gate.** Before closing, present 2–3 shift designs with a recommendation; the user picks.

### Phase H3 — Hierarchical FL
**Goal:** local, regional and global models trained together, with a threshold fallback.
**Deliverables:**
- `fl/hierarchy.py`:
  - Nodes train locally.
  - Each regional aggregator runs FedAvg over its region every round, if the threshold is met.
  - The global aggregator combines regional models every `global_every` regional
    rounds, weighted by sample count.
- Simulated time from §4e.
- Three models evaluated per node on its test set:
  - `local`: local-only
  - `regional`
  - `global`
- Existing `centralized` and `local_only` arms kept.
- `metrics/disagreement.py`:
  - Per-example agreement across the three models.
  - Disagreement rate per node, region and global.
  - Accuracy on agreeing vs disagreeing subsets (H3).
- Fallback events logged per round.
**Acceptance:**
- One command produces all arms plus the hierarchy into one result file.
- With one region and `global_every: 1`, hierarchical reduces to flat FedAvg (tested).
- A node in loss of signal drops its region below threshold and triggers the fallback
  without crashing.
- Runs on laptop CPU with synthetic data in < 10 min.
**Gate.**

### Phase H4 — Simulated secure aggregation
**Goal:** the aggregator sees only the sum of updates.
**Deliverables:** `fl/secagg.py`:
- Pairwise masks from a seeded PRG, added by each node and cancelling in the sum.
- Dropout recovery modelled as extra protocol round trips, charged in simulated time
  and bytes.
- Minimum group size = `min_participants`.
- Applied at regional and global levels (configurable).
- Docs state that it is a protocol simulation, not cryptographically secure.
**Acceptance:**
- Aggregates with secure aggregation on equal aggregates with it off, within float tolerance (tested).
- No individual unmasked update is stored or logged anywhere (tested).
- Simulated-time and bytes overhead reported per round and per Mars preset.
**Gate.**

### Phase H5 — On-the-fly model updates and terminal peek
**Goal:** demonstrate updates arriving during a mission, safely.
**Deliverables:**
- `fl/versioning.py`:
  - Every node holds versioned regional and global models.
  - An incoming version is accepted only if its validation score is at least the
    current score minus a tolerance (set on validation, recorded in PREREG.md).
  - Rejected updates are kept out and logged.
- `peek.py` with `fedastro peek runs/<run_id>`, **only when requested**, read-only.
  Per round it shows:
  - simulated time
  - each node's state (training / in transit / loss of signal / fallback)
  - local, regional and global accuracy
  - disagreement rate
  - update accept/reject events

  A `--follow` mode tails a run in progress. Peek must never change results.
**Acceptance:**
- Results are identical with and without peek (tested).
- At least one rejected update appears in a scripted run (e.g. a deliberately degraded
  update injected in a test).
**Gate.**

### Phase H6 — Model comparison
**Goal:** H4, across logistic regression, MLP and 1D-CNN.
**Deliverables:**
- `arch: logreg` (single linear layer, softmax).
- The existing MLP and 1D-CNN.
- Features feed logreg and MLP; sequences feed the 1D-CNN.
- `metrics/calibration.py` (ECE, reliability bins).
- Reported per arm: accuracy, macro-F1, AUC, ECE, worst-node accuracy, parameter count,
  payload bytes.
**Acceptance:** all three architectures run through every arm and the hierarchy, and
appear in the result schema.
**Gate.**

### Phase H7 — Test-then-train evaluation mode
**Goal:** model the streaming deployment, where labels are scarce and late.
**Deliverables:**
- `data/stream.py`: each node's data in time order.
- Labels revealed for a configurable fraction (default 10%) after a configurable delay
  in rounds; the rest are masked.
- `metrics/stream.py`: each example is predicted before its label is revealed (test-
  then-train accuracy), then joins training once labelled.
- A separate result section; this never replaces the fixed-split results.
**Acceptance:**
- No example is scored after the model has trained on it (tested).
- Label fraction 1.0 with delay 0 behaves sensibly against fixed-split numbers (reported).
**Gate.**

### Phase H8 — Closing the centralized vs federated gap
**Goal:** do everything reasonable to shrink the gap, and report what remains.
**Deliverables:**
- A pre-registered grid, tuned **on validation only**:
  - FedProx μ
  - local epochs
  - server optimizer (FedAdam / server momentum)
  - learning rate
- SCAFFOLD as a clearly labelled stretch goal.
- A gap table: centralized − federated and centralized − hierarchical, per model family.
**Acceptance:**
- The grid and selection rule are frozen in PREREG.md before running.
- The selected configuration is chosen by a rule applied to validation scores, recorded.
- The remaining gap is reported honestly.
**Gate.**

### Phase H9 — Data runs, part 1: OSD-435 and WESAD
**Goal:** the completed architecture on the two existing datasets, first.
**Deliverables:**
- **OSD-435:** `data/osdr.py` via the OSDR API, with provenance and checksum.
  - **Plan mode must first inspect the dataset** (samples, features, targets) and
    propose a task. Stop for approval.
  - Runs: (a) a FLUID-style 2-silo IID split, (b) a non-IID split on the same data,
    flat topology, three-way comparison.
- **WESAD:** the full architecture on `wesad12.yaml` (H3–H7, with and without shift),
  5 seeds, plus the locked flat results for reference.
- Laptop CPU or the cluster CPU queue. No GPU needed.
**Acceptance:**
- Every PREREG.md configuration for these datasets produces results.
- H1–H4 are each answered (supported / not supported / inconclusive) with numbers.
**Gate.** Report the results tables before starting CinC.

### Phase H10 — Data runs, part 2: CinC 2021
**Goal:** the primary dataset, on the GPU cluster.
**Deliverables:**
- `data/cinc2021.py` reading the PhysioNet/CinC Challenge 2021 public training set
  (WFDB, v1.0.3), with provenance and checksums.
- **Subset first:** about 2k recordings for a smoke run, then about 20k for the main
  runs. The full set only if time allows.
- Preprocessing:
  - Resample to 100 Hz, 10 s windows.
  - INCART's 30-minute Holter records are segmented, with every segment kept with its
    patient.
  - Patient grouping enforced where IDs exist (PTB, INCART).
- Decisions to surface in plan mode, each with a recommendation:
  - **Target:** recommended primary is binary normal sinus rhythm vs any scored
    abnormality; multi-label as stretch.
  - **Leads:** recommended 2-lead (I, II), matching wearable/suit realism and the
    Challenge's "Will two do?" framing.
  - **Feature set for logreg/MLP:** per-lead summary statistics plus RR-interval
    statistics.
- `scripts/fetch_data.py` supports `--subset N`.
- **Time one run on the smoke subset, then extrapolate the full grid's cost before
  launching it.** If a run exceeds 15 min, propose cutting to 3 seeds or 20 rounds.
- Cluster SLURM scripts per §8: one grid variant per job.
**Acceptance:**
- Patient disjointness across nodes and splits is asserted.
- The per-node label and size table shows natural label and quantity skew.
- The smoke subset runs end to end on the laptop CPU.
- The main grid completes on the cluster.
- H1–H4 are answered with numbers.
**Gate.**

### Phase H11 — Figures, clinical-validation protocol, docs
**Goal:** presentation-ready outputs.
**Deliverables:**
- `scripts/make_figures.py` reads `runs/` only and emits 16:9 PNG and PDF figures:
  - Three-way and hierarchical accuracy per region.
  - The disagreement analysis (H3).
  - Model comparison (H4).
  - The gap table.
  - Secure aggregation overhead vs Mars preset.
  - Threshold-fallback timeline.
  - The DP-collapse discussion figure from the Phase 5 results.
- `docs/clinical_validation_protocol.md`:
  - Expert review of label definitions.
  - Operating-point sensitivity and specificity.
  - Calibration.
  - Worst-site performance.
  - Clinical-dataset re-runs.
  - Analog studies (bed rest, HERA, Antarctic stations) as the route to real validation.
- README update: hypotheses, the §2 limitations, exact repro commands, caveats (§9).
**Acceptance:** figures render legibly in greyscale. Every number in a figure traces to
a result file. The README reproduces one headline result from a clean clone.
**Gate.** Slides are planned separately with the user.

---

## 6. Run artifact schema

Every run writes `runs/<run_id>/result.json`. Existing fields are a stable contract and
must not change. This branch **adds** sections (bump `schema_version`). Unused sections
stay present with nulls; never silently drop a field.

```json
{
  "schema_version": "2",
  "...existing fields unchanged...": "...",
  "topology": {"name": "cinc15", "regions": [], "aggregators": [], "min_participants": 5},
  "shift": {"enabled": false, "kind": null, "magnitude": 0.0, "provenance": "SIMULATED"},
  "hierarchy": {
    "rounds": [{"round": 1, "sim_wallclock_s": 0.0,
                "regions": {"leo": {"participants": [], "fallback": false, "bytes": 0}},
                "global_aggregated": false}],
    "per_node": {"node_name": {"local": {}, "regional": {}, "global": {}}},
    "per_region": {}, "global": {}
  },
  "disagreement": {"per_node": {}, "per_region": {}, "global": {},
                   "acc_agree": null, "acc_disagree": null},
  "secagg": {"enabled": false, "simulated": true, "overhead_bytes": 0,
             "overhead_sim_s": 0.0, "dropout_recoveries": 0},
  "updates": {"tolerance": null, "events": [{"round": 1, "node": "", "model": "",
              "version": 0, "accepted": true, "val_before": null, "val_after": null}]},
  "stream": {"enabled": false, "label_fraction": null, "label_delay_rounds": null,
             "test_then_train": {}},
  "calibration": {"ece": {}}
}
```

---

## 7. Working protocol for Claude Code

### Plan mode first, then code mode — every phase
1. **Plan mode.** Announce the phase. Re-read its section and acceptance checks. Read
   the relevant code and the latest PROGRESS.md entry. Produce a plan covering:
   - files to create or change
   - tests to add
   - config changes
   - risks
   - every design decision not fixed in this file, with options and a recommendation

   **Write no code. Stop and wait for the user's approval.**
2. **Code mode.** Implement the approved plan only. If something forces a deviation,
   stop and say so; do not silently expand scope.
3. **Gate.**
   - Run `pytest` and the acceptance checks.
   - Update PROGRESS.md: phase, what was built, actual acceptance numbers, decisions
     made, open questions, deferrals.
   - Suggest a conventional commit message, e.g. `feat(fl): phase H3 hierarchical FL`.
   - Stop. The user commits.

### General
- One phase per session. Assume the next session has no memory of this one;
  PROGRESS.md is the handoff.
- Tests before gates. A phase is not done if `pytest` fails.
- Pre-register before running. Experiment configs, seeds, metrics and selection rules
  go into PREREG.md and are frozen before result-producing runs.
- Flag scope creep, and how long a phase is really taking.
- Prefer deleting over disabling, **except** DP code, which stays (default off).

---

## 8. Cluster (PACE ICE) and known pitfalls

### Cluster
- Work in scratch (`$SCR`); home quota is full. Scratch is purged after inactivity, so
  copy results back promptly. Store CinC data in scratch.
- SLURM: account `bios`, QOS `coc-ice`. Do not pass `-p`. GPU jobs:
  `--gres=gpu:l40s:1` (V100 also works). Blackwell nodes do not work with the pinned
  torch 2.2.2.
- One grid variant per job: `fedastro grid <sweep> --index i`.
- Setup and data fetch on the login node; compute only in jobs.

### Pitfalls
- **Pins:** torch 2.2.2 (Intel Mac wheels, needs numpy < 2), opacus 1.5.2, flwr 1.23.0.
  Do not upgrade.
- Ray client resources must be set, or 15 simulated nodes will thrash a laptop.
- **Patient disjointness:** CinC sources mostly have one record per patient, but PTB and
  INCART have several. Group by patient, or every number is inflated. Assert in tests.
- INCART records are 30 min at 257 Hz; the other sources are 10 s at 500 Hz. Resample
  and segment before partitioning, keeping each segment with its patient.
- CinC labels are SNOMED-CT codes. Use the Challenge's scored-label mapping; record it in
  the manifest.
- Class imbalance is real in every dataset. Report macro-F1, AUC and ECE alongside
  accuracy.
- Simulated wall-clock and real wall-clock stay in separate fields.

---

## 9. Out of scope for this branch, and caveats

### Later work (do not start)
1. Radiation-hardened processor latency profiles and a bit-flip robustness test.
2. Gravity level, radiation dose and mission day as input features.
3. A rule for crew moving between regions (e.g. Mars transit → Mars surface): which
   model they switch to, and whether their personal model carries over.
4. Drift detection for incident response, using disagreement spikes as a trigger.
5. Agency-calibrated data (Plan B) lives in a separate repo.
6. A browser dashboard (the terminal peek replaces it for now).

### Out of scope
Real hardware, real cross-agency deployment, access-controlled human spaceflight data,
DP-SGD as a headline result, environmental sensor data, real clinical validation, large
models or LLMs.

### Caveats to carry into docs and the presentation
1. A neutral aggregator is assumed; an aggregator could still turn attacker.
2. Labelling costs crew time.
3. Not deployable today; an Earth–ISS simulation is the near-term option.
4. The location shift is simulated, not measured.
5. Without DP there is no formal privacy guarantee.
6. All datasets are terrestrial or animal proxies for astronaut data.
