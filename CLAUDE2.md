# CLAUDE2.md — Extension Plan: Agency Silos, FLUID Comparison, Statistical Rigour

> Second project instruction file for Claude Code. **Read `CLAUDE.md` first, then this
> file, then `PROGRESS.md`.** This file extends `CLAUDE.md`; it does not replace it.
> **Work one phase at a time. Stop at every gate.**

---

## 0. How this file relates to CLAUDE.md

- Everything in `CLAUDE.md` still applies: rules §1, framing §2, schema contract §6,
  working protocol §7, pitfalls §8.
- Phases 0–6 of `CLAUDE.md` are **complete**. Their findings are the *golden findings*
  (§3 below) and must stay reproducible.
- **This file supersedes `CLAUDE.md` only where it says so explicitly:**
  - `CLAUDE.md` Phases 7–9 are replaced by Phases 10–18 below.
  - The Streamlit dashboard is cut. `scripts/run_report.py` is the booth demo.
  - `CLAUDE.md` §9 "Out of scope" is narrowed in §9 below (open terrestrial analog data
    is now in scope; access-controlled data is still out).
- If the two files conflict anywhere else, **stop and ask**. Do not pick one silently.

---

## 1. What changes, and why

The repo currently demonstrates FL mechanisms (latency, DP, gradient-inversion attack,
async aggregation) on WESAD, split into silos that are **scenario labels** with
**artificial** heterogeneity.

The target is the same mechanisms, plus:

1. **Natural agency-analog silos.** Open terrestrial health data whose silos are real
   countries or cohorts, mapped to space agencies. The heterogeneity is natural, not
   partitioned.
2. **A direct comparison with FLUID** on NASA OSDR OSD-435: replicate FLUID's IID
   two-silo setup, then the non-IID case FLUID's authors flag as future work.
3. **Accuracy on new individuals, with statistical significance.** Subject-disjoint and
   group-disjoint evaluation, paired confidence intervals, and stronger baselines.
4. **Mission dynamics.** Agency silos inherit link conditions from mission locations;
   partners can join mid-run (plug-and-play); latency can change during a run.
5. **Stretch:** streaming data with drift detection, and a resource profile.

### Hypotheses

| ID | Hypothesis | Dataset | Poster-critical |
|---|---|---|---|
| H1 | (unchanged, `CLAUDE.md` §0) FL beats local-only on non-IID sparse data | WESAD | yes, already shown |
| H2 | On natural agency-analog silos, FL beats local-only **and** an ensemble of local models on held-out individuals and held-out cohorts | agency-analog | yes |
| H3 | FLUID's IID two-silo result does not carry over unchanged to non-IID splits of the same data | OSD-435 | yes |
| H4 | A partner joining mid-run gets a better model on day one from the federation than from its own data alone | agency-analog (+ WESAD) | yes |
| H5 | A drift detector fires within a bounded number of rounds, with a reported false-positive rate | WESAD (time-ordered) | stretch |
| — | Resource profile (time, memory, payload) is *descriptive*, not a hypothesis | all | stretch |

Every hypothesis can fail. **A failure is a result.** Record it; never tune to make FL win.

---

## 2. Additional non-negotiable rules (8–15)

Rules 1–7 of `CLAUDE.md` §1 still apply.

8. **Golden findings stay reproducible.** No change may alter the run id or the results
   of any existing config. Enforce it with the golden test tiers in §3. New features enter
   only through **new, optional config blocks that are excluded from the experiment hash
   when unused**, which is the mechanism the repo already uses. Never change the default
   of an existing field.
9. **Pre-register before the first real run.** For each new experiment, write
   `prereg/<phase>-<name>.md` and commit it *before* running on real data. It must state:
   - splits and held-out groups;
   - seeds and holdout redraws;
   - metrics, the primary comparison and its success criterion;
   - any thresholds (drift onset, join round, latency schedule).

   Results are reported against the pre-registration. Deviations are listed, never hidden.
10. **Provenance gains a `domain` field.** Values: `terrestrial_proxy` (WESAD),
    `terrestrial_analog` (agency-analog data), `spaceflight_rodent` (OSD-435),
    `synthetic`. The field is additive and optional, so old artifacts stay valid. Figure
    captions must state the domain.
11. **Natural silos are never re-mixed.**
    - A country or cohort is one silo, or is split only *within* itself.
    - A held-out group (cohort or agency) contributes **nothing** to training: no
      samples, no scaler statistics, no feature selection.
    - All preprocessing is fitted per silo on its training split, or computed as an
      explicit federated statistic (the same pattern as the existing scaler exchange).
    - Feature selection on pooled data is forbidden.
12. **Statistics for any "A beats B" claim.**
    - Compute **paired** differences over seeds × holdout redraws.
    - Report the mean difference with a 95% bootstrap CI, reusing
      `metrics/aggregate.bootstrap_ci`.
    - The primary comparison per hypothesis is named in the pre-registration. Everything
      else is labelled exploratory.
    - Report how many comparisons were made. Do not report p-values without CIs.
13. **Agency names are analog labels.**
    - Silos may carry agency names only with a `domain` of `terrestrial_analog` or
      `synthetic`, and every caption says "agency-analog".
    - A silo's mission location sets its link profile only; it never changes its data.
    - GEO hosts no crewed mission, so any GEO profile is labelled hypothetical.
    - Agencies with no open data (e.g. Roscosmos) are **absent**, not simulated. They may
      appear only as a future "join" event in docs.
14. **China is excluded from agency-analog silos by default.** It may be included only
    through an explicit config flag, recorded in the manifest.
15. **Time-ordered claims need time-ordered data.** Streaming and drift experiments use
    WESAD, which is time-ordered within each subject. Never invent an order for
    cross-sectional data such as microbiome samples.

---

## 3. Golden findings and the reproducibility lock

### Findings to preserve

Exact numbers live in `PROGRESS.md`; do not restate them from memory.

| ID | Finding | Source config(s) |
|---|---|---|
| G1 | Three-arm H1 on WESAD by-subject, same run id hash `ce31e25dda` laptop vs cluster | `configs/experiments/wesad_by_subject.yaml` |
| G2 | Federated performance vs Dirichlet α, with the quantity-control arm | `configs/sweeps/phase4_alpha.yaml`, `wesad_dirichlet_control.yaml` |
| G3 | FedAvg vs FedProx and the μ sweep | `configs/sweeps/phase4_by_subject.yaml`, `phase4_mu.yaml` |
| G4 | DP utility collapse at small ε; attack success depends on batch size (B=1 vs B=32) | `configs/sweeps/phase5_eps.yaml` (+ `phase5_eps_cluster.yaml` once run) |
| G5 | FedBuff speed comes from excluding far silos; a deadline wastes time on a dark silo; sync stalls without one | `configs/sweeps/phase6_presets.yaml`, `phase6_dark.yaml` |

### Test tiers (built in Phase 10)

- **Tier A (fast, every `make test`):**
  - For every existing config and sweep variant, the computed run id equals the one
    recorded in `tests/golden/run_ids.json`. This is hash computation only, no training.
  - The smoke config result is byte-identical to `tests/golden/smoke_result.json`.
- **Tier B (slow, `make golden`, cluster or overnight):**
  - Re-run a pinned subset (G1 at 5 seeds, plus one variant each of G2–G5).
  - Compare final metrics with `tests/golden/metrics.json` at a tolerance of 1e-3, the
    same tolerance used for the laptop-vs-cluster check.
  - Tier B runs at every phase gate that touches `src/`, and before the release tag.

**If a golden check fails, stop.** Report which finding moved and why. Never update a
golden file to make a test pass without explicit user approval recorded in `PROGRESS.md`.

---

## 4. Phase dependency graph

```
P10 Reproducibility lock ─┬─> P11 Dataset registry + provenance domain
                          │        │
                          │        ├─> P13 Agency-analog natural silos ─┐
                          │        │                                     ├─> P15 Agency topology,
                          ├─> P12 Evaluation & statistics upgrade ──────┤     mission join,
                          │        │                                     │     time-varying latency
                          │        └─> P14 OSD-435 + FLUID comparison    │          │
                          │                                              │          v
                          │                                              │    P16 Streaming + drift (stretch)
                          │                                              │          │
                          └──────────────────────────────────────────────┴─> P17 Resource profile (stretch)
                                                                                    │
                                                    P18 Figures, report, README, release tag
```

**Session order:** P10 → P11 → P12 → P13 → P14 → P15 → P16 → P17 → P18.

- P12 comes before P13 and P14 so new datasets get rigorous evaluation from their first
  run.
- P13 and P14 are independent once P11 and P12 are done. P15 needs P13 (agency silos)
  and P12 (day-one metric).
- **Poster cut line:** P10–P15 and P18 are required. P16 and P17 are stretch: if they
  aren't done, the poster lists them as future work.

---

## 5. Phases

Each phase: **Goal → Depends on → Deliverables → Acceptance → Gate.**
Every gate also requires: `make test` passes (including golden Tier A); Tier B passes if
`src/` changed; `PROGRESS.md` updated with real numbers; conventional commit; stop.

### Phase 10 — Reproducibility lock
**Goal:** freeze the current state so every later change is checked against it.
**Depends on:** nothing.
**Deliverables:**
- `tests/golden/run_ids.json`: run ids for every existing config and sweep variant.
- `tests/golden/smoke_result.json`.
- `tests/golden/metrics.json`: final metrics of the pinned Tier-B subset, taken from
  **clean-commit** runs. If only `-dirty` runs exist, list the configs that need a
  clean-commit re-run and stop for the user to run them on the cluster.
- `make golden` target.
- The frozen `privacy.attack` block and commit sha recorded in `PROGRESS.md` (Phase 5
  deferred item 7).
- SLURM scripts under `scripts/slurm/` for Tier B and the pending `p5epsc` sweep: account
  `bios`, QOS `coc-ice`, no `-p`, one variant per job via `fedastro grid ... --index i`.
- A git tag `v0.6-golden` on the locked commit.

**Acceptance:**
- Tier A passes.
- Tier B is either passing, or explicitly pending cluster runs with the job scripts ready.
- No file under `src/` changed in this phase.

**Gate.**

### Phase 11 — Dataset registry and provenance domain
**Goal:** add datasets without editing the pipeline each time.
**Depends on:** P10.
**Deliverables:**
- A registry in `data/__init__.py` (decorator or dict). `pipeline.load_dataset` and the
  `config.DATASETS` validation read from it. `synthetic` and `wesad` are registered
  through it.
- `provenance.domain` (rule 10), optional in the schema and `run_report.py`.
- A generic `GroupedDatasetBundle` capability: `group_id` per sample (country/cohort),
  alongside the existing `subject_id`.
- A partitioner `by_group` (one silo per group, or per mapped agency) and
  `holdout_groups` (whole groups excluded from training).

**Acceptance:**
- Golden Tier A unchanged.
- A test asserts that the registry path gives byte-identical smoke results.
- A test asserts that held-out groups contribute no samples and no scaler statistics.

**Gate.**

### Phase 12 — Evaluation and statistics upgrade
**Goal:** make "generalises to new individuals" a defensible, measured claim.
**Depends on:** P10 (P11 helps but isn't required).
**Deliverables:**
- **`ensemble_of_locals` arm:** average the local models' predicted probabilities, with
  no FL. It is a new optional arm and doesn't change the existing arms.
- **Holdout redraws:** an `evaluation.holdout_draws` option (default 1 = current
  behaviour) re-drawing which subjects or groups are held out.
- **Paired statistics** (`metrics/stats.py`):
  - federated − local_only, federated − ensemble_of_locals, and federated − centralized;
  - computed over seeds × draws, with bootstrap CIs, written to a new optional `stats`
    block.
- **Fairness reporting:** worst-silo and best-to-worst gap as first-class metrics in the
  artifact and in `run_report.py`.
- **Held-out-group evaluation:** when `holdout_groups` is set, report per-group metrics.

**Acceptance:**
- Golden Tier A unchanged, because every new block defaults off.
- Run once on WESAD by-subject with 5 seeds × 3 draws.
- Report whether H1 survives the ensemble baseline and the redraws, honestly either way.

**Gate.** Pre-register the WESAD redraw run (rule 9) before running it.

### Phase 13 — Agency-analog natural silos
**Goal:** real, naturally heterogeneous silos mapped to agencies.
**Depends on:** P11, P12.

**Decision required at the start of the session.** Present these options and wait for
the user's choice:

| Option | Silos → agencies | Access | Notes |
|---|---|---|---|
| A. Colorectal cancer gut microbiome (curatedMetagenomicData, Bioconductor) | USA→NASA, Japan→JAXA, EU cohorts→ESA, Canada→CSA, India→ISRO; China cohort dropped (rule 14) | Open; R export step | Same label (CRC vs control) and uniform processing. Weak India and Canada cohorts; lab effects are confounded with country |
| B. PhysioNet/CinC 2016 heart sounds | USA, Denmark, France, Greece, Iran (+ Brazil and Bangladesh hold-outs) | Open, `wget` | Feature extraction from audio; tiny Greek and French silos; no patient ids |
| C. WHO STEPS hypertension | Space-programme countries | Licensed, needs approval | Only if files have arrived |

**Deliverables (after the decision):**
- A loader registered via P11, with provenance (source, version, checksum,
  `domain: terrestrial_analog`).
- `scripts/fetch_data.py` support. For option A, a pinned `Rscript` export to CSV, with
  the R package version recorded.
- **Leak-free features:**
  - per-sample transforms only (e.g. log or CLR of abundances);
  - any feature filter is either a fixed pre-registered list or a **federated**
    prevalence statistic, never computed on pooled data.
- `configs/silos/agency_analog.yaml`, with agency names and no link profile yet (P15
  adds it).
- Experiments: the three arms + `ensemble_of_locals`, with held-out individuals within
  each silo and one whole held-out cohort or country.
- A silo-size learning curve (pre-registered subsample sizes) to back the word "sparse".

**Acceptance:**
- Per-silo n and label counts are printed.
- Disjointness of individuals and of groups is asserted in tests.
- H2 is reported against its pre-registration with paired CIs.
- Golden Tier A is unchanged.

**Gate.**

### Phase 14 — OSD-435 and the FLUID comparison
**Goal:** state the contribution in FLUID's own terms.
**Depends on:** P11, P12.
**Deliverables, in order. Stop at step 2.**
1. Read the FLUID preprint (bioRxiv 10.1101/2025.01.14.633017) and record in
   `PROGRESS.md`:
   - FLUID's task and label;
   - its features and its split;
   - its metric and its reported result, quoted with page or figure reference.

   Record only what the paper says. Do not fill gaps.
2. Fetch OSD-435 metadata through the OSDR API and report: sample count, groups
   (flight / ground / other), assay type, file sizes. **Stop and wait for the user to
   approve the split design.**
3. After approval: a loader with provenance (`domain: spaceflight_rodent`, study version,
   checksums). Gene or feature selection happens inside each silo, or uses a fixed list
   chosen before seeing labels.
4. **Replication:** FLUID's IID two-silo 50/50 split, matching its task as closely as the
   data allows. List every difference from FLUID.
5. **Extension:** the pre-registered non-IID split(s). Use the same three arms + ensemble.

**Acceptance:**
- The replication and the extension sit side by side in one report.
- With tiny n, report CIs and say plainly where nothing can be concluded.
- Every reference to FLUID follows `CLAUDE.md` §2.

**Gate.**

### Phase 15 — Agency topology, mission join, time-varying latency
**Goal:** cross-agency federation under mission conditions, with plug-and-play partners.
**Depends on:** P12, P13 (P14 is not required).
**Deliverables:**
- **Topology:** `configs/silos/agency_locations.yaml`. Each agency-analog silo gets a
  *scenario* mission location (LEO, lunar orbit, lunar surface, Mars orbit, Mars
  surface; GEO only labelled hypothetical). The location maps to a link profile built
  from the existing presets and loss-of-signal models. Location assignments are scenario
  choices, written in the pre-registration.
- **Join:** `comms.join` (silo, from_round or from_s).
  - A joining silo receives the current global model and the existing global scaler,
    and contributes nothing before it joins.
  - Supported for FedAvg, FedProx and FedBuff.
- **Day-one metric:** the global model's performance on the joiner's held-out data
  **before its first update**, compared with the joiner's local-only model and with the
  final federated model. Paired CIs.
- **Time-varying latency:** a piecewise-linear latency schedule per silo (e.g. a transfer
  from LEO to lunar orbit).
- Recorded in a new optional `membership` block: join events and schedule
  breakpoints.

**Acceptance:**
- Existing comms-off and comms-on run ids are unchanged (golden Tier A).
- The scheduling of a run with a join is identical to the no-join run before the join.
- H4 is reported against its pre-registration on agency-analog data. A WESAD replicate
  is optional.

**Gate.**

### Phase 16 — Streaming data and drift (stretch)
**Goal:** show the model can be updated as new data arrives, and that drift is noticed.
**Depends on:** P15 (it reuses the time-scheduling machinery), P11.
**Deliverables:**
- **Streaming:** `data.stream` releases each silo's training data in time-ordered
  windows. This is WESAD only (rule 15). Each round trains on the data released so far.
- **Drift:** `fl/drift.py`.
  - A covariate shift that ramps in one silo from a pre-registered onset.
  - A windowed performance-drop detector setting a per-silo flag.
  - Fill the existing `drift` schema fields.
- **Mitigation (stretch within the stretch):** one simple rule fixed a priori, such as
  resetting the drifted silo's local optimiser or down-weighting it until it recovers.

**Acceptance:**
- An undefended run degrades after onset.
- Detection delay is reported in rounds.
- The false-positive rate on a matched no-drift control is reported.
- Golden Tier A is unchanged.

**Gate.**

### Phase 17 — Resource profile (stretch)
**Goal:** an honest descriptive answer to "can this run on constrained hardware?"
**Depends on:** P12. Run it after P16 so it profiles the final code.
**Deliverables:**
- An optional `resources` block, recorded per silo per round:
  - parameter count and update payload in bytes;
  - local training time;
  - peak RSS memory.
- A profiling config with `torch.set_num_threads(1)`.
- A table in `run_report.py`.

No optimisation (quantisation, pruning) in this phase; list it as future work.

**Acceptance:**
- The numbers are produced for WESAD and for the agency-analog data.
- The README states that onboard power and radiation-tolerant hardware are not
  evaluated.

**Gate.**

### Phase 18 — Figures, report, README, release
**Goal:** a poster-ready, reproducible release.
**Depends on:** all phases that were done.
**Deliverables:**
- `scripts/make_figures.py` (vector PDFs, readable in greyscale at print size).
  Panels:
  - H1 three-arm curve with CIs;
  - H2 agency-analog per-silo and held-out-cohort results with CIs;
  - H3 FLUID replication vs non-IID;
  - H4 day-one value of joining;
  - privacy before/after (B=1 next to B=32);
  - simulated time-to-target per preset;
  - worst-silo fairness;
  - H5 drift (if done).
- Every caption states n seeds, holdout draws, data domain and provenance.
- `run_report.py` extended to cover the new blocks.
- `README.md`:
  - the hypotheses;
  - the honest limitations (`CLAUDE.md` §2 plus: analog data; agency mapping is a
    scenario; lab effects confounded with country; OSD-435 is rodent data with tiny n;
    DP and the attack were tested on synchronous FedAvg only; subject-level ε is vacuous;
    GEO is hypothetical; Roscosmos is absent);
  - exact repro commands per figure.
- Golden Tier A and Tier B pass.
- A tag `v1.0-asgsr2026`, created by the user, not by Claude Code.

**Gate.**

---

## 6. Schema additions (additive only)

All of these are optional. They are `null` or absent when unused and are excluded from
the experiment hash when unused. No existing field is renamed, re-typed or removed.

```json
{
  "provenance": [{"dataset": "str", "version": "str", "checksum": "str",
                  "domain": "terrestrial_proxy | terrestrial_analog | spaceflight_rodent | synthetic"}],
  "groups": {"silo_to_groups": {}, "holdout_groups": []},
  "arms": {"ensemble_of_locals": {"final": {}, "per_holdout_group": {}}},
  "evaluation": {"holdout_draw": 0, "per_holdout_group": {}},
  "stats": {"primary": "str", "comparisons": [
            {"a": "federated", "b": "local_only", "metric": "str",
             "mean_diff": 0.0, "ci95": [0.0, 0.0], "n_pairs": 0}]},
  "fairness": {"worst_silo": {}, "gap": {}},
  "membership": {"joins": [], "latency_schedule": {}},
  "stream": {"enabled": false, "windows": []},
  "resources": {"params": null, "update_bytes": null, "per_silo": {}},
  "prereg": {"file": "str", "deviations": []}
}
```

Update `run_report.py` (and `make_figures.py` once it exists) in the same phase that adds
a block.

---

## 7. Experiment matrix

| Capability | WESAD | Agency-analog | OSD-435 |
|---|---|---|---|
| Three arms + ensemble | P12 | P13 | P14 |
| Held-out individuals | existing + P12 redraws | P13 | P14 (if n allows) |
| Held-out whole group | — | P13 | — |
| Silo-size curve | optional | P13 | — |
| DP + attack | existing (G4) | optional, one ε variant | no (n too small) |
| Link presets / FedBuff | existing (G5) | P15 | no |
| Join / time-varying latency | optional replicate | P15 | no |
| Streaming + drift | P16 | no (not time-ordered) | no |
| Resource profile | P17 | P17 | no |

---

## 8. Working protocol additions

- **Start of every session:**
  1. Read `CLAUDE.md`, `CLAUDE2.md` and `PROGRESS.md`.
  2. Name the phase being started and re-read its acceptance checks.
  3. Run golden Tier A before any edit, to confirm a clean starting point.
- **Pre-registration first:** for any phase running a new real-data experiment, write and
  commit the `prereg/` file before the first real run.
- **Decision stops:** P13 (dataset choice), P14 step 2 (split design), and any time a
  golden check fails. At a stop, state the options with your recommendation and wait.
- **Scope flags:** if a phase is taking much longer than its role in the poster
  warrants, say so and propose a cut rather than silently expanding.
- **PROGRESS.md entries** for these phases are headed `## Phase 1x — ...`, newest first,
  matching the existing style.

---

## 9. Out of scope (narrowed from CLAUDE.md §9)

**Still out:**
- access-controlled human or astronaut data (NASA LSDA/LSAH, ESA HREDA restricted data,
  credentialed PhysioNet);
- real hardware;
- real cross-agency deployment;
- large models;
- secure aggregation implementation (describe the protocol only);
- model compression;
- personalised or hierarchical FL (future work);
- anything that cannot be rehearsed offline.

**Now in scope:** open, terrestrial, country- or cohort-level health data used as an
agency analog, under rules 10–14. Agency-calibrated *synthetic* silos belong to the
separate Plan B repository, not here.
