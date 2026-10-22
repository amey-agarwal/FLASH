# PROGRESS

Handoff log between sessions. Newest phase first. Assume no memory of prior sessions.

**Phase 5: attack fairness accepted by the user (2026-10-17). Awaiting commit.**
Phase 4 is also awaiting commit. Next: Phase 6, deep-space comms (plan at the end of
the Phase 5 section), when the user approves.
**GPU cluster:** the shakedown (Phase 3 plan) is still pending; the user cannot run it yet.
On the cluster, run one grid variant per job with `fedastro grid ... --index i`.

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

### Open questions for the user
- Robustness items 3 (multi-step FedAvg update), 5 (restarts), 7 (frozen settings,
  ready now), 8 (membership inference): implement any before the cluster run?

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

