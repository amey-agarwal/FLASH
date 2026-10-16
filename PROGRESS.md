# PROGRESS

Handoff log between sessions. Newest phase first. Assume no memory of prior sessions.

**Next: Phase 5 — Privacy (Opacus DP + gradient inversion)**, after review of Phase 4.
**GPU cluster:** the shakedown (Phase 3 plan) is still pending. Phase 4 ran fully on the laptop
in about 75 min; on the cluster, run one grid variant per job with `fedastro grid ... --index i`.

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

