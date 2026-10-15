# PROGRESS

Handoff log between sessions. Newest phase first. Assume no memory of prior sessions.

**Next: Phase 3 — Federated core (FedAvg)**, after review of Phase 2.

---

## Phase 2 — Model + non-FL baselines — COMPLETE, awaiting review (2026-10-15)

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
4. **Federated standardisation for Phase 3 (recommended).** Clients share
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

