# PROGRESS

Handoff log between sessions. Newest phase first. Assume no memory of prior sessions.

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

### Next: Phase 1 — Data layer (WESAD)
