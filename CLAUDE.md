# CLAUDE.md — Cross-Silo Sparse Federated Learning for Autonomous Astronaut Healthcare

> Project instruction file for Claude Code. Read this fully before touching any code.
> **Work one phase at a time. Do not attempt the whole project in a single session.**

---

## 0. What this repo is

A reproducible simulation that tests one hypothesis, backing an ASGSR 2026 poster
(42nd Annual Meeting, 2–5 Dec 2026, Crystal Gateway Marriott, Arlington VA).

**Hypothesis (H1):** Cross-silo federated training on non-IID, sparse physiological
data produces a model that generalises better than any single silo's locally-trained
model, at a quantifiable privacy and communication cost.

The repo must produce, for every experiment, an honest **three-way comparison**:

| Arm | Meaning | Role |
|---|---|---|
| `centralized` | all silo data pooled, trained normally | upper bound |
| `federated` | FedAvg / FedProx across silos, no data movement | the thing under test |
| `local_only` | each silo trains alone, evaluated on the global held-out set | lower bound |

If `federated` loses to `local_only` under some condition, **that is a result, not a
bug.** Report it. Never tune to make FL look good.

### Execution environment
- **Development:** laptop, CPU, small subsets, fast iteration.
- **Training/eval:** a rented online GPU cluster, headless, from a git clone.
- Therefore: **no notebooks in the critical path**, no interactive prompts, no
  hardcoded paths, everything driven by config files + CLI. The dashboard runs
  locally and reads result artifacts; it is never required to produce results.

---

## 1. Non-negotiable rules

1. **Verify library APIs against the installed version.** Flower's API changed
   significantly across 1.x (`start_simulation` is deprecated in favour of
   `ServerApp`/`ClientApp` + `flwr run`). Check `flwr.__version__` and the installed
   package's own docs/docstrings before writing orchestration code. Do not write from
   memory of an older tutorial.
2. **No synthetic data standing in for real data without a loud label.** Every
   dataset loader must record its provenance (source, version, checksum) into the run
   manifest. A `--synthetic` smoke-test mode is allowed and must tag results
   `provenance: SYNTHETIC`.
3. **Seeded and deterministic.** Every run takes `--seed`; set Python/NumPy/Torch
   seeds. Multi-seed runs are the default for any number that goes on the poster
   (minimum 5 seeds, report mean ± std).
4. **Results are files, not stdout.** Every run writes a JSON artifact conforming to
   the schema in §6 under `runs/<run_id>/`. The dashboard and figure scripts consume
   only those files.
5. **Never fabricate numbers, citations, or results** in code comments, docstrings,
   READMEs, or figure captions. If a value is a placeholder, name it `PLACEHOLDER_`.
6. **Keep the model small.** A 1D-CNN or 2–3 layer MLP. Large models break the
   gradient-inversion demo (Phase 5), blow up Opacus per-sample-gradient memory, and
   make the live demo slow. Model capacity is not the contribution here.
7. **Stop at phase gates.** At the end of each phase, run the acceptance checks,
   update `PROGRESS.md`, commit, and report back. Do not roll into the next phase
   unprompted.

---

## 2. Scientific framing to preserve in code and docs

These are defensibility constraints. Violating them costs the poster.

- **FLUID (Casaletto et al., bioRxiv 10.1101/2025.01.14.633017) is prior art, not our
  result.** It is a real ISS-deployed FL framework (OpenFL on HPE Spaceborne
  Computer-2, deployed 2024-03-15) but its biomedical run used **IID mouse cardiac
  data (NASA OSDR OSD-435) artificially split 50/50 Earth/ISS**. It did *not*
  demonstrate non-IID multi-agency human FL. Any reference in this repo must say so.
  Our contribution is the non-IID, multi-silo, delay-and-drift case FLUID's authors
  flag as future work.
- **Astronaut N is tiny (tens).** Terrestrial/analog data is a stated proxy, never
  passed off as spaceflight data. Silo names are *scenario labels* for a simulation.
- **Differential privacy ε is a dial, not a certificate.** Report ε and δ explicitly;
  δ < 1/N_train. Do not describe any configuration as simply "private."
- Claims in docs must be of the form "we demonstrate the mechanism under simulated
  conditions," never "we validate an operational system."

---

## 3. Repo layout (target state)

```
.
├── CLAUDE.md               # this file
├── PROGRESS.md             # phase log — Claude Code updates this every session
├── README.md               # written last (Phase 9)
├── pyproject.toml          # pinned deps; Flower app config if required by version
├── configs/
│   ├── base.yaml           # defaults
│   ├── silos/              # silo topology definitions
│   └── experiments/        # one YAML per poster experiment
├── src/fedastro/
│   ├── data/
│   │   ├── base.py         # DatasetBundle interface + provenance record
│   │   ├── wesad.py        # PRIMARY dataset
│   │   ├── osdr.py         # OSD-435 via OSDR API
│   │   ├── eicu.py         # OPTIONAL, credential-gated
│   │   ├── partition.py    # per-subject / per-site / Dirichlet(α) partitioners
│   │   └── synthetic.py    # smoke-test only
│   ├── models/model.py
│   ├── fl/
│   │   ├── client.py
│   │   ├── server.py
│   │   ├── strategy.py     # FedAvg, FedProx, buffered-async
│   │   ├── comms.py        # latency / jitter / loss-of-signal
│   │   └── drift.py        # concept-drift injection + detector
│   ├── privacy/
│   │   ├── dp.py           # Opacus wrapper + accountant
│   │   └── attack.py       # gradient inversion (DLG/iDLG-style)
│   ├── baselines/          # centralized + local_only arms
│   ├── metrics/            # schema, aggregation, comms accounting
│   └── cli.py              # single entrypoint: `fedastro run <config>`
├── dashboard/app.py        # Streamlit, reads runs/, never writes
├── scripts/
│   ├── fetch_data.py
│   ├── sweep.py            # multi-seed / multi-config launcher for the cluster
│   └── make_figures.py     # runs/ -> poster-ready PDFs
├── tests/
└── runs/                   # gitignored artifacts
```

---

## 4. Silo topology (the simulated federation)

Five silos, defined in `configs/silos/`. Each is a scenario label with a data
partition and a comms profile.

| Silo | Data share | Latency (one-way) | LOS behaviour |
|---|---|---|---|
| `nasa_earth` | large, most representative | ~0 | none |
| `esa_analog` | medium | ~0 | rare |
| `iss_commercial` | medium | ~0.25 s | short, occasional |
| `lunar_habitat` | small | ~1.3 s | periodic |
| `mars_transit` | very small | 180 s – 1337 s (configurable) | long blackouts |

Latency figures are grounded: Earth–Moon ≈ 1.2–1.3 s; Earth–Mars one-way ranges from
~3 min 2 s at closest approach to ~22 min 17 s at maximum. Mars presets must expose
`near`, `mean`, `far`. Reference the real FLUID loss-of-signal event (~7.5 min pause)
as a calibration point for the LOS model.

---

## 5. Phased work plan

Each phase: **Goal → Deliverables → Acceptance checks → Gate.**
At the gate, run the checks, update `PROGRESS.md`, commit with a conventional message,
and stop.

### Phase 0 — Scaffold
**Goal:** installable, testable skeleton that runs end-to-end on synthetic data.
**Deliverables:** `pyproject.toml` with pinned deps (`flwr[simulation]`, `torch`,
`opacus`, `numpy`, `pandas`, `scikit-learn`, `pyyaml`, `streamlit`, `pytest`);
package skeleton; `cli.py` with `run`/`sweep` subcommands; config loader with
dataclass validation; seeding utility; run-manifest writer; `.gitignore`; CI-less
`make test` equivalent.
**Acceptance:** `pytest` passes; `fedastro run configs/experiments/smoke.yaml`
produces a valid `runs/<id>/result.json` using `synthetic.py`; a fresh clone +
install reproduces it byte-identically for a fixed seed.
**Gate.**

### Phase 1 — Data layer
**Goal:** real data in, honestly partitioned.
**Deliverables:** `DatasetBundle` interface (X, y, subject_id, provenance);
`wesad.py` (download via `scripts/fetch_data.py`, window + feature-extract the
chest/wrist signals, 3-class target: baseline/stress/amusement); `partition.py` with
`by_subject`, `by_site`, `dirichlet(alpha)`; a global held-out test set that **no silo
trains on**; provenance + checksum recording.
**Acceptance:** per-silo label distributions printed and visibly non-uniform; no
subject appears in two silos; held-out set disjoint from all silo training data
(assert in tests); loader completes in < 60 s on laptop CPU for one silo.
**Gate.** Report the per-silo class distribution table before proceeding.

### Phase 2 — Model + non-FL baselines
**Goal:** the upper and lower bounds, before any federation exists.
**Deliverables:** small 1D-CNN and MLP in `models/model.py` (selectable by config);
`baselines/centralized.py`; `baselines/local_only.py` (trains one model per silo,
evaluates each on the global held-out set); shared `train_local()` used later by FL
clients so arms are genuinely comparable.
**Acceptance:** centralized ≥ local_only on average across silos; 5-seed mean ± std
reported for both; training one arm < 5 min on laptop CPU.
**Gate.**

### Phase 3 — Federated core
**Goal:** FedAvg across the five silos, matching the baselines' training code.
**Deliverables:** `fl/client.py` (`get_parameters`/`fit`/`evaluate`);
`fl/server.py`; `fl/strategy.py` starting with FedAvg plus a metrics-aggregation hook
that records **per-silo** accuracy each round, not just the global mean;
communication accounting in `metrics/` (bytes per round = parameter payload × clients
× 2, rounds-to-target-accuracy).
**Acceptance:** H1 is now testable — one command produces all three arms into one
result file; federated curve lands between local_only and centralized (or does not,
and that is recorded); 5 silos × 20 rounds completes in < 10 min on laptop CPU.
**Gate.** This is the **minimum viable poster**. Everything after is upside.

### Phase 4 — Non-IID robustness
**Goal:** show where FL breaks and what fixes it.
**Deliverables:** FedProx (proximal term, μ configurable) in `strategy.py`; a
Dirichlet-α sweep config; per-silo fairness metric (accuracy spread across silos).
**Acceptance:** a sweep over α showing federated performance degrading as
heterogeneity rises; FedAvg vs FedProx compared at matched rounds and seeds, with an
honest verdict either way.
**Gate.**

### Phase 5 — Privacy
**Goal:** the demo's emotional core — leakage, then defence.
**Deliverables:** `privacy/dp.py` wrapping Opacus (`make_private_with_epsilon`,
δ = 1/N_train, `max_grad_norm` configurable, `get_epsilon()` recorded per run; handle
the BatchNorm incompatibility via `ModuleValidator`); `privacy/attack.py` implementing
gradient inversion against a single client update — optimise a dummy input to match
the shared gradient, report reconstruction error (MSE, and correlation per channel for
time-series); an attack-vs-ε sweep.
**Acceptance:** attack visibly succeeds at ε=∞ (no DP) on a small batch, and
reconstruction error rises monotonically as ε falls; utility-vs-ε curve produced over
a defensible range (sweep ε ∈ {0.5, 1, 2, 3, 5, 10, 13}; note in docs that ε in the
5–13 band is *utility-prioritised*, weaker than the ε<1 theoretical ideal); attack
runs on laptop CPU in < 2 min.
**Gate.** Ask for review of attack fairness before building figures — a rigged attack
(batch size 1, untrained net, cherry-picked sample) is the most likely methodological
objection from a judge.

### Phase 6 — Deep-space comms realism
**Goal:** make "communication-aware synchronisation" a measured quantity.
**Deliverables:** `fl/comms.py` — per-silo one-way delay, jitter, bandwidth cap, and
loss-of-signal windows with configurable duration/probability; simulated wall-clock
accounting (do **not** actually sleep for 22 minutes — model time, and keep a
`--real-sleep` flag only for the live demo); a buffered-asynchronous aggregation
strategy tolerating stragglers (FedBuff-style: aggregate once K updates land rather
than waiting for all); Earth/Moon/Mars-near/Mars-far presets.
**Acceptance:** time-to-target-accuracy reported in simulated wall-clock for each
preset; sync vs buffered-async compared under Mars-far with LOS enabled; a silo going
dark mid-run does not crash the federation.
**Gate.**

### Phase 7 — Drift
**Goal:** the long-duration-mission argument.
**Deliverables:** `fl/drift.py` — injectable covariate/label shift in a silo that
ramps over rounds; a simple detector (windowed performance drop / distribution
distance) raising a per-silo flag; optional drift-aware aggregation variant.
**Acceptance:** an undefended run visibly degrades after drift onset; the detector
fires within a bounded number of rounds; false-positive rate on a no-drift control
run is reported.
**Gate.**

### Phase 8 — Dashboard + figures
**Goal:** the poster-session artifact and the printed panels.
**Deliverables:** `dashboard/app.py` (Streamlit) reading `runs/` — silo topology view
lighting up per round, three-way accuracy curves, per-silo fairness, ε meter, comms
cost, attack before/after panel, drift flags; replay mode driven by a saved run so
nothing is computed live; `scripts/make_figures.py` emitting vector PDFs for four
poster panels (hero three-way curve; privacy before/after; latency/time-to-target;
silo fairness).
**Acceptance:** dashboard launches from a clean clone with only `runs/` present and
no GPU; replay of a saved run is deterministic; a scripted "demo loop" runs unattended
for 10 minutes without error; figures render legibly at print size in greyscale.
**Gate.**

### Phase 9 — Cluster hardening, docs, stretch
**Goal:** make it runnable by someone else, and scale it.
**Deliverables:** `scripts/sweep.py` for multi-seed/multi-config batch runs with
resumability and per-run logs; device selection (`cpu`/`cuda`) throughout; a cluster
quickstart; `README.md` with the hypothesis, honest limitations (§2), and exact repro
commands; OSD-435 loader via the OSDR API as the NASA-credibility silo; eICU loader
**only if** PhysioNet credentialing has cleared.
**Stretch (clearly labelled, cut freely):** scaling study to 20/50 simulated silos
with wall-clock and memory curves; SCAFFOLD; personalised per-silo heads; secure
aggregation (masked updates) — describe the protocol in docs even if unimplemented.
**Gate.**

---

## 6. Run artifact schema

Every run writes `runs/<run_id>/result.json`. Treat this as a stable contract —
the dashboard and figure scripts depend on it; changes require updating both.

```json
{
  "run_id": "str",
  "git_sha": "str",
  "timestamp_utc": "str",
  "config": { "...resolved config, fully expanded..." },
  "seed": 0,
  "provenance": [{"dataset": "WESAD", "version": "str", "checksum": "str"}],
  "silos": [{"name": "mars_transit", "n_train": 0, "label_counts": {}}],
  "arms": {
    "centralized": {"final": {"accuracy": 0.0, "auc": 0.0}, "history": []},
    "local_only":  {"per_silo": {}, "mean": {}},
    "federated":   {
      "rounds": [{"round": 1, "global": {}, "per_silo": {},
                  "bytes": 0, "sim_wallclock_s": 0.0, "participants": []}],
      "final": {}
    }
  },
  "privacy": {"enabled": false, "epsilon": null, "delta": null,
              "max_grad_norm": null, "attack": {"mse": null}},
  "comms": {"preset": "earth", "los_events": []},
  "drift": {"enabled": false, "onset_round": null, "detected_round": null},
  "notes": "str"
}
```

Unused sections stay present with nulls. Never silently drop a field.

---

## 7. Working protocol for Claude Code

- **One phase per session.** Announce which phase you are starting and re-read its
  acceptance checks first.
- **Tests before gates.** Each phase adds tests under `tests/`. A phase is not done if
  `pytest` fails.
- **Commit per phase** with a message like `feat(fl): phase 3 federated core (FedAvg)`.
- **Update `PROGRESS.md`** every session: phase, what was built, acceptance results
  (actual numbers), open questions, anything deferred. This is the handoff document
  between sessions — assume the next session has no memory of this one.
- **Surface decisions, don't bury them.** If a phase forces a design choice not
  specified here (feature-extraction window length, which WESAD signals to use,
  FedProx μ), state the options and your recommendation, and record the choice in
  `PROGRESS.md`.
- **Flag scope creep.** If something would take longer than its phase warrants, say so
  rather than silently expanding.
- **Prefer deleting over disabling.** No dead code paths behind `if False`.

---

## 8. Known pitfalls

- Flower's simulation engine is Ray-backed; client resource limits must be set or many
  simulated clients will thrash a laptop. Tune `client_resources` by environment.
- Opacus stores per-sample gradients — memory scales with batch size × params. Keep
  both small. BatchNorm is rejected outright; use GroupNorm/LayerNorm or
  `ModuleValidator.fix`.
- Gradient inversion gets much harder with large batches and trained networks. The
  demo must state the threat model explicitly (single update, small batch, known
  architecture) rather than implying universal leakage.
- Per-subject partitioning means the global held-out set must also be
  subject-disjoint, or every accuracy number is inflated. Assert this in tests.
- Simulated wall-clock ≠ real wall-clock. Keep them in separate fields and never plot
  them on the same axis without labelling.
- WESAD class imbalance is real; report AUC/macro-F1 alongside accuracy.

---

## 9. Out of scope

Real hardware (no Raspberry Pis — laptop only), real cross-agency deployment, any
access-controlled human spaceflight data requiring a Data Access Committee or IRB,
large models, and anything that cannot be rehearsed offline before the poster session.
