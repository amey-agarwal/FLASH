# FLASH: cross-silo sparse federated learning for autonomous astronaut healthcare

This is a reproducible simulation behind the ASGSR 2026 presentation *"A Cross-Silo Sparse
Federated Learning System Architecture for Autonomous Astronaut Healthcare Systems"* (42nd
Annual Meeting, Arlington VA, 2–5 December 2026).

**"Sparse" means sparse data:** small, siloed astronaut populations. It does not mean sparse
model updates.

The simulation models space agencies and missions as data silos. The silos never share data;
they share model updates through federated learning (FL). Nodes report to regional aggregators
in low Earth orbit (LEO), lunar orbit and Mars orbit, and the regional aggregators report to a
neutral global aggregator. On top of that the architecture adds:
- secure aggregation, as a protocol simulation;
- a validation gate on model updates;
- a threshold fallback when a region loses signal;
- a measure of disagreement between models, as a flag for clinician review.

**What it is:** a demonstration of the mechanism under simulated conditions.

**What it is not:** a validated operational or clinical system. Every dataset is a terrestrial or
animal **proxy** for astronaut data. Silo, region and agency names are scenario labels, not real
deployments.

## Hypotheses and results

The hypotheses were pre-registered in [PREREG.md](PREREG.md) before any result-producing run. The
decision rule (PREREG §2): each comparison is paired by seed with a hierarchical bootstrap 95% CI.
A comparison is supported if the CI is above 0, not supported if it is below 0, and inconclusive
otherwise. There are 5 seeds per configuration.

- **H1:** federated training beats each silo's local-only model.
- **H2:** under a location-dependent shift, regional models beat (a) local-only models and (b) the
  global model on their own region's test data. With no shift (the control), the global model is at
  least as good as the regional ones.
- **H3** (descriptive): accuracy is lower where the local, regional and global models disagree.
- **H4:** H1 and H2 hold across model families (logistic regression, MLP, 1D-CNN).

| | WESAD (12 nodes) | CinC 2021 (20k-record subset, 15 nodes) |
|---|---|---|
| H1, local-only scored on all regions' test data pooled (pre-registered primary) | supported, every family | supported, every family |
| H1, each node scored on its own test data (added in Phase H11a) | supported, every family | **not supported** on accuracy, every family |
| H2 | supported, every family | **not supported** (logreg, MLP); inconclusive (1D-CNN) |
| H3 | supported | supported |
| H4 | H1 and H2 hold in every family | H1 (primary) holds; H2 and H1 own-test do not |

The numbers and CIs are in [PROGRESS.md](PROGRESS.md), under "Phase H11a" → "Reruns on PACE". The
run directories are not in git; [BACKUPS.md](BACKUPS.md) says where they are kept.

### How to read the CinC results
- **Each CinC node holds many patients from a single hospital source.** On its own patients, a
  node's local model does about as well as the federated or regional model.
- **The federated advantage in the primary H1 row comes from testing local-only models on other
  agencies' patients.**
- **An exploratory follow-up** (not pre-registered; fig 8; PROGRESS.md "Small-node check") found
  that, with no shift, local models on most nodes almost never predict the minority (normal) class.
  Federation raises minority-class recall on most nodes. So the scarce data is the rarer condition,
  not the number of records.
- **The simulated concept shift makes the binary CinC task nearly separable** in the shifted
  regions. H2 and H3 margins on CinC are mostly an artifact of the shift.
- **Losses are reported as results.** Nothing was tuned to make FL win; every choice was made on
  validation data only.

### Other datasets
- **OSD-435** (NASA OSDR mouse cardiac data, the FLUID setup): flat topology only, too small for
  the hierarchy. Results are in PROGRESS.md "Phase H9", "OSD-435 results".
- **The flat five-silo WESAD results** from earlier phases are locked by a regression test (below).

## Limitations

- **Proxy data.** WESAD (wearable stress), CinC 2021 (ECG) and OSD-435 (mouse cardiac) stand in
  for astronaut data. Astronaut N is tiny, and none of this is astronaut data.
- **The location shift is simulated.** None of the datasets contains real location-dependent
  physiology. Every shifted result says so.
- **Privacy rests on two assumptions:** a neutral aggregator and secure aggregation. Neither gives
  a formal guarantee. The aggregator could turn attacker, and the final global model can still
  leak through membership inference.
- **Secure aggregation is a protocol simulation, not cryptography.** Masks come from a seeded
  pseudo-random generator, not real key agreement.
- **DP-SGD is set aside in this branch.** The Phase 5 result is a discussion point (fig 7): DP-SGD
  drives the attack to chance, but the federated model collapses toward the majority class. The
  DP code stays, off by default.
- **Disagreement is a flag, not a second opinion.** The local, regional and global models share
  training data, so their agreement is correlated.
- **Not deployable today.** The architecture applies once crewed lunar and Mars missions run. An
  Earth–ISS simulation is the near-term option. Labelling costs crew time.
- **Prior art.** FLUID (Casaletto et al., bioRxiv 10.1101/2025.01.14.633017) ran FL on the ISS
  with IID mouse data. NASA Frontier Development Lab 2021 did cross-institution FL for
  astronaut-health research. This project's contribution is the measured behaviour of non-IID,
  quantity-skewed, multi-agency FL with regional models under deep-space delay.
- **Clinical validation** is a written protocol here, not a result:
  [docs/clinical_validation_protocol.md](docs/clinical_validation_protocol.md).

## Modelling assumptions

All simulated times are scenario values unless a source is named. Simulated time is reported
separately from real wall-clock time.

**Links and time**
1. **No receiver contention.** Each node–aggregator link has its own bandwidth, so transfers to
   one aggregator overlap fully. Simulated round times are therefore a lower bound. Nodes send
   only to their regional aggregator, and the regional aggregators only to the global aggregator.
2. **The link rate is symmetric:** 256 kbps both ways. Real relay links are often asymmetric. A
   1D-CNN update (11.7 KB) takes about 0.4 s, against light-times of about 1.3 s (Moon) and
   182–1337 s (Mars) and a 120 s regional deadline.
3. **Light-time is fixed for each run** (one Mars distance preset). There is no orbital motion.
4. **Loss-of-signal schedules are simulated** (Poisson or periodic). Only the 450 s outage length
   is sourced, from FLUID's ISS loss of signal. The lunar profile, out of view 48 min per 2 h
   orbit, is a scenario value.
5. **Nothing is lost in transit.** Links are store-and-forward: a message waits out an outage and
   is dropped only after its 6 h lifetime. There are no bit errors or packet loss.
6. **Jitter** is half-normal (|N(0, 0.05 s)|) and independent for each transfer.
7. **Local training time is modelled:** 0.01 s per example per epoch, with the same hardware on
   every node.
8. **Aggregation itself takes no simulated time.**
9. **Solar conjunction is off by default.** It lasts about 2 weeks, longer than any run.

**Protocol**

10. **Rounds are synchronised** across regions with a 120 s regional deadline. A late update is
    dropped for that round.
11. **Threshold fallback:** a region with fewer reports than `min_participants` uses the last
    global model it received, and that round's updates are discarded.
12. **Nodes fail only by losing signal.** There are no crashes and no faulty or malicious
    (Byzantine) nodes. The aggregator is assumed neutral.
13. **Full participation:** every reachable node takes part in every round.

**Data**

14. **Labels are available at training time** in the fixed-split runs. Only the stream mode
    models scarce, delayed labels.
15. **Each node's data is fixed for the run.** No crew joins or leaves, and there is no drift.
16. **The set of missions is fixed for the run.** No mission joins or leaves mid-run, and nodes
    drop out only through loss of signal.

Later work includes simulating missions that come and go, and a producer–consumer communication
model: nodes push updates when ready, and aggregators listen with bounded receive queues instead
of synchronised rounds.

## Setup

The development machine is an Intel Mac, so the pins are fixed: Python 3.11, torch 2.2.2
(numpy < 2), flwr 1.23.0, opacus 1.5.2. GPU runs used the Georgia Tech PACE ICE cluster (SLURM).

```bash
uv sync --frozen
uv run --frozen pytest -q                         # default suite; slow tests: -m "slow or not slow"
```

Data is downloaded into `data/raw/` and never extracted or modified. Each dataset's licence must be
accepted explicitly:

```bash
uv run --frozen python scripts/fetch_data.py wesad --accept-license      # ~2.25 GB
uv run --frozen python scripts/fetch_data.py osd435 --accept-license
uv run --frozen python scripts/fetch_data.py cinc2021 --subset 20000 --accept-license
```

## Reproduce a headline result from a clean clone

This reproduces the locked flat five-silo WESAD result for seed 0. It takes a few minutes on a
laptop CPU, plus the download:

```bash
git clone <repo> && cd <repo>
uv sync --frozen
uv run --frozen python scripts/fetch_data.py wesad --accept-license
uv run --frozen fedastro run configs/experiments/wesad_by_subject.yaml --seed 0 --output-dir runs/repro
```

Expected `result.json` test accuracy, from `tests/golden/flat_wesad_g1.json` (absolute tolerance
1e-3):

| Arm | `result.json` field | Accuracy |
|---|---|---|
| centralized | `arms.centralized.final.accuracy` | 0.849 |
| federated (FedAvg) | `arms.federated.final.accuracy` | 0.670 |
| local-only (mean over silos) | `arms.local_only.mean.accuracy` | 0.631 |

`uv run --frozen pytest tests/test_regression_flat.py -m "slow or not slow"` checks all 5 seeds.

The hierarchical experiments run as one sweep per configuration (PREREG §6), on the cluster. The
per-variant commands and SLURM scripts are in PROGRESS.md (Phases H9 and H10) and
`scripts/slurm/`:

```bash
uv run --frozen fedastro grid configs/sweeps/h9_wesad12_logreg_full.yaml --index 0
uv run --frozen fedastro hypotheses configs/hypotheses/h9_wesad12.yaml --output-dir runs/<new> runs/<old>
```

## Figures

```bash
uv run --frozen python scripts/make_figures.py --tables   # writes figures/ (16:9 PNG + PDF + JSON)
```

The script only reads `runs/`; it never trains a model. Each figure has a JSON sidecar with every
plotted number and the result files it came from. The figures are greyscale-safe. Fig 8 is
exploratory and says so on the figure.

## Repository map

- `src/fedastro/`: data loaders, partitioning and shift, FL and the hierarchy, secure aggregation,
  update versioning, metrics, CLI.
- `configs/`: base config, silos, topologies (`cinc15`, `wesad12`), experiments and sweeps.
- `PREREG.md`: hypotheses, metrics, decision rules, configurations and seeds, frozen before the
  runs, with every later deviation disclosed in §7.
- `PROGRESS.md`: the phase-by-phase record, newest first, with every number reported.
