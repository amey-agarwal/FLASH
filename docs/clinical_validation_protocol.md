# Clinical validation protocol (written protocol, not a result)

This project demonstrates a federated learning architecture **under simulated conditions** on
proxy datasets: WESAD, PhysioNet/CinC 2021 and NASA OSDR OSD-435. **None of its models has been
clinically validated, and none may be used for diagnosis or monitoring.** This document sets out
what validation would require before any model trained by this architecture could support a
clinical decision on a crewed mission. It describes future work; it reports no results.

## 1. Scope and intended use

- **Intended role:** decision support for crew medical officers and ground clinicians. It never
  diagnoses on its own and never triggers treatment.
- **Signals in scope:** physiological only (ECG, wearable signals), as in the project. Environmental
  sensor data is out of scope.
- **Disagreement flags:** when the local, regional and global models disagree, the case is flagged
  for clinician review. The three models share training data, so their agreement is correlated.
  It must never be presented as independent confirmation.

## 2. Label definitions: expert review

1. **CinC 2021 target.** Normal sinus rhythm (SNOMED-CT 426783006, no other scored code) against
   any other scored abnormality, using the Challenge's scored-label mapping.
   - Cardiologists review whether this binary target is clinically meaningful for in-flight
     monitoring.
   - They also review whether sinus bradycardia and tachycardia, both counted as abnormal here,
     should be treated differently in microgravity.
2. **WESAD stress labels** come from a laboratory protocol (baseline, stress, amusement). A
   clinical psychologist assesses whether they transfer to operational stress in spaceflight.
3. **Agreement on labels.** At least two independent annotators label a stratified sample of
   records. Agreement is reported as Cohen's kappa before any performance claim, with the
   disagreements adjudicated.

## 3. Operating point: sensitivity and specificity

- **Choosing the threshold.** Report sensitivity and specificity, with confidence intervals, at a
  threshold fixed **on validation data**. The threshold should reflect the mission's cost of a
  missed abnormality against the cost of a false alarm (crew time, consumables, anxiety).
- **Report the minority class explicitly.** In this project, accuracy rewarded local models that
  almost never predicted the minority (normal) class. On most CinC nodes normal-class recall was
  under 0.05 while accuracy stayed near 0.84 (exploratory analysis, PROGRESS.md H11a). Validation
  must report per-class recall, balanced accuracy and macro-F1. Accuracy alone is not acceptable.
- **Curves.** Report ROC and precision–recall curves per site, not only pooled.

## 4. Calibration

- Report expected calibration error (ECE) and reliability diagrams, per arm and per site. The code
  already computes ECE and reliability bins (`metrics/calibration.py`).
- If a model is to output a probability for clinicians, recalibrate it (for example with
  temperature scaling) on validation data only, and re-check it on held-out data.

## 5. Worst-site and per-agency performance

- Report every metric per site (agency analog), with the worst site named. A model that is good
  on average but fails one agency's crew is not acceptable.
- **Sites with a single class or very few examples** (for example INCART in this project: no
  normal records) cannot validate specificity. They must be flagged, and must not be averaged
  into a reassuring figure.
- Report subgroup performance where the data allows (sex, age band). Astronaut cohorts are small,
  so wide intervals must be reported as such.

## 6. Re-runs on clinical datasets

1. **Re-run the full pipeline on clinical-grade data** with expert labels, with the
   pre-registration discipline used here (PREREG.md):
   - hypotheses, metrics, configurations and seeds frozen before any result-producing run;
   - every deviation disclosed.
2. **Patient-level splits only.** No patient may appear in more than one node or split.
3. **External validation** on at least one site that took no part in training. The CinC results
   here test new patients from known sources, not unseen sources.
4. **Uncertainty from test-subject sampling.** Report CIs that resample test subjects or patients,
   not only training seeds.

## 7. Route to real validation: analog studies

Real astronaut data is scarce and access-controlled. The realistic path runs through analogs that
reproduce parts of the spaceflight setting:

- **Bed-rest studies** (head-down tilt): cardiovascular deconditioning, a physiology shift closer
  to microgravity than this project's simulated shift.
- **HERA** (NASA's Human Exploration Research Analog): isolation, confinement and communication
  delay, with crews that can wear the same sensors.
- **Antarctic stations** (for example Concordia, Halley): long-duration isolation, limited
  bandwidth and real loss of contact, with several nations' crews as natural silos.
- **Earth–ISS demonstration:** the near-term option, following the FLUID precedent (OpenFL on HPE
  Spaceborne Computer-2).

Each analog study should define in advance:
- the clinical endpoint;
- the comparison (federated against local-only, against centralized where data sharing is
  allowed);
- the success criteria and stopping rules;
- ethics approval and crew consent, including consent to the federated use of their data.

## 8. Privacy and governance (prerequisites, not validation)

- **The privacy assumptions are not guarantees.** This project's privacy rests on a neutral
  aggregator and on secure aggregation (simulated here). Neither gives a formal guarantee, and the
  final model can still leak through membership inference. A clinical deployment needs a real
  cryptographic implementation, an agreed governance model for the aggregator, and a decision on
  differential privacy. DP-SGD collapsed utility in this project's Phase 5.
- **Agencies must agree in advance** on data use, model ownership, update approval (the validation
  gate) and incident response.

## 9. Not claimed

The project does not claim clinical accuracy, safety or effectiveness, or readiness for flight. It
claims only that the mechanism works under simulated conditions on proxy data.
