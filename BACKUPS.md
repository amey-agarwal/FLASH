# Result backups

Archives of run outputs that are no longer, or not only, kept as loose folders. `runs/` is
git-ignored, so these zips are the durable copy of the results. Every number in PROGRESS.md traces
to a `result.json` inside one of them. Copy each zip to off-laptop storage too (Google Drive or an
external disk).

To restore, run `unzip` from the repo root (`~/Desktop/FLUIC`). The paths inside each zip already
start at the right level.

| Zip (location) | Size | Runs (`result.json`) | Contents | Restore |
|---|---|---|---|---|
| `runs/h10-f7db96a_backup.zip` (in the repo, git-ignored) | 59 MB | 528 | **Phase H10c, commit `f7db96a`, CinC 2021 20k subset (`cinc15`), PACE, 2026-11-25.** Stage A tuning (`gapA_*`, 180), stage B (`gapB_*`, 288), centralized/local-only baselines (`eval_gapC_*`, 30) and the evaluation runs at seeds 0–4 (`eval_gapB_gapA_*`, 30). Also `gap_cinc15.json` / `.txt` (the gap table in PROGRESS.md H10c), the selection JSONs, `select_stageB.txt` and `generated/` (stage B sweeps, eval sweeps, gap spec). | `cd runs && unzip h10-f7db96a_backup.zip` → `runs/h10-f7db96a/` |
| `runs/h10t-e51b0ad_backup.zip` (in the repo, git-ignored) | 0.4 MB | 3 | **Phase H10c step 1, commit `e51b0ad`, PACE job 7255811.** The 20k timing runs (logreg, MLP, 1D-CNN; seed 100); timing only, scores not read. | `cd runs && unzip h10t-e51b0ad_backup.zip` → `runs/h10t-e51b0ad/` |
| `runs/h8-62cf33b_backup.zip` (in the repo, git-ignored) | 22 MB | 528 | **Phase H8, commit `62cf33b`, WESAD12, PACE.** Stage A tuning (`gapA_*`), stage B (`gapB_*`), centralized/local-only baselines (`gapC_*`), 468 tuning runs in all; plus the 60 evaluation runs at seeds 0–4 (`eval_*`, the presentation numbers), `gap_wesad12.json` (gap table), the selection JSONs and `generated/` (the eval sweeps, gap spec and stage B sweeps). | `cd runs && unzip h8-62cf33b_backup.zip` → `runs/h8-62cf33b/` |
| `~/Desktop/fluic_results_backup_2026-11-21.zip` | 34 MB | 341 | **Everything in `runs/` from Phases 3–6, H4 and H9 OSD-435**, plus the PACE Phase 5 downloads: `p4alpha`, `p4mu`, `p4subj` (Phase 4), `p5eps` (Phase 5 DP), `p6pre`, `p6dark` (Phase 6 latency), `wesad_by_subject`, `wesad_by_subject_cnn` (locked flat results), `h4secagg`, `h9_osd435*` and `h9_osd435_report.txt`; `PACE-runs-eval/` (`p5epsc-*` per-run zips from PACE); and a nested copy of `runs/h8-62cf33b_backup.zip`. Excludes `runs/smoke*` and `runs/*.log`. | `unzip ~/Desktop/fluic_results_backup_2026-11-21.zip` |
| `~/Desktop/fluic_h9_wesad_2026-11-21.zip` | 15 MB | 435 | **Phase H9 WESAD, commit `84a40ec`, PACE jobs 7226543–60.** 18 sweeps × 3 families (centralized, local-only, federated, hierarchical at 6 shift settings; full architecture; stream), the per-sweep `*-summary.json` files, `hypotheses_wesad12.txt` / `.json` (H1–H4 verdicts) and `h9_summary.txt` (the PROGRESS.md H9 part 3 tables). | `unzip ~/Desktop/fluic_h9_wesad_2026-11-21.zip` → `runs/h9-84a40ec/` |

## Notes
- **The PACE copies** are in `$SCR/FLASH/runs/h8-62cf33b/`, `$SCR/FLASH/runs/h9-84a40ec/`,
  `$SCR/FLASH/runs/h10-f7db96a/` and `$SCR/FLASH/runs/h10t-e51b0ad/` (plus the two H10 zips in
  `$SCR/FLASH/runs/`). Scratch is purged after inactivity, so don't rely on them.
- **Unzipped working copies** of the H10 zips are in `runs/h10-f7db96a/` and `runs/h10t-e51b0ad/`
  (387 MB and 3 MB). H10c step 3 reads `runs/h10-f7db96a/generated/`.
- **Duplicates of the above, safe to delete:** `PACE-run-evals2/` and `PACE-run-evals3/` are already
  deleted. `recentPACEruns/` is now empty (the H9 WESAD download was removed earlier, and the H10 zips were
  moved into `runs/` on 2026-11-25), so the folder can go.
- **Checking a zip:** `unzip -l <zip> | grep -c 'result.json$'` should give the run count in
  the table.
