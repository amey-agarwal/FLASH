#!/bin/bash
# Phase 5 publishable sweep p5epsc (configs/sweeps/phase5_eps_cluster.yaml): one epsilon
# variant per array task (0 = no DP, 1..7 = epsilon 0.5..13), 5 seeds each, 10 attack
# trials per cell. About 3 h per variant on CPU (PROGRESS.md, Phase 5); a GPU does not help.
#
# Freeze the attack settings first (PROGRESS.md, clean-commit batch). Then, from the repo root:
#   mkdir -p runs/slurm && sbatch scripts/slurm/p5epsc.sbatch
#   sbatch --array=3 scripts/slurm/p5epsc.sbatch     # re-run one variant; done seeds are skipped
# Afterwards (reads JSON only):
#   uv run --frozen fedastro grid configs/sweeps/phase5_eps_cluster.yaml --report --output-dir runs/clean-<sha>
#SBATCH -J p5epsc
#SBATCH -A bios
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH --array=0-7
#SBATCH -o runs/slurm/%x-%A_%a.out
set -uo pipefail
source scripts/slurm/common.sh
uv run --frozen fedastro grid configs/sweeps/phase5_eps_cluster.yaml \
    --index "$SLURM_ARRAY_TASK_ID" --resume --output-dir "$OUT"
