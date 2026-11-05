#!/bin/bash
# Golden Tier B (CLAUDE2.md section 3): one pinned item per array task, 0..4 = G1..G5
# (scripts/golden.py PINNED). G1 is `fedastro sweep` (5 seeds); G2-G5 are one variant
# each via `fedastro grid --index i`. Longest task: G4 (DP + attack), about 1 h.
#
# From the repo root, on a clean checkout:
#   mkdir -p runs/slurm && sbatch scripts/slurm/golden.sbatch
# Then on the login node (reads JSON only):
#   uv run --frozen python scripts/golden.py freeze --runs-dir runs/golden-<sha>   # first time only
#   uv run --frozen python scripts/golden.py check  --runs-dir runs/golden-<sha>   # every later gate
#SBATCH -J golden
#SBATCH -A bios
#SBATCH -q coc-ice
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH --array=0-4
#SBATCH -o runs/slurm/%x-%A_%a.out
set -uo pipefail
OUT_PREFIX=golden
source scripts/slurm/common.sh
ITEMS=(G1 G2 G3 G4 G5)
uv run --frozen python scripts/golden.py run --item "${ITEMS[$SLURM_ARRAY_TASK_ID]}" \
    --resume --output-dir "$OUT"
