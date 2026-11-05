# Sourced by every scripts/slurm/*.sbatch. Submit from the repo root:
#   mkdir -p runs/slurm && sbatch scripts/slurm/<job>.sbatch
#
# Sets REPO, SHA and OUT, and refuses to run on a dirty tree, so every result carries a
# clean git_sha. OUT defaults to runs/<OUT_PREFIX>-<sha>; set FEDASTRO_OUT to override.

# PACE ICE's /etc/bashrc reads an unset variable, which kills a job running under set -u
# (job 6816077). Source ~/.bashrc (it puts uv on PATH) with -e/-u off, then restore them.
_opts=$(set +o)
set +eu
[ -f ~/.bashrc ] && source ~/.bashrc
eval "$_opts"

REPO=${FEDASTRO_REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO" || exit 3
SHA=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "tree is dirty at $SHA: refusing to produce results" >&2
    exit 3
fi
OUT=${FEDASTRO_OUT:-runs/${OUT_PREFIX:-clean}-$SHA}
mkdir -p "$OUT"

# One short Ray temp dir per job, so parallel jobs on one node do not share Ray state.
export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$RAY_TMPDIR"
trap 'rm -rf "$RAY_TMPDIR"' EXIT

echo "job $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID:-none} on $(hostname), commit $SHA, out $OUT"
