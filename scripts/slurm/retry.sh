# Sourced by scripts/slurm/gap.sbatch (and tests/test_simguard.py).
#
# retry_tempfail CMD...  runs CMD up to RETRY_ATTEMPTS times (default 3) while it exits 75
# (EX_TEMPFAIL: a Ray start-up failure, src/fedastro/fl/simguard.py). Every attempt gets its own
# Ray temp dir under the job's RAY_TMPDIR. CMD must be idempotent: here `fedastro grid --resume`,
# which skips finished runs, and the runs are seeded, so a retry gives identical results.
# Any other exit code ends the loop with that code. Never `ray stop`: it would kill other
# users' Ray processes on a shared node.
retry_tempfail() {
    local max=${RETRY_ATTEMPTS:-3} wait_s=${RETRY_WAIT_S:-30} base=${RAY_TMPDIR:-/tmp/ray_retry_$$}
    local k=1 rc
    while :; do
        export RAY_TMPDIR="$base/a$k"
        mkdir -p "$RAY_TMPDIR"
        "$@"
        rc=$?
        if [ "$rc" -ne 75 ] || [ "$k" -ge "$max" ]; then
            [ "$rc" -eq 75 ] && echo "giving up after $k attempts (Ray start-up failure); resubmit this index" >&2
            export RAY_TMPDIR="$base"
            return "$rc"
        fi
        echo "retry $k after Ray start-up failure (exit 75); waiting ${wait_s}s" >&2
        sleep "$wait_s"
        k=$((k + 1))
    done
}
