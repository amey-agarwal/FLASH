# CI-less task runner. Requires uv (https://docs.astral.sh/uv/).
.PHONY: install test smoke repro-check clean fetch-wesad grid golden

install:
	uv sync --frozen

test:
	uv run --frozen pytest

smoke:
	uv run --frozen fedastro run configs/experiments/smoke.yaml --seed 0

# Runs the smoke config twice into separate dirs and byte-compares result.json.
repro-check:
	SOURCE_DATE_EPOCH=0 uv run --frozen fedastro run configs/experiments/smoke.yaml --seed 0 --output-dir runs/_repro_a
	SOURCE_DATE_EPOCH=0 uv run --frozen fedastro run configs/experiments/smoke.yaml --seed 0 --output-dir runs/_repro_b
	cmp runs/_repro_a/*/result.json runs/_repro_b/*/result.json && echo "BYTE-IDENTICAL"

clean:
	rm -rf runs/_repro_a runs/_repro_b .pytest_cache

fetch-wesad:
	uv run --frozen python scripts/fetch_data.py wesad --accept-license

# Long laptop sweeps: caffeinate (macOS) stops the machine sleeping mid-run, which
# otherwise stalls Ray and times the round out. Usage: make grid SWEEP=configs/sweeps/x.yaml
grid:
	caffeinate -i uv run --frozen fedastro grid $(SWEEP) --resume

# Golden Tier B (CLAUDE2.md section 3): re-run the pinned subset (scripts/golden.py) into
# a fresh dir and compare final metrics with tests/golden/metrics.json at 1e-3. Slow
# (about 1.5 h of CPU), needs the WESAD archive. On the cluster use scripts/slurm/golden.sbatch.
# caffeinate only where it exists (macOS), so the target also runs on Linux.
CAFFEINATE := $(shell command -v caffeinate >/dev/null 2>&1 && echo caffeinate -i)
golden:
	rm -rf runs/_golden
	$(CAFFEINATE) uv run --frozen python scripts/golden.py run --output-dir runs/_golden
	uv run --frozen python scripts/golden.py check --runs-dir runs/_golden
