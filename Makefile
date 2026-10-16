# CI-less task runner. Requires uv (https://docs.astral.sh/uv/).
.PHONY: install test smoke repro-check clean fetch-wesad grid

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
