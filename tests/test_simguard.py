"""Fail-fast and retry on a Ray start-up failure (infrastructure; no result changes)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from fedastro.fl.simguard import EXIT_SIM_HUNG, is_ray_startup_failure, simulation_watchdog

REPO = Path(__file__).resolve().parents[1]


def _chained(msg: str) -> RuntimeError:
    try:
        try:
            raise RuntimeError(msg)
        except RuntimeError as inner:
            raise RuntimeError("hierarchical arm failed") from inner
    except RuntimeError as outer:
        return outer


def test_startup_failure_matched_through_the_chain():
    assert is_ray_startup_failure(_chained("query.stats: 0/15 replies within 600.0s"))
    assert is_ray_startup_failure(RuntimeError("query.stats: 0/12 replies within 600s"))


def test_partial_replies_and_other_errors_not_matched():
    assert not is_ray_startup_failure(_chained("fit: 3/15 replies within 600.0s"))
    assert not is_ray_startup_failure(_chained("fit: 10/15 replies within 600.0s"))
    assert not is_ray_startup_failure(ValueError("0/15 replies within 600s"))
    assert not is_ray_startup_failure(RuntimeError("fit: client error(s): node 1: boom"))
    assert not is_ray_startup_failure(None)


def test_hung_simulation_exits_75():
    script = textwrap.dedent("""
        import time
        from fedastro.fl.simguard import simulation_watchdog
        out = {}
        with simulation_watchdog(out, "test", grace_s=1.0, poll_s=0.1):
            out["error"] = RuntimeError("query.stats: 0/15 replies within 600.0s")
            time.sleep(60)  # the hang
        print("not reached")
    """)
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=50)
    assert p.returncode == EXIT_SIM_HUNG
    assert time.monotonic() - t0 < 40
    assert "simulation hung after ServerApp error (test)" in p.stderr
    assert "not reached" not in p.stdout


def test_no_error_no_exit_and_watchdog_stops():
    out: dict = {}
    with simulation_watchdog(out, "ok", grace_s=0.2, poll_s=0.05):
        time.sleep(0.2)
    time.sleep(0.5)  # a live watchdog would have no error to act on anyway; the thread is gone
    import threading
    assert not [t for t in threading.enumerate() if t.name == "simguard-ok"]


def test_error_that_returns_still_raises_normally():
    out: dict = {}
    with pytest.raises(RuntimeError, match="arm failed"):
        with simulation_watchdog(out, "returns", grace_s=5.0, poll_s=0.05):
            out["error"] = RuntimeError("query.stats: 0/15 replies within 600.0s")
            time.sleep(0.2)
        raise RuntimeError("arm failed")
    time.sleep(0.3)  # the watchdog stopped when the block exited: still alive here
    import threading
    assert not [t for t in threading.enumerate() if t.name == "simguard-returns"]


def _retry(tmp_path: Path, codes: list[int]) -> tuple[int, int, str]:
    counter = tmp_path / "n"
    counter.write_text("0")
    stub = tmp_path / "stub.sh"
    stub.write_text(textwrap.dedent(f"""
        n=$(( $(cat {counter}) + 1 )); echo $n > {counter}
        codes=({' '.join(map(str, codes))})
        exit ${{codes[$((n - 1))]}}
    """))
    cmd = (f"source {REPO / 'scripts/slurm/retry.sh'}; RAY_TMPDIR={tmp_path / 'ray'} "
           f"RETRY_WAIT_S=0 retry_tempfail bash {stub}")
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)
    return p.returncode, int(counter.read_text()), p.stderr


def test_retry_loop_retries_75_then_succeeds(tmp_path):
    rc, n, err = _retry(tmp_path, [75, 75, 0])
    assert (rc, n) == (0, 3)
    assert "retry 1 after Ray start-up failure" in err and "retry 2" in err


def test_retry_loop_gives_up_after_3(tmp_path):
    rc, n, err = _retry(tmp_path, [75, 75, 75, 0])
    assert (rc, n) == (75, 3)
    assert "giving up after 3 attempts" in err


def test_retry_loop_does_not_retry_other_failures(tmp_path):
    rc, n, _ = _retry(tmp_path, [1, 0])
    assert (rc, n) == (1, 1)
