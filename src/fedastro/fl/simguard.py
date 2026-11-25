"""Fail fast on a Ray start-up failure in a Flower simulation (infrastructure only).

On shared PACE nodes a Ray worker sometimes fails to register with its raylet. The ServerApp's
first exchange then gets 0 replies and raises after ``fl.timeout_s``. Usually
``run_simulation`` returns and the arm raises. Sometimes it does not: one of Flower's
simulation-engine worker threads stays blocked on the dead actor, and Flower waits for it
forever (H10c task 7294746_9 idled for 6 h).

- ``simulation_watchdog`` ends such a hung process with ``EXIT_SIM_HUNG`` (75, EX_TEMPFAIL).
- ``is_ray_startup_failure`` lets ``fedastro grid`` exit 75 for the ordinary, non-hung case
  too.
- ``scripts/slurm/gap.sbatch`` retries a task that exits 75, with ``--resume``. Runs are
  seeded and this failure happens before any training, so a retry gives identical results.

Nothing here reads or changes a config, a score or a run id.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from contextlib import contextmanager
from typing import Any, Iterator

EXIT_SIM_HUNG = 75  # EX_TEMPFAIL: the SLURM script retries this exit code
GRACE_S = 120.0  # real seconds after a ServerApp error before a non-returning simulation is ended
_SHUTDOWN_WAIT_S = 20.0
_NO_REPLIES = re.compile(r"\b0/\d+ replies within ")


def is_ray_startup_failure(exc: BaseException | None) -> bool:
    """True if ``exc`` or an exception it was raised from (or during) is an exchange in
    which *no* node replied: the Ray start-up failure.

    A partial reply count (some nodes answered) is not matched: that is a real problem."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, RuntimeError) and _NO_REPLIES.search(str(exc)):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _shutdown_ray() -> None:
    try:
        import ray

        ray.shutdown()
    except Exception:  # best effort only; the process is exiting anyway
        pass


def _end_process(label: str, error: BaseException) -> None:
    print(f"simulation hung after ServerApp error ({label}): {type(error).__name__}: {error}; "
          f"exiting {EXIT_SIM_HUNG} for retry", file=sys.stderr, flush=True)
    t = threading.Thread(target=_shutdown_ray, daemon=True)
    t.start()
    t.join(_SHUTDOWN_WAIT_S)
    sys.stdout.flush()
    os._exit(EXIT_SIM_HUNG)


@contextmanager
def simulation_watchdog(out: dict[str, Any], label: str, grace_s: float = GRACE_S,
                        poll_s: float = 1.0) -> Iterator[None]:
    """Guard a ``run_simulation`` call whose ServerApp stores its exception in ``out["error"]``.

    If ``out["error"]`` is set and the block has not exited ``grace_s`` seconds later, the
    process ends with ``EXIT_SIM_HUNG``. A block that exits, normally or by raising, stops
    the watchdog."""
    done = threading.Event()

    def watch() -> None:
        while not done.wait(poll_s):
            if "error" in out:
                if not done.wait(grace_s):
                    _end_process(label, out["error"])
                return

    t = threading.Thread(target=watch, name=f"simguard-{label}", daemon=True)
    t.start()
    try:
        yield
    finally:
        done.set()
        t.join(poll_s + 1.0)


__all__ = ["EXIT_SIM_HUNG", "GRACE_S", "is_ray_startup_failure", "simulation_watchdog"]
