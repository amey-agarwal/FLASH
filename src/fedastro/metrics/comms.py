"""Communication accounting for the federated arm.

bytes per round = sum over participants of (download payload + upload payload). With
every client participating and the full model exchanged, this is
parameter payload x clients x 2 (CLAUDE.md Phase 3). Payload is the raw float32 tensor
bytes; protocol framing overhead is not modelled.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def payload_bytes(arrays: list[np.ndarray]) -> int:
    return int(sum(a.nbytes for a in arrays))


def rounds_to_target(rounds: list[dict[str, Any]], threshold: float | None,
                     metric: str = "accuracy") -> dict[str, Any]:
    """First round whose global holdout ``metric`` reaches ``threshold``, with the
    cumulative bytes spent up to and including that round and its simulated
    wall-clock time. None if never reached."""
    out: dict[str, Any] = {"threshold": threshold, "metric": metric, "round": None,
                           "cumulative_bytes": None, "sim_wallclock_s": None}
    if threshold is None:
        return out
    cumulative = 0
    for r in rounds:
        cumulative += r["bytes"]
        value = r["global"].get(metric)
        if value is not None and value >= threshold:
            out["round"] = r["round"]
            out["cumulative_bytes"] = cumulative
            out["sim_wallclock_s"] = r.get("sim_wallclock_s")  # simulated seconds, None if comms off
            break
    return out
