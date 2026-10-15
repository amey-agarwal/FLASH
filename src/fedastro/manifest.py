"""Run identity and artifact writing.

Two files per run under ``<output_dir>/<run_id>/``:

- ``result.json``: the section 6 contract. Deterministic for a fixed config, seed,
  commit and ``SOURCE_DATE_EPOCH``; this is what reproducibility checks byte-compare.
- ``manifest.json``: the execution environment (host, versions, real elapsed time).
  It is expected to differ between machines and is never byte-compared.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from fedastro.config import ExperimentConfig
from fedastro.metrics.schema import SCHEMA_VERSION, validate_result

TRACKED_PACKAGES = (
    "fedastro", "flwr", "ray", "torch", "opacus", "numpy", "pandas",
    "scikit-learn", "pyyaml", "streamlit",
)


def make_run_id(cfg: ExperimentConfig, seed: int) -> str:
    return f"{cfg.name}-seed{seed}-{cfg.config_hash()[:10]}"


def git_sha() -> str:
    """HEAD commit of the repo containing this package (or the cwd), '-dirty' if modified.

    Untracked files are ignored so gitignored artifacts never mark a run dirty.
    """
    for where in (Path(__file__).resolve().parent, Path.cwd()):
        try:
            sha = subprocess.run(
                ["git", "-C", str(where), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(where), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            return sha + ("-dirty" if status else "")
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return "unknown"


def timestamp_utc() -> str:
    """Current UTC time, or SOURCE_DATE_EPOCH if set (reproducible-builds convention)."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        t = dt.datetime.fromtimestamp(int(epoch), tz=dt.timezone.utc)
    else:
        t = dt.datetime.now(tz=dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def environment() -> dict[str, Any]:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "packages": versions,
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def write_run(result: dict[str, Any], output_dir: str | Path, elapsed_s: float) -> Path:
    """Validate and write result.json + manifest.json. Returns the run directory."""
    validate_result(result)
    run_dir = Path(output_dir) / result["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": result["run_id"],
        "schema_version": SCHEMA_VERSION,
        "git_sha": result["git_sha"],
        "started_utc": result["timestamp_utc"],
        "real_elapsed_s": round(elapsed_s, 3),
        "output_dir": str(output_dir),
        "argv": sys.argv,
        "environment": environment(),
    }
    (run_dir / "result.json").write_text(dumps(result))
    (run_dir / "manifest.json").write_text(dumps(manifest))
    return run_dir
