"""Filesystem locations. Data paths come from the environment, never from configs,
so moving data on a cluster does not change any run's identity."""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    """``$FEDASTRO_DATA_ROOT`` if set, else ``./data`` relative to the working directory."""
    return Path(os.environ.get("FEDASTRO_DATA_ROOT", "data")).resolve()


def raw_dir() -> Path:
    return data_root() / "raw"


def cache_dir() -> Path:
    return data_root() / "cache"
