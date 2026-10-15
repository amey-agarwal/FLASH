from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo() -> Path:
    return REPO


@pytest.fixture
def smoke_config(repo: Path) -> Path:
    return repo / "configs" / "experiments" / "smoke.yaml"
