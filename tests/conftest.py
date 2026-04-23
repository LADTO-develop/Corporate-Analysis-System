"""pytest fixtures shared across the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cas.utils.logging import configure_logging
from cas.utils.seeds import set_seeds


@pytest.fixture(autouse=True, scope="session")
def _setup_logging_and_seeds() -> None:
    configure_logging(level="WARNING", json_output=False)
    set_seeds(42)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    return project_root / "configs"
