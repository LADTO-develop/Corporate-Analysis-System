"""pytest fixtures shared across the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from bfd.utils.logging import configure_logging
from bfd.utils.seeds import set_seeds


@pytest.fixture(autouse=True, scope="session")
def _setup_logging_and_seeds() -> None:
    configure_logging(level="WARNING", json_output=False)
    set_seeds(42)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    return project_root / "configs"
