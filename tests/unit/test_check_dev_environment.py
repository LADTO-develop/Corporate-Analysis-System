"""Tests for the development environment checker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from packaging.requirements import Requirement


def _load_check_dev_environment_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_dev_environment.py"
    spec = importlib.util.spec_from_file_location("check_dev_environment", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_requirement_status_accepts_declared_range() -> None:
    checker = _load_check_dev_environment_module()

    status, failure = checker._requirement_status(
        package_name="altair",
        installed_version="5.5.0",
        requirement=Requirement("altair>=5.4.0,<6.0.0"),
    )

    assert status == "5.5.0 (requires <6.0.0,>=5.4.0)"
    assert failure is None


def test_requirement_status_flags_out_of_range_versions() -> None:
    checker = _load_check_dev_environment_module()

    status, failure = checker._requirement_status(
        package_name="altair",
        installed_version="6.0.0",
        requirement=Requirement("altair>=5.4.0,<6.0.0"),
    )

    assert status == "6.0.0 (requires <6.0.0,>=5.4.0) [out of range]"
    assert failure == "altair version 6.0.0 does not satisfy <6.0.0,>=5.4.0"
