"""Validate that one Python environment can run CAS development and agent workflows."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
INSTALL_HINT = 'python -m pip install -e ".[dev,agent,ml,viz,dashboard]"'
DEFAULT_CHECK_EXTRAS = ("dev", "agent", "ml", "viz", "dashboard")

REQUIRED_IMPORTS = {
    "project": {
        "corporate-analysis-system": "cas",
    },
    "runtime": {
        "langgraph": "langgraph",
        "pandas": "pandas",
        "numpy": "numpy",
        "pydantic": "pydantic",
        "pyyaml": "yaml",
        "requests": "requests",
        "structlog": "structlog",
        "xgboost": "xgboost",
    },
    "dev": {
        "ruff": "ruff",
        "mypy": "mypy",
        "pytest": "pytest",
        "pytest-cov": "pytest_cov",
    },
    "agent": {
        "agno": "agno",
        "anthropic": "anthropic",
        "openai": "openai",
        "google-genai": "google.genai",
    },
    "dashboard-viz-ml": {
        "altair": "altair",
        "matplotlib": "matplotlib",
        "plotly": "plotly",
        "scikit-learn": "sklearn",
        "shap": "shap",
        "streamlit": "streamlit",
    },
}


def main() -> None:
    """Run environment checks and exit non-zero when the environment is incomplete."""
    args = _parse_args()
    print("CAS development environment check")
    print(f"- executable: {sys.executable}")
    print(f"- python: {sys.version.split()[0]}")

    failures = _python_version_failures()
    requirement_specs = _project_requirement_specs()
    failures.extend(_import_failures(requirement_specs))
    if args.live_agno:
        failures.extend(_live_agno_failures())

    if failures:
        print("\nEnvironment check failed:")
        for failure in failures:
            print(f"- {failure}")
        print(f"\nInstall or refresh the unified environment with:\n  {INSTALL_HINT}")
        raise SystemExit(1)

    print("Environment check passed.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-agno",
        action="store_true",
        help="Also require live Agno API keys via scripts/check_agno_stage2.py.",
    )
    return parser.parse_args()


def _python_version_failures() -> list[str]:
    if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
        return [
            "Python must be >=3.12,<3.13 to match pyproject.toml and CI; "
            f"found {sys.version.split()[0]}."
        ]
    return []


def _import_failures(requirement_specs: dict[str, Requirement]) -> list[str]:
    failures: list[str] = []
    for group, packages in REQUIRED_IMPORTS.items():
        print(f"\n[{group}]")
        for package_name, import_name in packages.items():
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                print(f"- {package_name}: missing")
                failures.append(f"Missing {package_name} ({import_name})")
                continue
            installed_version = _package_version(package_name)
            requirement = requirement_specs.get(package_name)
            status, failure = _requirement_status(
                package_name=package_name,
                installed_version=installed_version,
                requirement=requirement,
            )
            print(f"- {package_name}: {status}")
            if failure:
                failures.append(failure)
    return failures


def _project_requirement_specs(
    extras: tuple[str, ...] = DEFAULT_CHECK_EXTRAS,
) -> dict[str, Requirement]:
    """Return dependency requirements declared in pyproject.toml, keyed by package name."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    raw_requirements = list(project.get("dependencies", []))
    optional_dependencies = project.get("optional-dependencies", {})
    if isinstance(optional_dependencies, dict):
        for extra_name in extras:
            requirements = optional_dependencies.get(extra_name, [])
            if isinstance(requirements, list):
                raw_requirements.extend(requirements)

    requirement_specs: dict[str, Requirement] = {}
    for raw_requirement in raw_requirements:
        if not isinstance(raw_requirement, str):
            continue
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement:
            continue
        requirement_specs[requirement.name] = requirement
    return requirement_specs


def _requirement_status(
    *,
    package_name: str,
    installed_version: str,
    requirement: Requirement | None,
) -> tuple[str, str | None]:
    if requirement is None or not requirement.specifier:
        return installed_version, None
    status = f"{installed_version} (requires {requirement.specifier})"
    try:
        installed = Version(installed_version)
    except InvalidVersion:
        return status, None
    if installed in requirement.specifier:
        return status, None
    return (
        f"{status} [out of range]",
        f"{package_name} version {installed_version} does not satisfy {requirement.specifier}",
    )


def _live_agno_failures() -> list[str]:
    command = [sys.executable, str(ROOT / "scripts" / "check_agno_stage2.py")]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    print("\n[live-agno]")
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())
    if completed.returncode == 0:
        return []
    return ["Live Agno preflight failed; check packages and API key environment variables."]


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "installed"


if __name__ == "__main__":
    main()
