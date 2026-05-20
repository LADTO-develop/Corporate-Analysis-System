"""Validate that one Python environment can run CAS development and agent workflows."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL_HINT = 'python -m pip install -e ".[dev,agent,ml,viz,dashboard]"'

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
    failures.extend(_import_failures())
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


def _import_failures() -> list[str]:
    failures: list[str] = []
    for group, packages in REQUIRED_IMPORTS.items():
        print(f"\n[{group}]")
        for package_name, import_name in packages.items():
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                print(f"- {package_name}: missing")
                failures.append(f"Missing {package_name} ({import_name})")
                continue
            print(f"- {package_name}: {_package_version(package_name)}")
    return failures


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
