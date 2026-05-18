"""Preflight check for live Agno Stage 2 runs."""

from __future__ import annotations

import importlib.util
import os
import sys
from importlib.metadata import PackageNotFoundError, version

from dotenv import load_dotenv

REQUIRED_PACKAGES = ("agno", "anthropic")
REQUIRED_ENV_VARS = ("ANTHROPIC_API_KEY",)


def main() -> None:
    """Validate Agno dependencies and local live-mode environment variables."""
    load_dotenv()
    package_errors = _missing_packages()
    env_errors = _missing_env_vars()
    runner = os.environ.get("CAS_STAGE2_RUNNER", "deterministic").strip().lower()
    fallback = os.environ.get("CAS_STAGE2_FALLBACK_ON_ERROR", "1").strip()

    print("CAS Stage 2 Agno preflight")
    print(f"- CAS_STAGE2_RUNNER={runner or 'deterministic'}")
    print(f"- CAS_STAGE2_MODEL={os.environ.get('CAS_STAGE2_MODEL', 'claude-sonnet-4-5-20250929')}")
    print(f"- CAS_STAGE2_FALLBACK_ON_ERROR={fallback or '1'}")
    for package_name in REQUIRED_PACKAGES:
        print(f"- {package_name}={_package_version(package_name)}")

    if package_errors or env_errors:
        for message in (*package_errors, *env_errors):
            print(f"ERROR: {message}", file=sys.stderr)
        raise SystemExit(1)

    if runner != "agno":
        print("WARN: CAS_STAGE2_RUNNER is not 'agno'; live Stage 2 will not call Agno.")
    print("Agno Stage 2 preflight passed.")


def _missing_packages() -> list[str]:
    missing: list[str] = []
    for package_name in REQUIRED_PACKAGES:
        if importlib.util.find_spec(package_name) is None:
            missing.append(
                f"Missing package '{package_name}'. Install with: python -m pip install -e \".[agent]\""
            )
    return missing


def _missing_env_vars() -> list[str]:
    missing: list[str] = []
    for name in REQUIRED_ENV_VARS:
        if not os.environ.get(name, "").strip():
            missing.append(f"Missing {name}. Add it to your local .env or shell environment.")
    return missing


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "missing"


if __name__ == "__main__":
    main()
