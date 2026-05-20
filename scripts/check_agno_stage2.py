"""Preflight check for live Agno Stage 2 runs."""

from __future__ import annotations

import importlib.util
import os
import sys
from importlib.metadata import PackageNotFoundError, version

from dotenv import load_dotenv

BASE_REQUIRED_PACKAGES = {"agno": "agno"}
PROVIDER_PACKAGES = {
    "anthropic": {"anthropic": "anthropic"},
    "openai": {"openai": "openai"},
    "google": {"google-genai": "google.genai"},
}
PROVIDER_ENV_VARS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}


def main() -> None:
    """Validate Agno dependencies and local live-mode environment variables."""
    load_dotenv()
    runner = os.environ.get("CAS_STAGE2_RUNNER", "deterministic").strip().lower()
    agno_mode = os.environ.get("CAS_STAGE2_AGNO_MODE", "multi_llm_committee").strip().lower()
    providers = _required_providers(runner=runner, agno_mode=agno_mode)
    package_specs = _required_package_specs(providers)
    package_errors = _missing_packages(package_specs)
    env_errors = _missing_env_vars(providers)
    fallback = os.environ.get("CAS_STAGE2_FALLBACK_ON_ERROR", "1").strip()

    print("CAS Stage 2 Agno preflight")
    print(f"- CAS_STAGE2_RUNNER={runner or 'deterministic'}")
    print(f"- CAS_STAGE2_AGNO_MODE={agno_mode or 'multi_llm_committee'}")
    print(f"- CAS_STAGE2_MODEL={os.environ.get('CAS_STAGE2_MODEL', 'claude-sonnet-4-5-20250929')}")
    print(f"- CAS_STAGE2_QUANT_PROVIDER={os.environ.get('CAS_STAGE2_QUANT_PROVIDER', 'anthropic')}")
    print(
        f"- CAS_STAGE2_QUANT_MODEL={os.environ.get('CAS_STAGE2_QUANT_MODEL', os.environ.get('CAS_STAGE2_MODEL', 'claude-sonnet-4-5-20250929'))}"
    )
    print(
        f"- CAS_STAGE2_EVIDENCE_PROVIDER={os.environ.get('CAS_STAGE2_EVIDENCE_PROVIDER', 'openai')}"
    )
    print(
        f"- CAS_STAGE2_EVIDENCE_MODEL={os.environ.get('CAS_STAGE2_EVIDENCE_MODEL', 'gpt-5.4-mini')}"
    )
    print(f"- CAS_STAGE2_CHAIR_PROVIDER={os.environ.get('CAS_STAGE2_CHAIR_PROVIDER', 'google')}")
    print(
        f"- CAS_STAGE2_CHAIR_MODEL={os.environ.get('CAS_STAGE2_CHAIR_MODEL', 'gemini-flash-latest')}"
    )
    print(f"- CAS_STAGE2_FALLBACK_ON_ERROR={fallback or '1'}")
    for package_name in package_specs:
        print(f"- {package_name}={_package_version(package_name)}")

    if package_errors or env_errors:
        for message in (*package_errors, *env_errors):
            print(f"ERROR: {message}", file=sys.stderr)
        raise SystemExit(1)

    if runner != "agno":
        print("WARN: CAS_STAGE2_RUNNER is not 'agno'; live Stage 2 will not call Agno.")
    print("Agno Stage 2 preflight passed.")


def _required_providers(*, runner: str, agno_mode: str) -> list[str]:
    if runner != "agno":
        return []
    if agno_mode in {"multi", "multi_llm", "multi_llm_committee"}:
        return [
            _normalize_provider(os.environ.get("CAS_STAGE2_QUANT_PROVIDER", "anthropic")),
            _normalize_provider(os.environ.get("CAS_STAGE2_EVIDENCE_PROVIDER", "openai")),
            _normalize_provider(os.environ.get("CAS_STAGE2_CHAIR_PROVIDER", "google")),
        ]
    return [_normalize_provider(os.environ.get("CAS_STAGE2_MODEL_PROVIDER", "anthropic"))]


def _required_package_specs(providers: list[str]) -> dict[str, str]:
    package_specs = dict(BASE_REQUIRED_PACKAGES)
    for provider in providers:
        package_specs.update(PROVIDER_PACKAGES[provider])
    return package_specs


def _missing_packages(package_specs: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for package_name, import_name in package_specs.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(
                f"Missing package '{package_name}'. Install with: python -m pip install -e \".[agent]\""
            )
    return missing


def _missing_env_vars(providers: list[str]) -> list[str]:
    missing: list[str] = []
    for provider in providers:
        names = PROVIDER_ENV_VARS[provider]
        if not any(os.environ.get(name, "").strip() for name in names):
            missing.append(
                f"Missing {' or '.join(names)}. Add it to your local .env or shell environment."
            )
    return missing


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace("-", "_")
    aliases = {
        "anthropic": "anthropic",
        "claude": "anthropic",
        "openai": "openai",
        "gpt": "openai",
        "google": "google",
        "gemini": "google",
    }
    if normalized not in aliases:
        raise SystemExit(
            "Unsupported provider. Use anthropic/claude, openai/gpt, or google/gemini."
        )
    return aliases[normalized]


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "missing"


if __name__ == "__main__":
    main()
