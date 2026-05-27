"""Preflight check for live Agno Stage 2 runs."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from dotenv import load_dotenv

from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
from cas.llm.model_catalog import (
    ModelCatalog,
    ProviderConfig,
    is_multi_llm_committee_mode,
    load_model_catalog,
    normalize_stage2_provider,
)

BASE_REQUIRED_PACKAGES = {"agno": ("agno",)}
SCENARIO_CHOICES = ("all", "active", "openai-single", "gemini-single", "multi-role")
ImportChecker = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class RoleRequirement:
    """Provider/model pair required by one Stage 2 role in a smoke scenario."""

    role: str
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class PreflightScenario:
    """One explicit Agno Stage 2 preflight scenario."""

    name: str
    description: str
    runner: str
    agno_mode: str
    roles: tuple[RoleRequirement, ...]


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Preflight result for one scenario."""

    scenario: PreflightScenario
    package_errors: tuple[str, ...]
    env_errors: tuple[str, ...]
    capability_warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.package_errors and not self.env_errors


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_CHOICES,
        default="all",
        help=(
            "Which smoke/preflight scenario to validate. Default all checks OpenAI single, "
            "Gemini single, and catalog multi-role defaults."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Validate Agno dependencies and local live-mode environment variables."""
    args = parse_args()
    load_dotenv()
    catalog = load_model_catalog()
    env = os.environ
    runtime_config = Stage2RuntimeConfig.from_env(env)
    scenarios = _selected_scenarios(args.scenario, catalog=catalog, env=env)
    results = [
        _evaluate_scenario(
            scenario,
            catalog=catalog,
            env=env,
            runtime_config=runtime_config,
        )
        for scenario in scenarios
    ]

    print("CAS Stage 2 Agno preflight")
    print(f"- scenario={args.scenario}")
    print(f"- fallback_on_error={runtime_config.fallback_on_error}")
    print(f"- llm_cache_enabled={runtime_config.llm_cache_enabled}")
    print(f"- agent_retries={runtime_config.agent_retries}")
    print(f"- agent_retry_delay_seconds={runtime_config.agent_retry_delay_seconds}")
    print(f"- agent_timeout_seconds={runtime_config.agent_timeout_seconds}")
    print(f"- provider_max_retries={runtime_config.provider_max_retries}")
    for result in results:
        _print_result(result, catalog=catalog)

    errors = [
        f"{result.scenario.name}: {message}"
        for result in results
        for message in (*result.package_errors, *result.env_errors)
    ]
    if errors:
        for message in errors:
            print(f"ERROR: {message}", file=sys.stderr)
        raise SystemExit(1)
    print("Agno Stage 2 preflight passed.")


def _selected_scenarios(
    scenario_name: str,
    *,
    catalog: ModelCatalog,
    env: Mapping[str, str],
) -> list[PreflightScenario]:
    if scenario_name == "all":
        return [
            _openai_single_scenario(catalog),
            _gemini_single_scenario(catalog),
            _multi_role_scenario(catalog),
        ]
    if scenario_name == "active":
        return [_active_scenario(catalog=catalog, env=env)]
    if scenario_name == "openai-single":
        return [_openai_single_scenario(catalog)]
    if scenario_name == "gemini-single":
        return [_gemini_single_scenario(catalog)]
    if scenario_name == "multi-role":
        return [_multi_role_scenario(catalog)]
    raise ValueError(f"Unknown scenario: {scenario_name}")


def _openai_single_scenario(catalog: ModelCatalog) -> PreflightScenario:
    single_default = catalog.stage2_defaults.single
    model = (
        single_default.model
        if single_default.provider == "openai"
        else _default_model_for_provider("openai", catalog=catalog)
    )
    return PreflightScenario(
        name="openai-single",
        description="Single-provider Agno Stage 2 smoke with OpenAI.",
        runner="agno",
        agno_mode="single",
        roles=(RoleRequirement(role="single", provider="openai", model=model),),
    )


def _gemini_single_scenario(catalog: ModelCatalog) -> PreflightScenario:
    return PreflightScenario(
        name="gemini-single",
        description="Single-provider Agno Stage 2 smoke with Gemini.",
        runner="agno",
        agno_mode="single",
        roles=(
            RoleRequirement(
                role="single",
                provider="google",
                model=_default_model_for_provider("google", catalog=catalog),
            ),
        ),
    )


def _multi_role_scenario(catalog: ModelCatalog) -> PreflightScenario:
    defaults = catalog.stage2_defaults
    return PreflightScenario(
        name="multi-role",
        description="Catalog multi-role Stage 2 smoke: Quant/Evidence/Chair role routing.",
        runner="agno",
        agno_mode="multi_llm_committee",
        roles=(
            RoleRequirement(
                role="quant_credit",
                provider=defaults.role("quant_credit").provider,
                model=defaults.role("quant_credit").model,
            ),
            RoleRequirement(
                role="evidence_audit",
                provider=defaults.role("evidence_audit").provider,
                model=defaults.role("evidence_audit").model,
            ),
            RoleRequirement(
                role="chair_report",
                provider=defaults.role("chair_report").provider,
                model=defaults.role("chair_report").model,
            ),
        ),
    )


def _active_scenario(*, catalog: ModelCatalog, env: Mapping[str, str]) -> PreflightScenario:
    runner = env.get("CAS_STAGE2_RUNNER", "deterministic").strip().lower() or "deterministic"
    agno_mode = (
        env.get("CAS_STAGE2_AGNO_MODE", catalog.stage2_defaults.agno_mode).strip().lower()
        or catalog.stage2_defaults.agno_mode
    )
    if runner != "agno":
        return PreflightScenario(
            name="active",
            description="Current environment is deterministic; no live Agno provider required.",
            runner=runner,
            agno_mode=agno_mode,
            roles=(),
        )
    if is_multi_llm_committee_mode(agno_mode):
        defaults = catalog.stage2_defaults
        roles = (
            RoleRequirement(
                role="quant_credit",
                provider=_normalize_provider(
                    env.get("CAS_STAGE2_QUANT_PROVIDER", defaults.role("quant_credit").provider),
                    catalog=catalog,
                ),
                model=env.get("CAS_STAGE2_QUANT_MODEL", defaults.role("quant_credit").model),
            ),
            RoleRequirement(
                role="evidence_audit",
                provider=_normalize_provider(
                    env.get(
                        "CAS_STAGE2_EVIDENCE_PROVIDER",
                        defaults.role("evidence_audit").provider,
                    ),
                    catalog=catalog,
                ),
                model=env.get("CAS_STAGE2_EVIDENCE_MODEL", defaults.role("evidence_audit").model),
            ),
            RoleRequirement(
                role="chair_report",
                provider=_normalize_provider(
                    env.get("CAS_STAGE2_CHAIR_PROVIDER", defaults.role("chair_report").provider),
                    catalog=catalog,
                ),
                model=env.get("CAS_STAGE2_CHAIR_MODEL", defaults.role("chair_report").model),
            ),
        )
    else:
        single_default = catalog.stage2_defaults.single
        roles = (
            RoleRequirement(
                role="single",
                provider=_normalize_provider(
                    env.get("CAS_STAGE2_MODEL_PROVIDER", single_default.provider),
                    catalog=catalog,
                ),
                model=env.get("CAS_STAGE2_MODEL", single_default.model),
            ),
        )
    return PreflightScenario(
        name="active",
        description="Current environment Stage 2 Agno route.",
        runner=runner,
        agno_mode=agno_mode,
        roles=roles,
    )


def _evaluate_scenario(
    scenario: PreflightScenario,
    *,
    catalog: ModelCatalog,
    env: Mapping[str, str],
    runtime_config: Stage2RuntimeConfig,
    import_checker: ImportChecker | None = None,
) -> ScenarioResult:
    package_errors = tuple(
        _missing_packages(
            _required_package_specs(_required_providers(scenario), catalog=catalog),
            import_checker=import_checker,
        )
    )
    env_errors = tuple(_missing_env_vars(_required_providers(scenario), catalog=catalog, env=env))
    capability_warnings = tuple(
        _capability_warnings(scenario, catalog=catalog, runtime_config=runtime_config)
    )
    return ScenarioResult(
        scenario=scenario,
        package_errors=package_errors,
        env_errors=env_errors,
        capability_warnings=capability_warnings,
    )


def _required_providers(scenario: PreflightScenario) -> list[str]:
    if scenario.runner != "agno":
        return []
    providers = [_normalize_provider_no_catalog(role.provider) for role in scenario.roles]
    return list(dict.fromkeys(providers))


def _required_package_specs(
    providers: list[str],
    *,
    catalog: ModelCatalog,
) -> dict[str, tuple[str, ...]]:
    package_specs = dict(BASE_REQUIRED_PACKAGES)
    for provider in providers:
        config = _provider_config(provider, catalog=catalog)
        package_specs[config.package_name] = config.required_imports or (config.runtime_provider,)
    return package_specs


def _missing_packages(
    package_specs: dict[str, tuple[str, ...]],
    *,
    import_checker: ImportChecker | None = None,
) -> list[str]:
    checker = import_checker or _default_import_checker
    missing: list[str] = []
    for package_name, import_names in package_specs.items():
        missing_imports = tuple(
            import_name for import_name in import_names if not checker(import_name)
        )
        if missing_imports:
            missing.append(
                f"Missing package '{package_name}' imports {', '.join(missing_imports)}. "
                'Install with: python -m pip install -e ".[agent]"'
            )
    return missing


def _missing_env_vars(
    providers: list[str],
    *,
    catalog: ModelCatalog,
    env: Mapping[str, str],
) -> list[str]:
    missing: list[str] = []
    for provider in providers:
        names = _provider_config(provider, catalog=catalog).api_key_env_vars
        if not any(env.get(name, "").strip() for name in names):
            missing.append(
                f"Missing {' or '.join(names)}. Add it to your local .env or shell environment."
            )
    return missing


def _capability_warnings(
    scenario: PreflightScenario,
    *,
    catalog: ModelCatalog,
    runtime_config: Stage2RuntimeConfig,
) -> list[str]:
    warnings: list[str] = []
    for role in scenario.roles:
        provider = _provider_config(role.provider, catalog=catalog)
        known_models = _runtime_model_ids(provider)
        if known_models and role.model not in known_models:
            warnings.append(
                f"{role.role}: model {role.model!r} is not listed in catalog for {provider.label}; "
                "treating it as a custom model id."
            )
        capabilities = _provider_runtime_capabilities(provider.runtime_provider)
        if capabilities["timeout"] != "enabled":
            warnings.append(f"{role.role}: timeout is not enabled for {provider.label}.")
        if capabilities["provider_retry"] == "none" and runtime_config.provider_max_retries > 0:
            warnings.append(
                f"{role.role}: provider retry is not wired for {provider.label}; "
                "agent retry still applies."
            )
    return warnings


def _print_result(result: ScenarioResult, *, catalog: ModelCatalog) -> None:
    scenario = result.scenario
    status = "PASS" if result.passed else "FAIL"
    print("")
    print(f"[{status}] {scenario.name}")
    print(f"- {scenario.description}")
    print(f"- runner={scenario.runner}")
    print(f"- agno_mode={scenario.agno_mode}")
    if not scenario.roles:
        print("- roles=(none)")
    for role in scenario.roles:
        provider = _provider_config(role.provider, catalog=catalog)
        capabilities = _provider_runtime_capabilities(provider.runtime_provider)
        key_status = _api_key_status(provider, env=os.environ)
        model_status = "catalog" if role.model in _runtime_model_ids(provider) else "custom"
        print(
            f"- role={role.role} provider={provider.label}({provider.runtime_provider}) "
            f"model={role.model} model_source={model_status}"
        )
        print(
            f"  package={provider.package_name}:{_package_version(provider.package_name)} "
            f"imports={','.join(provider.required_imports) or provider.runtime_provider}"
        )
        print(f"  api_key={key_status}")
        print(
            "  capabilities="
            f"structured_outputs:{capabilities['structured_outputs']}, "
            f"timeout:{capabilities['timeout']}, "
            f"provider_retry:{capabilities['provider_retry']}"
        )
    for warning in result.capability_warnings:
        print(f"WARN: {warning}")


def _default_model_for_provider(provider: str, *, catalog: ModelCatalog) -> str:
    config = _provider_config(provider, catalog=catalog)
    if config.agent_models:
        return _runtime_model_id(config.agent_models[0].id)
    raise SystemExit(f"Provider {provider!r} has no configured agent model.")


def _runtime_model_ids(provider: ProviderConfig) -> set[str]:
    return {_runtime_model_id(option.id) for option in provider.agent_models}


def _runtime_model_id(model_id: str) -> str:
    return model_id.split(":", 1)[1] if ":" in model_id else model_id


def _provider_runtime_capabilities(provider: str) -> dict[str, str]:
    normalized = _normalize_provider_no_catalog(provider)
    if normalized == "openai":
        return {
            "structured_outputs": "native",
            "timeout": "enabled",
            "provider_retry": "max_retries",
        }
    if normalized == "google":
        return {
            "structured_outputs": "native",
            "timeout": "enabled",
            "provider_retry": "retries",
        }
    if normalized == "anthropic":
        return {
            "structured_outputs": "native",
            "timeout": "enabled",
            "provider_retry": "agent_retry_only",
        }
    return {
        "structured_outputs": "unknown",
        "timeout": "unknown",
        "provider_retry": "unknown",
    }


def _api_key_status(provider: ProviderConfig, *, env: Mapping[str, str]) -> str:
    names = provider.api_key_env_vars
    if any(env.get(name, "").strip() for name in names):
        return "set:" + "/".join(names)
    return "missing:" + "/".join(names)


def _normalize_provider(provider: str, *, catalog: ModelCatalog) -> str:
    try:
        return normalize_stage2_provider(provider, catalog=catalog)
    except ValueError as error:
        raise SystemExit(
            "Unsupported provider. Use anthropic/claude, openai/gpt, or google/gemini."
        ) from error


def _normalize_provider_no_catalog(provider: str) -> str:
    return provider.strip().lower().replace("-", "_")


def _provider_config(provider: str, *, catalog: ModelCatalog) -> ProviderConfig:
    config = catalog.provider(provider)
    if config is None:
        raise SystemExit(
            "Unsupported provider. Use anthropic/claude, openai/gpt, or google/gemini."
        )
    return config


def _default_import_checker(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "missing"


if __name__ == "__main__":
    main()
