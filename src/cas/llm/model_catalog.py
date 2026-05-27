"""Config-backed LLM model catalog shared by dashboard and Stage 2."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cas.utils.io import read_yaml

DEFAULT_MODEL_CATALOG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "dashboard" / "llm_models.yaml"
)
MODEL_CATALOG_PATH_ENV_VAR = "CAS_LLM_MODEL_CONFIG"
LEGACY_MODEL_CATALOG_PATH_ENV_VAR = "CAS_DASHBOARD_LLM_MODEL_CONFIG"
DEFAULT_STAGE2_RUNNER = "deterministic"
DEFAULT_STAGE2_AGNO_MODE = "single"
DEFAULT_STAGE2_SINGLE_PROVIDER = "openai"
DEFAULT_STAGE2_SINGLE_MODEL = "gpt-4.1-mini"
DEFAULT_STAGE2_MULTI_ROLE_DEFAULTS = {
    "quant_credit": ("google", "gemini-2.5-flash"),
    "evidence_audit": ("anthropic", "claude-sonnet-4-6"),
    "chair_report": ("openai", "gpt-4.1-mini"),
}
ImportChecker = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class ModelOption:
    """A model ID plus its display label."""

    id: str
    label: str


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Provider-level model catalog settings."""

    key: str
    label: str
    aliases: tuple[str, ...]
    api_key_env_vars: tuple[str, ...]
    agent_enabled: bool
    required_imports: tuple[str, ...]
    agent_models: tuple[ModelOption, ...]
    package_name: str
    runtime_provider: str

    def first_api_key_env_var(self) -> str:
        """Return the preferred API key environment variable name."""
        return self.api_key_env_vars[0] if self.api_key_env_vars else ""

    def api_key_from_env(self, env: Mapping[str, str] | None = None) -> str:
        """Return the first non-empty configured API key from the environment."""
        source = os.environ if env is None else env
        for env_var in self.api_key_env_vars:
            value = source.get(env_var, "").strip()
            if value:
                return value
        return ""


@dataclass(frozen=True, slots=True)
class ProviderAvailability:
    """Local preflight result for a provider."""

    provider: ProviderConfig
    missing_imports: tuple[str, ...]
    has_api_key: bool

    @property
    def is_available(self) -> bool:
        """Return whether the provider can be exposed in the model UI."""
        return not self.missing_imports and self.has_api_key

    @property
    def missing_reasons(self) -> tuple[str, ...]:
        """Return concise Korean reasons for hidden providers."""
        reasons: list[str] = []
        if self.missing_imports:
            reasons.append("패키지 없음: " + ", ".join(self.missing_imports))
        if not self.has_api_key:
            reasons.append("API 키 없음: " + ", ".join(self.provider.api_key_env_vars))
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class Stage2RoleModelDefault:
    """Default provider/model pair for a Stage 2 role."""

    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class Stage2ModelDefaults:
    """Stage 2 single-model and multi-role defaults loaded from the catalog."""

    agno_mode: str
    single: Stage2RoleModelDefault
    roles: dict[str, Stage2RoleModelDefault]

    def role(self, role: str) -> Stage2RoleModelDefault:
        """Return the configured default for a Stage 2 role."""
        normalized = _normalize_role(role)
        try:
            return self.roles[normalized]
        except KeyError as error:
            raise ValueError(f"Unknown Stage 2 role: {role}") from error


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """Parsed shared LLM model catalog."""

    providers: dict[str, ProviderConfig]
    agent_defaults: dict[str, str]
    stage2_defaults: Stage2ModelDefaults

    def provider(self, key_or_alias: str) -> ProviderConfig | None:
        """Resolve a provider by canonical key, runtime provider, or alias."""
        normalized = key_or_alias.strip().lower().replace("-", "_")
        if normalized in self.providers:
            return self.providers[normalized]
        for provider in self.providers.values():
            candidates = {provider.runtime_provider, *provider.aliases}
            normalized_candidates = {
                candidate.strip().lower().replace("-", "_") for candidate in candidates
            }
            if normalized in normalized_candidates:
                return provider
        return None


DashboardModelCatalog = ModelCatalog


def _default_import_checker(import_name: str) -> bool:
    """Return whether an import can be resolved without importing it."""
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    """Validate that a raw config value is a mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} 설정은 mapping이어야 합니다.")
    return value


def _as_string_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
    """Validate and normalize a raw string sequence."""
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{field_name} 설정은 문자열 목록이어야 합니다.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} 항목은 문자열이어야 합니다.")
        cleaned = item.strip()
        if cleaned:
            result.append(cleaned)
    return tuple(result)


def _as_model_options(value: object, *, field_name: str) -> tuple[ModelOption, ...]:
    """Validate and normalize configured model option dictionaries."""
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{field_name} 설정은 모델 목록이어야 합니다.")

    options: list[ModelOption] = []
    for item in value:
        item_mapping = _as_mapping(item, field_name=field_name)
        model_id = item_mapping.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"{field_name} 항목에는 id가 필요합니다.")
        label = item_mapping.get("label", model_id)
        if not isinstance(label, str):
            raise ValueError(f"{field_name} 항목의 label은 문자열이어야 합니다.")
        options.append(ModelOption(id=model_id.strip(), label=label.strip() or model_id.strip()))
    return tuple(options)


def _as_bool(value: object, *, default: bool = False) -> bool:
    """Normalize a bool-like config value."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError("boolean 설정에는 true/false 값을 사용해야 합니다.")


def _parse_provider(provider_key: str, raw_provider: object) -> ProviderConfig:
    """Parse a single provider block."""
    provider = _as_mapping(raw_provider, field_name=f"providers.{provider_key}")
    label = provider.get("label", provider_key)
    if not isinstance(label, str):
        raise ValueError(f"providers.{provider_key}.label은 문자열이어야 합니다.")
    package_name = provider.get("package_name", provider_key)
    if not isinstance(package_name, str) or not package_name.strip():
        raise ValueError(f"providers.{provider_key}.package_name은 문자열이어야 합니다.")
    runtime_provider = provider.get("runtime_provider", provider_key)
    if not isinstance(runtime_provider, str) or not runtime_provider.strip():
        raise ValueError(f"providers.{provider_key}.runtime_provider는 문자열이어야 합니다.")

    return ProviderConfig(
        key=provider_key,
        label=label,
        aliases=_as_string_sequence(
            provider.get("aliases"),
            field_name=f"providers.{provider_key}.aliases",
        ),
        api_key_env_vars=_as_string_sequence(
            provider.get("api_key_env_vars"),
            field_name=f"providers.{provider_key}.api_key_env_vars",
        ),
        agent_enabled=_as_bool(provider.get("agent_enabled")),
        required_imports=_as_string_sequence(
            provider.get("required_imports"),
            field_name=f"providers.{provider_key}.required_imports",
        ),
        agent_models=_as_model_options(
            provider.get("agent_models"),
            field_name=f"providers.{provider_key}.agent_models",
        ),
        package_name=package_name.strip(),
        runtime_provider=runtime_provider.strip().lower().replace("-", "_"),
    )


def _parse_stage2_defaults(
    raw_config: Mapping[str, Any], catalog: ModelCatalog | None = None
) -> Stage2ModelDefaults:
    """Parse Stage 2 defaults, using conservative fallback values when omitted."""
    raw_stage2 = raw_config.get("stage2_defaults", {})
    stage2 = _as_mapping(raw_stage2, field_name="stage2_defaults")
    agno_mode_raw = stage2.get("agno_mode", DEFAULT_STAGE2_AGNO_MODE)
    if not isinstance(agno_mode_raw, str) or not agno_mode_raw.strip():
        raise ValueError("stage2_defaults.agno_mode은 문자열이어야 합니다.")

    single = _parse_stage2_role_default(
        stage2.get("single", {}),
        field_name="stage2_defaults.single",
        fallback_provider=DEFAULT_STAGE2_SINGLE_PROVIDER,
        fallback_model=DEFAULT_STAGE2_SINGLE_MODEL,
        catalog=catalog,
    )
    multi = _as_mapping(
        stage2.get("multi_llm_committee", {}),
        field_name="stage2_defaults.multi_llm_committee",
    )

    roles: dict[str, Stage2RoleModelDefault] = {}
    for role, fallback in DEFAULT_STAGE2_MULTI_ROLE_DEFAULTS.items():
        roles[role] = _parse_stage2_role_default(
            multi.get(role, {}),
            field_name=f"stage2_defaults.multi_llm_committee.{role}",
            fallback_provider=fallback[0],
            fallback_model=fallback[1],
            catalog=catalog,
        )

    return Stage2ModelDefaults(
        agno_mode=agno_mode_raw.strip(),
        single=single,
        roles=roles,
    )


def _parse_stage2_role_default(
    value: object,
    *,
    field_name: str,
    fallback_provider: str,
    fallback_model: str,
    catalog: ModelCatalog | None,
) -> Stage2RoleModelDefault:
    """Parse one Stage 2 provider/model default."""
    role_raw = _as_mapping(value, field_name=field_name)
    provider_raw = role_raw.get("provider", fallback_provider)
    model_raw = role_raw.get("model", fallback_model)
    if not isinstance(provider_raw, str) or not provider_raw.strip():
        raise ValueError(f"{field_name}.provider는 문자열이어야 합니다.")
    if not isinstance(model_raw, str) or not model_raw.strip():
        raise ValueError(f"{field_name}.model은 문자열이어야 합니다.")
    provider = normalize_stage2_provider(provider_raw, catalog=catalog)
    return Stage2RoleModelDefault(provider=provider, model=model_raw.strip())


def load_model_catalog(path: str | Path | None = None) -> ModelCatalog:
    """Load the shared LLM model catalog from YAML."""
    raw_path = (
        path
        or os.environ.get(MODEL_CATALOG_PATH_ENV_VAR)
        or os.environ.get(LEGACY_MODEL_CATALOG_PATH_ENV_VAR)
        or DEFAULT_MODEL_CATALOG_PATH
    )
    raw_config = read_yaml(raw_path)
    providers_raw = _as_mapping(raw_config.get("providers"), field_name="providers")

    providers: dict[str, ProviderConfig] = {}
    for key, raw_provider in providers_raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("provider key는 비어 있지 않은 문자열이어야 합니다.")
        normalized_key = key.strip().lower().replace("-", "_")
        providers[normalized_key] = _parse_provider(normalized_key, raw_provider)

    agent_defaults_raw = _as_mapping(
        raw_config.get("agent_defaults", {}),
        field_name="agent_defaults",
    )
    agent_defaults = {
        str(key): str(value)
        for key, value in agent_defaults_raw.items()
        if isinstance(key, str) and isinstance(value, str)
    }

    placeholder_catalog = ModelCatalog(
        providers=providers,
        agent_defaults=agent_defaults,
        stage2_defaults=Stage2ModelDefaults(
            agno_mode=DEFAULT_STAGE2_AGNO_MODE,
            single=Stage2RoleModelDefault(
                provider=DEFAULT_STAGE2_SINGLE_PROVIDER,
                model=DEFAULT_STAGE2_SINGLE_MODEL,
            ),
            roles={
                role: Stage2RoleModelDefault(provider=provider, model=model)
                for role, (provider, model) in DEFAULT_STAGE2_MULTI_ROLE_DEFAULTS.items()
            },
        ),
    )
    return ModelCatalog(
        providers=providers,
        agent_defaults=agent_defaults,
        stage2_defaults=_parse_stage2_defaults(raw_config, catalog=placeholder_catalog),
    )


def load_dashboard_model_catalog(path: str | Path | None = None) -> ModelCatalog:
    """Backward-compatible alias for dashboard callers."""
    return load_model_catalog(path)


def normalize_stage2_provider(
    provider: str,
    *,
    catalog: ModelCatalog | None = None,
) -> str:
    """Normalize a provider into the runtime provider name used by Stage 2."""
    active_catalog = catalog or load_model_catalog()
    configured = active_catalog.provider(provider)
    if configured is not None:
        return configured.runtime_provider

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
        raise ValueError(
            "Unsupported CAS Stage 2 model provider. "
            "Use one of: anthropic/claude, openai/gpt, google/gemini."
        )
    return aliases[normalized]


def stage2_single_model_default(catalog: ModelCatalog | None = None) -> Stage2RoleModelDefault:
    """Return the catalog-backed single-model Stage 2 default."""
    active_catalog = catalog or load_model_catalog()
    return active_catalog.stage2_defaults.single


def stage2_role_model_default(
    role: str,
    *,
    catalog: ModelCatalog | None = None,
) -> Stage2RoleModelDefault:
    """Return the catalog-backed multi-LLM Stage 2 default for a role."""
    active_catalog = catalog or load_model_catalog()
    return active_catalog.stage2_defaults.role(role)


def is_multi_llm_committee_mode(agno_mode: str) -> bool:
    """Return whether a Stage 2 routing mode uses role-specific models."""
    return agno_mode.strip().lower() in {"multi", "multi_llm", "multi_llm_committee"}


def provider_availability(
    provider: ProviderConfig,
    *,
    env: Mapping[str, str] | None = None,
    import_checker: ImportChecker | None = None,
) -> ProviderAvailability:
    """Run local preflight for provider visibility."""
    checker = import_checker or _default_import_checker
    missing_imports = tuple(
        import_name for import_name in provider.required_imports if not checker(import_name)
    )
    return ProviderAvailability(
        provider=provider,
        missing_imports=missing_imports,
        has_api_key=bool(provider.api_key_from_env(env)),
    )


def available_agent_providers(
    catalog: ModelCatalog,
    *,
    env: Mapping[str, str] | None = None,
    import_checker: ImportChecker | None = None,
) -> tuple[ProviderConfig, ...]:
    """Return agent providers enabled in config and passing preflight."""
    providers: list[ProviderConfig] = []
    for provider in catalog.providers.values():
        if not provider.agent_enabled:
            continue
        status = provider_availability(provider, env=env, import_checker=import_checker)
        if status.is_available and provider.agent_models:
            providers.append(provider)
    return tuple(providers)


def unavailable_agent_provider_statuses(
    catalog: ModelCatalog,
    *,
    env: Mapping[str, str] | None = None,
    import_checker: ImportChecker | None = None,
) -> tuple[ProviderAvailability, ...]:
    """Return preflight statuses for configured providers hidden from the UI."""
    statuses: list[ProviderAvailability] = []
    for provider in catalog.providers.values():
        if not provider.agent_enabled:
            continue
        status = provider_availability(provider, env=env, import_checker=import_checker)
        if not status.is_available or not provider.agent_models:
            statuses.append(status)
    return tuple(statuses)


def available_agent_model_options(
    catalog: ModelCatalog,
    *,
    env: Mapping[str, str] | None = None,
    import_checker: ImportChecker | None = None,
) -> tuple[ModelOption, ...]:
    """Return agent model options whose provider passes preflight."""
    options: list[ModelOption] = []
    for provider in available_agent_providers(
        catalog,
        env=env,
        import_checker=import_checker,
    ):
        options.extend(provider.agent_models)
    return tuple(options)


def _normalize_role(role: str) -> str:
    """Normalize public role aliases to Stage 2 role names."""
    normalized = role.strip().lower().replace("-", "_")
    aliases = {
        "quant": "quant_credit",
        "quant_credit": "quant_credit",
        "evidence": "evidence_audit",
        "research": "evidence_audit",
        "evidence_audit": "evidence_audit",
        "chair": "chair_report",
        "manager": "chair_report",
        "chair_report": "chair_report",
    }
    return aliases.get(normalized, normalized)
