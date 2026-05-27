from __future__ import annotations

from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
from cas.dashboard.model_catalog import (
    available_agent_model_options,
    load_dashboard_model_catalog,
    unavailable_agent_provider_statuses,
)


def test_default_model_catalog_includes_required_agent_providers() -> None:
    catalog = load_dashboard_model_catalog()

    assert catalog.agent_defaults["quant"] == "openai:gpt-4o"
    assert catalog.agent_defaults["research"] == "anthropic:claude-sonnet-4-6"
    assert catalog.agent_defaults["manager"] == "gemini:gemini-2.5-flash"
    assert catalog.provider("openai") is not None
    assert catalog.provider("anthropic") == catalog.provider("claude")
    assert catalog.provider("gemini") == catalog.provider("google")
    assert catalog.stage2_defaults.single.provider == "openai"
    assert catalog.stage2_defaults.single.model == "gpt-4.1-mini"
    assert catalog.stage2_defaults.role("quant_credit").provider == "google"
    assert catalog.stage2_defaults.role("quant_credit").model == "gemini-2.5-flash"
    assert catalog.stage2_defaults.role("evidence_audit").provider == "anthropic"
    assert catalog.stage2_defaults.role("evidence_audit").model == "claude-sonnet-4-6"
    assert catalog.stage2_defaults.role("chair_report").provider == "openai"
    assert catalog.stage2_defaults.role("chair_report").model == "gpt-4.1-mini"


def test_available_agent_model_options_filters_by_api_key() -> None:
    catalog = load_dashboard_model_catalog()
    env = {"OPENAI_API_KEY": "sk-test"}

    options = available_agent_model_options(
        catalog,
        env=env,
        import_checker=lambda _name: True,
    )

    assert [option.id for option in options] == [
        "openai:gpt-4.1-mini",
        "openai:gpt-4o",
        "openai:gpt-4o-mini",
    ]

    hidden_statuses = unavailable_agent_provider_statuses(
        catalog,
        env=env,
        import_checker=lambda _name: True,
    )
    hidden_by_provider = {status.provider.key: status for status in hidden_statuses}
    assert "claude" in hidden_by_provider
    assert "API 키 없음: ANTHROPIC_API_KEY" in hidden_by_provider["claude"].missing_reasons
    assert "google" in hidden_by_provider
    assert (
        "API 키 없음: GOOGLE_API_KEY, GEMINI_API_KEY"
        in hidden_by_provider["google"].missing_reasons
    )


def test_available_agent_model_options_filters_by_provider_dependency() -> None:
    catalog = load_dashboard_model_catalog()
    env = {
        "OPENAI_API_KEY": "sk-test",
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "GOOGLE_API_KEY": "sk-google-test",
    }

    options = available_agent_model_options(
        catalog,
        env=env,
        import_checker=lambda name: name != "google.genai",
    )

    assert "gemini:gemini-2.5-flash" not in {option.id for option in options}
    assert "openai:gpt-4o" in {option.id for option in options}
    assert "anthropic:claude-sonnet-4-6" in {option.id for option in options}


def test_stage2_runtime_config_uses_catalog_defaults_for_multi_llm() -> None:
    config = Stage2RuntimeConfig.from_env(
        {
            "CAS_STAGE2_RUNNER": "agno",
            "CAS_STAGE2_AGNO_MODE": "multi_llm_committee",
        }
    )

    assert config.model_provider == "openai"
    assert config.model == "gpt-4.1-mini"
    assert config.multi_role_provider_resolved("quant_credit") == "google"
    assert config.multi_role_model_resolved("quant_credit") == "gemini-2.5-flash"
    assert config.multi_role_provider_resolved("evidence_audit") == "anthropic"
    assert config.multi_role_model_resolved("evidence_audit") == "claude-sonnet-4-6"
    assert config.multi_role_provider_resolved("chair_report") == "openai"
    assert config.multi_role_model_resolved("chair_report") == "gpt-4.1-mini"
