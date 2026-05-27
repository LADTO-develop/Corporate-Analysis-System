"""Tests for explicit Stage 2 Agno preflight scenarios."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
from cas.llm.model_catalog import load_model_catalog


def _load_check_agno_stage2_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_agno_stage2.py"
    spec = importlib.util.spec_from_file_location("check_agno_stage2", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selected_scenarios_include_openai_gemini_and_multi_role() -> None:
    checker = _load_check_agno_stage2_module()
    catalog = load_model_catalog()

    scenarios = checker._selected_scenarios("all", catalog=catalog, env={})

    assert [scenario.name for scenario in scenarios] == [
        "openai-single",
        "gemini-single",
        "multi-role",
    ]
    assert scenarios[1].roles[0].provider == "google"
    assert scenarios[1].roles[0].model == "gemini-2.5-flash"
    assert [(role.role, role.provider, role.model) for role in scenarios[2].roles] == [
        ("quant_credit", "google", "gemini-2.5-flash"),
        ("evidence_audit", "anthropic", "claude-sonnet-4-6"),
        ("chair_report", "openai", "gpt-4.1-mini"),
    ]


def test_gemini_preflight_requires_google_or_gemini_key() -> None:
    checker = _load_check_agno_stage2_module()
    catalog = load_model_catalog()
    scenario = checker._selected_scenarios("gemini-single", catalog=catalog, env={})[0]

    result = checker._evaluate_scenario(
        scenario,
        catalog=catalog,
        env={},
        runtime_config=Stage2RuntimeConfig(),
        import_checker=lambda _name: True,
    )

    assert result.package_errors == ()
    assert len(result.env_errors) == 1
    assert "GOOGLE_API_KEY or GEMINI_API_KEY" in result.env_errors[0]
    assert result.passed is False


def test_openai_and_gemini_capabilities_are_timeout_and_retry_aware() -> None:
    checker = _load_check_agno_stage2_module()

    assert checker._provider_runtime_capabilities("openai") == {
        "structured_outputs": "native",
        "timeout": "enabled",
        "provider_retry": "max_retries",
    }
    assert checker._provider_runtime_capabilities("google") == {
        "structured_outputs": "native",
        "timeout": "enabled",
        "provider_retry": "retries",
    }
