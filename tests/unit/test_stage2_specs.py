"""Tests for Stage 2 role contracts."""

from __future__ import annotations

from cas.agents.stage2_specs import (
    STAGE2_AGENT_ROLES,
    STAGE2_AGENT_SPECS,
    get_stage2_agent_spec,
)


def test_stage2_agent_roles_are_fixed_in_execution_order() -> None:
    assert STAGE2_AGENT_ROLES == ("quant_credit", "evidence_audit", "chair_report")
    assert tuple(spec.role for spec in STAGE2_AGENT_SPECS) == STAGE2_AGENT_ROLES


def test_stage2_specs_define_future_agno_contracts() -> None:
    for role in STAGE2_AGENT_ROLES:
        spec = get_stage2_agent_spec(role)

        assert spec.display_name.endswith("Agent")
        assert spec.purpose
        assert spec.required_inputs
        assert spec.output_fields
        assert "model_view" in " ".join((*spec.required_inputs, spec.future_agno_instruction))


def test_chair_report_spec_owns_committee_view_fields() -> None:
    spec = get_stage2_agent_spec("chair_report")

    assert "final_committee_label" in spec.output_fields
    assert "committee_decision_type" in spec.output_fields
    assert "committee_risk_signal" in spec.output_fields
    assert "veto_triggered" in spec.output_fields
    assert "hidden_tail_risk_flag" in spec.output_fields
    assert "conflict_resolution" in spec.output_fields
    assert "decision_trace" in spec.output_fields
    assert "final_review_memo" in spec.output_fields


def test_evidence_audit_spec_exposes_evidence_limitations() -> None:
    spec = get_stage2_agent_spec("evidence_audit")

    assert "evidence_strength" in spec.output_fields
    assert "model_challenge" in spec.output_fields
    assert "evidence_limitations" in spec.output_fields
