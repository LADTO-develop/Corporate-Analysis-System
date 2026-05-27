"""Unit tests for Stage 2 committee helpers."""

from __future__ import annotations

from typing import Any

import pytest

from cas.agents import stage2_runner as stage2_runner_module
from cas.agents.nodes import committee_node as committee_node_module
from cas.agents.nodes.committee_node import (
    _evidence_audit_agent,
    _quant_credit_agent,
    _recommendation_from_score,
    run,
)
from cas.agents.stage2_bundle import build_stage2_input_bundle
from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
    ReviewQAOutput,
    RiskRecallQAOutput,
)
from cas.agents.stage2_runner import Stage2LLMResponse
from cas.agents.state import AgentState


def test_recommendation_thresholds() -> None:
    thresholds = {"priority": 0.75, "watch": 0.60, "review": 0.45}
    assert _recommendation_from_score(0.82, thresholds) == "priority"
    assert _recommendation_from_score(0.65, thresholds) == "watch"
    assert _recommendation_from_score(0.50, thresholds) == "review"
    assert _recommendation_from_score(0.30, thresholds) == "defer"


def test_quant_credit_agent_generates_quant_summary() -> None:
    state: AgentState = {
        "company_id": "250",
        "company_name": "삼천당제약(주)",
        "market": "KOSDAQ",
        "source_feature_row": {
            "market": "KOSDAQ",
            "industry_macro_category": "manufacturing",
            "firm_size_group": "mid_sized",
            "current_ratio": 2.82,
            "cash_ratio": 0.69,
            "capital_impairment_ratio": -0.12,
            "gross_profit": 96966293.0,
            "dividend_payer": 1,
        },
        "peer_comparison_rows": [
            {
                "feature": "gross_profit",
                "industry_median": 55000000.0,
                "market_median": 30000000.0,
                "industry_percentile": 83.4,
            }
        ],
    }
    xgb_result = {
        "probability_speculative": 0.0474,
        "prediction_label": "투자적격",
        "top_drivers": [
            ("capital_impairment_ratio", -0.5535),
            ("gross_profit", -0.4322),
            ("dividend_payer", -0.4259),
        ],
    }

    state["xgboost_result"] = xgb_result
    structured_output = _quant_credit_agent(build_stage2_input_bundle(state))
    agent = structured_output.to_agent_output()

    assert structured_output.role == "quant_credit"
    assert structured_output.key_risk_factors or structured_output.mitigating_factors
    assert agent.role == "quant_credit"
    assert "투자적격" in agent.summary
    assert "위험확률" in agent.summary
    assert len(agent.findings) == 3
    assert any("핵심 위험 요인" in item for item in agent.findings)
    assert any("완화 요인" in item for item in agent.findings)
    assert "산업 중앙값" in " ".join(agent.findings)


def test_evidence_audit_agent_flags_liquidity_mismatch() -> None:
    state: AgentState = {
        "source_feature_row": {
            "current_ratio": 0.82,
            "cash_ratio": 0.11,
            "debt_ratio": 1.40,
            "short_term_borrowings_share": 0.71,
            "cashflow_coverage_ratio": 0.80,
            "interest_coverage_ratio": 2.10,
            "ocf_to_total_liabilities": 0.04,
            "ocf_to_total_borrowings": 0.09,
            "is_2y_consecutive_ocf_deficit": 0,
            "icr_under_1": 0,
        },
        "xgboost_result": {"prediction_label": "투자적격"},
    }

    structured_output = _evidence_audit_agent(build_stage2_input_bundle(state))
    agent = structured_output.to_agent_output()

    assert structured_output.role == "evidence_audit"
    assert structured_output.debt_liquidity_cross_check
    assert agent.role == "evidence_audit"
    assert "투자적격" in agent.summary
    assert "추가 경계" in agent.summary
    assert any("유동비율이 1.0 미만" in item for item in agent.findings)
    assert any("단기차입금 비중이 높아 차환 리스크" in item for item in agent.findings)


def test_evidence_audit_agent_preserves_downside_but_notes_support() -> None:
    state: AgentState = {
        "source_feature_row": {
            "current_ratio": 2.10,
            "cash_ratio": 0.62,
            "debt_ratio": 2.80,
            "short_term_borrowings_share": 0.22,
            "cashflow_coverage_ratio": 5.40,
            "interest_coverage_ratio": 4.10,
            "ocf_to_total_liabilities": 0.14,
            "ocf_to_total_borrowings": 0.27,
            "is_2y_consecutive_ocf_deficit": 0,
            "icr_under_1": 0,
        },
        "xgboost_result": {"prediction_label": "부적격"},
    }

    structured_output = _evidence_audit_agent(build_stage2_input_bundle(state))
    agent = structured_output.to_agent_output()

    assert structured_output.role == "evidence_audit"
    assert structured_output.debt_liquidity_cross_check
    assert agent.role == "evidence_audit"
    assert "부적격" in agent.summary
    assert "완화 신호" in agent.summary
    assert any("현금흐름 커버리지가 5배 이상" in item for item in agent.findings)
    assert any("영업현금흐름이 총부채 대비 0.1 이상" in item for item in agent.findings)


def test_evidence_audit_agent_scores_direct_external_risk_evidence() -> None:
    state: AgentState = {
        "company_id": "000250",
        "company_name": "삼천당제약(주)",
        "source_feature_row": {
            "stock_code": "000250",
            "current_ratio": 1.8,
            "cash_ratio": 0.4,
            "short_term_borrowings_share": 0.25,
            "cashflow_coverage_ratio": 4.2,
            "interest_coverage_ratio": 3.4,
        },
        "xgboost_result": {"prediction_label": "투자적격", "probability_speculative": 0.31},
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "삼천당제약(주) 횡령 혐의 발생",
                    "summary": "삼천당제약(주) 공시: 횡령 혐의 발생",
                    "reliability": "high",
                    "company_match": True,
                    "critical_terms": ["횡령"],
                    "critical_context_confirmed": True,
                    "veto_candidate": True,
                    "evidence_score": 0.91,
                    "evidence_quality": "high",
                }
            ],
            "direct_match_count": 1,
            "verified_item_count": 1,
            "veto_candidate_count": 1,
            "high_confidence_critical_count": 1,
            "critical_terms": ["횡령"],
            "has_critical_risk": True,
        },
    }

    structured_output = _evidence_audit_agent(build_stage2_input_bundle(state))
    agent = structured_output.to_agent_output()

    assert structured_output.evidence_strength == "strong"
    assert structured_output.critical_evidence_count == 1
    assert structured_output.watch_context_count == 0
    assert structured_output.hard_distress_detected is True
    assert structured_output.recommended_evidence_treatment == "critical_veto_review"
    assert "보수 검토" in structured_output.model_challenge
    assert "보류 또는 부적격 검토" in structured_output.audit_conclusion
    assert any("외부근거 위험" in item for item in agent.findings)
    assert any(
        "recommended_evidence_treatment=critical_veto_review" in item for item in agent.findings
    )


def test_committee_view_exposes_final_decision_fields() -> None:
    state: AgentState = {
        "company_id": "250",
        "company_name": "삼천당제약(주)",
        "market": "KOSDAQ",
        "source_feature_row": {
            "market": "KOSDAQ",
            "industry_macro_category": "manufacturing",
            "firm_size_group": "mid_sized",
            "current_ratio": 0.82,
            "cash_ratio": 0.11,
            "short_term_borrowings_share": 0.71,
        },
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.31,
            "top_drivers": [("current_ratio", 0.22)],
        },
        "rule_result": {
            "recommendation": "review",
            "confidence": 0.62,
            "blocking_flags": [],
        },
        "news_cache_snapshot": {"status": "not_implemented"},
    }

    result = run(state)
    committee_view = result["committee_view"]

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["veto_triggered"] is False
    assert "conflict_resolution" in committee_view
    assert committee_view["key_risk_factors"]
    assert committee_view["evidence_summary"][0]["source"] == "model_view"
    assert "deterministic runner" in result["audit"][0].summary


def test_agno_review_qa_skips_plain_investment_model_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_triplet_agents(**_kwargs: Any) -> Stage2LLMResponse:
        return Stage2LLMResponse(
            quant_credit=QuantCreditOutput(
                quant_summary="Triplet quant summary",
                model_rationale="Triplet model rationale",
                key_risk_factors=["유동성 경계"],
                mitigating_factors=["현금흐름 방어"],
                confidence=0.78,
            ),
            evidence_audit=EvidenceAuditOutput(
                evidence_summary="Triplet evidence summary",
                evidence_status="ready",
                evidence_reliability="기준일 이전 직접 근거를 우선했습니다.",
                evidence_strength="weak",
                model_challenge="치명 외부근거는 제한적입니다.",
                audit_conclusion="보류 사유는 재무 경계 신호 중심입니다.",
                debt_liquidity_cross_check=["유동성 추가 점검"],
                macro_industry_sensitivity=[],
                external_evidence_findings=[],
                confidence=0.7,
            ),
            chair_report=ChairReportOutput(
                report_summary="Triplet chair summary",
                model_preservation_note="model_view 보존",
                committee_scope_note="committee_view 보완",
                final_review_memo_seed="보류 의견으로 정리합니다.",
                confidence=0.74,
            ),
        )

    def fake_review_qa_agent_with_cache(**_kwargs: Any) -> tuple[ReviewQAOutput, dict[str, Any]]:
        raise AssertionError("ReviewQA should not run for a plain investment-model hold.")

    monkeypatch.setenv("CAS_ALLOW_LIVE_STAGE2_IN_TESTS", "1")
    monkeypatch.setenv("CAS_STAGE2_RUNNER", "agno")
    monkeypatch.setenv("CAS_STAGE2_REVIEW_QA_ENABLED", "1")
    monkeypatch.setenv("CAS_STAGE2_LLM_CACHE_ENABLED", "0")
    monkeypatch.setattr(
        stage2_runner_module,
        "_run_triplet_agents_with_agno",
        fake_triplet_agents,
    )
    monkeypatch.setattr(
        committee_node_module,
        "_run_review_qa_agent_with_cache",
        fake_review_qa_agent_with_cache,
    )
    state: AgentState = {
        "company_id": "KOSDAQ-000250-2023",
        "company_name": "삼천당제약(주)",
        "market": "KOSDAQ",
        "source_feature_row": {
            "market": "KOSDAQ",
            "current_ratio": 0.82,
            "cash_ratio": 0.11,
            "short_term_borrowings_share": 0.71,
        },
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.31,
        },
        "rule_result": {
            "recommendation": "review",
            "confidence": 0.62,
            "blocking_flags": [],
        },
        "news_cache_snapshot": {"status": "ready", "items": []},
    }

    result = committee_node_module.run(state)

    assert [agent.role for agent in result["agent_outputs"]] == [
        "quant_credit",
        "evidence_audit",
        "chair_report",
    ]
    assert result["committee_view"]["final_committee_label"] == "보류"
    assert result["agent_summary"]["synthesis"] == "Triplet chair summary"
    runtime = result["stage2_runtime_diagnostics"]
    assert runtime["review_qa_triggered"] is False
    assert runtime["review_qa_trigger_reasons"] == []
    assert "review_qa" not in runtime["agent_elapsed_seconds"]


def test_review_qa_does_not_trigger_for_eligible_ambiguous_evidence_only() -> None:
    state: AgentState = {
        "xgboost_result": {"prediction_label": "투자적격", "probability_speculative": 0.12},
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "주요사항보고서(전환사채권발행결정)",
                    "summary": "단일 medium 자금조달 공시입니다.",
                    "company_match": True,
                    "disclosure_severity": "caution",
                    "evidence_quality": "medium",
                    "evidence_score": 0.50,
                }
            ],
        },
    }
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
        "final_review_memo": "외부근거를 검토했으나 최종 위원회 판단은 적격입니다.",
    }

    reasons = committee_node_module._review_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert reasons == []


def test_review_qa_does_not_trigger_for_mitigation_hold_ambiguous_evidence_only() -> None:
    state: AgentState = {
        "source_feature_row": {
            "current_ratio": 2.4,
            "cash_ratio": 0.35,
            "cashflow_coverage_ratio": 1.7,
            "ocf_to_total_liabilities": 0.09,
            "ocf_to_sales": 0.05,
            "interest_coverage_ratio": 4.2,
            "icr_under_1": 0,
            "equity_ratio": 0.52,
            "debt_ratio": 0.9,
            "capital_impairment_ratio": 0.0,
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.34,
            "threshold": 0.32,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "주요사항보고서(전환사채권발행결정)",
                    "summary": "단일 medium 자금조달 공시입니다.",
                    "company_match": True,
                    "disclosure_severity": "caution",
                    "evidence_quality": "medium",
                    "evidence_score": 0.50,
                }
            ],
        },
    }
    committee_view = {
        "final_committee_label": "보류",
        "committee_decision_type": "mitigation_hold",
        "final_review_memo": "부적격 확정이 아닌 과민경고 완화 보류입니다.",
    }

    reasons = committee_node_module._review_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert reasons == []


def test_review_qa_triggers_for_risk_hold_with_watch_context_defense() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.30,
            "threshold": 0.32,
        },
        "source_feature_row": {
            "current_ratio": 1.8,
            "cash_ratio": 0.22,
            "cashflow_coverage_ratio": 2.1,
            "ocf_to_total_liabilities": 0.12,
            "ocf_to_sales": 0.08,
            "interest_coverage_ratio": 3.4,
            "icr_under_1": 0,
            "equity_ratio": 0.55,
            "debt_ratio": 0.8,
            "capital_impairment_ratio": 0.0,
            "total_borrowings_ratio": 0.28,
            "short_term_borrowings_share": 0.35,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "단일판매ㆍ공급계약해지",
                    "company_match": True,
                    "evidence_score": 0.68,
                    "provider_relevance": "caution",
                    "disclosure_severity": "caution",
                    "disclosure_event_class": "watch_context",
                    "disclosure_materiality": "watch_context",
                    "materiality_ratio": 0.0592,
                }
            ],
            "direct_match_count": 1,
            "verified_item_count": 1,
            "veto_candidate_count": 0,
            "high_confidence_critical_count": 0,
        },
    }
    committee_view = {
        "final_committee_label": "보류",
        "committee_decision_type": "risk_hold",
        "agent_disagreement_level": "medium",
        "agent_disagreement_reasons": ["chair_risk_without_critical_evidence"],
        "final_review_memo": "watch-context 공시와 경계 신호 때문에 위험 보류입니다.",
    }

    reasons = committee_node_module._review_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert reasons == ["risk_hold_without_critical_evidence"]


def test_review_qa_skips_low_disagreement_risk_hold_with_watch_context_defense() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.30,
            "threshold": 0.32,
        },
        "source_feature_row": {
            "current_ratio": 1.8,
            "cash_ratio": 0.22,
            "cashflow_coverage_ratio": 2.1,
            "ocf_to_total_liabilities": 0.12,
            "ocf_to_sales": 0.08,
            "interest_coverage_ratio": 3.4,
            "icr_under_1": 0,
            "equity_ratio": 0.55,
            "debt_ratio": 0.8,
            "capital_impairment_ratio": 0.0,
            "total_borrowings_ratio": 0.28,
            "short_term_borrowings_share": 0.35,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "단일판매ㆍ공급계약해지",
                    "company_match": True,
                    "evidence_score": 0.68,
                    "provider_relevance": "caution",
                    "disclosure_severity": "caution",
                    "disclosure_event_class": "watch_context",
                    "disclosure_materiality": "watch_context",
                    "materiality_ratio": 0.0592,
                }
            ],
            "direct_match_count": 1,
            "verified_item_count": 1,
            "veto_candidate_count": 0,
            "high_confidence_critical_count": 0,
        },
    }
    committee_view = {
        "final_committee_label": "보류",
        "committee_decision_type": "risk_hold",
        "agent_disagreement_level": "low",
        "agent_disagreement_reasons": [],
        "final_review_memo": "watch-context 공시와 경계 신호 때문에 위험 보류입니다.",
    }

    reasons = committee_node_module._review_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert reasons == []


def test_review_qa_skips_medium_disagreement_without_relevant_reason() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.30,
            "threshold": 0.32,
        },
        "source_feature_row": {
            "current_ratio": 1.8,
            "cash_ratio": 0.22,
            "cashflow_coverage_ratio": 2.1,
            "ocf_to_total_liabilities": 0.12,
            "ocf_to_sales": 0.08,
            "interest_coverage_ratio": 3.4,
            "icr_under_1": 0,
            "equity_ratio": 0.55,
            "debt_ratio": 0.8,
            "capital_impairment_ratio": 0.0,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "단일판매ㆍ공급계약해지",
                    "company_match": True,
                    "evidence_score": 0.68,
                    "provider_relevance": "caution",
                    "disclosure_severity": "caution",
                    "disclosure_event_class": "watch_context",
                    "disclosure_materiality": "watch_context",
                }
            ],
        },
    }
    committee_view = {
        "final_committee_label": "보류",
        "committee_decision_type": "risk_hold",
        "agent_disagreement_level": "medium",
        "agent_disagreement_reasons": ["agent_confidence_gap"],
        "final_review_memo": "watch-context 공시와 경계 신호 때문에 위험 보류입니다.",
    }

    reasons = committee_node_module._review_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert reasons == []


def test_review_qa_skips_high_model_risk_hold_without_actionable_downgrade_path() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.62,
            "threshold": 0.31,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [],
            "direct_match_count": 0,
            "verified_item_count": 0,
            "veto_candidate_count": 0,
            "high_confidence_critical_count": 0,
        },
    }
    committee_view = {
        "final_committee_label": "보류",
        "committee_decision_type": "risk_hold",
        "agent_disagreement_score": 0.65,
        "final_review_memo": "위험 보류입니다.",
    }

    reasons = committee_node_module._review_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert reasons == []


def test_review_qa_triggers_for_high_investment_model_risk_hold_with_watch_context() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.30,
            "threshold": 0.32,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "최대주주변경",
                    "company_match": True,
                    "evidence_score": 0.62,
                    "provider_relevance": "caution",
                    "disclosure_severity": "caution",
                    "disclosure_event_class": "watch_context",
                    "disclosure_materiality": "watch_context",
                }
            ],
            "direct_match_count": 1,
            "verified_item_count": 1,
            "veto_candidate_count": 0,
            "high_confidence_critical_count": 0,
        },
    }
    committee_view = {
        "final_committee_label": "보류",
        "committee_decision_type": "risk_hold",
        "agent_disagreement_score": 0.65,
        "final_review_memo": "위험 보류입니다.",
    }

    reasons = committee_node_module._review_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert reasons == [
        "agent_disagreement_high_without_critical_evidence",
        "risk_hold_without_critical_evidence",
    ]


def test_review_qa_advisory_downgrades_risk_hold_subtype_only() -> None:
    committee_view = {
        "final_committee_label": "보류",
        "committee_decision_type": "risk_hold",
        "committee_decision_type_label": "위험 보류",
        "committee_risk_signal": True,
        "veto_triggered": False,
        "hidden_tail_risk_flag": False,
        "conflict_resolution": "반복 자금조달 공시 때문에 위험 보류로 정리했습니다.",
        "final_review_memo": "최종 위원회 판단은 보류입니다.",
        "mitigating_factors": ["현금흐름 방어축은 확인됩니다."],
        "decision_trace": [],
    }
    review_qa_output = ReviewQAOutput(
        qa_summary="위험 보류 강도가 다소 과합니다.",
        trigger_reasons=[
            "risk_hold_without_critical_evidence",
        ],
        label_memo_consistency="최종 라벨과 메모가 충돌하지 않습니다.",
        risk_hold_assessment="overstated",
        evidence_cutoff_check="기준일 이전 근거만 사용했습니다.",
        overhold_guardrail_assessment="경계등급 보류가 더 적절합니다.",
        recommended_action="downgrade_risk_hold_to_boundary_hold",
        confidence=0.72,
    )
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_review_qa_advisory(
        committee_view=committee_view,
        review_qa_output=review_qa_output,
        runtime_diagnostics=runtime,
    )

    assert adjusted["final_committee_label"] == "보류"
    assert adjusted["committee_decision_type"] == "boundary_hold"
    assert adjusted["committee_decision_type_label"] == "경계등급 보류"
    assert adjusted["committee_risk_signal"] is True
    assert "ReviewQA" in adjusted["conflict_resolution"]
    assert adjusted["decision_trace"][-1]["gate"] == "review_qa_subtype_adjustment"
    assert runtime["review_qa_advisory_applied"] is True
    assert runtime["review_qa_adjusted_decision_type"] == "boundary_hold"
    assert runtime["review_qa_advisory_apply_reason"] == "review_qa_overstated_risk_hold"


def test_review_qa_advisory_does_not_downgrade_hidden_tail_risk() -> None:
    committee_view = {
        "final_committee_label": "보류",
        "committee_decision_type": "risk_hold",
        "committee_decision_type_label": "위험 보류",
        "committee_risk_signal": True,
        "veto_triggered": False,
        "hidden_tail_risk_flag": True,
        "conflict_resolution": "숨은 꼬리위험 때문에 위험 보류로 정리했습니다.",
        "final_review_memo": "최종 위원회 판단은 보류입니다.",
        "decision_trace": [],
    }
    review_qa_output = ReviewQAOutput(
        qa_summary="위험 보류 강도가 다소 과합니다.",
        trigger_reasons=["risk_hold_without_critical_evidence"],
        label_memo_consistency="충돌 없음",
        risk_hold_assessment="overstated",
        evidence_cutoff_check="기준일 이전 근거만 사용했습니다.",
        overhold_guardrail_assessment="경계등급 보류 후보입니다.",
        recommended_action="downgrade_risk_hold_to_boundary_hold",
        confidence=0.72,
    )
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_review_qa_advisory(
        committee_view=committee_view,
        review_qa_output=review_qa_output,
        runtime_diagnostics=runtime,
    )

    assert adjusted["committee_decision_type"] == "risk_hold"
    assert adjusted["committee_risk_signal"] is True
    assert runtime["review_qa_advisory_applied"] is False


def test_review_qa_advisory_downgrades_watch_context_only_risk_hold() -> None:
    committee_view = {
        "final_committee_label": "보류",
        "committee_decision_type": "risk_hold",
        "committee_decision_type_label": "위험 보류",
        "committee_risk_signal": True,
        "veto_triggered": False,
        "hidden_tail_risk_flag": False,
        "conflict_resolution": "공시 제목상 위험 가능성이 있어 위험 보류로 정리했습니다.",
        "final_review_memo": "최종 위원회 판단은 보류입니다.",
        "mitigating_factors": [],
        "decision_trace": [],
    }
    review_qa_output = ReviewQAOutput(
        qa_summary="공시는 watch_context 수준이라 위험 보류보다는 경계 보류가 맞습니다.",
        trigger_reasons=[
            "risk_hold_without_critical_evidence",
        ],
        label_memo_consistency="충돌 없음",
        risk_hold_assessment="adequate",
        evidence_cutoff_check="기준일 이전 근거만 사용했습니다.",
        overhold_guardrail_assessment="치명 외부근거가 없어 guardrail 적용 가능합니다.",
        recommended_action="downgrade_risk_hold_to_boundary_hold",
        confidence=0.5,
    )
    news_cache = {
        "status": "ready",
        "items": [
            {
                "source": "opendart",
                "title": "단일판매ㆍ공급계약해지",
                "company_match": True,
                "evidence_score": 0.68,
                "provider_relevance": "caution",
                "disclosure_severity": "caution",
                "disclosure_event_class": "watch_context",
                "disclosure_materiality": "watch_context",
                "materiality_ratio": 0.0592,
            }
        ],
        "direct_match_count": 1,
        "verified_item_count": 1,
        "veto_candidate_count": 0,
        "high_confidence_critical_count": 0,
    }
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_review_qa_advisory(
        committee_view=committee_view,
        review_qa_output=review_qa_output,
        news_cache_snapshot=news_cache,
        runtime_diagnostics=runtime,
    )

    assert adjusted["committee_decision_type"] == "boundary_hold"
    assert adjusted["committee_risk_signal"] is True
    assert runtime["review_qa_advisory_applied"] is True
    assert runtime["review_qa_advisory_apply_reason"] == "watch_context_only_risk_hold_override"


def test_review_qa_advisory_keeps_substantive_external_risk_hold() -> None:
    committee_view = {
        "final_committee_label": "보류",
        "committee_decision_type": "risk_hold",
        "committee_decision_type_label": "위험 보류",
        "committee_risk_signal": True,
        "veto_triggered": False,
        "hidden_tail_risk_flag": False,
        "conflict_resolution": "종속회사 영업정지 공시 때문에 위험 보류로 정리했습니다.",
        "final_review_memo": "최종 위원회 판단은 보류입니다.",
        "decision_trace": [],
    }
    review_qa_output = ReviewQAOutput(
        qa_summary="위험 보류를 낮출 수 있는지 확인했습니다.",
        trigger_reasons=["risk_hold_without_critical_evidence"],
        label_memo_consistency="충돌 없음",
        risk_hold_assessment="adequate",
        evidence_cutoff_check="기준일 이전 근거만 사용했습니다.",
        overhold_guardrail_assessment="중요도 높은 공시가 있어 guardrail 적용이 어렵습니다.",
        recommended_action="downgrade_risk_hold_to_boundary_hold",
        confidence=0.9,
    )
    news_cache = {
        "status": "ready",
        "items": [
            {
                "source": "opendart",
                "title": "영업정지(종속회사의주요경영사항)",
                "company_match": True,
                "evidence_score": 1.0,
                "provider_relevance": "risk",
                "disclosure_severity": "adverse",
                "disclosure_event_class": "substantive_adverse",
                "disclosure_materiality": "substantive_adverse",
                "materiality_ratio": 0.1137,
            }
        ],
        "direct_match_count": 1,
        "verified_item_count": 1,
        "veto_candidate_count": 0,
        "high_confidence_critical_count": 0,
    }
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_review_qa_advisory(
        committee_view=committee_view,
        review_qa_output=review_qa_output,
        news_cache_snapshot=news_cache,
        runtime_diagnostics=runtime,
    )

    assert adjusted["committee_decision_type"] == "risk_hold"
    assert adjusted["committee_risk_signal"] is True
    assert runtime["review_qa_advisory_applied"] is False
    assert runtime["review_qa_advisory_apply_reason"] == ""


def test_review_qa_triggers_for_reject_with_watch_context_only_evidence() -> None:
    state: AgentState = {
        "source_feature_row": {
            "current_ratio": 4.1,
            "cash_ratio": 1.9,
            "equity_ratio": 0.60,
            "debt_ratio": 0.66,
            "capital_impairment_ratio": 0.0,
            "total_borrowings_ratio": 0.16,
            "short_term_borrowings_share": 0.01,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
            "cashflow_coverage_ratio": -20.0,
            "ocf_to_total_liabilities": -0.07,
            "ocf_to_sales": -0.03,
            "icr_under_1": 1,
            "interest_coverage_ratio": -12.1,
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.91,
            "threshold": 0.25,
        },
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.91,
            "threshold": 0.25,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "주주명부폐쇄기간또는기준일설정",
                    "company_match": True,
                    "evidence_score": 0.62,
                    "provider_relevance": "routine",
                    "disclosure_severity": "routine",
                    "disclosure_event_class": "routine_context",
                    "disclosure_materiality": "routine_context",
                }
            ],
            "direct_match_count": 1,
            "verified_item_count": 1,
            "veto_candidate_count": 0,
            "high_confidence_critical_count": 0,
        },
    }
    committee_view = {
        "final_committee_label": "부적격",
        "committee_decision_type": "reject",
        "committee_risk_signal": True,
        "agent_disagreement_level": "medium",
        "agent_disagreement_reasons": ["chair_reject_without_critical_evidence"],
        "final_review_memo": "모델 고확률과 재무 약점으로 부적격입니다.",
    }

    reasons = committee_node_module._review_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert "reject_without_critical_evidence" in reasons


@pytest.mark.parametrize(
    "recommended_action",
    [
        "downgrade_reject_to_boundary_hold",
        "downgrade_risk_hold_to_boundary_hold",
        "keep_committee_view",
    ],
)
def test_review_qa_advisory_downgrades_reject_with_watch_context_only_evidence(
    recommended_action: str,
) -> None:
    committee_view = {
        "final_committee_label": "부적격",
        "committee_decision_type": "reject",
        "committee_decision_type_label": "부적격",
        "committee_risk_signal": True,
        "veto_triggered": False,
        "hidden_tail_risk_flag": False,
        "conflict_resolution": "모델 원판단과 위원회 라벨이 일치합니다.",
        "final_review_memo": "모델 고확률과 재무 약점으로 부적격입니다.",
        "mitigating_factors": [],
        "decision_trace": [],
    }
    state: AgentState = {
        "source_feature_row": {
            "current_ratio": 4.1,
            "cash_ratio": 1.9,
            "equity_ratio": 0.60,
            "debt_ratio": 0.66,
            "capital_impairment_ratio": -5.2,
            "total_borrowings_ratio": 0.16,
            "short_term_borrowings_share": 0.01,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
            "cashflow_coverage_ratio": -20.0,
            "ocf_to_total_liabilities": -0.07,
            "ocf_to_sales": -0.03,
            "icr_under_1": 1,
            "interest_coverage_ratio": -12.1,
        }
    }
    review_qa_output = ReviewQAOutput(
        qa_summary="치명 외부근거가 약해 부적격 확정보다는 보류가 적절합니다.",
        trigger_reasons=["reject_without_critical_evidence"],
        label_memo_consistency="충돌 없음",
        risk_hold_assessment="not_applicable",
        evidence_cutoff_check="기준일 이전 근거만 사용했습니다.",
        overhold_guardrail_assessment="watch-context 공시만 확인됩니다.",
        recommended_action=recommended_action,  # type: ignore[arg-type]
        confidence=0.74,
    )
    news_cache = {
        "status": "ready",
        "items": [
            {
                "source": "opendart",
                "title": "최대주주변경",
                "company_match": True,
                "evidence_score": 0.66,
                "provider_relevance": "caution",
                "disclosure_severity": "caution",
                "disclosure_event_class": "watch_context",
                "disclosure_materiality": "watch_context",
            }
        ],
        "direct_match_count": 1,
        "verified_item_count": 1,
        "veto_candidate_count": 0,
        "high_confidence_critical_count": 0,
    }
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_review_qa_advisory(
        committee_view=committee_view,
        review_qa_output=review_qa_output,
        bundle=build_stage2_input_bundle(state),
        news_cache_snapshot=news_cache,
        runtime_diagnostics=runtime,
    )

    assert adjusted["final_committee_label"] == "보류"
    assert adjusted["committee_decision_type"] == "boundary_hold"
    assert adjusted["committee_risk_signal"] is True
    assert adjusted["decision_trace"][-1]["gate"] == "review_qa_reject_adjustment"
    assert runtime["review_qa_advisory_applied"] is True
    assert runtime["review_qa_advisory_apply_reason"] in {
        "review_qa_reject_watch_context_only_override",
        "review_qa_reject_defensive_boundary_override",
    }


def test_review_qa_advisory_keeps_reject_with_substantive_external_risk() -> None:
    committee_view = {
        "final_committee_label": "부적격",
        "committee_decision_type": "reject",
        "committee_decision_type_label": "부적격",
        "committee_risk_signal": True,
        "veto_triggered": False,
        "hidden_tail_risk_flag": False,
        "conflict_resolution": "실질 adverse 공시 때문에 부적격입니다.",
        "final_review_memo": "실질 adverse 공시가 확인됩니다.",
        "decision_trace": [],
    }
    review_qa_output = ReviewQAOutput(
        qa_summary="낮출 수 있는지 점검했습니다.",
        trigger_reasons=["reject_without_critical_evidence"],
        label_memo_consistency="충돌 없음",
        risk_hold_assessment="not_applicable",
        evidence_cutoff_check="기준일 이전 근거만 사용했습니다.",
        overhold_guardrail_assessment="중요도 높은 공시가 있습니다.",
        recommended_action="downgrade_reject_to_boundary_hold",
        confidence=0.9,
    )
    news_cache = {
        "status": "ready",
        "items": [
            {
                "source": "opendart",
                "title": "단일판매ㆍ공급계약해지",
                "company_match": True,
                "evidence_score": 0.9,
                "provider_relevance": "risk",
                "disclosure_severity": "adverse",
                "disclosure_event_class": "substantive_adverse",
                "disclosure_materiality": "substantive_adverse",
                "materiality_ratio": 0.18,
            }
        ],
        "direct_match_count": 1,
        "verified_item_count": 1,
        "veto_candidate_count": 0,
        "high_confidence_critical_count": 0,
    }
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_review_qa_advisory(
        committee_view=committee_view,
        review_qa_output=review_qa_output,
        bundle=build_stage2_input_bundle({"source_feature_row": {}}),
        news_cache_snapshot=news_cache,
        runtime_diagnostics=runtime,
    )

    assert adjusted["final_committee_label"] == "부적격"
    assert adjusted["committee_decision_type"] == "reject"
    assert runtime["review_qa_advisory_applied"] is False
    assert runtime["review_qa_advisory_apply_reason"] == ""


def test_review_qa_advisory_keeps_reject_without_balance_sheet_defense() -> None:
    committee_view = {
        "final_committee_label": "부적격",
        "committee_decision_type": "reject",
        "committee_decision_type_label": "부적격",
        "committee_risk_signal": True,
        "veto_triggered": False,
        "hidden_tail_risk_flag": False,
        "conflict_resolution": "모델 원판단과 위원회 라벨이 일치합니다.",
        "final_review_memo": "모델 고확률과 재무 약점으로 부적격입니다.",
        "decision_trace": [],
    }
    review_qa_output = ReviewQAOutput(
        qa_summary="치명 외부근거는 약하지만 재무 방어축도 약합니다.",
        trigger_reasons=["reject_without_critical_evidence"],
        label_memo_consistency="충돌 없음",
        risk_hold_assessment="not_applicable",
        evidence_cutoff_check="기준일 이전 근거만 사용했습니다.",
        overhold_guardrail_assessment="재무 방어축이 부족합니다.",
        recommended_action="keep_committee_view",
        confidence=0.8,
    )
    state: AgentState = {
        "source_feature_row": {
            "current_ratio": 0.5,
            "cash_ratio": 0.02,
            "equity_ratio": 0.10,
            "debt_ratio": 7.0,
            "capital_impairment_ratio": 0.7,
            "total_borrowings_ratio": 0.82,
            "short_term_borrowings_share": 0.96,
            "is_2y_consecutive_operating_loss": 1,
            "is_2y_consecutive_ocf_deficit": 1,
            "cashflow_coverage_ratio": -2.0,
            "interest_coverage_ratio": -5.0,
            "icr_under_1": 1,
        }
    }
    news_cache = {
        "status": "ready",
        "items": [
            {
                "source": "opendart",
                "title": "주주명부폐쇄기간또는기준일설정",
                "company_match": True,
                "evidence_score": 0.66,
                "provider_relevance": "routine",
                "disclosure_severity": "routine",
                "disclosure_event_class": "routine_context",
                "disclosure_materiality": "routine_context",
            }
        ],
        "direct_match_count": 1,
        "verified_item_count": 1,
        "veto_candidate_count": 0,
        "high_confidence_critical_count": 0,
    }
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_review_qa_advisory(
        committee_view=committee_view,
        review_qa_output=review_qa_output,
        bundle=build_stage2_input_bundle(state),
        news_cache_snapshot=news_cache,
        runtime_diagnostics=runtime,
    )

    assert adjusted["final_committee_label"] == "부적격"
    assert adjusted["committee_decision_type"] == "reject"
    assert runtime["review_qa_advisory_applied"] is False


def test_risk_recall_qa_triggers_for_eligible_near_threshold_weak_financials() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.39,
            "threshold": 0.45,
        },
        "source_feature_row": {
            "current_ratio": 0.82,
            "cash_ratio": 0.08,
            "cashflow_coverage_ratio": -0.2,
            "interest_coverage_ratio": 0.7,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "주요사항보고서(전환사채권발행결정)",
                    "company_match": True,
                    "evidence_score": 0.62,
                    "disclosure_severity": "caution",
                    "disclosure_materiality": "watch_context",
                }
            ],
        },
    }
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
    }

    reasons = committee_node_module._risk_recall_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert "eligible_near_threshold" in reasons
    assert "eligible_near_threshold_with_weak_financials" in reasons
    assert "eligible_with_recall_watch_evidence" in reasons


def test_risk_recall_qa_does_not_trigger_for_defensive_eligible_case() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.12,
            "threshold": 0.45,
        },
        "source_feature_row": {
            "current_ratio": 2.4,
            "cash_ratio": 0.45,
            "cashflow_coverage_ratio": 2.1,
            "interest_coverage_ratio": 7.5,
            "debt_ratio": 0.35,
        },
        "news_cache_snapshot": {"status": "ready", "items": []},
    }
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
    }

    reasons = committee_node_module._risk_recall_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert reasons == []


def test_risk_recall_qa_does_not_trigger_for_routine_audit_watch_only() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.39,
            "threshold": 0.45,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "감사보고서제출",
                    "company_match": True,
                    "evidence_score": 0.75,
                    "provider_relevance": "risk",
                    "disclosure_severity": "adverse",
                    "disclosure_event_class": "routine_context",
                    "disclosure_materiality": "routine_context",
                }
            ],
        },
    }
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
    }

    reasons = committee_node_module._risk_recall_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert reasons == []


def test_risk_recall_qa_does_not_trigger_for_near_threshold_boundary_only() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.39,
            "threshold": 0.45,
        },
        "source_feature_row": {
            "current_ratio": 2.1,
            "cash_ratio": 0.35,
            "cashflow_coverage_ratio": 1.5,
            "interest_coverage_ratio": 5.0,
            "debt_ratio": 0.45,
        },
        "prior_rating_reference": {
            "prior_credit_rating": "BBB-",
            "prior_rating_boundary_group": "near_investment_BBB_plus_to_BBB_minus",
        },
        "news_cache_snapshot": {"status": "ready", "items": []},
    }
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
    }

    reasons = committee_node_module._risk_recall_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert reasons == []


def test_risk_recall_qa_triggers_for_prior_hard_distress_with_weak_financials() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.39,
            "threshold": 0.45,
        },
        "source_feature_row": {
            "current_ratio": 0.82,
            "cash_ratio": 0.08,
            "cashflow_coverage_ratio": 1.5,
            "interest_coverage_ratio": 5.0,
            "debt_ratio": 0.45,
        },
        "prior_rating_reference": {
            "has_prior_rating": True,
            "prior_credit_rating": "CCC",
            "prior_credit_rating_rank": 18,
            "prior_rating_date": "2022-11-30",
        },
        "news_cache_snapshot": {"status": "ready", "items": []},
    }
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
    }

    reasons = committee_node_module._risk_recall_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert "eligible_prior_hard_distress_context" in reasons


def test_risk_recall_qa_triggers_for_substantive_evidence_without_near_threshold() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.12,
            "threshold": 0.45,
        },
        "source_feature_row": {
            "current_ratio": 2.0,
            "cash_ratio": 0.30,
            "cashflow_coverage_ratio": 1.0,
            "interest_coverage_ratio": 4.0,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "영업정지(종속회사의주요경영사항)",
                    "company_match": True,
                    "evidence_score": 1.0,
                    "disclosure_materiality": "substantive_adverse",
                    "materiality_ratio": 0.1137,
                }
            ],
        },
    }
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
    }

    reasons = committee_node_module._risk_recall_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert "eligible_with_substantive_evidence" in reasons


def test_risk_recall_qa_treats_materiality_ratio_as_substantive() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.20,
            "threshold": 0.45,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "영업정지(종속회사의주요경영사항)",
                    "company_match": True,
                    "evidence_score": 1.0,
                    "provider_relevance": "risk",
                    "disclosure_severity": "adverse",
                    "disclosure_event_class": "watch_context",
                    "disclosure_materiality": "watch_context",
                    "materiality_ratio": 0.1137,
                }
            ],
        },
    }
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
    }

    reasons = committee_node_module._risk_recall_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert "eligible_with_substantive_evidence" in reasons


def test_risk_recall_qa_does_not_treat_uncorroborated_financing_as_substantive() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.20,
            "threshold": 0.45,
        },
        "source_feature_row": {
            "current_ratio": 2.1,
            "cash_ratio": 0.35,
            "cashflow_coverage_ratio": 1.8,
            "ocf_to_total_liabilities": 0.09,
            "ocf_to_sales": 0.04,
            "interest_coverage_ratio": 5.0,
            "equity_ratio": 0.55,
            "debt_ratio": 0.8,
            "total_borrowings_ratio": 0.22,
            "capital_impairment_ratio": 0.0,
            "net_margin": 0.05,
            "icr_under_1": 0,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "주요사항보고서(유상증자결정)",
                    "company_match": True,
                    "evidence_score": 1.0,
                    "provider_relevance": "risk",
                    "disclosure_severity": "adverse",
                    "disclosure_event_class": "material_financing",
                    "disclosure_materiality": "substantive_adverse",
                    "materiality_ratio": 0.20,
                    "critical_context_confirmed": False,
                    "veto_candidate": False,
                }
            ],
        },
    }
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
    }

    reasons = committee_node_module._risk_recall_qa_trigger_reasons(
        bundle=build_stage2_input_bundle(state),
        committee_view=committee_view,
    )

    assert "eligible_with_substantive_evidence" not in reasons


def test_risk_recall_qa_advisory_escalates_eligible_to_boundary_hold() -> None:
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
        "committee_decision_type_label": "적격",
        "committee_risk_signal": False,
        "veto_triggered": False,
        "hidden_tail_risk_flag": False,
        "conflict_resolution": "정량 모델과 외부근거를 종합해 적격으로 정리했습니다.",
        "final_review_memo": "최종 위원회 판단은 적격입니다.",
        "key_risk_factors": [],
        "decision_trace": [],
    }
    output = RiskRecallQAOutput(
        qa_summary="기준선 근처 적격이라 경계 보류가 더 안전합니다.",
        trigger_reasons=[
            "eligible_near_threshold",
            "eligible_near_threshold_with_weak_financials",
        ],
        eligible_safety_assessment="needs_boundary_review",
        financial_resilience_check="유동성/현금흐름 방어축이 약합니다.",
        evidence_recall_check="치명 외부근거는 없지만 watch_context 공시가 있습니다.",
        rating_boundary_check="기준선 근처입니다.",
        recommended_action="escalate_eligible_to_boundary_hold",
        confidence=0.66,
    )
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.39,
            "threshold": 0.45,
        },
        "source_feature_row": {
            "current_ratio": 0.82,
            "cash_ratio": 0.08,
            "cashflow_coverage_ratio": -0.2,
        },
    }
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_risk_recall_qa_advisory(
        committee_view=committee_view,
        risk_recall_qa_output=output,
        bundle=build_stage2_input_bundle(state),
        runtime_diagnostics=runtime,
    )

    assert adjusted["final_committee_label"] == "보류"
    assert adjusted["committee_decision_type"] == "boundary_hold"
    assert adjusted["committee_risk_signal"] is True
    assert adjusted["decision_trace"][-1]["gate"] == "risk_recall_qa_escalation"
    assert "최종 위원회 판단은 적격입니다" not in adjusted["final_review_memo"]
    assert "초기 위원회 판단은 적격이었습니다" in adjusted["final_review_memo"]
    assert "최종 표시 라벨을 보류로 올립니다" in adjusted["final_review_memo"]
    assert runtime["risk_recall_qa_advisory_applied"] is True
    assert runtime["risk_recall_qa_advisory_apply_reason"] == "risk_recall_boundary_safety_review"


def test_risk_recall_qa_advisory_blocks_low_quality_news_only_boundary_escalation() -> None:
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
        "committee_decision_type_label": "적격",
        "committee_risk_signal": False,
        "conflict_resolution": "정량 모델과 외부근거를 종합해 적격으로 정리했습니다.",
        "final_review_memo": "최종 위원회 판단은 적격입니다.",
        "key_risk_factors": [],
        "decision_trace": [],
    }
    output = RiskRecallQAOutput(
        qa_summary="저품질 뉴스 스니펫에 hard distress 단어가 있어 보류를 권고했습니다.",
        trigger_reasons=["eligible_with_recall_watch_evidence"],
        eligible_safety_assessment="needs_boundary_review",
        financial_resilience_check="뚜렷한 복수 재무취약성은 없습니다.",
        evidence_recall_check="뉴스 요약에 횡령 단어가 있습니다.",
        rating_boundary_check="경계등급 맥락은 제한적입니다.",
        recommended_action="escalate_eligible_to_boundary_hold",
        confidence=0.78,
    )
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.18,
            "threshold": 0.45,
        },
        "source_feature_row": {
            "current_ratio": 1.8,
            "cash_ratio": 0.3,
            "cashflow_coverage_ratio": 1.2,
            "interest_coverage_ratio": 4.0,
            "debt_ratio": 0.8,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "naver_news",
                    "title": "공시종합",
                    "summary": "타사 직원 횡령혐의 공시와 함께 대상 기업 이름이 목록에 언급됐습니다.",
                    "company_match": True,
                    "evidence_quality": "low",
                    "evidence_score": 0.54,
                    "provider_relevance": "unknown",
                    "disclosure_severity": "veto",
                    "critical_terms": ["횡령"],
                    "veto_candidate": False,
                    "critical_context_confirmed": False,
                }
            ],
        },
    }
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_risk_recall_qa_advisory(
        committee_view=committee_view,
        risk_recall_qa_output=output,
        bundle=build_stage2_input_bundle(state),
        runtime_diagnostics=runtime,
    )

    assert adjusted["final_committee_label"] == "적격"
    assert adjusted["committee_decision_type"] == "eligible"
    assert runtime["risk_recall_qa_advisory_applied"] is False


def test_risk_recall_qa_advisory_blocks_low_quality_news_only_risk_escalation() -> None:
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
        "committee_decision_type_label": "적격",
        "committee_risk_signal": False,
        "conflict_resolution": "정량 모델과 외부근거를 종합해 적격으로 정리했습니다.",
        "final_review_memo": "최종 위원회 판단은 적격입니다.",
        "key_risk_factors": [],
        "decision_trace": [],
    }
    output = RiskRecallQAOutput(
        qa_summary="저품질 뉴스 스니펫을 중대 외부근거로 보아 위험 보류를 권고했습니다.",
        trigger_reasons=["eligible_with_substantive_evidence"],
        eligible_safety_assessment="material_missed_risk",
        financial_resilience_check="복수 재무취약성은 확인되지 않습니다.",
        evidence_recall_check="뉴스 요약에 배임 단어가 있습니다.",
        rating_boundary_check="경계등급 맥락은 제한적입니다.",
        recommended_action="escalate_eligible_to_risk_hold",
        confidence=0.82,
    )
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.18,
            "threshold": 0.45,
        },
        "source_feature_row": {
            "current_ratio": 1.8,
            "cash_ratio": 0.3,
            "cashflow_coverage_ratio": 1.2,
            "interest_coverage_ratio": 4.0,
            "debt_ratio": 0.8,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "naver_news",
                    "title": "업계 인터뷰",
                    "summary": "업계 전반의 배임횡령 이슈를 설명하면서 대상 기업 이름이 언급됐습니다.",
                    "company_match": True,
                    "evidence_quality": "low",
                    "evidence_score": 0.54,
                    "provider_relevance": "unknown",
                    "disclosure_severity": "veto",
                    "critical_terms": ["배임", "횡령"],
                    "veto_candidate": False,
                    "critical_context_confirmed": False,
                }
            ],
        },
    }
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_risk_recall_qa_advisory(
        committee_view=committee_view,
        risk_recall_qa_output=output,
        bundle=build_stage2_input_bundle(state),
        runtime_diagnostics=runtime,
    )

    assert adjusted["final_committee_label"] == "적격"
    assert adjusted["committee_decision_type"] == "eligible"
    assert runtime["risk_recall_qa_advisory_applied"] is False


def test_deterministic_risk_recall_guardrail_escalates_near_threshold_two_weak_axes() -> None:
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
        "committee_decision_type_label": "적격",
        "committee_risk_signal": False,
        "conflict_resolution": "정량 모델과 외부근거를 종합해 적격으로 정리했습니다.",
        "final_review_memo": "최종 위원회 판단은 적격입니다.",
        "key_risk_factors": [],
        "decision_trace": [],
    }
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.39,
            "threshold": 0.45,
        },
        "source_feature_row": {
            "current_ratio": 0.82,
            "cash_ratio": 0.08,
            "cashflow_coverage_ratio": 1.5,
            "interest_coverage_ratio": 5.0,
            "debt_ratio": 0.45,
            "total_borrowings_ratio": 0.20,
        },
        "news_cache_snapshot": {"status": "disabled", "items": []},
    }
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_deterministic_risk_recall_guardrail(
        committee_view=committee_view,
        bundle=build_stage2_input_bundle(state),
        runtime_diagnostics=runtime,
    )

    assert adjusted["final_committee_label"] == "보류"
    assert adjusted["committee_decision_type"] == "risk_hold"
    assert adjusted["committee_decision_type_label"] == "위험 보류"
    assert adjusted["committee_risk_signal"] is True
    assert adjusted["risk_hold_reason_tags"] == ["financial_stress_hold"]
    assert adjusted["decision_trace"][-2]["gate"] == "risk_recall_guardrail_escalation"
    assert "RiskRecallQA" not in adjusted["risk_hold_reason_summary"]
    assert runtime["risk_recall_guardrail_applied"] is True
    assert runtime["risk_recall_guardrail_apply_reason"] == "risk_recall_severe_financial_weakness"
    assert runtime["risk_recall_guardrail_weak_axes"] == [
        "low_current_ratio",
        "low_cash_ratio",
    ]


def test_deterministic_risk_recall_guardrail_keeps_far_threshold_three_axes_eligible() -> None:
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
        "committee_decision_type_label": "적격",
        "committee_risk_signal": False,
        "conflict_resolution": "정량 모델과 외부근거를 종합해 적격으로 정리했습니다.",
        "final_review_memo": "최종 위원회 판단은 적격입니다.",
        "key_risk_factors": [],
        "decision_trace": [],
    }
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.20,
            "threshold": 0.45,
        },
        "source_feature_row": {
            "current_ratio": 0.82,
            "cash_ratio": 0.08,
            "cashflow_coverage_ratio": -0.2,
            "interest_coverage_ratio": 5.0,
            "debt_ratio": 0.45,
            "total_borrowings_ratio": 0.20,
        },
        "news_cache_snapshot": {"status": "disabled", "items": []},
    }
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_deterministic_risk_recall_guardrail(
        committee_view=committee_view,
        bundle=build_stage2_input_bundle(state),
        runtime_diagnostics=runtime,
    )

    assert adjusted["final_committee_label"] == "적격"
    assert adjusted["committee_decision_type"] == "eligible"
    assert runtime["risk_recall_guardrail_applied"] is False
    assert runtime["risk_recall_guardrail_apply_reason"] == ""


def test_risk_recall_qa_advisory_escalates_prior_hard_distress_to_risk_hold() -> None:
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
        "committee_decision_type_label": "적격",
        "committee_risk_signal": False,
        "veto_triggered": False,
        "hidden_tail_risk_flag": False,
        "conflict_resolution": "정량 모델과 외부근거를 종합해 적격으로 정리했습니다.",
        "final_review_memo": "최종 위원회 판단은 적격입니다.",
        "key_risk_factors": [],
        "decision_trace": [],
    }
    output = RiskRecallQAOutput(
        qa_summary="과거 CCC 등급과 현재 재무 약점이 함께 남아 있습니다.",
        trigger_reasons=["eligible_prior_hard_distress_context"],
        eligible_safety_assessment="material_missed_risk",
        financial_resilience_check="유동성 방어축이 약합니다.",
        evidence_recall_check="치명 외부근거는 제한적입니다.",
        rating_boundary_check="기준일 이전 CCC 등급 컨텍스트가 있습니다.",
        recommended_action="escalate_eligible_to_risk_hold",
        confidence=0.78,
    )
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.39,
            "threshold": 0.45,
        },
        "source_feature_row": {
            "current_ratio": 0.82,
            "cash_ratio": 0.08,
            "cashflow_coverage_ratio": 1.5,
            "interest_coverage_ratio": 5.0,
            "debt_ratio": 0.45,
        },
        "prior_rating_reference": {
            "has_prior_rating": True,
            "prior_credit_rating": "CCC",
            "prior_credit_rating_rank": 18,
            "prior_rating_date": "2022-11-30",
        },
        "news_cache_snapshot": {"status": "ready", "items": []},
    }
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_risk_recall_qa_advisory(
        committee_view=committee_view,
        risk_recall_qa_output=output,
        bundle=build_stage2_input_bundle(state),
        runtime_diagnostics=runtime,
    )

    assert adjusted["final_committee_label"] == "보류"
    assert adjusted["committee_decision_type"] == "risk_hold"
    assert adjusted["risk_hold_reason_tags"] == ["prior_hard_distress_hold"]
    assert runtime["risk_recall_qa_advisory_applied"] is True
    assert (
        runtime["risk_recall_qa_advisory_apply_reason"] == "risk_recall_prior_hard_distress_context"
    )


def test_risk_recall_qa_advisory_escalates_substantive_external_risk_to_risk_hold() -> None:
    committee_view = {
        "final_committee_label": "적격",
        "committee_decision_type": "eligible",
        "committee_decision_type_label": "적격",
        "committee_risk_signal": False,
        "veto_triggered": False,
        "hidden_tail_risk_flag": False,
        "conflict_resolution": "정량 모델과 외부근거를 종합해 적격으로 정리했습니다.",
        "final_review_memo": "최종 위원회 판단은 적격입니다.",
        "key_risk_factors": [],
        "decision_trace": [],
    }
    output = RiskRecallQAOutput(
        qa_summary="중대 외부근거가 적격 판단에 충분히 반영되지 않았습니다.",
        trigger_reasons=["eligible_with_substantive_evidence"],
        eligible_safety_assessment="material_missed_risk",
        financial_resilience_check="재무 방어축은 별도 확인이 필요합니다.",
        evidence_recall_check="매출 대비 10% 이상 영업정지 공시가 있습니다.",
        rating_boundary_check="기준선 맥락보다 외부근거가 중요합니다.",
        recommended_action="escalate_eligible_to_risk_hold",
        confidence=0.78,
    )
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.20,
            "threshold": 0.45,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "영업정지(종속회사의주요경영사항)",
                    "company_match": True,
                    "evidence_score": 1.0,
                    "provider_relevance": "risk",
                    "disclosure_severity": "adverse",
                    "disclosure_event_class": "substantive_adverse",
                    "disclosure_materiality": "substantive_adverse",
                    "materiality_ratio": 0.1137,
                }
            ],
        },
    }
    runtime: dict[str, object] = {}

    adjusted = committee_node_module._apply_risk_recall_qa_advisory(
        committee_view=committee_view,
        risk_recall_qa_output=output,
        bundle=build_stage2_input_bundle(state),
        runtime_diagnostics=runtime,
    )

    assert adjusted["final_committee_label"] == "보류"
    assert adjusted["committee_decision_type"] == "risk_hold"
    assert adjusted["committee_risk_signal"] is True
    assert adjusted["risk_hold_reason_tags"] == ["external_materiality_hold"]
    assert adjusted["decision_trace"][-1]["gate"] == "risk_hold_reason_tagging"
    assert runtime["risk_recall_qa_advisory_applied"] is True
    assert (
        runtime["risk_recall_qa_advisory_apply_reason"] == "risk_recall_substantive_external_risk"
    )
