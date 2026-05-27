"""Tests for Stage 2 input bundle normalization."""

from __future__ import annotations

from cas.agents.stage2_bundle import build_stage2_input_bundle
from cas.agents.state import AgentState


def test_stage2_input_bundle_normalizes_state_for_agents() -> None:
    state: AgentState = {
        "company_id": "KOSPI-005930-2024",
        "company_name": "삼성전자",
        "market": "KOSPI",
        "analysis_year": 2025,
        "model_view": {"y_proba": 0.21},
        "xgboost_result": {"prediction_label": "투자적격", "threshold": 0.325},
        "source_feature_row": {"market": "KOSPI", "current_ratio": 2.1},
        "prior_rating_reference": {
            "has_prior_rating": True,
            "prior_credit_rating": "BBB-",
        },
        "peer_comparison_rows": [
            {"feature": "current_ratio", "industry_median": 1.5},
            {"feature": None, "industry_median": 0.0},
        ],
        "news_cache_snapshot": {"status": "not_implemented"},
    }

    bundle = build_stage2_input_bundle(state)

    assert bundle.company_name == "삼성전자"
    assert bundle.prediction_label == "투자적격"
    assert bundle.probability_speculative == 0.21
    assert bundle.threshold == 0.325
    assert bundle.news_status == "not_implemented"
    assert set(bundle.peer_rows_by_feature) == {"current_ratio"}
    assert bundle.prior_rating_reference["prior_credit_rating"] == "BBB-"
    assert bundle.credit_policy_snapshot == {}


def test_stage2_input_bundle_threshold_falls_back_to_model_view() -> None:
    state: AgentState = {
        "model_view": {"prediction_label": "투자적격", "threshold": 0.31},
        "xgboost_result": {},
    }

    bundle = build_stage2_input_bundle(state)

    assert bundle.threshold == 0.31


def test_stage2_input_bundle_exports_prompt_payload() -> None:
    state: AgentState = {
        "company_id": "KOSDAQ-000250-2023",
        "source_feature_row": {"company_name": "삼천당제약(주)", "market": "KOSDAQ"},
    }

    payload = build_stage2_input_bundle(state).to_prompt_payload()

    assert payload["company"]["company_id"] == "KOSDAQ-000250-2023"
    assert payload["company"]["company_name"] == "삼천당제약(주)"
    assert payload["company"]["market"] == "KOSDAQ"
    assert "model_view" in payload
    assert "prior_rating_reference" in payload
    assert payload["credit_policy_snapshot"] == {}


def test_stage2_input_bundle_includes_credit_policy_snapshot() -> None:
    state: AgentState = {
        "company_id": "KOSDAQ-000000-2025",
        "company_name": "테스트기업",
        "market": "KOSDAQ",
        "analysis_year": 2026,
        "company_profile": {},
        "source_feature_row": {},
        "peer_comparison_rows": [],
        "model_view": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.42,
        },
        "xgboost_result": {},
        "rule_result": {},
        "news_cache_snapshot": {"status": "not_requested"},
        "credit_policy_snapshot": {
            "policy_version": "credit_signal_policy_v1",
            "signals": [],
            "label_override_allowed": False,
            "risk_signal_count": 0,
            "mitigating_signal_count": 0,
            "critical_signal_count": 0,
        },
    }

    bundle = build_stage2_input_bundle(state)
    payload = bundle.to_prompt_payload()

    assert bundle.credit_policy_snapshot["policy_version"] == "credit_signal_policy_v1"
    assert bundle.credit_policy_snapshot["label_override_allowed"] is False
    assert bundle.credit_policy_snapshot["risk_signal_count"] == 0
    assert payload["credit_policy_snapshot"]["label_override_allowed"] is False
    assert payload["credit_policy_snapshot"]["critical_signal_count"] == 0


def test_stage2_input_bundle_exports_compact_prompt_payload_with_materiality() -> None:
    state: AgentState = {
        "company_id": "KOSDAQ-317120-2023",
        "company_name": "(주)라닉스",
        "market": "KOSDAQ",
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.946,
            "threshold": 0.25,
            "unused_large_field": "x" * 1000,
        },
        "xgboost_result": {
            "top_drivers": [
                {"feature": "interest_coverage_ratio", "shap_value": 0.12},
            ],
        },
        "source_feature_row": {
            "stock_code": "317120",
            "current_ratio": 1.9,
            "interest_coverage_ratio": -1.92,
            "debt_ratio": 1.75,
            "unused_raw_column": "drop-me",
        },
        "news_cache_snapshot": {
            "status": "ready",
            "as_of_date": "2022-12-31",
            "items": [
                {
                    "source": "opendart",
                    "title": "주요사항보고서(전환사채권발행결정)",
                    "summary": "라닉스 직접 관련 자금조달 공시입니다.",
                    "company_match": True,
                    "evidence_score": 0.72,
                    "provider_relevance": "caution",
                    "disclosure_severity": "caution",
                    "disclosure_event_class": "material_financing",
                    "disclosure_materiality": "substantive_adverse",
                    "materiality_ratio": 0.1361,
                    "materiality_basis": "희석률: 13.61%",
                    "unused_item_field": "drop-me",
                }
            ],
        },
    }

    payload = build_stage2_input_bundle(state).to_compact_prompt_payload(role="evidence_audit")

    assert payload["stage1_model"]["prediction_label"] == "부적격"
    assert "unused_large_field" not in payload["stage1_model"]
    assert payload["financial_metrics"]["interest_coverage_ratio"] == -1.92
    assert "unused_raw_column" not in payload["financial_metrics"]
    assert payload["materiality_summary"]["max_materiality_ratio"] == 0.1361
    assert payload["materiality_summary"]["financing_evidence_count"] == 1
    assert payload["news_cache_snapshot"]["items"][0]["materiality_basis"] == "희석률: 13.61%"
    assert "unused_item_field" not in payload["news_cache_snapshot"]["items"][0]


def test_stage2_input_bundle_adds_normalized_signal_summary() -> None:
    state: AgentState = {
        "company_id": "KOSDAQ-317120-2023",
        "company_name": "(주)라닉스",
        "market": "KOSDAQ",
        "model_view": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.24,
            "threshold": 0.30,
            "stage2_review_trigger": True,
            "stage2_secondary_trigger": True,
            "stage2_review_priority": "high",
            "trigger_reason_code": "manufacturing_fn_rescue",
        },
        "source_feature_row": {
            "current_ratio": 0.8,
            "cash_ratio": 0.05,
            "cashflow_coverage_ratio": -0.2,
            "interest_coverage_ratio": 0.5,
            "debt_ratio": 2.5,
            "short_term_borrowings_share": 0.95,
        },
        "prior_rating_reference": {
            "has_prior_rating": True,
            "prior_credit_rating": "BBB-",
            "prior_credit_rating_rank": 10,
            "prior_rating_boundary_group": "exact_bbb_minus_bb_plus_boundary",
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "주요사항보고서(전환사채권발행결정)",
                    "summary": "직접 관련 자금조달 공시입니다.",
                    "company_match": True,
                    "evidence_score": 0.72,
                    "provider_relevance": "caution",
                    "disclosure_severity": "caution",
                    "disclosure_event_class": "material_financing",
                    "disclosure_materiality": "substantive_adverse",
                    "materiality_ratio": 0.1361,
                    "materiality_basis": "희석률: 13.61%",
                }
            ],
        },
    }

    payload = build_stage2_input_bundle(state).to_compact_prompt_payload(role="risk_recall_qa")
    summary = payload["normalized_signal_summary"]

    assert payload["prompt_context_version"] == "stage2_compact_prompt_context_v2"
    assert "weak_cashflow" in summary["weak_financial_axes"]
    assert "weak_interest_coverage" in summary["weak_financial_axes"]
    assert summary["weak_financial_axis_count"] >= 4
    assert summary["materiality_profile"]["max_materiality_ratio"] == 0.1361
    assert summary["evidence_treatment"]["recommended_evidence_treatment"] in {
        "watch_context",
        "substantive_review",
        "critical_veto_review",
    }
    assert summary["boundary_context"]["has_rating_boundary_context"] is True
    assert summary["secondary_trigger_profile"]["stage2_secondary_trigger"] is True
    assert summary["secondary_trigger_profile"]["eligible_near_threshold"] is True


def test_stage2_compact_prompt_payload_is_role_scoped() -> None:
    state: AgentState = {
        "company_id": "KOSDAQ-000250-2023",
        "source_feature_row": {"company_name": "삼천당제약(주)", "market": "KOSDAQ"},
        "news_cache_snapshot": {"status": "ready", "items": []},
        "peer_comparison_rows": [{"feature": "current_ratio", "industry_median": 1.5}],
    }

    quant_payload = build_stage2_input_bundle(state).to_compact_prompt_payload(role="quant_credit")
    evidence_payload = build_stage2_input_bundle(state).to_compact_prompt_payload(
        role="evidence_audit"
    )

    assert "peer_comparison_rows" in quant_payload
    assert "news_cache_snapshot" not in quant_payload
    assert "news_cache_snapshot" in evidence_payload
    assert "peer_comparison_rows" not in evidence_payload


def test_stage2_input_bundle_preserves_prior_rating_reference_fallbacks() -> None:
    profile_state: AgentState = {
        "company_id": "KOSPI-000000-2024",
        "company_profile": {
            "prior_rating_reference": {
                "has_prior_rating": True,
                "prior_credit_rating": "BB+",
            }
        },
    }
    profile_bundle = build_stage2_input_bundle(profile_state)

    assert profile_bundle.prior_rating_reference["prior_credit_rating"] == "BB+"

    model_view_state: AgentState = {
        "company_id": "KOSPI-000001-2024",
        "model_view": {
            "prior_rating_reference": {
                "has_prior_rating": True,
                "prior_credit_rating": "BBB-",
            }
        },
    }
    model_view_bundle = build_stage2_input_bundle(model_view_state)

    assert model_view_bundle.prior_rating_reference["prior_credit_rating"] == "BBB-"
