"""Tests for dashboard value formatting helpers."""

from __future__ import annotations

import math

import pandas as pd

from cas.dashboard.chart_data import finite_chart_frame, finite_float_or_none
from cas.dashboard.committee_copy import committee_decision_type_info
from cas.dashboard.committee_panel import (
    agent_disagreement_level_info,
    agent_disagreement_reason_label,
    review_qa_trigger_reason_label,
)
from cas.dashboard.evidence_panel import (
    _external_evidence_items_frame,
    _external_evidence_materiality_basis,
    _external_materiality_summary,
)
from cas.dashboard.formatting import COVERAGE_CAP_LABEL, format_ratio_value
from cas.dashboard.labels import (
    format_stage2_risk_band,
    to_committee_base_label,
    to_industry_label,
    to_market_display_label,
    to_stage2_risk_band,
)


def test_coverage_ratios_are_displayed_as_multiples() -> None:
    assert format_ratio_value(15.29068, "interest_coverage_ratio") == "15.29배"
    assert format_ratio_value(35.59144, "cashflow_coverage_ratio") == "35.59배"
    assert format_ratio_value(2.5, "interest_coverage_ratio", signed=True) == "+2.50배"


def test_ordinary_ratios_still_display_as_percentages() -> None:
    assert format_ratio_value(0.057773, "net_margin") == "5.78%"
    assert format_ratio_value(0.025, "net_margin", signed=True) == "+2.50%p"


def test_capped_coverage_ratios_are_not_rendered_as_huge_percentages() -> None:
    assert format_ratio_value(1_000_000, "interest_coverage_ratio") == COVERAGE_CAP_LABEL


def test_negative_capital_impairment_keeps_its_signed_percentage() -> None:
    assert format_ratio_value(-42.108489, "capital_impairment_ratio") == "-4210.85%"
    assert format_ratio_value(0.452343, "capital_impairment_ratio") == "45.23%"
    assert format_ratio_value(-0.2, "capital_impairment_ratio", signed=True) == "-20.00%p"


def test_committee_decision_copy_is_user_friendly() -> None:
    boundary = committee_decision_type_info("경계등급 보류", risk_signal=False)
    mitigation = committee_decision_type_info("과민경고 완화 보류", risk_signal=False)

    assert "딱 잘라 말하기 어려운" in boundary["body"]
    assert "BBB-/BB+" in boundary["detail"]
    assert "바로 부적격으로 단정하긴 이릅니다" in mitigation["body"]
    assert "SHAP" in mitigation["action"]


def test_agent_disagreement_copy_is_user_friendly() -> None:
    high = agent_disagreement_level_info("high", score=0.65)
    inferred_medium = agent_disagreement_level_info("", score=0.30)

    assert high["label"] == "높음"
    assert high["tone"] == "high"
    assert "추가 QA 검토" in high["body"]
    assert inferred_medium["label"] == "중간"
    assert (
        agent_disagreement_reason_label("quant_risk_evidence_watch_context")
        == "정량 모델은 위험을 보지만 외부근거는 치명급이 아니에요."
    )
    assert (
        review_qa_trigger_reason_label("agent_disagreement_high_without_critical_evidence")
        == "내부 의견 차이가 큰데 치명 외부근거는 제한적이라 다시 확인했어요."
    )
    assert (
        review_qa_trigger_reason_label("ambiguous_external_evidence")
        == "외부근거가 애매해 위험으로 볼지 다시 확인했어요."
    )


def test_dashboard_label_helpers_match_user_facing_copy() -> None:
    assert to_market_display_label("KOSPI") == "코스피"
    assert to_industry_label("it_services") == "IT·서비스업"
    assert to_committee_base_label("투자적격") == "적격"
    assert to_committee_base_label("unknown") == "보류"
    assert to_stage2_risk_band("고위험") == "high_risk"
    assert format_stage2_risk_band("watch") == "관찰"


def test_chart_numeric_helpers_drop_non_finite_values() -> None:
    assert finite_float_or_none("12.5") == 12.5
    assert finite_float_or_none(math.inf) is None

    frame = finite_chart_frame(
        [
            {"label": "valid", "값": "1.25"},
            {"label": "missing", "값": pd.NA},
            {"label": "infinite", "값": math.inf},
        ],
        ["값"],
    )

    assert frame["label"].tolist() == ["valid"]
    assert frame["값"].tolist() == [1.25]


def test_external_evidence_table_exposes_materiality_context() -> None:
    snapshot = {
        "items": [
            {
                "source": "opendart",
                "title": "주요사항보고서(유상증자결정)",
                "company_match": True,
                "evidence_quality": "high",
                "reliability": "high",
                "materiality_ratio": 0.155,
                "materiality_basis": "발행금액/자기자본: 15.50%",
                "dilution_ratio": 0.2123,
                "dilution_basis": "희석률: 21.23%",
                "disclosure_materiality": "substantive_adverse",
                "disclosure_event_class": "material_financing",
            },
            {
                "source": "opendart",
                "title": "타인에대한채무보증결정",
                "company_match": True,
                "evidence_quality": "medium",
                "reliability": "high",
                "materiality_ratio": "0.1490",
                "materiality_basis": "채무보증금액/자기자본: 14.90%",
                "disclosure_materiality": "substantive_adverse",
                "disclosure_event_class": "material_debt_guarantee",
            },
            {
                "source": "opendart",
                "title": "단일판매공급계약해지",
                "company_match": True,
                "evidence_quality": "medium",
                "reliability": "medium",
                "materiality_ratio": "0.0240",
                "materiality_basis": "계약해지금액/매출액: 2.40%",
                "disclosure_materiality": "procedural_or_one_off",
                "disclosure_event_class": "low_materiality_contract_cancellation",
            },
        ]
    }

    frame = _external_evidence_items_frame(snapshot)

    assert "상세 중요도" in frame.columns
    assert "중요도 단계" in frame.columns
    assert "공시 성격" in frame.columns
    assert "희석률: 21.23%" in frame.loc[0, "상세 중요도"]
    assert frame.loc[1, "상세 중요도"] == "채무보증금액/자기자본: 14.90%"
    assert frame.loc[2, "중요도 단계"] == "절차/일회성"


def test_external_materiality_summary_highlights_scale_and_event_type() -> None:
    financing_item = {
        "source": "opendart",
        "title": "주요사항보고서(유상증자결정)",
        "materiality_ratio": 0.155,
        "materiality_basis": "발행금액/자기자본: 15.50%",
        "dilution_ratio": 0.2123,
        "dilution_basis": "희석률: 21.23%",
        "disclosure_materiality": "substantive_adverse",
        "disclosure_event_class": "material_financing",
    }
    snapshot = {
        "items": [
            financing_item,
            {
                "source": "opendart",
                "title": "타인에대한채무보증결정",
                "materiality_ratio": "0.1490",
                "materiality_basis": "채무보증금액/자기자본: 14.90%",
                "disclosure_materiality": "substantive_adverse",
                "disclosure_event_class": "material_debt_guarantee",
            },
            {
                "source": "opendart",
                "title": "단일판매공급계약해지",
                "materiality_ratio": "0.0240",
                "materiality_basis": "계약해지금액/매출액: 2.40%",
                "disclosure_materiality": "procedural_or_one_off",
                "disclosure_event_class": "low_materiality_contract_cancellation",
            },
        ]
    }

    summary = _external_materiality_summary(snapshot)

    assert _external_evidence_materiality_basis(financing_item).endswith("희석률: 21.23%")
    assert summary["has_materiality"] is True
    assert "희석률: 21.23%" in str(summary["max_basis"])
    assert summary["substantive_count"] == 2
    assert summary["watch_or_low_count"] == 1
    assert "중요 자금조달" in str(summary["top_events"])
