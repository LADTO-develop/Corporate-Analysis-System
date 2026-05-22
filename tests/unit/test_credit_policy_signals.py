from __future__ import annotations

from cas.agents.signals.credit_policy_signals import (
    CreditPolicyConfig,
    CreditPolicyCriterion,
    evaluate_credit_policy,
)


def test_credit_policy_flags_high_debt_ratio_with_industry_gate() -> None:
    policy = CreditPolicyConfig(
        policy_version="test_policy",
        label_override_allowed=False,
        criteria=[
            CreditPolicyCriterion(
                criterion_id="leverage.debt_ratio.high",
                feature="debt_ratio",
                operator=">=",
                threshold=2.0,
                industry_percentile_min=75.0,
                direction="risk_increasing",
                severity="high",
                score_delta=0.08,
                reason_kr="부채비율 고위험 테스트",
                basis=["test"],
            )
        ],
    )

    snapshot = evaluate_credit_policy(
        source_feature_row={"debt_ratio": 2.5},
        peer_comparison_rows=[
            {
                "feature": "debt_ratio",
                "industry_percentile": 82.0,
            }
        ],
        policy=policy,
    )

    assert snapshot.policy_version == "test_policy"
    assert snapshot.label_override_allowed is False
    assert snapshot.risk_signal_count == 1
    assert snapshot.mitigating_signal_count == 0
    assert snapshot.net_policy_delta == 0.08
    assert snapshot.signals[0].criterion_id == "leverage.debt_ratio.high"


def test_credit_policy_does_not_flag_when_percentile_gate_fails() -> None:
    policy = CreditPolicyConfig(
        policy_version="test_policy",
        criteria=[
            CreditPolicyCriterion(
                criterion_id="leverage.debt_ratio.high",
                feature="debt_ratio",
                operator=">=",
                threshold=2.0,
                industry_percentile_min=75.0,
                direction="risk_increasing",
                severity="high",
                score_delta=0.08,
                reason_kr="부채비율 고위험 테스트",
                basis=["test"],
            )
        ],
    )

    snapshot = evaluate_credit_policy(
        source_feature_row={"debt_ratio": 2.5},
        peer_comparison_rows=[
            {
                "feature": "debt_ratio",
                "industry_percentile": 60.0,
            }
        ],
        policy=policy,
    )

    assert snapshot.signals == []
    assert snapshot.risk_signal_count == 0
    assert snapshot.net_policy_delta == 0.0


def test_credit_policy_treats_nan_values_as_missing_not_truthy() -> None:
    policy = CreditPolicyConfig(
        policy_version="test_policy",
        criteria=[
            CreditPolicyCriterion(
                criterion_id="coverage.icr.under_1",
                feature="interest_coverage_ratio",
                operator="<",
                threshold=1.0,
                direction="risk_increasing",
                severity="critical",
                score_delta=0.12,
                reason_kr="이자보상배율 결측 방어 테스트",
                basis=["test"],
            ),
            CreditPolicyCriterion(
                criterion_id="coverage.icr.flag",
                feature="icr_under_1",
                operator="truthy",
                direction="risk_increasing",
                severity="critical",
                score_delta=0.12,
                reason_kr="이자보상배율 플래그 결측 방어 테스트",
                basis=["test"],
            ),
        ],
    )

    snapshot = evaluate_credit_policy(
        source_feature_row={
            "interest_coverage_ratio": float("nan"),
            "icr_under_1": float("nan"),
        },
        peer_comparison_rows=[],
        policy=policy,
    )

    assert snapshot.signals == []
    assert snapshot.critical_signal_count == 0
    assert snapshot.net_policy_delta == 0.0
