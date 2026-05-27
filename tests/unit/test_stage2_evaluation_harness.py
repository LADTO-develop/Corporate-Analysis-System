"""Tests for Stage 2 feature_46/full_review_trigger_73 harness metrics."""

from __future__ import annotations

import pandas as pd

from cas.agents.stage2_evaluation_harness import (
    build_harness_report,
    explanation_quality_metrics,
    provider_summary_frame,
    summarize_batch_results,
    summarize_by_category,
)


def test_summarize_batch_results_reports_requested_harness_metrics() -> None:
    results = pd.DataFrame(
        [
            {
                "sample_category": "fn_caught_by_stage2_review",
                "model_error_type": "false_negative",
                "final_committee_label": "보류",
                "committee_effect": "fn_escalated",
                "committee_success": True,
                "committee_review_safe_success": True,
                "stage2_total_elapsed_seconds": 10.0,
                "case_elapsed_seconds": 11.0,
                "stage2_llm_cache_hit": False,
                "stage2_review_qa_cache_hit": False,
                "stage2_risk_recall_qa_cache_hit": True,
                "stage2_review_qa_triggered": False,
                "stage2_risk_recall_qa_triggered": True,
                "error_message": "",
                "stage2_error_message": "",
            },
            {
                "sample_category": "fp_needing_committee_mitigation",
                "model_error_type": "false_positive",
                "final_committee_label": "보류",
                "committee_effect": "fp_mitigated",
                "committee_success": True,
                "committee_review_safe_success": True,
                "stage2_total_elapsed_seconds": 20.0,
                "case_elapsed_seconds": 22.0,
                "stage2_llm_cache_hit": True,
                "stage2_review_qa_cache_hit": False,
                "stage2_risk_recall_qa_cache_hit": False,
                "stage2_review_qa_triggered": True,
                "stage2_risk_recall_qa_triggered": False,
                "error_message": "",
                "stage2_error_message": "",
            },
            {
                "sample_category": "true_negative_overescalation_guardrail",
                "model_error_type": "true_negative",
                "final_committee_label": "",
                "committee_effect": "run_failed",
                "committee_success": False,
                "committee_review_safe_success": False,
                "stage2_total_elapsed_seconds": None,
                "case_elapsed_seconds": 5.0,
                "stage2_llm_cache_hit": False,
                "stage2_review_qa_cache_hit": False,
                "stage2_risk_recall_qa_cache_hit": False,
                "stage2_review_qa_triggered": False,
                "stage2_risk_recall_qa_triggered": False,
                "error_message": "timeout",
                "stage2_error_message": "",
            },
        ]
    )

    summary = summarize_batch_results(
        results,
        run_id="openai_gpt_4_1_mini",
        runner="agno",
        provider="openai",
        model="gpt-4.1-mini",
        stage2_policy_version="stage2_policy_v1",
    )

    assert summary["strict_success_rate"] == 0.6667
    assert summary["review_safe_success_rate"] == 0.6667
    assert summary["fn_rescue_success_rate"] == 1.0
    assert summary["fp_over_hold_count"] == 1
    assert summary["fp_over_hold_rate"] == 1.0
    assert summary["stage2_latency_mean_seconds"] == 15.0
    assert summary["stage2_latency_p95_seconds"] == 19.5
    assert summary["stage2_latency_max_seconds"] == 20.0
    assert summary["case_latency_p95_seconds"] == 20.9
    assert summary["review_qa_trigger_rows"] == 1
    assert summary["risk_recall_qa_trigger_rows"] == 1
    assert summary["any_qa_trigger_rows"] == 2
    assert summary["any_qa_trigger_rate"] == 0.6667
    assert summary["stage2_policy_version"] == "stage2_policy_v1"
    assert summary["llm_cache_hit_rows"] == 1
    assert summary["any_cache_hit_rows"] == 2
    assert summary["run_failure_rows"] == 1
    assert "memo_quality_score" in summary
    assert "quality_gate_pass" in summary


def test_explanation_quality_metrics_gate_report_ready_memos() -> None:
    results = pd.DataFrame(
        [
            {
                "final_committee_label": "보류",
                "committee_decision_type": "risk_hold",
                "risk_hold_reason_tags": "financial_stress_hold",
                "risk_hold_reason_summary": "부채와 차입 부담, 현금흐름 둔화가 함께 확인됩니다.",
                "agent_disagreement_score": 0.2,
                "agent_disagreement_summary": "정량과 외부근거 간 일부 상충을 조정했습니다.",
                "conflict_resolution": "최종 라벨은 보류이며 위험 보류로 검토합니다.",
                "decision_trace": '[{"gate":"risk_hold","triggered":true}]',
                "final_review_memo": (
                    "최종 라벨은 보류입니다. 부채와 차입 부담, 현금흐름 둔화, "
                    "이자 coverage 약화가 함께 보여 위험 보류로 검토합니다. "
                    "외부근거는 미수집/비활성 상태이므로 확인된 공시나 기사로 단정하지 않고, "
                    "재무 방어력과 기준선 이탈 여부를 추가 확인해야 합니다."
                ),
                "evidence_status": "disabled",
                "evidence_items": 0,
            }
        ]
    )

    metrics = explanation_quality_metrics(results)

    assert metrics["quality_gate_pass"] is True
    assert metrics["memo_quality_score"] >= 0.55
    assert metrics["evidence_grounding_score"] >= 0.5
    assert metrics["decision_consistency_score"] >= 0.75
    assert metrics["hallucination_flag_rate"] == 0.0


def test_explanation_quality_metrics_flags_hallucinated_external_evidence() -> None:
    results = pd.DataFrame(
        [
            {
                "final_committee_label": "적격",
                "committee_decision_type": "eligible",
                "final_review_memo": (
                    "최종 라벨은 적격입니다. 확인된 공시 확인 결과 신용등급을 확정합니다."
                ),
                "decision_trace": '[{"gate":"eligible","triggered":true}]',
                "evidence_status": "disabled",
                "evidence_items": 0,
            }
        ]
    )

    metrics = explanation_quality_metrics(results)

    assert metrics["quality_gate_pass"] is False
    assert metrics["hallucination_flag_rows"] == 1
    assert "hallucination_flag_rate" in metrics["quality_gate_fail_reasons"]


def test_explanation_quality_metrics_prefers_llm_judge_columns() -> None:
    results = pd.DataFrame(
        [
            {
                "final_committee_label": "보류",
                "committee_decision_type": "risk_hold",
                "final_review_memo": "짧은 메모",
                "evidence_status": "ready",
                "evidence_items": 1,
                "llm_judge_memo_quality": 5,
                "llm_judge_evidence_grounding": 4,
                "llm_judge_decision_consistency": 5,
                "llm_judge_hallucination_flag": False,
            }
        ]
    )

    metrics = explanation_quality_metrics(results)

    assert metrics["quality_judge_source"] == "llm_judge_columns"
    assert metrics["memo_quality_score"] == 1.0
    assert metrics["evidence_grounding_score"] == 0.8
    assert metrics["decision_consistency_score"] == 1.0
    assert metrics["quality_gate_pass"] is True


def test_provider_and_category_summary_have_stable_shapes() -> None:
    results = pd.DataFrame(
        [
            {
                "sample_category": "fn_caught_by_stage2_review",
                "committee_success": True,
                "committee_review_safe_success": True,
            }
        ]
    )
    provider = provider_summary_frame(
        [
            {
                "run_id": "deterministic",
                "runner": "deterministic",
                "provider": "deterministic",
                "model": "deterministic",
                "rows": 1,
                "stage2_policy_version": "stage2_policy_v1",
            }
        ]
    )
    category = summarize_by_category(results, run_id="deterministic")

    assert next(iter(provider.columns)) == "run_id"
    assert provider.loc[0, "rows"] == 1
    assert provider.loc[0, "stage2_policy_version"] == "stage2_policy_v1"
    assert category.loc[0, "sample_category"] == "fn_caught_by_stage2_review"
    assert category.loc[0, "strict_success_rate"] == 1.0


def test_harness_report_includes_stage2_policy_version(tmp_path) -> None:
    report = build_harness_report(
        provider_summary=pd.DataFrame(),
        category_summary=pd.DataFrame(),
        sample_summary={"sample_rows": 0},
        skipped_runs=[],
        output_dir=tmp_path,
        stage2_policy_version="stage2_policy_v1",
        prompt_contract_versions={
            "quant_credit": "stage2_role_prompt_contract_v2:quant_credit",
        },
    )

    assert "Stage2 policy version: `stage2_policy_v1`" in report
    assert "quant_credit=stage2_role_prompt_contract_v2:quant_credit" in report
