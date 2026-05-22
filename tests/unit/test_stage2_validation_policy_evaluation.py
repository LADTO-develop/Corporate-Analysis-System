"""Tests for validation/test Stage 2 policy diagnostics."""

from __future__ import annotations

import pandas as pd
from scripts import export_stage2_validation_test_policy_evaluation as policy


def test_build_trace_gate_contribution_counts_fn_and_fp_help() -> None:
    frame = pd.DataFrame(
        [
            {
                "split": "valid",
                "is_speculative": 1,
                "policy_stage1_model": False,
                "current_committee_label": "보류",
                "current_committee_decision_type": "risk_hold",
                "current_committee_risk_signal": True,
                "trace_hidden_tail_risk_triggered": True,
                "trace_overwarning_mitigation_triggered": False,
            },
            {
                "split": "valid",
                "is_speculative": 0,
                "policy_stage1_model": True,
                "current_committee_label": "보류",
                "current_committee_decision_type": "mitigation_hold",
                "current_committee_risk_signal": False,
                "trace_hidden_tail_risk_triggered": False,
                "trace_overwarning_mitigation_triggered": True,
            },
        ]
    )

    contribution = policy.build_trace_gate_contribution(frame)

    hidden_tail = contribution.loc[contribution["gate"].eq("hidden_tail_risk")].iloc[0]
    mitigation = contribution.loc[contribution["gate"].eq("overwarning_mitigation")].iloc[0]

    assert hidden_tail["fn_escalated_count"] == 1
    assert hidden_tail["fp_softened_count"] == 0
    assert mitigation["fn_escalated_count"] == 0
    assert mitigation["fp_softened_count"] == 1
    assert mitigation["dominant_effect"] == "fp_softening"
