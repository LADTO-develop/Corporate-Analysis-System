"""Versioned Stage 2 policy thresholds loaded from YAML."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from cas.utils.io import read_yaml

DEFAULT_STAGE2_POLICY_PATH = Path("configs/agent/stage2_policy.yaml")

_DEFAULT_STAGE2_POLICY: dict[str, Any] = {
    "policy_version": "stage2_policy_v1",
    "review_qa": {
        "advisory": {
            "overstated_risk_hold_min_confidence": 0.55,
            "risk_hold_min_confidence": 0.45,
            "reject_min_confidence": 0.45,
        },
        "disagreement": {
            "high_score_floor": 0.55,
            "medium_score_floor": 0.25,
        },
        "boundary_defense": {
            "min_defensive_axes": 3,
            "current_ratio_floor": 1.20,
            "cash_ratio_floor": 0.15,
            "equity_ratio_floor": 0.40,
            "debt_ratio_ceiling": 1.50,
            "capital_impairment_ratio_ceiling": 0.0,
            "total_borrowings_ratio_ceiling": 0.50,
            "short_term_borrowings_share_ceiling": 0.70,
            "cashflow_coverage_ratio_floor": 1.0,
            "ocf_to_total_liabilities_floor": 0.05,
            "ocf_to_sales_floor": 0.0,
            "interest_coverage_ratio_floor": 1.0,
        },
        "reject_boundary_defense": {
            "min_defensive_axes": 3,
            "current_ratio_floor": 1.20,
            "cash_ratio_floor": 0.15,
            "equity_ratio_floor": 0.40,
            "debt_ratio_ceiling": 1.50,
            "capital_impairment_ratio_ceiling": 0.0,
            "total_borrowings_ratio_ceiling": 0.50,
            "short_term_borrowings_share_ceiling": 0.70,
            "cashflow_coverage_ratio_floor": 0.0,
            "ocf_to_total_liabilities_floor": 0.0,
            "ocf_to_sales_floor": 0.0,
        },
        "extreme_distress": {
            "capital_impairment_ratio_floor": 0.50,
            "equity_ratio_ceiling": 0.15,
            "debt_ratio_floor": 5.0,
            "short_term_borrowings_share_floor": 0.95,
            "interest_coverage_ratio_ceiling": 1.0,
            "cashflow_coverage_ratio_floor": 0.0,
            "ocf_to_total_liabilities_floor": 0.0,
            "ocf_to_sales_floor": 0.0,
        },
    },
    "risk_recall_qa": {
        "advisory": {
            "risk_hold_min_confidence": 0.70,
            "boundary_hold_min_confidence": 0.60,
            "near_threshold_min_weak_axes": 2,
            "multi_axis_min_weak_axes": 3,
            "boundary_rating_min_weak_axes": 2,
            "severe_financial_weakness_min_axes": 4,
        },
        "evidence": {
            "opendart_min_score": 0.55,
            "medium_high_min_score": 0.55,
            "fallback_min_score": 0.65,
            "watch_min_score": 0.45,
        },
        "trigger": {
            "near_threshold_margin": 0.10,
            "current_ratio_floor": 1.0,
            "cash_ratio_floor": 0.10,
            "cashflow_coverage_ratio_floor": 0.0,
            "ocf_to_total_liabilities_floor": 0.0,
            "ocf_to_sales_floor": 0.0,
            "interest_coverage_ratio_floor": 1.0,
            "debt_ratio_floor": 2.0,
            "total_borrowings_ratio_floor": 0.65,
            "short_term_borrowings_share_floor": 0.90,
        },
    },
    "committee_guardrails": {
        "model_threshold": {
            "default": 0.315,
        },
        "secondary_review": {
            "probability_floor_absolute": 0.28,
            "threshold_buffer": 0.10,
            "rule_confidence_floor": 0.60,
            "risk_signal_threshold_buffer": 0.04,
            "liquidity_current_ratio_floor": 1.0,
            "liquidity_cash_ratio_floor": 0.10,
            "prior_boundary_probability_floor": 0.20,
        },
        "isolated_interest_cover_defense": {
            "interest_coverage_ratio_ceiling": 1.0,
            "current_ratio_floor": 1.20,
            "cash_ratio_floor": 0.15,
            "cashflow_coverage_ratio_floor": 1.0,
            "ocf_to_total_liabilities_floor": 0.05,
            "total_borrowings_ratio_ceiling": 0.10,
            "capital_impairment_ratio_ceiling": 0.0,
            "net_margin_floor": -0.05,
        },
        "isolated_icr_review_buffer": {
            "interest_coverage_ratio_ceiling": 1.0,
            "capital_impairment_ratio_ceiling": 0.0,
            "net_margin_floor": -0.05,
            "cashflow_coverage_ratio_floor": 1.0,
            "ocf_to_total_liabilities_floor": 0.05,
            "equity_ratio_floor": 0.70,
            "debt_ratio_ceiling": 0.50,
            "total_borrowings_ratio_ceiling": 0.20,
        },
        "cashflow_backed_liquidity_buffer": {
            "current_ratio_ceiling": 1.0,
            "cash_ratio_floor": 0.25,
            "cashflow_coverage_ratio_floor": 1.0,
            "ocf_to_total_liabilities_floor": 0.05,
            "ocf_to_sales_floor": 0.0,
            "interest_coverage_ratio_floor": 3.0,
            "equity_ratio_floor": 0.40,
            "debt_ratio_ceiling": 1.50,
            "short_term_borrowings_share_ceiling": 0.80,
            "total_borrowings_ratio_ceiling": 0.30,
            "capital_impairment_ratio_ceiling": 0.0,
            "net_margin_floor": -0.05,
        },
        "secondary_overhold_supports": {
            "min_required_supports": 2,
            "current_ratio_floor": 1.20,
            "cash_ratio_floor": 0.15,
            "cashflow_coverage_ratio_floor": 1.0,
            "ocf_to_total_liabilities_floor": 0.05,
            "ocf_to_sales_floor": 0.0,
            "interest_coverage_ratio_floor": 1.0,
            "equity_ratio_floor": 0.40,
            "debt_ratio_ceiling": 1.50,
            "total_borrowings_ratio_ceiling": 0.50,
            "capital_impairment_ratio_ceiling": 0.0,
        },
        "secondary_overhold_blocker": {
            "net_margin_floor": -0.10,
            "ocf_to_sales_floor": 0.0,
            "ocf_to_total_liabilities_floor": 0.0,
            "interest_coverage_ratio_floor": 3.0,
            "equity_ratio_floor": 0.40,
            "debt_ratio_floor": 1.50,
        },
        "missing_statement_placeholder": {
            "assets_total_ceiling": 0.0,
            "gross_profit_ceiling": 0.0,
            "interest_coverage_ratio_floor": 999_999.0,
            "cashflow_coverage_ratio_floor": 999_999.0,
        },
        "severe_financial_watch": {
            "capital_impairment_ratio_floor": 0.0,
            "interest_coverage_ratio_floor": 1.0,
            "current_ratio_floor": 0.70,
            "cash_ratio_floor": 0.05,
            "cashflow_coverage_ratio_floor": 0.0,
            "ocf_to_total_liabilities_floor": 0.0,
        },
        "risk_hold_financial_stress": {
            "reject_confirmation_min_signals": 2,
            "min_financial_flags": 2,
            "interest_coverage_ratio_floor": 1.0,
            "cashflow_coverage_ratio_floor": 0.0,
            "ocf_to_total_liabilities_floor": 0.0,
            "ocf_to_sales_floor": 0.0,
            "net_margin_floor": -0.10,
            "capital_impairment_ratio_floor": 0.0,
            "equity_ratio_floor": 0.25,
            "debt_ratio_floor": 1.50,
            "current_ratio_floor": 1.0,
            "cash_ratio_floor": 0.10,
        },
        "cash_rich_loss_stage_overwarning_buffer": {
            "probability_floor": 0.85,
            "current_ratio_floor": 2.0,
            "cash_ratio_floor": 0.50,
            "equity_ratio_floor": 0.60,
            "debt_ratio_ceiling": 0.50,
            "total_borrowings_ratio_ceiling": 0.10,
            "cashflow_coverage_ratio_floor": 0.0,
            "ocf_to_total_liabilities_floor": 0.0,
            "ocf_to_sales_floor": 0.0,
        },
        "prior_boundary_overwarning_buffer": {
            "threshold_additive_margin": 0.20,
            "probability_floor_absolute": 0.55,
        },
        "model_only_overwarning_buffer": {
            "threshold_additive_margin": 0.10,
            "probability_ceiling": 0.90,
        },
        "cashflow_backed_fp_resilience": {
            "capital_impairment_ratio_ceiling": 0.0,
            "cashflow_coverage_ratio_floor": 0.0,
            "ocf_to_total_liabilities_floor": 0.05,
            "ocf_to_sales_floor": 0.0,
            "equity_ratio_floor": 0.40,
            "debt_ratio_ceiling": 1.50,
            "total_borrowings_ratio_ceiling": 0.40,
            "short_term_borrowings_share_ceiling": 0.70,
        },
        "extreme_financial_distress": {
            "capital_impairment_ratio_floor": 0.50,
            "equity_ratio_ceiling": 0.15,
            "debt_ratio_floor": 5.0,
            "short_term_borrowings_share_floor": 0.95,
            "cashflow_coverage_ratio_floor": 0.0,
            "ocf_to_total_liabilities_floor": 0.0,
            "ocf_to_sales_floor": 0.0,
            "interest_coverage_ratio_ceiling": 1.0,
        },
        "financial_resilience_overwarning": {
            "current_ratio_floor": 1.20,
            "cash_ratio_floor": 0.15,
            "equity_ratio_floor": 0.40,
            "debt_ratio_ceiling": 1.50,
            "total_borrowings_ratio_ceiling": 0.50,
            "capital_impairment_ratio_ceiling": 0.0,
            "interest_coverage_ratio_floor": 1.0,
            "net_margin_floor": 0.0,
            "ocf_to_sales_floor": 0.0,
            "short_term_borrowings_share_ceiling": 0.80,
            "blocker_net_margin_floor": -0.10,
            "blocker_equity_ratio_floor": 0.25,
            "blocker_total_borrowings_ratio_floor": 0.65,
            "blocker_short_term_borrowings_share_floor": 0.90,
            "min_support_count": 8,
            "max_blocker_count": 0,
        },
        "cashflow_backed_near_threshold_tn_defense": {
            "interest_coverage_ratio_ceiling": 1.0,
            "capital_impairment_ratio_ceiling": 0.0,
            "net_margin_floor": -0.10,
            "cashflow_coverage_ratio_floor": 1.0,
            "ocf_to_total_liabilities_floor": 0.05,
            "ocf_to_sales_floor": 0.0,
            "cash_ratio_floor": 0.05,
            "total_borrowings_ratio_ceiling": 0.55,
        },
        "prior_rating": {
            "speculative_min_rank": 11,
            "stable_investment_max_rank": 8,
        },
    },
}


@dataclass(frozen=True)
class Stage2Policy:
    """Resolved Stage 2 policy values."""

    policy_version: str
    values: Mapping[str, Any]

    def value(self, *path: str) -> object:
        """Return a raw policy value by nested key path."""
        current: object = self.values
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                joined = ".".join(path)
                raise KeyError(f"Stage 2 policy key not found: {joined}")
            current = current[key]
        return current

    def float(self, *path: str) -> float:
        """Return a policy value as float."""
        return float(self.value(*path))

    def int(self, *path: str) -> int:
        """Return a policy value as int."""
        return int(self.value(*path))


@lru_cache(maxsize=4)
def load_stage2_policy(path: str | Path = DEFAULT_STAGE2_POLICY_PATH) -> Stage2Policy:
    """Load the versioned Stage 2 policy from YAML."""
    raw = read_yaml(path)
    values = _deep_merge(_DEFAULT_STAGE2_POLICY, raw)
    version = str(values.get("policy_version") or _DEFAULT_STAGE2_POLICY["policy_version"])
    return Stage2Policy(policy_version=version, values=values)


def stage2_policy_version() -> str:
    """Return the active Stage 2 policy version string."""
    return load_stage2_policy().policy_version


def _deep_merge(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(defaults)
    for key, value in overrides.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


__all__ = [
    "DEFAULT_STAGE2_POLICY_PATH",
    "Stage2Policy",
    "load_stage2_policy",
    "stage2_policy_version",
]
