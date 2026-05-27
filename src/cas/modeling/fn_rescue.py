"""Deterministic Stage 2 FN-rescue scores for low-score manufacturing cases."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

FN_RESCUE_TARGET_MARKET = "KOSDAQ"
FN_RESCUE_TARGET_INDUSTRY = "manufacturing"
FN_RESCUE_DEFAULT_PROB_CEILING = 0.30
FN_RESCUE_DEFAULT_SCORE_THRESHOLD = 0.65
FN_RESCUE_DEFAULT_MIN_GROUPS = 2
FN_RESCUE_GROUP_THRESHOLD = 0.70
FN_RESCUE_POLICY_NAME = (
    "kosdaq_manufacturing_low_stage1_probability_financial_stress_rescue_gate"
)

FN_RESCUE_RAW_COLUMNS = [
    "accounts_receivable_ratio",
    "inventory_ratio",
    "contract_assets_ratio",
    "ar_days_diff",
    "inventory_days_diff",
    "ap_days_diff",
    "delta_accruals_ratio",
    "is_ocf_turn_negative",
    "is_operating_income_turn_negative",
    "ocf_to_total_borrowings_diff",
    "ocf_to_total_liabilities_diff",
    "cash_ratio_diff",
    "delta_st_borrowings_share",
    "total_borrowings_growth",
    "current_ratio_diff",
    "capital_impairment_diff",
]
FN_RESCUE_SCORE_COLUMNS = [
    "fn_rescue_working_capital_stress_score",
    "fn_rescue_cashflow_turn_stress_score",
    "fn_rescue_borrowing_pressure_score",
    "fn_rescue_score",
    "fn_rescue_group_count",
]


def require_fn_rescue_columns(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> None:
    required = list(columns or FN_RESCUE_RAW_COLUMNS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"FN rescue source columns are missing: {missing}")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _positive_part(frame: pd.DataFrame, column: str) -> pd.Series:
    return _numeric(frame, column).clip(lower=0)


def _risk_rank(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    high_value_is_risk: bool = True,
) -> pd.Series:
    ranks = values.groupby([frame["fiscal_year"], frame["industry_macro_category"]]).rank(
        pct=True,
        method="average",
    )
    if high_value_is_risk:
        return ranks
    return 1.0 - ranks


def _mean_score(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame.loc[:, columns].mean(axis=1, skipna=True)


def add_manufacturing_fn_rescue_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic stress scores for KOSDAQ manufacturing FN rescue review."""
    require_fn_rescue_columns(frame)
    output = frame.copy().replace([np.inf, -np.inf], np.nan)

    output["_fn_risk_receivables"] = _risk_rank(output, _numeric(output, "accounts_receivable_ratio"))
    output["_fn_risk_inventory"] = _risk_rank(output, _numeric(output, "inventory_ratio"))
    output["_fn_risk_contract_assets"] = _risk_rank(output, _numeric(output, "contract_assets_ratio"))
    output["_fn_risk_ar_days_worsening"] = _risk_rank(output, _positive_part(output, "ar_days_diff"))
    output["_fn_risk_inventory_days_worsening"] = _risk_rank(
        output,
        _positive_part(output, "inventory_days_diff"),
    )
    output["_fn_risk_ap_days_worsening"] = _risk_rank(output, _positive_part(output, "ap_days_diff"))
    output["fn_rescue_working_capital_stress_score"] = _mean_score(
        output,
        [
            "_fn_risk_receivables",
            "_fn_risk_inventory",
            "_fn_risk_contract_assets",
            "_fn_risk_ar_days_worsening",
            "_fn_risk_inventory_days_worsening",
            "_fn_risk_ap_days_worsening",
        ],
    )

    output["_fn_risk_delta_accruals_abs"] = _risk_rank(
        output,
        _numeric(output, "delta_accruals_ratio").abs(),
    )
    output["_fn_risk_ocf_turn_negative"] = _numeric(output, "is_ocf_turn_negative").clip(0, 1)
    output["_fn_risk_operating_income_turn_negative"] = _numeric(
        output,
        "is_operating_income_turn_negative",
    ).clip(0, 1)
    output["_fn_risk_ocf_borrowings_drop"] = _risk_rank(
        output,
        _numeric(output, "ocf_to_total_borrowings_diff"),
        high_value_is_risk=False,
    )
    output["_fn_risk_ocf_liabilities_drop"] = _risk_rank(
        output,
        _numeric(output, "ocf_to_total_liabilities_diff"),
        high_value_is_risk=False,
    )
    output["_fn_risk_cash_ratio_drop"] = _risk_rank(
        output,
        _numeric(output, "cash_ratio_diff"),
        high_value_is_risk=False,
    )
    output["fn_rescue_cashflow_turn_stress_score"] = _mean_score(
        output,
        [
            "_fn_risk_delta_accruals_abs",
            "_fn_risk_ocf_turn_negative",
            "_fn_risk_operating_income_turn_negative",
            "_fn_risk_ocf_borrowings_drop",
            "_fn_risk_ocf_liabilities_drop",
            "_fn_risk_cash_ratio_drop",
        ],
    )

    output["_fn_risk_st_borrowings_rise"] = _risk_rank(
        output,
        _positive_part(output, "delta_st_borrowings_share"),
    )
    output["_fn_risk_total_borrowings_growth"] = _risk_rank(
        output,
        _numeric(output, "total_borrowings_growth"),
    )
    output["_fn_risk_current_ratio_drop"] = _risk_rank(
        output,
        _numeric(output, "current_ratio_diff"),
        high_value_is_risk=False,
    )
    output["_fn_risk_capital_impairment_worsening"] = _risk_rank(
        output,
        _positive_part(output, "capital_impairment_diff"),
    )
    output["fn_rescue_borrowing_pressure_score"] = _mean_score(
        output,
        [
            "_fn_risk_st_borrowings_rise",
            "_fn_risk_total_borrowings_growth",
            "_fn_risk_current_ratio_drop",
            "_fn_risk_capital_impairment_worsening",
        ],
    )

    output["fn_rescue_score"] = (
        output["fn_rescue_working_capital_stress_score"] * 0.35
        + output["fn_rescue_cashflow_turn_stress_score"] * 0.35
        + output["fn_rescue_borrowing_pressure_score"] * 0.30
    )
    group_scores = output.loc[
        :,
        [
            "fn_rescue_working_capital_stress_score",
            "fn_rescue_cashflow_turn_stress_score",
            "fn_rescue_borrowing_pressure_score",
        ],
    ]
    output["fn_rescue_group_count"] = group_scores.ge(FN_RESCUE_GROUP_THRESHOLD).sum(axis=1)
    return output


def build_manufacturing_fn_rescue_gate(
    frame: pd.DataFrame,
    *,
    probability_column: str = "prob_speculative",
    prediction_column: str = "pred_label_tuned",
    score_column: str = "fn_rescue_score",
    group_count_column: str = "fn_rescue_group_count",
    probability_ceiling: float = FN_RESCUE_DEFAULT_PROB_CEILING,
    score_threshold: float = FN_RESCUE_DEFAULT_SCORE_THRESHOLD,
    min_group_count: int = FN_RESCUE_DEFAULT_MIN_GROUPS,
) -> pd.Series:
    """Return a review-only trigger for likely low-score manufacturing FN cases."""
    market = frame["market"].astype(str).eq(FN_RESCUE_TARGET_MARKET)
    industry = frame["industry_macro_category"].astype(str).eq(FN_RESCUE_TARGET_INDUSTRY)
    stage1_normal = pd.to_numeric(frame[prediction_column], errors="coerce").fillna(0).astype(int).eq(0)
    low_probability = pd.to_numeric(frame[probability_column], errors="coerce").le(
        probability_ceiling
    )
    high_rescue_score = pd.to_numeric(frame[score_column], errors="coerce").ge(score_threshold)
    enough_groups = pd.to_numeric(frame[group_count_column], errors="coerce").ge(min_group_count)
    return market & industry & stage1_normal & low_probability & high_rescue_score & enough_groups
