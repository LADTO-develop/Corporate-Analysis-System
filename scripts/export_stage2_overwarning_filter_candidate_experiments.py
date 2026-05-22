"""Compare candidate Stage 2 over-warning mitigation policies.

The policies in this script do not change the Stage 1 XGBoost prediction.
They only estimate which Stage 1 false positives could be softened from
`부적격` to `보류` by the committee, and how many true positives would be
incorrectly softened at the same time.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTION_SCORES_PATH = ROOT / "data/outputs/dashboard/feature_43_mvp/prediction_scores.csv"
FEATURE_MASTER_PATH = ROOT / "data/input/credit_43_features/feature_43_master.csv"
TARGET_LABEL_REFERENCE_PATH = ROOT / "data/evaluation/target_label_reference.csv"
OUTPUT_DIR = ROOT / "data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents"

KEY_COLUMNS = ["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]
FEATURE_COLUMNS = [
    "current_ratio",
    "cash_ratio",
    "equity_ratio",
    "debt_ratio",
    "total_borrowings_ratio",
    "capital_impairment_ratio",
    "interest_coverage_ratio",
    "net_margin",
    "ocf_to_sales",
    "ocf_to_total_liabilities",
    "ocf_to_total_borrowings",
    "cashflow_coverage_ratio",
    "short_term_borrowings_share",
    "dividend_payer",
    "is_2y_consecutive_operating_loss",
    "is_2y_consecutive_ocf_deficit",
    "icr_under_1",
]
RATING_RANK = {
    "AAA": 1,
    "AA+": 2,
    "AA": 3,
    "AA-": 4,
    "A+": 5,
    "A": 6,
    "A-": 7,
    "BBB+": 8,
    "BBB": 9,
    "BBB-": 10,
    "BB+": 11,
    "BB": 12,
    "BB-": 13,
    "B+": 14,
    "B": 15,
    "B-": 16,
    "CCC+": 17,
    "CCC": 18,
    "CCC-": 19,
    "CC": 20,
    "C": 21,
    "D": 22,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-scores", type=Path, default=PREDICTION_SCORES_PATH)
    parser.add_argument("--feature-master", type=Path, default=FEATURE_MASTER_PATH)
    parser.add_argument("--target-label-reference", type=Path, default=TARGET_LABEL_REFERENCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def normalize_stock_code(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )


def read_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    frame = frame.copy()
    frame["stock_code"] = normalize_stock_code(frame["stock_code"])
    for column in ["fiscal_year", "eval_year", "is_speculative", "pred_label_tuned"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in [
        "prob_speculative",
        "threshold",
        "prob_speculative_overwarning_filter",
        "threshold_overwarning_filter",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def read_feature_master(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    frame = frame.copy()
    frame["stock_code"] = normalize_stock_code(frame["stock_code"])
    for column in ["fiscal_year", "eval_year", *FEATURE_COLUMNS]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.loc[:, [*KEY_COLUMNS, *[column for column in FEATURE_COLUMNS if column in frame]]]


def read_label_reference(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[*KEY_COLUMNS, "credit_rating", "credit_rating_rank"])
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    frame = frame.copy()
    frame["stock_code"] = normalize_stock_code(frame["stock_code"])
    for column in ["fiscal_year", "eval_year"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "credit_rating_rank" not in frame.columns and "credit_rating" in frame.columns:
        frame["credit_rating_rank"] = frame["credit_rating"].map(RATING_RANK)
    if "credit_rating_rank" in frame.columns:
        frame["credit_rating_rank"] = pd.to_numeric(frame["credit_rating_rank"], errors="coerce")
    keep_columns = [*KEY_COLUMNS, "credit_rating", "credit_rating_rank", "rating_agency_group"]
    return frame.loc[:, [column for column in keep_columns if column in frame.columns]]


def build_frame(
    scores: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    test = scores.loc[scores["split"].astype(str).eq("test")].copy()
    test = test.merge(features, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if not labels.empty:
        test = test.merge(labels, on=KEY_COLUMNS, how="left", validate="many_to_one")
    if "credit_rating_rank" not in test.columns:
        test["credit_rating_rank"] = np.nan
    if "credit_rating" not in test.columns:
        test["credit_rating"] = ""
    test["rating_boundary_group"] = rating_boundary_group(test["credit_rating_rank"])
    test["stage1_risk"] = stage1_risk(test)
    test["actual_speculative"] = test["is_speculative"].astype(int).eq(1)
    add_financial_signal_counts(test)
    return test


def rating_boundary_group(rank: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(rank, errors="coerce")
    return pd.Series(
        np.select(
            [
                numeric.le(7),
                numeric.between(8, 10, inclusive="both"),
                numeric.between(11, 13, inclusive="both"),
                numeric.ge(14),
            ],
            [
                "upper_investment_A_or_above",
                "near_investment_BBB_plus_to_BBB_minus",
                "near_speculative_BB_plus_to_BB_minus",
                "deep_speculative_B_plus_or_lower",
            ],
            default="missing_rating",
        ),
        index=rank.index,
    )


def stage1_risk(frame: pd.DataFrame) -> pd.Series:
    if "pred_label_tuned" in frame.columns:
        return frame["pred_label_tuned"].fillna(0).astype(int).eq(1)
    return frame["prob_speculative"].ge(frame["threshold"])


def add_financial_signal_counts(frame: pd.DataFrame) -> None:
    support_flags = {
        "support_current_ratio_ge_1_2": at_least(frame, "current_ratio", 1.2),
        "support_cash_ratio_ge_0_15": at_least(frame, "cash_ratio", 0.15),
        "support_equity_ratio_ge_0_40": at_least(frame, "equity_ratio", 0.40),
        "support_debt_ratio_le_1_50": at_most(frame, "debt_ratio", 1.50),
        "support_total_borrowings_le_0_50": at_most(frame, "total_borrowings_ratio", 0.50),
        "support_capital_impairment_le_0": at_most(frame, "capital_impairment_ratio", 0.0),
        "support_icr_ge_1": at_least(frame, "interest_coverage_ratio", 1.0),
        "support_net_margin_ge_0": at_least(frame, "net_margin", 0.0),
        "support_ocf_to_sales_ge_0": at_least(frame, "ocf_to_sales", 0.0),
        "support_no_2y_operating_loss": flag_is_false(frame, "is_2y_consecutive_operating_loss"),
        "support_no_2y_ocf_deficit": flag_is_false(frame, "is_2y_consecutive_ocf_deficit"),
        "support_no_icr_under_1": flag_is_false(frame, "icr_under_1"),
        "support_short_term_borrowings_le_0_80": at_most(
            frame,
            "short_term_borrowings_share",
            0.80,
        ),
        "support_dividend_payer": flag_is_true(frame, "dividend_payer"),
    }
    blocker_flags = {
        "block_2y_operating_loss": flag_is_true(frame, "is_2y_consecutive_operating_loss"),
        "block_2y_ocf_deficit": flag_is_true(frame, "is_2y_consecutive_ocf_deficit"),
        "block_icr_under_1": flag_is_true(frame, "icr_under_1"),
        "block_net_margin_lt_minus_0_10": below(frame, "net_margin", -0.10),
        "block_equity_ratio_lt_0_25": below(frame, "equity_ratio", 0.25),
        "block_capital_impairment_gt_0": above(frame, "capital_impairment_ratio", 0.0),
        "block_total_borrowings_gt_0_65": above(frame, "total_borrowings_ratio", 0.65),
        "block_short_term_borrowings_gt_0_90": above(frame, "short_term_borrowings_share", 0.90),
        "block_current_ratio_lt_0_80": below(frame, "current_ratio", 0.80),
        "block_cash_ratio_lt_0_05": below(frame, "cash_ratio", 0.05),
        "block_ocf_to_sales_lt_minus_0_20": below(frame, "ocf_to_sales", -0.20),
    }
    for column, values in {**support_flags, **blocker_flags}.items():
        frame[column] = values
    v1_blockers = [
        "block_2y_operating_loss",
        "block_2y_ocf_deficit",
        "block_icr_under_1",
        "block_net_margin_lt_minus_0_10",
        "block_equity_ratio_lt_0_25",
        "block_capital_impairment_gt_0",
        "block_total_borrowings_gt_0_65",
        "block_short_term_borrowings_gt_0_90",
    ]
    extended_blockers = [
        *v1_blockers,
        "block_current_ratio_lt_0_80",
        "block_cash_ratio_lt_0_05",
        "block_ocf_to_sales_lt_minus_0_20",
    ]
    support_columns = [column for column in support_flags if column != "support_dividend_payer"]
    frame["financial_support_count"] = frame.loc[:, support_columns].sum(axis=1).astype(int)
    frame["financial_support_count_with_dividend"] = (
        frame.loc[:, list(support_flags)].sum(axis=1).astype(int)
    )
    frame["financial_blocker_count"] = frame.loc[:, v1_blockers].sum(axis=1).astype(int)
    frame["financial_blocker_count_extended"] = (
        frame.loc[:, extended_blockers].sum(axis=1).astype(int)
    )


def at_least(frame: pd.DataFrame, column: str, threshold: float) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce").ge(threshold).fillna(False)


def at_most(frame: pd.DataFrame, column: str, threshold: float) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce").le(threshold).fillna(False)


def above(frame: pd.DataFrame, column: str, threshold: float) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce").gt(threshold).fillna(False)


def below(frame: pd.DataFrame, column: str, threshold: float) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce").lt(threshold).fillna(False)


def flag_is_true(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce").ge(0.5).fillna(False)


def flag_is_false(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce").lt(0.5).fillna(False)


def policy_definitions() -> list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        (
            "dashboard_composite_filter",
            "현재 대시보드 조합형 재무 스트레스 필터가 정상으로 본 stage1 위험 기업",
            lambda f: f.get("stage2_overwarning_filter_candidate", False).astype(bool),
        ),
        (
            "financial_resilience_v1_support8_block0",
            "재무 방어 신호 8개 이상, 강한 차단 신호 0개",
            lambda f: f["financial_support_count"].ge(8) & f["financial_blocker_count"].eq(0),
        ),
        (
            "financial_resilience_strict_support9_block0",
            "재무 방어 신호 9개 이상, 강한 차단 신호 0개",
            lambda f: f["financial_support_count"].ge(9) & f["financial_blocker_count"].eq(0),
        ),
        (
            "financial_resilience_lax_support7_block0",
            "재무 방어 신호 7개 이상, 강한 차단 신호 0개",
            lambda f: f["financial_support_count"].ge(7) & f["financial_blocker_count"].eq(0),
        ),
        (
            "financial_resilience_v1_support8_block0_prob_lt_0_85",
            "재무 방어 신호 8개 이상, 차단 0개, 1차 확률 85% 미만",
            lambda f: f["financial_support_count"].ge(8)
            & f["financial_blocker_count"].eq(0)
            & f["prob_speculative"].lt(0.85),
        ),
        (
            "financial_resilience_v1_support8_block0_prob_lt_0_90",
            "재무 방어 신호 8개 이상, 차단 0개, 1차 확률 90% 미만",
            lambda f: f["financial_support_count"].ge(8)
            & f["financial_blocker_count"].eq(0)
            & f["prob_speculative"].lt(0.90),
        ),
        (
            "extended_blocker_support8_block0",
            "재무 방어 신호 8개 이상, 확장 차단 신호 0개",
            lambda f: f["financial_support_count"].ge(8)
            & f["financial_blocker_count_extended"].eq(0),
        ),
        (
            "liquidity_capital_profit_core",
            "유동성·현금·자본·이자보상·순이익률 핵심 방어 조건 충족",
            liquidity_capital_profit_core,
        ),
        (
            "liquidity_capital_profit_core_prob_lt_0_85",
            "유동성·자본·이익 방어 조건 충족, 1차 확률 85% 미만",
            lambda f: liquidity_capital_profit_core(f) & f["prob_speculative"].lt(0.85),
        ),
        (
            "liquidity_capital_profit_core_prob_lt_0_80",
            "유동성·자본·이익 방어 조건 충족, 1차 확률 80% 미만",
            lambda f: liquidity_capital_profit_core(f) & f["prob_speculative"].lt(0.80),
        ),
        (
            "liquidity_capital_profit_core_plus_support8",
            "유동성·자본·이익 방어 조건과 재무 방어 신호 8개 이상 동시 충족",
            lambda f: liquidity_capital_profit_core(f) & f["financial_support_count"].ge(8),
        ),
        (
            "liquidity_capital_profit_core_plus_support8_prob_lt_0_85",
            "유동성·자본·이익 방어 조건, 재무 방어 8개 이상, 1차 확률 85% 미만",
            lambda f: liquidity_capital_profit_core(f)
            & f["financial_support_count"].ge(8)
            & f["prob_speculative"].lt(0.85),
        ),
        (
            "liquidity_capital_profit_core_plus_ocf_not_deep_negative",
            "유동성·자본·이익 방어 조건과 OCF/매출액 -5% 이상",
            lambda f: liquidity_capital_profit_core(f) & at_least(f, "ocf_to_sales", -0.05),
        ),
        (
            "cashflow_quality_core",
            "이자보상·OCF·자본비율 방어 조건 충족",
            lambda f: at_least(f, "interest_coverage_ratio", 1.0)
            & at_least(f, "ocf_to_sales", 0.0)
            & at_least(f, "ocf_to_total_liabilities", 0.0)
            & at_least(f, "equity_ratio", 0.40)
            & f["financial_blocker_count"].eq(0),
        ),
        (
            "low_borrowing_buffer",
            "낮은 차입금 비중과 유동성 버퍼 동시 충족",
            lambda f: at_most(f, "total_borrowings_ratio", 0.35)
            & at_most(f, "short_term_borrowings_share", 0.80)
            & at_least(f, "current_ratio", 1.2)
            & at_least(f, "cash_ratio", 0.15)
            & f["financial_blocker_count"].eq(0),
        ),
        (
            "dividend_balance_sheet_buffer",
            "배당 이력과 자본/부채/현금 방어 조건 충족",
            lambda f: flag_is_true(f, "dividend_payer")
            & at_least(f, "equity_ratio", 0.40)
            & at_most(f, "debt_ratio", 1.50)
            & at_least(f, "cash_ratio", 0.10)
            & f["financial_blocker_count"].eq(0),
        ),
        (
            "dashboard_or_financial_resilience_v1",
            "현재 대시보드 필터 또는 재무 방어 v1 중 하나라도 충족",
            lambda f: f.get("stage2_overwarning_filter_candidate", False).astype(bool)
            | (f["financial_support_count"].ge(8) & f["financial_blocker_count"].eq(0)),
        ),
    ]


def liquidity_capital_profit_core(frame: pd.DataFrame) -> pd.Series:
    return (
        at_least(frame, "current_ratio", 1.2)
        & at_least(frame, "cash_ratio", 0.15)
        & at_least(frame, "equity_ratio", 0.40)
        & at_most(frame, "debt_ratio", 1.50)
        & at_least(frame, "interest_coverage_ratio", 1.0)
        & at_least(frame, "net_margin", 0.0)
        & frame["financial_blocker_count"].eq(0)
    )


def summarize_policy(
    frame: pd.DataFrame,
    *,
    policy: str,
    description: str,
    candidate: pd.Series,
) -> dict[str, Any]:
    stage1 = frame["stage1_risk"].astype(bool)
    y_true = frame["actual_speculative"].astype(bool)
    candidate = candidate.fillna(False).astype(bool) & stage1
    stage1_risk_rows = int(stage1.sum())
    fp_total = int((stage1 & ~y_true).sum())
    tp_total = int((stage1 & y_true).sum())
    actual_positive_total = int(y_true.sum())
    candidate_count = int(candidate.sum())
    fp_mitigated = int((candidate & ~y_true).sum())
    tp_softened = int((candidate & y_true).sum())
    reject_after = stage1 & ~candidate
    tp_after = int((reject_after & y_true).sum())
    precision_before = safe_div(tp_total, stage1_risk_rows)
    recall_before = safe_div(tp_total, actual_positive_total)
    precision_after = safe_div(tp_after, int(reject_after.sum()))
    recall_after = safe_div(tp_after, actual_positive_total)
    return {
        "policy": policy,
        "description": description,
        "test_rows": len(frame),
        "stage1_risk_rows": stage1_risk_rows,
        "stage1_fp_total": fp_total,
        "stage1_tp_total": tp_total,
        "candidate_count": candidate_count,
        "fp_mitigated": fp_mitigated,
        "tp_softened": tp_softened,
        "fp_remaining": fp_total - fp_mitigated,
        "tp_preserved": tp_total - tp_softened,
        "candidate_precision_for_fp": safe_div(fp_mitigated, candidate_count),
        "fp_mitigation_rate": safe_div(fp_mitigated, fp_total),
        "tp_softening_rate": safe_div(tp_softened, tp_total),
        "net_fp_minus_tp": fp_mitigated - tp_softened,
        "weighted_net_fp_minus_2tp": fp_mitigated - (2 * tp_softened),
        "stage1_reject_precision_before": precision_before,
        "reject_precision_after_if_hold_is_not_reject": precision_after,
        "stage1_reject_recall_before": recall_before,
        "reject_recall_after_if_hold_is_not_reject": recall_after,
        "hold_or_reject_recall_after": recall_before,
    }


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_case_flags(frame: pd.DataFrame, policies: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        *KEY_COLUMNS,
        "split",
        "is_speculative",
        "credit_rating",
        "rating_boundary_group",
        "prob_speculative",
        "threshold",
        "stage1_risk",
        "financial_support_count",
        "financial_blocker_count",
        "financial_blocker_count_extended",
        *FEATURE_COLUMNS,
    ]
    output = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    for policy in policies:
        output[f"candidate__{policy['policy']}"] = policy["candidate"].astype(bool)
    return output


def build_report(metrics: pd.DataFrame, summary: dict[str, Any]) -> str:
    top = metrics.sort_values(
        ["weighted_net_fp_minus_2tp", "fp_mitigation_rate", "candidate_precision_for_fp"],
        ascending=[False, False, False],
    ).head(8)
    table_columns = [
        "policy",
        "candidate_count",
        "fp_mitigated",
        "tp_softened",
        "fp_mitigation_rate",
        "tp_softening_rate",
        "candidate_precision_for_fp",
        "weighted_net_fp_minus_2tp",
    ]
    table = markdown_table(top.loc[:, table_columns])
    return f"""# Stage 2 Over-Warning Filter Candidate Experiments

- Generated at: `{summary["generated_at_utc"]}`
- Test rows: `{summary["test_rows"]}`
- Stage 1 risk rows: `{summary["stage1_risk_rows"]}`
- Stage 1 FP / TP among risk rows: `{summary["stage1_fp_total"]}` / `{summary["stage1_tp_total"]}`

## Top Candidate Policies

{table}

## How To Read

- `fp_mitigated`: 실제 투자적격인데 1차 모델이 부적격으로 본 FP를 위원회 `보류` 후보로 낮춘 수입니다.
- `tp_softened`: 실제 투기등급인데 위원회가 `보류`로 낮출 위험이 있는 수입니다. 낮을수록 좋습니다.
- `candidate_precision_for_fp`: 보류 완화 후보 중 실제 FP 비율입니다.
- `hold_or_reject_recall_after`: `보류`도 2차 검토 대상으로 보면 Recall은 유지된다는 가정의 값입니다.

이 실험은 외부근거가 없거나 강한 악재가 없다는 전제의 오프라인 후보 비교입니다.
실제 위원회 판단에서는 `veto`, 직접 관련 외부 악재, hidden-tail-risk 조건이 우선합니다.
"""


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    scores = read_scores(args.prediction_scores)
    features = read_feature_master(args.feature_master)
    labels = read_label_reference(args.target_label_reference)
    frame = build_frame(scores, features, labels)

    policy_results: list[dict[str, Any]] = []
    policy_frames: list[dict[str, Any]] = []
    for name, description, builder in policy_definitions():
        candidate = builder(frame).fillna(False).astype(bool) & frame["stage1_risk"].astype(bool)
        policy_results.append(
            summarize_policy(frame, policy=name, description=description, candidate=candidate)
        )
        policy_frames.append({"policy": name, "description": description, "candidate": candidate})

    metrics = pd.DataFrame(policy_results).sort_values(
        ["weighted_net_fp_minus_2tp", "fp_mitigation_rate", "candidate_precision_for_fp"],
        ascending=[False, False, False],
    )
    cases = build_case_flags(frame, policy_frames)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "test_rows": len(frame),
        "stage1_risk_rows": int(frame["stage1_risk"].sum()),
        "stage1_fp_total": int((frame["stage1_risk"] & ~frame["actual_speculative"]).sum()),
        "stage1_tp_total": int((frame["stage1_risk"] & frame["actual_speculative"]).sum()),
        "best_policy_by_weighted_net": str(metrics.iloc[0]["policy"]) if not metrics.empty else "",
        "paths": {
            "metrics": "data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_overwarning_filter_candidate_experiments.csv",
            "cases": "data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_overwarning_filter_candidate_cases.csv",
            "summary": "data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_overwarning_filter_candidate_summary.json",
            "report": "data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_overwarning_filter_candidate_report.md",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "stage2_overwarning_filter_candidate_experiments.csv"
    cases_path = args.output_dir / "stage2_overwarning_filter_candidate_cases.csv"
    summary_path = args.output_dir / "stage2_overwarning_filter_candidate_summary.json"
    report_path = args.output_dir / "stage2_overwarning_filter_candidate_report.md"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    cases.to_csv(cases_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(build_report(metrics, summary), encoding="utf-8")

    print(f"[Saved] {metrics_path}")
    print(f"[Saved] {cases_path}")
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
