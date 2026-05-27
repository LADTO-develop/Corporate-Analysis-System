from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cas.modeling.stage1_xgboost import (  # noqa: E402
    DEFAULT_ROLLING_EVAL_YEARS,
    DEFAULT_STAGE1_RANDOM_STATE,
    DEFAULT_STAGE1_RECALL_FLOOR,
    classification_metrics,
    evaluate_calibrated_stage1_split,
    read_stage1_feature_columns,
    read_stage1_master,
    train_stage1_xgboost,
)

INPUT_DIR = ROOT / "data" / "input" / "credit_46_features"
MASTER_PATH = INPUT_DIR / "feature_46_master.csv"
FEATURE_LIST_PATH = INPUT_DIR / "feature_46_list.json"
RAW_TS2000_PATH = ROOT / "data" / "raw" / "ts2000" / "TS2000_Credit_Model_Dataset_Model_V1.csv"
OUTPUT_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_46_xgboost" / "diagnostics"

RANDOM_STATE = DEFAULT_STAGE1_RANDOM_STATE
RECALL_FLOOR = DEFAULT_STAGE1_RECALL_FLOOR
ROLLING_EVAL_YEARS = DEFAULT_ROLLING_EVAL_YEARS

MERGE_KEYS = ["market", "stock_code", "fiscal_year"]
ID_COLUMNS = [
    "market",
    "stock_code",
    "corp_name",
    "fiscal_year",
    "eval_year",
    "firm_size_group",
    "industry_macro_category",
]
FOCUS_SEGMENTS = [
    ("overall", "all", None, None),
    ("market", "KOSDAQ", "market", "KOSDAQ"),
    ("market", "KOSPI", "market", "KOSPI"),
    ("industry", "manufacturing", "industry_macro_category", "manufacturing"),
    ("industry", "it_services", "industry_macro_category", "it_services"),
]

RAW_MACRO_COLUMNS = [
    "market_spread_diff",
    "spec_spread_diff",
    "base_rate_diff",
    "treasury_3y_diff",
    "usd_krw_diff",
]
MACRO_SHOCK_COLUMNS = [
    "macro_market_spread_widening",
    "macro_spec_spread_widening",
    "macro_base_rate_hike",
    "macro_treasury_3y_hike",
    "macro_usd_krw_up",
]
VULNERABILITY_COLUMNS = [
    "macro_short_term_borrowings_pressure",
    "macro_total_borrowings_pressure",
    "macro_interest_coverage_pressure",
    "macro_cash_ratio_inverse",
]
INTERACTION_COLUMNS = [
    "macro_market_spread_widening_x_short_term_borrowings_share",
    "macro_market_spread_widening_x_total_borrowings_ratio",
    "macro_market_spread_widening_x_interest_coverage_pressure",
    "macro_market_spread_widening_x_cash_ratio_inverse",
    "macro_spec_spread_widening_x_short_term_borrowings_share",
    "macro_spec_spread_widening_x_total_borrowings_ratio",
    "macro_base_rate_hike_x_short_term_borrowings_share",
    "macro_base_rate_hike_x_interest_coverage_pressure",
    "macro_treasury_3y_hike_x_total_borrowings_ratio",
    "macro_usd_krw_up_x_cash_ratio_inverse",
]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    note: str
    added_columns: tuple[str, ...]


VARIANTS = [
    VariantSpec(
        name="baseline_46_native",
        note="공식 46개 입력셋 baseline",
        added_columns=(),
    ),
    VariantSpec(
        name="macro_regime_51_raw",
        note="금리, 스프레드, 환율 변화량 5개 raw macro feature 추가",
        added_columns=tuple(RAW_MACRO_COLUMNS),
    ),
    VariantSpec(
        name="macro_shock_components_51",
        note="악화 방향으로만 절단한 macro shock component 5개 추가",
        added_columns=tuple(MACRO_SHOCK_COLUMNS),
    ),
    VariantSpec(
        name="macro_vulnerability_interactions_56",
        note="macro shock과 차입, 이자보상, 현금 유동성 취약도의 interaction 10개 추가",
        added_columns=tuple(INTERACTION_COLUMNS),
    ),
    VariantSpec(
        name="macro_shock_plus_interactions_61",
        note="macro shock component 5개와 취약도 interaction 10개를 함께 추가",
        added_columns=tuple([*MACRO_SHOCK_COLUMNS, *INTERACTION_COLUMNS]),
    ),
    VariantSpec(
        name="macro_regime_plus_interactions_61",
        note="raw macro 변화량 5개와 취약도 interaction 10개를 함께 추가",
        added_columns=tuple([*RAW_MACRO_COLUMNS, *INTERACTION_COLUMNS]),
    ),
    VariantSpec(
        name="macro_full_pressure_70",
        note="raw macro, shock component, 취약도 proxy, interaction을 모두 추가",
        added_columns=tuple(
            [
                *RAW_MACRO_COLUMNS,
                *MACRO_SHOCK_COLUMNS,
                *VULNERABILITY_COLUMNS,
                *INTERACTION_COLUMNS,
            ]
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run macro regime and macro-vulnerability interaction experiments "
            "for the official credit_46_features XGBoost model."
        )
    )
    parser.add_argument("--master-path", type=Path, default=MASTER_PATH)
    parser.add_argument("--feature-list-path", type=Path, default=FEATURE_LIST_PATH)
    parser.add_argument("--raw-ts2000-path", type=Path, default=RAW_TS2000_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--eval-years", type=int, nargs="+", default=ROLLING_EVAL_YEARS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def read_master(path: Path) -> pd.DataFrame:
    return read_stage1_master(path, duplicate_keys=MERGE_KEYS)


def read_feature_columns(path: Path, master: pd.DataFrame) -> list[str]:
    return read_stage1_feature_columns(path, master)


def read_raw_macro_sources(path: Path) -> pd.DataFrame:
    columns = [*MERGE_KEYS, *RAW_MACRO_COLUMNS]
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str}, usecols=columns)
    raw["stock_code"] = raw["stock_code"].astype("string").str.zfill(6)
    duplicates = int(raw.duplicated(MERGE_KEYS).sum())
    if duplicates:
        raw = raw.sort_values(MERGE_KEYS).drop_duplicates(MERGE_KEYS, keep="last")
    for column in RAW_MACRO_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    return raw


def positive_part(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").clip(lower=0)


def add_macro_features(master: pd.DataFrame, raw_ts2000_path: Path) -> pd.DataFrame:
    raw = read_raw_macro_sources(raw_ts2000_path)
    output = master.merge(raw, on=MERGE_KEYS, how="left", validate="one_to_one")
    output = output.replace([np.inf, -np.inf], np.nan)

    output["macro_market_spread_widening"] = positive_part(output, "market_spread_diff")
    output["macro_spec_spread_widening"] = positive_part(output, "spec_spread_diff")
    output["macro_base_rate_hike"] = positive_part(output, "base_rate_diff")
    output["macro_treasury_3y_hike"] = positive_part(output, "treasury_3y_diff")
    output["macro_usd_krw_up"] = positive_part(output, "usd_krw_diff")

    short_term_share = pd.to_numeric(output["short_term_borrowings_share"], errors="coerce")
    total_borrowings = pd.to_numeric(output["total_borrowings_ratio"], errors="coerce")
    interest_coverage = pd.to_numeric(output["interest_coverage_ratio"], errors="coerce")
    cash_ratio = pd.to_numeric(output["cash_ratio"], errors="coerce")

    output["macro_short_term_borrowings_pressure"] = short_term_share.clip(lower=0, upper=1)
    output["macro_total_borrowings_pressure"] = total_borrowings.clip(lower=0, upper=2)
    output["macro_interest_coverage_pressure"] = (
        1.0 - interest_coverage.clip(lower=-4, upper=1)
    ).clip(
        lower=0,
        upper=5,
    )
    output["macro_cash_ratio_inverse"] = 1.0 / (1.0 + cash_ratio.clip(lower=0, upper=10))

    output["macro_market_spread_widening_x_short_term_borrowings_share"] = (
        output["macro_market_spread_widening"] * output["macro_short_term_borrowings_pressure"]
    )
    output["macro_market_spread_widening_x_total_borrowings_ratio"] = (
        output["macro_market_spread_widening"] * output["macro_total_borrowings_pressure"]
    )
    output["macro_market_spread_widening_x_interest_coverage_pressure"] = (
        output["macro_market_spread_widening"] * output["macro_interest_coverage_pressure"]
    )
    output["macro_market_spread_widening_x_cash_ratio_inverse"] = (
        output["macro_market_spread_widening"] * output["macro_cash_ratio_inverse"]
    )
    output["macro_spec_spread_widening_x_short_term_borrowings_share"] = (
        output["macro_spec_spread_widening"] * output["macro_short_term_borrowings_pressure"]
    )
    output["macro_spec_spread_widening_x_total_borrowings_ratio"] = (
        output["macro_spec_spread_widening"] * output["macro_total_borrowings_pressure"]
    )
    output["macro_base_rate_hike_x_short_term_borrowings_share"] = (
        output["macro_base_rate_hike"] * output["macro_short_term_borrowings_pressure"]
    )
    output["macro_base_rate_hike_x_interest_coverage_pressure"] = (
        output["macro_base_rate_hike"] * output["macro_interest_coverage_pressure"]
    )
    output["macro_treasury_3y_hike_x_total_borrowings_ratio"] = (
        output["macro_treasury_3y_hike"] * output["macro_total_borrowings_pressure"]
    )
    output["macro_usd_krw_up_x_cash_ratio_inverse"] = (
        output["macro_usd_krw_up"] * output["macro_cash_ratio_inverse"]
    )
    return output


def variant_feature_columns(base_columns: list[str], spec: VariantSpec) -> list[str]:
    return [*base_columns, *[column for column in spec.added_columns if column not in base_columns]]


def evaluate_model(
    *,
    model: object,
    policy: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics, eval_prob = evaluate_calibrated_stage1_split(
        model=model,
        policy=policy,
        evaluation=evaluation,
        columns=columns,
    )
    threshold = float(metrics["threshold_tuned"])
    predictions = eval_prob >= threshold
    scored = evaluation.loc[:, [*ID_COLUMNS, "is_speculative"]].copy()
    scored["prob_speculative"] = eval_prob
    scored["prediction"] = predictions.astype(int)
    return metrics, scored


def evaluate_variant_fold(
    *,
    frame: pd.DataFrame,
    base_columns: list[str],
    spec: VariantSpec,
    eval_year: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    policy_year = eval_year - 1
    train = frame.loc[frame["fiscal_year"] < policy_year].copy()
    policy = frame.loc[frame["fiscal_year"] == policy_year].copy()
    evaluation = frame.loc[frame["fiscal_year"] == eval_year].copy()
    if train.empty or policy.empty or evaluation.empty:
        raise ValueError(
            f"Empty rolling split for eval_year={eval_year}: "
            f"train={len(train)}, policy={len(policy)}, evaluation={len(evaluation)}"
        )
    columns = variant_feature_columns(base_columns, spec)
    model = train_stage1_xgboost(train=train, policy=policy, columns=columns, seed=seed)
    metrics, scored = evaluate_model(
        model=model, policy=policy, evaluation=evaluation, columns=columns
    )
    return (
        {
            "variant": spec.name,
            "note": spec.note,
            "added_features": ", ".join(spec.added_columns),
            "added_feature_count": len(spec.added_columns),
            "feature_count": len(columns),
            "eval_year": eval_year,
            "policy_year": policy_year,
            "train_rows": len(train),
            "policy_rows": len(policy),
            "eval_rows": len(evaluation),
            "eval_positive_rate": float(evaluation["is_speculative"].mean()),
            "best_iteration": getattr(model, "best_iteration", None),
            **metrics,
        },
        scored.assign(variant=spec.name, eval_year=eval_year),
    )


def evaluate_variant_final_test(
    *,
    frame: pd.DataFrame,
    base_columns: list[str],
    spec: VariantSpec,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train = frame.loc[frame["fiscal_year"] <= 2021].copy()
    policy = frame.loc[frame["fiscal_year"] == 2022].copy()
    evaluation = frame.loc[frame["fiscal_year"] >= 2023].copy()
    columns = variant_feature_columns(base_columns, spec)
    model = train_stage1_xgboost(train=train, policy=policy, columns=columns, seed=seed)
    metrics, scored = evaluate_model(
        model=model, policy=policy, evaluation=evaluation, columns=columns
    )
    return (
        {
            "variant": spec.name,
            "note": spec.note,
            "added_features": ", ".join(spec.added_columns),
            "added_feature_count": len(spec.added_columns),
            "feature_count": len(columns),
            "train_rows": len(train),
            "policy_rows": len(policy),
            "eval_rows": len(evaluation),
            "eval_positive_rate": float(evaluation["is_speculative"].mean()),
            "best_iteration": getattr(model, "best_iteration", None),
            **metrics,
        },
        scored.assign(variant=spec.name, eval_year="final_test"),
    )


def run_experiment(
    *,
    frame: pd.DataFrame,
    base_columns: list[str],
    eval_years: list[int],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    for spec in VARIANTS:
        for eval_year in eval_years:
            row, scored = evaluate_variant_fold(
                frame=frame,
                base_columns=base_columns,
                spec=spec,
                eval_year=eval_year,
                seed=seed,
            )
            fold_rows.append(row)
            scored_frames.append(scored)
        final_row, final_scored = evaluate_variant_final_test(
            frame=frame,
            base_columns=base_columns,
            spec=spec,
            seed=seed,
        )
        final_rows.append(final_row)
        scored_frames.append(final_scored)

    fold_metrics = pd.DataFrame(fold_rows)
    final_metrics = pd.DataFrame(final_rows)
    scored_all = pd.concat(scored_frames, ignore_index=True)
    summary = merge_final_metrics(summarize_rolling(fold_metrics), final_metrics)
    segments = build_segment_metrics(scored_all)
    return fold_metrics, final_metrics, summary, segments


def summarize_rolling(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "eval_pr_auc",
        "eval_roc_auc",
        "eval_brier",
        "eval_logloss",
        "eval_precision_at_threshold",
        "eval_recall_at_threshold",
        "eval_f1_at_threshold",
        "eval_false_positive_at_threshold",
        "eval_false_negative_at_threshold",
    ]
    rows: list[dict[str, Any]] = []
    for variant, group in fold_metrics.groupby("variant", sort=False):
        row: dict[str, Any] = {
            "variant": variant,
            "note": group["note"].iloc[0],
            "added_features": group["added_features"].iloc[0],
            "added_feature_count": int(group["added_feature_count"].iloc[0]),
            "feature_count": int(group["feature_count"].iloc[0]),
            "folds": len(group),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
            row[f"{column}_min"] = float(group[column].min())
            row[f"{column}_max"] = float(group[column].max())
        row["total_false_positive"] = int(group["eval_false_positive_at_threshold"].sum())
        row["total_false_negative"] = int(group["eval_false_negative_at_threshold"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def merge_final_metrics(rolling_summary: pd.DataFrame, final_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "variant",
        "threshold_tuned",
        "eval_pr_auc",
        "eval_roc_auc",
        "eval_brier",
        "eval_logloss",
        "eval_precision_at_threshold",
        "eval_recall_at_threshold",
        "eval_f1_at_threshold",
        "eval_false_positive_at_threshold",
        "eval_false_negative_at_threshold",
    ]
    final = final_metrics.loc[:, columns].rename(
        columns={column: f"final_{column}" for column in columns if column != "variant"}
    )
    summary = rolling_summary.merge(final, on="variant", how="left")
    baseline = summary.loc[summary["variant"].eq("baseline_46_native")].iloc[0]
    summary["rolling_f1_delta_vs_baseline"] = (
        summary["eval_f1_at_threshold_mean"] - baseline["eval_f1_at_threshold_mean"]
    )
    summary["rolling_pr_auc_delta_vs_baseline"] = (
        summary["eval_pr_auc_mean"] - baseline["eval_pr_auc_mean"]
    )
    summary["final_f1_delta_vs_baseline"] = (
        summary["final_eval_f1_at_threshold"] - baseline["final_eval_f1_at_threshold"]
    )
    summary["final_pr_auc_delta_vs_baseline"] = (
        summary["final_eval_pr_auc"] - baseline["final_eval_pr_auc"]
    )
    summary["final_fp_delta_vs_baseline"] = (
        summary["final_eval_false_positive_at_threshold"]
        - baseline["final_eval_false_positive_at_threshold"]
    )
    summary["final_fn_delta_vs_baseline"] = (
        summary["final_eval_false_negative_at_threshold"]
        - baseline["final_eval_false_negative_at_threshold"]
    )
    return sort_summary(summary)


def build_segment_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, eval_year), group in scored.groupby(["variant", "eval_year"], sort=False):
        for dimension, segment, column, value in FOCUS_SEGMENTS:
            segment_frame = (
                group if column is None else group.loc[group[column].astype(str).eq(str(value))]
            )
            if segment_frame.empty:
                continue
            y_true = segment_frame["is_speculative"].astype(int)
            predictions = segment_frame["prediction"].astype(int).to_numpy()
            metric = classification_metrics(y_true, predictions)
            rows.append(
                {
                    "variant": variant,
                    "eval_year": eval_year,
                    "dimension": dimension,
                    "segment": segment,
                    "rows": len(segment_frame),
                    "positives": int((y_true == 1).sum()),
                    "negatives": int((y_true == 0).sum()),
                    **metric,
                }
            )
    return pd.DataFrame(rows)


def sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        [
            "rolling_f1_delta_vs_baseline",
            "rolling_pr_auc_delta_vs_baseline",
            "final_f1_delta_vs_baseline",
            "final_fn_delta_vs_baseline",
        ],
        ascending=[False, False, False, True],
    )


def best_candidate(summary: pd.DataFrame) -> pd.Series:
    return sort_summary(summary).iloc[0]


def format_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def format_signed(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):+.4f}"


def format_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return str(int(value))


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(label for label, _, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for item in frame.to_dict(orient="records"):
        values = []
        for _, column, kind in columns:
            value = item.get(column)
            if kind == "metric":
                values.append(format_metric(value))
            elif kind == "signed":
                values.append(format_signed(value))
            elif kind == "int":
                values.append(format_int(value))
            else:
                values.append(str(value) if value is not None else "")
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def recommendation_text(best: pd.Series) -> str:
    if str(best["variant"]) == "baseline_46_native":
        return (
            "- Rolling OOT 기준에서도 현재 46-feature baseline이 가장 안정적입니다. "
            "macro interaction 후보는 공식 모델에 바로 반영하지 않는 편이 안전합니다."
        )
    rolling_f1_delta = float(best["rolling_f1_delta_vs_baseline"])
    final_f1_delta = float(best["final_f1_delta_vs_baseline"])
    final_fn_delta = int(best["final_fn_delta_vs_baseline"])
    if rolling_f1_delta >= 0.005 and final_f1_delta >= 0 and final_fn_delta <= 0:
        return (
            "- Rolling OOT와 Final Test가 모두 받쳐주는 후보입니다. "
            "별도 artifact 생성 후 승격 후보로 검토할 수 있습니다."
        )
    if rolling_f1_delta > 0 and final_fn_delta > 0:
        return (
            "- Rolling OOT에서는 좋아졌지만 Final Test에서 FN이 증가했습니다. "
            "Stage2 보조 신호나 추가 후보 실험으로만 보관하는 편이 안전합니다."
        )
    return (
        "- 개선 폭이 작거나 FP/FN trade-off가 있습니다. "
        "공식 모델 즉시 교체보다는 후보 기록으로 보관하는 편이 안전합니다."
    )


def build_report(
    *,
    fold_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    segments: pd.DataFrame,
    eval_years: list[int],
) -> str:
    baseline = summary.loc[summary["variant"].eq("baseline_46_native")].iloc[0]
    best = best_candidate(summary)
    baseline_rows = fold_metrics.loc[fold_metrics["variant"].eq("baseline_46_native")]
    best_rows = fold_metrics.loc[fold_metrics["variant"].eq(str(best["variant"]))]
    final_ranked = final_metrics.sort_values(
        ["eval_f1_at_threshold", "eval_pr_auc", "eval_recall_at_threshold"],
        ascending=False,
    )
    final_segment = segments.loc[
        segments["eval_year"].astype(str).eq("final_test")
        & segments["dimension"].isin(["overall", "market", "industry"])
    ].copy()

    return "\n".join(
        [
            "# 46-Feature Macro Interaction Feature Pack Experiments",
            "",
            "공식 `credit_46_features` 입력에 macro regime 변화량과 "
            "macro shock x 재무 취약도 interaction 후보를 추가해 walk-forward rolling OOT로 비교했습니다.",
            "",
            f"Rolling 평가연도는 `{', '.join(str(year) for year in eval_years)}`이고, "
            "Final Test는 공식 test split인 2023~2024 구간입니다.",
            "각 fold는 `과거 연도 학습 -> 직전 1년 Platt calibration/threshold 선택 -> 다음 1년 평가` 구조입니다.",
            "",
            "## 1. 결론",
            "",
            f"- Baseline rolling F1/PR-AUC: `{format_metric(baseline['eval_f1_at_threshold_mean'])}` / "
            f"`{format_metric(baseline['eval_pr_auc_mean'])}`",
            f"- Rolling 기준 최상위 후보: `{best['variant']}` "
            f"(rolling F1 `{format_metric(best['eval_f1_at_threshold_mean'])}`, "
            f"PR-AUC `{format_metric(best['eval_pr_auc_mean'])}`)",
            f"- Rolling F1 변화: `{format_signed(best['rolling_f1_delta_vs_baseline'])}`",
            f"- Final Test F1 변화: `{format_signed(best['final_f1_delta_vs_baseline'])}`",
            recommendation_text(best),
            "",
            "## 2. 후보별 Rolling + Final Test 비교",
            "",
            markdown_table(
                summary,
                [
                    ("Variant", "variant", "text"),
                    ("Added", "added_feature_count", "int"),
                    ("Roll PR", "eval_pr_auc_mean", "metric"),
                    ("Roll P", "eval_precision_at_threshold_mean", "metric"),
                    ("Roll R", "eval_recall_at_threshold_mean", "metric"),
                    ("Roll F1", "eval_f1_at_threshold_mean", "metric"),
                    ("Roll dF1", "rolling_f1_delta_vs_baseline", "signed"),
                    ("Roll FP", "total_false_positive", "int"),
                    ("Roll FN", "total_false_negative", "int"),
                    ("Final PR", "final_eval_pr_auc", "metric"),
                    ("Final P", "final_eval_precision_at_threshold", "metric"),
                    ("Final R", "final_eval_recall_at_threshold", "metric"),
                    ("Final F1", "final_eval_f1_at_threshold", "metric"),
                    ("Final dF1", "final_f1_delta_vs_baseline", "signed"),
                    ("Final FP", "final_eval_false_positive_at_threshold", "int"),
                    ("Final FN", "final_eval_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 3. Baseline 연도별 Rolling 성능",
            "",
            markdown_table(
                baseline_rows,
                [
                    ("Eval Year", "eval_year", "int"),
                    ("Threshold", "threshold_tuned", "metric"),
                    ("PR-AUC", "eval_pr_auc", "metric"),
                    ("Precision", "eval_precision_at_threshold", "metric"),
                    ("Recall", "eval_recall_at_threshold", "metric"),
                    ("F1", "eval_f1_at_threshold", "metric"),
                    ("FP", "eval_false_positive_at_threshold", "int"),
                    ("FN", "eval_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 4. 최상위 후보 연도별 Rolling 성능",
            "",
            markdown_table(
                best_rows,
                [
                    ("Eval Year", "eval_year", "int"),
                    ("Threshold", "threshold_tuned", "metric"),
                    ("PR-AUC", "eval_pr_auc", "metric"),
                    ("Precision", "eval_precision_at_threshold", "metric"),
                    ("Recall", "eval_recall_at_threshold", "metric"),
                    ("F1", "eval_f1_at_threshold", "metric"),
                    ("FP", "eval_false_positive_at_threshold", "int"),
                    ("FN", "eval_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 5. 참고용 Final Test 순위",
            "",
            markdown_table(
                final_ranked,
                [
                    ("Variant", "variant", "text"),
                    ("Threshold", "threshold_tuned", "metric"),
                    ("PR-AUC", "eval_pr_auc", "metric"),
                    ("Precision", "eval_precision_at_threshold", "metric"),
                    ("Recall", "eval_recall_at_threshold", "metric"),
                    ("F1", "eval_f1_at_threshold", "metric"),
                    ("FP", "eval_false_positive_at_threshold", "int"),
                    ("FN", "eval_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 6. Final Test 세그먼트",
            "",
            markdown_table(
                final_segment.sort_values(["variant", "dimension", "segment"]),
                [
                    ("Variant", "variant", "text"),
                    ("Segment", "segment", "text"),
                    ("Rows", "rows", "int"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
            ),
            "",
            "## 7. 후보 변수",
            "",
            "- Raw macro regime: " + ", ".join(f"`{column}`" for column in RAW_MACRO_COLUMNS),
            "- Macro shock component: "
            + ", ".join(f"`{column}`" for column in MACRO_SHOCK_COLUMNS),
            "- Vulnerability proxy: "
            + ", ".join(f"`{column}`" for column in VULNERABILITY_COLUMNS),
            "- Macro x vulnerability interaction: "
            + ", ".join(f"`{column}`" for column in INTERACTION_COLUMNS),
            "",
            "## 8. 해석 주의",
            "",
            "- 후보 선택은 rolling OOT 평균 기준으로 판단했습니다.",
            "- Final Test는 공식 test split 사후 확인용입니다.",
            "- Macro 변수는 연도별 공통 신호가 강하므로 특정 국면을 외우는지 주의해야 합니다.",
            "- F1이 좋아도 FN이 늘면 조기경보 모델로는 보수적으로 해석합니다.",
            "- 이 실험은 official artifact를 덮어쓰지 않는 feature-pack screening입니다.",
        ]
    )


def write_outputs(
    *,
    fold_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    segments: pd.DataFrame,
    output_dir: Path,
    eval_years: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fold_metrics": output_dir / "macro_interaction_feature_pack_rolling_fold_metrics.csv",
        "final_test": output_dir / "macro_interaction_feature_pack_final_test.csv",
        "summary": output_dir / "macro_interaction_feature_pack_summary.csv",
        "segments": output_dir / "macro_interaction_feature_pack_segment_metrics.csv",
        "report": output_dir / "macro_interaction_feature_pack_report.md",
        "metadata": output_dir / "macro_interaction_feature_pack_summary.json",
    }
    fold_metrics.to_csv(paths["fold_metrics"], index=False, encoding="utf-8-sig")
    final_metrics.to_csv(paths["final_test"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    segments.to_csv(paths["segments"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(
        build_report(
            fold_metrics=fold_metrics,
            final_metrics=final_metrics,
            summary=summary,
            segments=segments,
            eval_years=eval_years,
        ),
        encoding="utf-8",
    )

    baseline = summary.loc[summary["variant"].eq("baseline_46_native")].iloc[0]
    best = best_candidate(summary)
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": "feature_46_xgboost",
        "dataset": "credit_46_features",
        "eval_years": eval_years,
        "fold_policy": "train fiscal_year < eval_year-1, tune on eval_year-1, evaluate eval_year",
        "final_test_policy": "train fiscal_year <= 2021, tune on 2022, evaluate 2023-2024",
        "threshold_policy": f"max precision with policy-year recall >= {RECALL_FLOOR:.2f}",
        "candidate_count": int(summary.shape[0]),
        "raw_macro_features": RAW_MACRO_COLUMNS,
        "macro_shock_features": MACRO_SHOCK_COLUMNS,
        "vulnerability_features": VULNERABILITY_COLUMNS,
        "interaction_features": INTERACTION_COLUMNS,
        "baseline": baseline.to_dict(),
        "best_by_rolling_mean_f1": best.to_dict(),
        "output_files": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    master = add_macro_features(read_master(args.master_path), args.raw_ts2000_path)
    base_columns = read_feature_columns(args.feature_list_path, master)
    fold_metrics, final_metrics, summary, segments = run_experiment(
        frame=master,
        base_columns=base_columns,
        eval_years=args.eval_years,
        seed=args.seed,
    )
    write_outputs(
        fold_metrics=fold_metrics,
        final_metrics=final_metrics,
        summary=summary,
        segments=segments,
        output_dir=args.output_dir,
        eval_years=args.eval_years,
    )
    baseline = summary.loc[summary["variant"].eq("baseline_46_native")].iloc[0]
    best = best_candidate(summary)
    print(
        json.dumps(
            {
                "best_variant": best["variant"],
                "best_rolling_f1": float(best["eval_f1_at_threshold_mean"]),
                "baseline_rolling_f1": float(baseline["eval_f1_at_threshold_mean"]),
                "best_final_test_f1": float(best["final_eval_f1_at_threshold"]),
                "baseline_final_test_f1": float(baseline["final_eval_f1_at_threshold"]),
                "report": str(
                    (args.output_dir / "macro_interaction_feature_pack_report.md").relative_to(ROOT)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
