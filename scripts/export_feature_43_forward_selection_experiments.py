from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from export_feature_43_candidate_feature_pack_experiments import (
    INPUT_DIR,
    OUTPUT_DIR,
    RAW_PATH,
    RECALL_FLOOR,
    attach_candidate_columns,
    evaluate_variant,
    format_int,
    format_metric,
    markdown_table,
    read_id_frames,
    read_raw_features,
    read_split_frames,
    unique_preserve_order,
)

DEFAULT_PAIR_POOL_SIZE = 16
DEFAULT_GREEDY_POOL_SIZE = 24

CANDIDATE_COLUMNS = [
    "roe",
    "operating_roe",
    "ebitda_margin",
    "interest_burden_ratio",
    "gross_margin",
    "operating_margin_diff",
    "ebitda_margin_diff",
    "ocf_to_total_assets",
    "ocf_deficit_flag",
    "delta_accruals_ratio",
    "ocf_to_total_liabilities_diff",
    "ocf_to_total_borrowings_diff",
    "rolling_3y_cv_ocf_to_total_borrowings",
    "is_zombie_3y",
    "is_3y_consecutive_operating_loss",
    "is_3y_consecutive_ocf_deficit",
    "is_operating_income_turn_negative",
    "is_ocf_turn_negative",
    "negative_equity_flag",
    "is_negative_equity_entry",
    "is_current_ratio_below_1",
    "ar_days",
    "inventory_days",
    "ap_days",
    "ar_days_diff",
    "inventory_days_diff",
    "ap_days_diff",
    "accounts_receivable_ratio",
    "inventory_ratio",
    "contract_assets_ratio",
    "advances_from_customers_ratio",
    "base_rate",
    "treasury_3y",
    "corp_aa_3y",
    "corp_bbb_3y",
    "market_spread",
    "usd_krw",
    "ppi",
    "base_rate_diff",
    "treasury_3y_diff",
    "usd_krw_diff",
    "market_spread_diff",
    "spec_spread_diff",
    "audit_qualified_flag",
    "rolling_3y_cv_operating_margin",
    "capital_impairment_diff",
    "equity_growth",
    "non_paid_in_equity_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run single-feature and two-feature forward-selection experiments for the "
            "43-feature XGBoost credit model."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--pair-pool-size",
        type=int,
        default=DEFAULT_PAIR_POOL_SIZE,
        help="Number of validation-ranked single features used for exhaustive pair search.",
    )
    parser.add_argument(
        "--greedy-pool-size",
        type=int,
        default=DEFAULT_GREEDY_POOL_SIZE,
        help="Number of validation-ranked single features used for greedy second-step search.",
    )
    return parser.parse_args()


def rank_by_valid(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values(
        [
            "valid_f1_at_threshold",
            "valid_pr_auc",
            "valid_precision_at_threshold",
            "test_f1_at_threshold",
        ],
        ascending=False,
    )


def rank_by_test(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values(
        [
            "test_f1_at_threshold",
            "test_pr_auc",
            "test_precision_at_threshold",
            "valid_f1_at_threshold",
        ],
        ascending=False,
    )


def safe_variant_name(prefix: str, columns: list[str]) -> str:
    return f"{prefix}__{'__'.join(columns)}"


def evaluate_feature_set(
    *,
    variant: str,
    step: str,
    note: str,
    frames: dict[str, pd.DataFrame],
    id_frames: dict[str, pd.DataFrame],
    base_feature_columns: list[str],
    added_columns: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feature_columns = unique_preserve_order([*base_feature_columns, *added_columns])
    row, segment_rows = evaluate_variant(
        variant=variant,
        note=note,
        frames=frames,
        id_frames=id_frames,
        feature_columns=feature_columns,
        added_columns=added_columns,
    )
    row["selection_step"] = step
    row["added_features"] = ", ".join(added_columns)
    for segment_row in segment_rows:
        segment_row["selection_step"] = step
        segment_row["added_features"] = ", ".join(added_columns)
    return row, segment_rows


def run_experiments(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    id_frames: dict[str, pd.DataFrame],
    raw: pd.DataFrame,
    pair_pool_size: int,
    greedy_pool_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_feature_columns = [column for column in train.columns if column != "is_speculative"]
    base_frames = {"train": train, "valid": valid, "test": test}
    candidate_columns = [
        column
        for column in unique_preserve_order(CANDIDATE_COLUMNS)
        if column not in base_feature_columns
    ]
    candidate_frames = attach_candidate_columns(
        frames=base_frames,
        id_frames=id_frames,
        raw=raw,
        candidate_columns=candidate_columns,
    )

    metric_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []

    baseline_row, baseline_segments = evaluate_feature_set(
        variant="baseline_43_native",
        step="baseline",
        note="현재 43개 변수, XGBoost native missing 기준",
        frames=base_frames,
        id_frames=id_frames,
        base_feature_columns=base_feature_columns,
        added_columns=[],
    )
    metric_rows.append(baseline_row)
    segment_rows.extend(baseline_segments)

    single_rows: list[dict[str, Any]] = []
    for column in candidate_columns:
        row, segments = evaluate_feature_set(
            variant=safe_variant_name("single", [column]),
            step="single",
            note=f"단일 후보 변수 `{column}` 추가",
            frames=candidate_frames,
            id_frames=id_frames,
            base_feature_columns=base_feature_columns,
            added_columns=[column],
        )
        single_rows.append(row)
        metric_rows.append(row)
        segment_rows.extend(segments)

    single_metrics = pd.DataFrame(single_rows)
    ranked_single_columns = (
        rank_by_valid(single_metrics)["added_features"].head(pair_pool_size).tolist()
    )
    for left, right in itertools.combinations(ranked_single_columns, 2):
        columns = [left, right]
        row, segments = evaluate_feature_set(
            variant=safe_variant_name("pair_top_pool", columns),
            step="pair_top_pool",
            note=f"validation 상위 단일 후보 2개 조합: `{left}` + `{right}`",
            frames=candidate_frames,
            id_frames=id_frames,
            base_feature_columns=base_feature_columns,
            added_columns=columns,
        )
        metric_rows.append(row)
        segment_rows.extend(segments)

    best_single = rank_by_valid(single_metrics).iloc[0]
    first_column = str(best_single["added_features"])
    greedy_pool = [
        column
        for column in rank_by_valid(single_metrics)["added_features"]
        .head(greedy_pool_size)
        .tolist()
        if column != first_column
    ]
    for second_column in greedy_pool:
        columns = [first_column, second_column]
        row, segments = evaluate_feature_set(
            variant=safe_variant_name("greedy_second", columns),
            step="greedy_second",
            note=f"greedy 1순위 `{first_column}`에 `{second_column}` 추가",
            frames=candidate_frames,
            id_frames=id_frames,
            base_feature_columns=base_feature_columns,
            added_columns=columns,
        )
        metric_rows.append(row)
        segment_rows.extend(segments)

    metrics = pd.DataFrame(metric_rows)
    segments = pd.DataFrame(segment_rows)
    return metrics, segments


def segment_row(segments: pd.DataFrame, variant: str, segment: str) -> dict[str, Any]:
    frame = segments.loc[segments["variant"].eq(variant) & segments["segment"].eq(segment)]
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def recommendation_text(
    *,
    best_valid: pd.Series,
    valid_delta: float,
    test_delta: float,
) -> str:
    if str(best_valid["variant"]) == "baseline_43_native":
        return (
            "- Validation 기준으로도 현재 baseline이 최상위입니다. "
            "작은 후보 변수 추가 역시 운영 반영할 근거가 부족합니다."
        )
    if valid_delta >= 0.005 and test_delta >= 0.005:
        return (
            f"- `{best_valid['variant']}`는 validation과 test에서 모두 개선되어 "
            "다음 후보 모델로 별도 검증할 가치가 있습니다."
        )
    if valid_delta >= 0.005 and test_delta < 0:
        return (
            f"- `{best_valid['variant']}`는 validation에서는 좋아졌지만 test에서는 악화되었습니다. "
            "과적합 가능성이 있어 production 반영은 보류하는 편이 안전합니다."
        )
    return (
        f"- `{best_valid['variant']}`의 개선 폭이 작습니다. "
        "운영 모델 교체보다 후보 변수 검토 목록으로 남기는 편이 좋습니다."
    )


def build_report(
    *,
    metrics: pd.DataFrame,
    segments: pd.DataFrame,
    pair_pool_size: int,
    greedy_pool_size: int,
) -> str:
    baseline = metrics.loc[metrics["variant"].eq("baseline_43_native")].iloc[0]
    best_valid = rank_by_valid(metrics).iloc[0]
    best_test = rank_by_test(metrics).iloc[0]
    valid_delta = float(best_valid["valid_f1_at_threshold"]) - float(
        baseline["valid_f1_at_threshold"]
    )
    test_delta = float(best_valid["test_f1_at_threshold"]) - float(baseline["test_f1_at_threshold"])
    test_only_delta = float(best_test["test_f1_at_threshold"]) - float(
        baseline["test_f1_at_threshold"]
    )
    top_single = rank_by_valid(metrics.loc[metrics["selection_step"].eq("single")]).head(12)
    top_pair = rank_by_valid(
        metrics.loc[metrics["selection_step"].isin(["pair_top_pool", "greedy_second"])]
    ).head(12)
    baseline_kosdaq = segment_row(segments, "baseline_43_native", "KOSDAQ")
    best_valid_kosdaq = segment_row(segments, str(best_valid["variant"]), "KOSDAQ")

    return "\n".join(
        [
            "# Forward Feature Selection Experiments",
            "",
            "현재 43-feature XGBoost 모델에 원본 Model V1의 후보 변수를 아주 작게 추가하는 실험입니다.",
            "단일 후보 변수를 모두 평가한 뒤, validation 상위 단일 후보의 2개 조합과 greedy 2단계 후보를 비교했습니다.",
            "모든 실험은 XGBoost native missing, Platt scaling, validation 기준 "
            f"`recall >= {RECALL_FLOOR:.2f}` 조건에서 precision 최대 threshold를 사용했습니다.",
            "",
            "## 1. 결론",
            "",
            f"- Baseline valid/test F1: `{format_metric(baseline['valid_f1_at_threshold'])}` / "
            f"`{format_metric(baseline['test_f1_at_threshold'])}`",
            f"- Validation 기준 선택 후보: `{best_valid['variant']}` "
            f"(valid F1 `{format_metric(best_valid['valid_f1_at_threshold'])}`, "
            f"test F1 `{format_metric(best_valid['test_f1_at_threshold'])}`)",
            f"- Validation 선택 후보의 baseline 대비 변화: valid F1 `{valid_delta:+.4f}`, "
            f"test F1 `{test_delta:+.4f}`",
            f"- 참고용 test F1 최상위 후보: `{best_test['variant']}` "
            f"(test F1 `{format_metric(best_test['test_f1_at_threshold'])}`, "
            f"baseline 대비 `{test_only_delta:+.4f}`)",
            recommendation_text(
                best_valid=best_valid,
                valid_delta=valid_delta,
                test_delta=test_delta,
            ),
            "",
            "## 2. 단일 변수 상위 후보",
            "",
            markdown_table(
                top_single,
                [
                    ("Feature", "added_features", "text"),
                    ("Threshold", "threshold_tuned", "metric"),
                    ("Valid PR", "valid_pr_auc", "metric"),
                    ("Valid P", "valid_precision_at_threshold", "metric"),
                    ("Valid R", "valid_recall_at_threshold", "metric"),
                    ("Valid F1", "valid_f1_at_threshold", "metric"),
                    ("Test PR", "test_pr_auc", "metric"),
                    ("Test P", "test_precision_at_threshold", "metric"),
                    ("Test R", "test_recall_at_threshold", "metric"),
                    ("Test F1", "test_f1_at_threshold", "metric"),
                    ("Test FP", "test_false_positive_at_threshold", "int"),
                    ("Test FN", "test_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 3. 2개 조합 상위 후보",
            "",
            markdown_table(
                top_pair,
                [
                    ("Features", "added_features", "text"),
                    ("Step", "selection_step", "text"),
                    ("Threshold", "threshold_tuned", "metric"),
                    ("Valid PR", "valid_pr_auc", "metric"),
                    ("Valid P", "valid_precision_at_threshold", "metric"),
                    ("Valid R", "valid_recall_at_threshold", "metric"),
                    ("Valid F1", "valid_f1_at_threshold", "metric"),
                    ("Test PR", "test_pr_auc", "metric"),
                    ("Test P", "test_precision_at_threshold", "metric"),
                    ("Test R", "test_recall_at_threshold", "metric"),
                    ("Test F1", "test_f1_at_threshold", "metric"),
                    ("Test FP", "test_false_positive_at_threshold", "int"),
                    ("Test FN", "test_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 4. 전체 상위 후보",
            "",
            markdown_table(
                rank_by_valid(metrics).head(20),
                [
                    ("Variant", "variant", "text"),
                    ("Step", "selection_step", "text"),
                    ("Added", "added_feature_count", "int"),
                    ("Features", "added_features", "text"),
                    ("Valid F1", "valid_f1_at_threshold", "metric"),
                    ("Test PR", "test_pr_auc", "metric"),
                    ("Test P", "test_precision_at_threshold", "metric"),
                    ("Test R", "test_recall_at_threshold", "metric"),
                    ("Test F1", "test_f1_at_threshold", "metric"),
                    ("Test FP", "test_false_positive_at_threshold", "int"),
                    ("Test FN", "test_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 5. KOSDAQ 오류 관점",
            "",
            f"- Baseline KOSDAQ FP/FN: `{format_int(baseline_kosdaq.get('false_positive'))}` / "
            f"`{format_int(baseline_kosdaq.get('false_negative'))}`",
            f"- Validation 선택 후보 KOSDAQ FP/FN: "
            f"`{format_int(best_valid_kosdaq.get('false_positive'))}` / "
            f"`{format_int(best_valid_kosdaq.get('false_negative'))}`",
            "",
            "## 6. 실험 범위",
            "",
            f"- 단일 후보: `{int(metrics['selection_step'].eq('single').sum())}`개",
            f"- 2개 조합 후보: validation 상위 단일 후보 `{pair_pool_size}`개 기반 조합 + "
            f"greedy second-step `{greedy_pool_size}`개 풀",
            "- test 기준 상위 후보는 사후 참고용이며, 모델 선택 기준으로는 사용하지 않습니다.",
        ]
    )


def write_outputs(
    *,
    metrics: pd.DataFrame,
    segments: pd.DataFrame,
    output_dir: Path,
    pair_pool_size: int,
    greedy_pool_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "forward_selection_metrics.csv"
    segments_path = output_dir / "forward_selection_segment_metrics.csv"
    report_path = output_dir / "forward_selection_report.md"
    summary_path = output_dir / "forward_selection_summary.json"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    segments.to_csv(segments_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_report(
            metrics=metrics,
            segments=segments,
            pair_pool_size=pair_pool_size,
            greedy_pool_size=greedy_pool_size,
        ),
        encoding="utf-8",
    )

    baseline = metrics.loc[metrics["variant"].eq("baseline_43_native")].iloc[0]
    best_valid = rank_by_valid(metrics).iloc[0]
    best_test = rank_by_test(metrics).iloc[0]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_source": str(RAW_PATH.relative_to(Path(__file__).resolve().parents[1])),
        "threshold_policy": f"max precision with validation recall >= {RECALL_FLOOR:.2f}",
        "pair_pool_size": pair_pool_size,
        "greedy_pool_size": greedy_pool_size,
        "baseline": baseline.to_dict(),
        "best_by_validation": best_valid.to_dict(),
        "best_by_test_reference_only": best_test.to_dict(),
        "output_files": {
            "metrics": str(metrics_path),
            "segment_metrics": str(segments_path),
            "report": str(report_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    train, valid, test = read_split_frames(args.input_dir)
    id_frames = read_id_frames(args.input_dir)
    raw = read_raw_features(args.raw_path)
    metrics, segments = run_experiments(
        train=train,
        valid=valid,
        test=test,
        id_frames=id_frames,
        raw=raw,
        pair_pool_size=args.pair_pool_size,
        greedy_pool_size=args.greedy_pool_size,
    )
    write_outputs(
        metrics=metrics,
        segments=segments,
        output_dir=args.output_dir,
        pair_pool_size=args.pair_pool_size,
        greedy_pool_size=args.greedy_pool_size,
    )
    best_valid = rank_by_valid(metrics).iloc[0]
    baseline = metrics.loc[metrics["variant"].eq("baseline_43_native")].iloc[0]
    print(
        json.dumps(
            {
                "best_by_validation": best_valid["variant"],
                "best_valid_f1": float(best_valid["valid_f1_at_threshold"]),
                "best_test_f1": float(best_valid["test_f1_at_threshold"]),
                "baseline_valid_f1": float(baseline["valid_f1_at_threshold"]),
                "baseline_test_f1": float(baseline["test_f1_at_threshold"]),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
