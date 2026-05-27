from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from export_feature_43_candidate_feature_pack_experiments import (
    INPUT_DIR,
    JOIN_KEYS,
    OUTPUT_DIR,
    RAW_PATH,
    RECALL_FLOOR,
    apply_platt_calibration,
    choose_threshold,
    classification_metrics,
    fit_platt_calibration,
    format_metric,
    markdown_table,
    probability_metrics,
    read_raw_features,
    train_xgboost,
    unique_preserve_order,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = INPUT_DIR / "feature_46_master.csv"
ROLLING_EVAL_YEARS = [2019, 2020, 2021, 2022]
ID_COLUMNS = [
    "market",
    "stock_code",
    "corp_name",
    "fiscal_year",
    "eval_year",
    "firm_size_group",
    "industry_macro_category",
]

ROLLING_VARIANTS: dict[str, dict[str, Any]] = {
    "baseline_43_native": {
        "note": "현재 43개 변수 baseline",
        "columns": [],
    },
    "val_best_interest_burden_ap_days_diff": {
        "note": "기존 2022 validation 기준 상위 조합",
        "columns": ["interest_burden_ratio", "ap_days_diff"],
    },
    "test_reference_base_rate_treasury_diff": {
        "note": "이전 실험에서 test 기준 참고 성능이 좋았던 조합",
        "columns": ["base_rate", "treasury_3y_diff"],
    },
    "balanced_delta_accruals_ppi": {
        "note": "validation 상위권이면서 test 하락폭이 작았던 조합",
        "columns": ["delta_accruals_ratio", "ppi"],
    },
    "single_non_paid_in_equity_ratio": {
        "note": "단일 후보 중 FP 증가가 작았던 자본 관련 후보",
        "columns": ["non_paid_in_equity_ratio"],
    },
    "single_ar_days": {
        "note": "단일 후보 중 test recall 유지가 비교적 좋았던 운전자본 후보",
        "columns": ["ar_days"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run walk-forward rolling OOT validation for selected feature candidates. "
            "Each fold trains on years before the policy year, calibrates/tunes threshold on "
            "the policy year, and evaluates the next fiscal year."
        )
    )
    parser.add_argument("--master-path", type=Path, default=MASTER_PATH)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--eval-years",
        type=int,
        nargs="+",
        default=ROLLING_EVAL_YEARS,
        help="Fiscal years to evaluate. The previous year is used as the policy year.",
    )
    return parser.parse_args()


def read_master(master_path: Path) -> pd.DataFrame:
    master = pd.read_csv(master_path, encoding="utf-8-sig", dtype={"stock_code": str})
    duplicates = master.duplicated(JOIN_KEYS).sum()
    if duplicates:
        raise ValueError(f"feature_43_master has duplicate rows for join keys: {duplicates}")
    return master


def feature_columns(master: pd.DataFrame) -> list[str]:
    excluded = {*ID_COLUMNS, "label_eval_year", "is_speculative"}
    return [column for column in master.columns if column not in excluded]


def candidate_columns() -> list[str]:
    columns: list[str] = []
    for variant in ROLLING_VARIANTS.values():
        columns.extend(variant["columns"])
    return unique_preserve_order(columns)


def attach_raw_candidates(master: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    columns = candidate_columns()
    missing_columns = [column for column in columns if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Candidate columns are missing from raw Model V1: {missing_columns}")
    if not columns:
        return master.copy()

    raw_subset = raw.loc[:, [*JOIN_KEYS, *columns]].copy()
    for column in columns:
        raw_subset[column] = pd.to_numeric(raw_subset[column], errors="coerce")
    joined = master.merge(raw_subset, on=JOIN_KEYS, how="left", indicator=True)
    unmatched = int(joined["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"feature_43_master has unmatched raw rows: {unmatched}")
    return joined.drop(columns=["_merge"])


def evaluate_fold(
    *,
    variant: str,
    note: str,
    columns: list[str],
    frame: pd.DataFrame,
    base_features: list[str],
    eval_year: int,
) -> dict[str, Any]:
    policy_year = eval_year - 1
    train_frame = frame.loc[frame["fiscal_year"] < policy_year].copy()
    policy_frame = frame.loc[frame["fiscal_year"] == policy_year].copy()
    eval_frame = frame.loc[frame["fiscal_year"] == eval_year].copy()
    if train_frame.empty or policy_frame.empty or eval_frame.empty:
        raise ValueError(
            f"Empty rolling split for eval_year={eval_year}: "
            f"train={len(train_frame)}, policy={len(policy_frame)}, eval={len(eval_frame)}"
        )

    features = unique_preserve_order([*base_features, *columns])
    y_train = train_frame["is_speculative"].astype(int)
    y_policy = policy_frame["is_speculative"].astype(int)
    y_eval = eval_frame["is_speculative"].astype(int)
    x_train = train_frame.loc[:, features]
    x_policy = policy_frame.loc[:, features]
    x_eval = eval_frame.loc[:, features]

    model = train_xgboost(x_train, y_train, x_policy, y_policy)
    policy_raw_probabilities = model.predict_proba(x_policy)[:, 1]
    eval_raw_probabilities = model.predict_proba(x_eval)[:, 1]
    coef, intercept = fit_platt_calibration(y_policy, policy_raw_probabilities)
    policy_probabilities = apply_platt_calibration(policy_raw_probabilities, coef, intercept)
    eval_probabilities = apply_platt_calibration(eval_raw_probabilities, coef, intercept)
    threshold, policy_threshold_metrics = choose_threshold(y_policy, policy_probabilities)
    eval_predictions = eval_probabilities >= threshold

    policy_probability_metrics = probability_metrics(y_policy, policy_probabilities)
    eval_probability_metrics = probability_metrics(y_eval, eval_probabilities)
    eval_classification_metrics = classification_metrics(y_eval, eval_predictions)

    return {
        "variant": variant,
        "note": note,
        "added_features": ", ".join(columns),
        "added_feature_count": len(columns),
        "eval_year": eval_year,
        "policy_year": policy_year,
        "train_year_min": int(train_frame["fiscal_year"].min()),
        "train_year_max": int(train_frame["fiscal_year"].max()),
        "train_rows": len(train_frame),
        "policy_rows": len(policy_frame),
        "eval_rows": len(eval_frame),
        "eval_positive_rate": float(y_eval.mean()),
        "best_iteration": getattr(model, "best_iteration", None),
        "threshold_tuned_on_policy_year": threshold,
        "policy_precision_at_threshold": policy_threshold_metrics["precision"],
        "policy_recall_at_threshold": policy_threshold_metrics["recall"],
        "policy_f1_at_threshold": policy_threshold_metrics["f1"],
        "policy_pr_auc": policy_probability_metrics["pr_auc"],
        "policy_roc_auc": policy_probability_metrics["roc_auc"],
        "eval_pr_auc": eval_probability_metrics["pr_auc"],
        "eval_roc_auc": eval_probability_metrics["roc_auc"],
        "eval_brier": eval_probability_metrics["brier"],
        "eval_logloss": eval_probability_metrics["logloss"],
        "eval_precision": eval_classification_metrics["precision"],
        "eval_recall": eval_classification_metrics["recall"],
        "eval_f1": eval_classification_metrics["f1"],
        "eval_true_negative": eval_classification_metrics["true_negative"],
        "eval_false_positive": eval_classification_metrics["false_positive"],
        "eval_false_negative": eval_classification_metrics["false_negative"],
        "eval_true_positive": eval_classification_metrics["true_positive"],
    }


def run_rolling_validation(
    *,
    master: pd.DataFrame,
    raw: pd.DataFrame,
    eval_years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = attach_raw_candidates(master, raw)
    base_features = feature_columns(master)
    rows = []
    for eval_year in eval_years:
        for variant, spec in ROLLING_VARIANTS.items():
            rows.append(
                evaluate_fold(
                    variant=variant,
                    note=spec["note"],
                    columns=spec["columns"],
                    frame=frame,
                    base_features=base_features,
                    eval_year=eval_year,
                )
            )
    fold_metrics = pd.DataFrame(rows)
    summary = summarize_metrics(fold_metrics)
    return fold_metrics, summary


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "eval_pr_auc",
        "eval_roc_auc",
        "eval_precision",
        "eval_recall",
        "eval_f1",
        "eval_false_positive",
        "eval_false_negative",
    ]
    rows = []
    for variant, group in fold_metrics.groupby("variant", sort=False):
        row: dict[str, Any] = {
            "variant": variant,
            "note": group["note"].iloc[0],
            "added_features": group["added_features"].iloc[0],
            "folds": len(group),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
            row[f"{column}_min"] = float(group[column].min())
            row[f"{column}_max"] = float(group[column].max())
        row["total_false_positive"] = int(group["eval_false_positive"].sum())
        row["total_false_negative"] = int(group["eval_false_negative"].sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["eval_f1_mean", "eval_pr_auc_mean", "eval_precision_mean"],
        ascending=False,
    )


def build_report(fold_metrics: pd.DataFrame, summary: pd.DataFrame) -> str:
    baseline = summary.loc[summary["variant"].eq("baseline_43_native")].iloc[0]
    best = summary.iloc[0]
    best_delta = float(best["eval_f1_mean"]) - float(baseline["eval_f1_mean"])
    baseline_rows = fold_metrics.loc[fold_metrics["variant"].eq("baseline_43_native")]
    best_rows = fold_metrics.loc[fold_metrics["variant"].eq(str(best["variant"]))]

    if str(best["variant"]) == "baseline_43_native":
        recommendation = (
            "- Rolling validation 평균 기준으로 현재 baseline이 가장 안정적입니다. "
            "후보 변수 추가는 운영 모델에 반영하지 않는 편이 좋습니다."
        )
    elif best_delta >= 0.005:
        recommendation = (
            f"- `{best['variant']}`가 rolling 평균 F1을 `{best_delta:+.4f}` 개선했습니다. "
            "다만 연도별 변동과 test 성능을 함께 확인한 뒤 후보 모델로만 검토하는 편이 안전합니다."
        )
    else:
        recommendation = (
            f"- `{best['variant']}`가 평균 기준 상위지만 개선 폭이 `{best_delta:+.4f}`로 작습니다. "
            "운영 모델 교체 근거로는 약합니다."
        )

    return "\n".join(
        [
            "# Rolling OOT Validation Experiments",
            "",
            "1년 validation에 대한 과신을 줄이기 위해 walk-forward rolling OOT 방식으로 비교했습니다.",
            "각 fold는 `과거 연도 학습 -> 직전 1년 Platt calibration/threshold 선택 -> 다음 1년 평가` 구조입니다.",
            "기존 단일 validation은 특정 경기/시장 국면에 우연히 잘 맞은 후보를 과대평가할 수 있기 때문에,",
            "여러 평가연도에서 같은 후보가 반복적으로 안정적인지 확인하는 용도로 rolling validation을 사용했습니다.",
            "최종 test 구간은 후보 선택에 쓰지 않고 마지막 확인용으로만 남깁니다.",
            "",
            "## 1. 결론",
            "",
            f"- Baseline rolling mean F1: `{format_metric(baseline['eval_f1_mean'])}` "
            f"(mean PR-AUC `{format_metric(baseline['eval_pr_auc_mean'])}`)",
            f"- Rolling 평균 최상위 후보: `{best['variant']}` "
            f"(mean F1 `{format_metric(best['eval_f1_mean'])}`, "
            f"mean PR-AUC `{format_metric(best['eval_pr_auc_mean'])}`)",
            f"- 최상위 후보의 baseline 대비 mean F1 변화: `{best_delta:+.4f}`",
            recommendation,
            "",
            "## 2. Rolling 평균 성능",
            "",
            markdown_table(
                summary,
                [
                    ("Variant", "variant", "text"),
                    ("Features", "added_features", "text"),
                    ("Folds", "folds", "int"),
                    ("PR-AUC mean", "eval_pr_auc_mean", "metric"),
                    ("ROC-AUC mean", "eval_roc_auc_mean", "metric"),
                    ("Precision mean", "eval_precision_mean", "metric"),
                    ("Recall mean", "eval_recall_mean", "metric"),
                    ("F1 mean", "eval_f1_mean", "metric"),
                    ("F1 min", "eval_f1_min", "metric"),
                    ("Total FP", "total_false_positive", "int"),
                    ("Total FN", "total_false_negative", "int"),
                ],
            ),
            "",
            "## 3. Baseline 연도별 성능",
            "",
            markdown_table(
                baseline_rows,
                [
                    ("Eval Year", "eval_year", "int"),
                    ("Policy Year", "policy_year", "int"),
                    ("Rows", "eval_rows", "int"),
                    ("PR-AUC", "eval_pr_auc", "metric"),
                    ("ROC-AUC", "eval_roc_auc", "metric"),
                    ("Precision", "eval_precision", "metric"),
                    ("Recall", "eval_recall", "metric"),
                    ("F1", "eval_f1", "metric"),
                    ("FP", "eval_false_positive", "int"),
                    ("FN", "eval_false_negative", "int"),
                ],
            ),
            "",
            "## 4. 최상위 후보 연도별 성능",
            "",
            markdown_table(
                best_rows,
                [
                    ("Eval Year", "eval_year", "int"),
                    ("Policy Year", "policy_year", "int"),
                    ("Rows", "eval_rows", "int"),
                    ("PR-AUC", "eval_pr_auc", "metric"),
                    ("ROC-AUC", "eval_roc_auc", "metric"),
                    ("Precision", "eval_precision", "metric"),
                    ("Recall", "eval_recall", "metric"),
                    ("F1", "eval_f1", "metric"),
                    ("FP", "eval_false_positive", "int"),
                    ("FN", "eval_false_negative", "int"),
                ],
            ),
            "",
            "## 5. 해석 주의",
            "",
            "- 이 실험은 test 2023~2024를 모델 선택에 쓰지 않기 위한 rolling validation입니다.",
            "- 각 fold의 threshold는 평가 연도 직전 1년에서만 선택했습니다.",
            "- 단일 validation 1년만 보면 우연한 연도 효과나 경기 국면 효과를 후보 변수 성능으로 착각할 수 있습니다.",
            "- rolling 평균은 후보 선별의 1차 기준이고, 운영 반영 여부는 final test와 오류 사례 해석까지 함께 봅니다.",
            "- 후보가 평균에서 좋아도 특정 연도에서 FN이 크게 늘면 조기경보 모델로는 보수적으로 봐야 합니다.",
        ]
    )


def write_outputs(
    *,
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    eval_years: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "rolling_validation_metrics.csv"
    summary_path = output_dir / "rolling_validation_summary.csv"
    report_path = output_dir / "rolling_validation_report.md"
    meta_path = output_dir / "rolling_validation_summary.json"

    fold_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path.write_text(build_report(fold_metrics, summary), encoding="utf-8")

    baseline = summary.loc[summary["variant"].eq("baseline_43_native")].iloc[0]
    best = summary.iloc[0]
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "eval_years": eval_years,
        "fold_policy": "train fiscal_year < eval_year-1, tune on eval_year-1, evaluate eval_year",
        "threshold_policy": f"max precision with policy-year recall >= {RECALL_FLOOR:.2f}",
        "baseline": baseline.to_dict(),
        "best_by_rolling_mean_f1": best.to_dict(),
        "output_files": {
            "metrics": str(metrics_path.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    }
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    master = read_master(args.master_path)
    raw = read_raw_features(args.raw_path)
    fold_metrics, summary = run_rolling_validation(
        master=master,
        raw=raw,
        eval_years=args.eval_years,
    )
    write_outputs(
        fold_metrics=fold_metrics,
        summary=summary,
        output_dir=args.output_dir,
        eval_years=args.eval_years,
    )
    baseline = summary.loc[summary["variant"].eq("baseline_43_native")].iloc[0]
    best = summary.iloc[0]
    print(
        json.dumps(
            {
                "best_variant": best["variant"],
                "best_mean_f1": float(best["eval_f1_mean"]),
                "best_mean_pr_auc": float(best["eval_pr_auc_mean"]),
                "baseline_mean_f1": float(baseline["eval_f1_mean"]),
                "baseline_mean_pr_auc": float(baseline["eval_pr_auc_mean"]),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
