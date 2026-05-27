from __future__ import annotations

import argparse
import itertools
import json
import sys
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
    evaluate_calibrated_stage1_split,
    read_stage1_feature_columns,
    read_stage1_master,
    train_stage1_xgboost,
)

INPUT_DIR = ROOT / "data" / "input" / "credit_46_features"
MASTER_PATH = INPUT_DIR / "feature_46_master.csv"
FEATURE_LIST_PATH = INPUT_DIR / "feature_46_list.json"
OUTPUT_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_46_xgboost" / "diagnostics"

RANDOM_STATE = DEFAULT_STAGE1_RANDOM_STATE
RECALL_FLOOR = DEFAULT_STAGE1_RECALL_FLOOR
ROLLING_EVAL_YEARS = DEFAULT_ROLLING_EVAL_YEARS
ID_COLUMNS = [
    "market",
    "stock_code",
    "corp_name",
    "fiscal_year",
    "eval_year",
    "firm_size_group",
    "industry_macro_category",
]

BASELINE_PARAMS: dict[str, Any] = {
    "max_depth": 4,
    "min_child_weight": 3.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "scale_pos_weight_multiplier": 1.0,
}

SEARCH_GRID: dict[str, list[Any]] = {
    "max_depth": [2, 3, 4],
    "min_child_weight": [5.0, 8.0, 12.0, 15.0],
    "gamma": [0.0, 0.5, 1.0, 3.0],
    "reg_alpha": [0.0, 0.05, 0.2, 1.0],
    "reg_lambda": [1.0, 3.0, 6.0, 10.0],
    "subsample": [0.65, 0.8, 0.9],
    "colsample_bytree": [0.65, 0.8, 0.9],
    "scale_pos_weight_multiplier": [0.8, 1.0, 1.2],
}

HAND_PICKED_PARAMS: list[dict[str, Any]] = [
    {
        "max_depth": 3,
        "min_child_weight": 8.0,
        "gamma": 0.5,
        "reg_alpha": 0.05,
        "reg_lambda": 3.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight_multiplier": 1.0,
    },
    {
        "max_depth": 3,
        "min_child_weight": 12.0,
        "gamma": 1.0,
        "reg_alpha": 0.2,
        "reg_lambda": 6.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight_multiplier": 1.0,
    },
    {
        "max_depth": 2,
        "min_child_weight": 10.0,
        "gamma": 1.0,
        "reg_alpha": 0.2,
        "reg_lambda": 6.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight_multiplier": 1.0,
    },
    {
        "max_depth": 2,
        "min_child_weight": 15.0,
        "gamma": 3.0,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
        "subsample": 0.65,
        "colsample_bytree": 0.65,
        "scale_pos_weight_multiplier": 1.0,
    },
    {
        "max_depth": 3,
        "min_child_weight": 8.0,
        "gamma": 1.0,
        "reg_alpha": 0.2,
        "reg_lambda": 6.0,
        "subsample": 0.65,
        "colsample_bytree": 0.8,
        "scale_pos_weight_multiplier": 1.2,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run rolling OOT regularization tuning for the official 46-feature XGBoost model."
        )
    )
    parser.add_argument("--master-path", type=Path, default=MASTER_PATH)
    parser.add_argument("--feature-list-path", type=Path, default=FEATURE_LIST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=96,
        help="Deterministic random sample size from the full regularization grid.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--eval-years", type=int, nargs="+", default=ROLLING_EVAL_YEARS)
    return parser.parse_args()


def read_master(path: Path) -> pd.DataFrame:
    return read_stage1_master(
        path,
        duplicate_keys=["market", "stock_code", "corp_name", "fiscal_year"],
    )


def read_feature_columns(path: Path, master: pd.DataFrame) -> list[str]:
    return read_stage1_feature_columns(path, master)


def candidate_key(params: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(params[key] for key in sorted(BASELINE_PARAMS))


def candidate_grid(max_candidates: int, seed: int) -> list[tuple[str, dict[str, Any]]]:
    keys = list(SEARCH_GRID)
    raw_candidates = [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(SEARCH_GRID[key] for key in keys))
    ]
    seen = {candidate_key(BASELINE_PARAMS)}
    candidates: list[tuple[str, dict[str, Any]]] = [("baseline_current", dict(BASELINE_PARAMS))]

    for index, params in enumerate(HAND_PICKED_PARAMS, start=1):
        key = candidate_key(params)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((f"hand_regularized_{index:02d}", dict(params)))

    search_candidates = [params for params in raw_candidates if candidate_key(params) not in seen]
    rng = np.random.default_rng(seed)
    if max_candidates > 0 and len(search_candidates) > max_candidates:
        selected = sorted(
            rng.choice(len(search_candidates), size=max_candidates, replace=False).tolist()
        )
        search_candidates = [search_candidates[index] for index in selected]

    for index, params in enumerate(search_candidates, start=1):
        key = candidate_key(params)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((f"grid_regularized_{index:03d}", dict(params)))
    return candidates


def evaluate_rolling_candidate(
    *,
    master: pd.DataFrame,
    columns: list[str],
    candidate_id: str,
    params: dict[str, Any],
    eval_years: list[int],
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for eval_year in eval_years:
        policy_year = eval_year - 1
        train = master.loc[master["fiscal_year"] < policy_year].copy()
        policy = master.loc[master["fiscal_year"] == policy_year].copy()
        evaluation = master.loc[master["fiscal_year"] == eval_year].copy()
        if train.empty or policy.empty or evaluation.empty:
            raise ValueError(
                f"Empty rolling split for eval_year={eval_year}: "
                f"train={len(train)}, policy={len(policy)}, evaluation={len(evaluation)}"
            )
        model = train_stage1_xgboost(
            train=train,
            policy=policy,
            columns=columns,
            params=params,
            seed=seed,
        )
        metrics, _ = evaluate_calibrated_stage1_split(
            model=model,
            policy=policy,
            evaluation=evaluation,
            columns=columns,
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "is_baseline": candidate_id == "baseline_current",
                **params,
                "eval_year": eval_year,
                "policy_year": policy_year,
                "train_rows": len(train),
                "policy_rows": len(policy),
                "eval_rows": len(evaluation),
                "eval_positive_rate": float(evaluation["is_speculative"].mean()),
                "best_iteration": getattr(model, "best_iteration", None),
                **metrics,
            }
        )
    return rows


def evaluate_final_holdout(
    *,
    master: pd.DataFrame,
    columns: list[str],
    candidate_id: str,
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    train = master.loc[master["fiscal_year"] <= 2021].copy()
    policy = master.loc[master["fiscal_year"] == 2022].copy()
    evaluation = master.loc[master["fiscal_year"] >= 2023].copy()
    model = train_stage1_xgboost(
        train=train,
        policy=policy,
        columns=columns,
        params=params,
        seed=seed,
    )
    metrics, _ = evaluate_calibrated_stage1_split(
        model=model,
        policy=policy,
        evaluation=evaluation,
        columns=columns,
    )
    return {
        "candidate_id": candidate_id,
        "is_baseline": candidate_id == "baseline_current",
        **params,
        "train_rows": len(train),
        "policy_rows": len(policy),
        "eval_rows": len(evaluation),
        "eval_positive_rate": float(evaluation["is_speculative"].mean()),
        "best_iteration": getattr(model, "best_iteration", None),
        **metrics,
    }


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
    rows = []
    param_columns = list(BASELINE_PARAMS)
    for candidate_id, group in fold_metrics.groupby("candidate_id", sort=False):
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "is_baseline": bool(group["is_baseline"].iloc[0]),
            "folds": len(group),
        }
        row.update({column: group[column].iloc[0] for column in param_columns})
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
            row[f"{column}_min"] = float(group[column].min())
            row[f"{column}_max"] = float(group[column].max())
        row["total_false_positive"] = int(group["eval_false_positive_at_threshold"].sum())
        row["total_false_negative"] = int(group["eval_false_negative_at_threshold"].sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "eval_f1_at_threshold_mean",
            "eval_pr_auc_mean",
            "eval_recall_at_threshold_mean",
            "eval_precision_at_threshold_mean",
        ],
        ascending=False,
    )


def merge_final_metrics(rolling_summary: pd.DataFrame, final_holdout: pd.DataFrame) -> pd.DataFrame:
    final_columns = [
        "candidate_id",
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
    renamed = final_holdout.loc[:, final_columns].rename(
        columns={column: f"final_{column}" for column in final_columns if column != "candidate_id"}
    )
    return rolling_summary.merge(renamed, on="candidate_id", how="left")


def ranked_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        [
            "eval_f1_at_threshold_mean",
            "eval_pr_auc_mean",
            "final_eval_f1_at_threshold",
            "eval_recall_at_threshold_mean",
        ],
        ascending=False,
    )


def format_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


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
            elif kind == "int":
                values.append(format_int(value))
            else:
                values.append(str(value) if value is not None else "")
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def best_candidate(summary: pd.DataFrame) -> pd.Series:
    return ranked_summary(summary).iloc[0]


def recommendation_text(best: pd.Series, baseline: pd.Series) -> str:
    rolling_f1_delta = float(best["eval_f1_at_threshold_mean"]) - float(
        baseline["eval_f1_at_threshold_mean"]
    )
    final_f1_delta = float(best["final_eval_f1_at_threshold"]) - float(
        baseline["final_eval_f1_at_threshold"]
    )
    final_fn_delta = int(best["final_eval_false_negative_at_threshold"]) - int(
        baseline["final_eval_false_negative_at_threshold"]
    )
    if str(best["candidate_id"]) == "baseline_current":
        return (
            "- Rolling OOT 기준으로도 현재 baseline이 가장 안정적입니다. "
            "공식 모델은 유지하고 다음 개선은 feature/calibration 쪽으로 넘기는 편이 좋습니다."
        )
    if rolling_f1_delta >= 0.005 and final_f1_delta >= 0 and final_fn_delta <= 0:
        return (
            "- Rolling OOT와 final holdout이 모두 받쳐주는 후보입니다. "
            "공식 모델 교체 후보로 별도 재학습/대시보드 검증을 진행할 가치가 있습니다."
        )
    if rolling_f1_delta >= 0.005 and final_f1_delta < 0:
        return (
            "- Rolling OOT에서는 좋아졌지만 final holdout에서 악화되었습니다. "
            "운영 반영은 보류하고 후보 기록으로만 남기는 편이 안전합니다."
        )
    return (
        "- 개선 폭이 작거나 FN trade-off가 있습니다. "
        "공식 모델 즉시 교체보다는 추가 OOT/feature 실험 후 판단하는 편이 안전합니다."
    )


def build_report(
    *,
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    final_holdout: pd.DataFrame,
    eval_years: list[int],
    max_candidates: int,
) -> str:
    ranked = ranked_summary(summary)
    baseline = summary.loc[summary["candidate_id"].eq("baseline_current")].iloc[0]
    best = best_candidate(summary)
    rolling_f1_delta = float(best["eval_f1_at_threshold_mean"]) - float(
        baseline["eval_f1_at_threshold_mean"]
    )
    final_f1_delta = float(best["final_eval_f1_at_threshold"]) - float(
        baseline["final_eval_f1_at_threshold"]
    )
    baseline_rows = fold_metrics.loc[fold_metrics["candidate_id"].eq("baseline_current")]
    best_rows = fold_metrics.loc[fold_metrics["candidate_id"].eq(str(best["candidate_id"]))]
    final_ranked = final_holdout.sort_values(
        ["eval_f1_at_threshold", "eval_pr_auc", "eval_recall_at_threshold"],
        ascending=False,
    ).head(10)

    return "\n".join(
        [
            "# 46-Feature Regularized XGBoost Rolling Tuning",
            "",
            "공식 `credit_46_features` 입력은 그대로 두고 XGBoost regularization만 바꾼 후보를",
            "walk-forward rolling OOT 기준으로 비교한 실험입니다.",
            "",
            "각 rolling fold는 `과거 연도 학습 -> 직전 1년 Platt calibration/threshold 선택 -> 다음 1년 평가` 구조입니다.",
            f"모델 선택용 rolling 평가연도는 `{', '.join(str(year) for year in eval_years)}`이고,",
            "2023~2024 final holdout은 사후 확인용으로만 사용했습니다.",
            f"검색 후보는 baseline 1개, hand-picked regularized 후보 {len(HAND_PICKED_PARAMS)}개, "
            f"deterministic random grid sample `{max_candidates}`개입니다.",
            "",
            "## 1. 결론",
            "",
            f"- Baseline rolling mean F1: `{format_metric(baseline['eval_f1_at_threshold_mean'])}` "
            f"(PR-AUC `{format_metric(baseline['eval_pr_auc_mean'])}`)",
            f"- Rolling 기준 최상위 후보: `{best['candidate_id']}` "
            f"(mean F1 `{format_metric(best['eval_f1_at_threshold_mean'])}`, "
            f"PR-AUC `{format_metric(best['eval_pr_auc_mean'])}`)",
            f"- Rolling F1 변화: `{rolling_f1_delta:+.4f}`",
            f"- Final holdout F1 변화: `{final_f1_delta:+.4f}`",
            recommendation_text(best, baseline),
            "",
            "## 2. Rolling 기준 상위 후보",
            "",
            markdown_table(
                ranked.head(12),
                [
                    ("Candidate", "candidate_id", "text"),
                    ("Depth", "max_depth", "int"),
                    ("Child", "min_child_weight", "metric"),
                    ("Gamma", "gamma", "metric"),
                    ("Alpha", "reg_alpha", "metric"),
                    ("Lambda", "reg_lambda", "metric"),
                    ("Sub", "subsample", "metric"),
                    ("Col", "colsample_bytree", "metric"),
                    ("SPW x", "scale_pos_weight_multiplier", "metric"),
                    ("Roll PR", "eval_pr_auc_mean", "metric"),
                    ("Roll P", "eval_precision_at_threshold_mean", "metric"),
                    ("Roll R", "eval_recall_at_threshold_mean", "metric"),
                    ("Roll F1", "eval_f1_at_threshold_mean", "metric"),
                    ("Roll FP", "total_false_positive", "int"),
                    ("Roll FN", "total_false_negative", "int"),
                    ("Final P", "final_eval_precision_at_threshold", "metric"),
                    ("Final R", "final_eval_recall_at_threshold", "metric"),
                    ("Final F1", "final_eval_f1_at_threshold", "metric"),
                    ("Final FP", "final_eval_false_positive_at_threshold", "int"),
                    ("Final FN", "final_eval_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 3. Baseline 연도별 rolling 성능",
            "",
            markdown_table(
                baseline_rows,
                [
                    ("Eval Year", "eval_year", "int"),
                    ("Policy Year", "policy_year", "int"),
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
            "## 4. 최상위 후보 연도별 rolling 성능",
            "",
            markdown_table(
                best_rows,
                [
                    ("Eval Year", "eval_year", "int"),
                    ("Policy Year", "policy_year", "int"),
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
            "## 5. 참고용 final holdout 상위 후보",
            "",
            "아래 표는 test 구간 사후 확인용입니다. 후보 선택 기준으로는 사용하지 않습니다.",
            "",
            markdown_table(
                final_ranked,
                [
                    ("Candidate", "candidate_id", "text"),
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
            "## 6. 해석 주의",
            "",
            "- 모델 선택은 rolling OOT 평균 기준으로만 판단했습니다.",
            "- final holdout 2023~2024는 사후 확인용입니다.",
            "- F1이 좋아도 FN이 늘면 조기경보 모델로는 보수적으로 해석합니다.",
            "- 이 실험은 official artifact를 덮어쓰지 않는 challenger screening입니다.",
        ]
    )


def write_outputs(
    *,
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    final_holdout: pd.DataFrame,
    output_dir: Path,
    eval_years: list[int],
    max_candidates: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = ranked_summary(summary)
    paths = {
        "fold_metrics": output_dir / "regularized_xgboost_rolling_tuning_fold_metrics.csv",
        "summary": output_dir / "regularized_xgboost_rolling_tuning_summary.csv",
        "final_holdout": output_dir / "regularized_xgboost_rolling_tuning_final_holdout.csv",
        "report": output_dir / "regularized_xgboost_rolling_tuning_report.md",
        "metadata": output_dir / "regularized_xgboost_rolling_tuning_summary.json",
    }
    fold_metrics.to_csv(paths["fold_metrics"], index=False, encoding="utf-8-sig")
    ranked.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    final_holdout.to_csv(paths["final_holdout"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(
        build_report(
            fold_metrics=fold_metrics,
            summary=summary,
            final_holdout=final_holdout,
            eval_years=eval_years,
            max_candidates=max_candidates,
        ),
        encoding="utf-8",
    )

    baseline = summary.loc[summary["candidate_id"].eq("baseline_current")].iloc[0]
    best = best_candidate(summary)
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": "feature_46_xgboost",
        "dataset": "credit_46_features",
        "eval_years": eval_years,
        "fold_policy": "train fiscal_year < eval_year-1, tune on eval_year-1, evaluate eval_year",
        "final_holdout_policy": "train fiscal_year <= 2021, tune on 2022, evaluate 2023-2024",
        "threshold_policy": f"max precision with policy-year recall >= {RECALL_FLOOR:.2f}",
        "candidate_count": int(summary.shape[0]),
        "baseline": baseline.to_dict(),
        "best_by_rolling_mean_f1": best.to_dict(),
        "output_files": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def run_experiment(
    *,
    master: pd.DataFrame,
    columns: list[str],
    eval_years: list[int],
    max_candidates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = candidate_grid(max_candidates=max_candidates, seed=seed)
    fold_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for candidate_id, params in candidates:
        fold_rows.extend(
            evaluate_rolling_candidate(
                master=master,
                columns=columns,
                candidate_id=candidate_id,
                params=params,
                eval_years=eval_years,
                seed=seed,
            )
        )
        final_rows.append(
            evaluate_final_holdout(
                master=master,
                columns=columns,
                candidate_id=candidate_id,
                params=params,
                seed=seed,
            )
        )
    fold_metrics = pd.DataFrame(fold_rows)
    final_holdout = pd.DataFrame(final_rows)
    summary = merge_final_metrics(summarize_rolling(fold_metrics), final_holdout)
    return fold_metrics, summary, final_holdout


def main() -> None:
    args = parse_args()
    master = read_master(args.master_path)
    columns = read_feature_columns(args.feature_list_path, master)
    fold_metrics, summary, final_holdout = run_experiment(
        master=master,
        columns=columns,
        eval_years=args.eval_years,
        max_candidates=args.max_candidates,
        seed=args.seed,
    )
    write_outputs(
        fold_metrics=fold_metrics,
        summary=summary,
        final_holdout=final_holdout,
        output_dir=args.output_dir,
        eval_years=args.eval_years,
        max_candidates=args.max_candidates,
    )
    baseline = summary.loc[summary["candidate_id"].eq("baseline_current")].iloc[0]
    best = best_candidate(summary)
    print(
        json.dumps(
            {
                "candidate_count": int(summary.shape[0]),
                "best_candidate": best["candidate_id"],
                "best_rolling_f1": float(best["eval_f1_at_threshold_mean"]),
                "baseline_rolling_f1": float(baseline["eval_f1_at_threshold_mean"]),
                "best_final_f1": float(best["final_eval_f1_at_threshold"]),
                "baseline_final_f1": float(baseline["final_eval_f1_at_threshold"]),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
