from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cas.modeling.fn_rescue import (  # noqa: E402
    FN_RESCUE_DEFAULT_MIN_GROUPS,
    FN_RESCUE_DEFAULT_PROB_CEILING,
    FN_RESCUE_DEFAULT_SCORE_THRESHOLD,
    FN_RESCUE_POLICY_NAME,
    FN_RESCUE_RAW_COLUMNS,
    FN_RESCUE_SCORE_COLUMNS,
    add_manufacturing_fn_rescue_scores,
    build_manufacturing_fn_rescue_gate,
)
from cas.modeling.stage1_xgboost import (  # noqa: E402
    DEFAULT_ROLLING_EVAL_YEARS,
    DEFAULT_STAGE1_RANDOM_STATE,
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
TARGET_SEGMENT_QUERY = "market == 'KOSDAQ' and industry_macro_category == 'manufacturing'"


@dataclass(frozen=True)
class RescuePolicy:
    policy_id: str
    note: str
    probability_ceiling: float
    score_threshold: float
    min_group_count: int = FN_RESCUE_DEFAULT_MIN_GROUPS


RESCUE_POLICIES = [
    RescuePolicy(
        policy_id="baseline_stage1_only",
        note="공식 46 Stage1 기준선. Rescue gate를 적용하지 않음",
        probability_ceiling=0.0,
        score_threshold=1.0,
        min_group_count=99,
    ),
    RescuePolicy(
        policy_id="strict_low_prob_010_score_078",
        note="Stage1 확률 0.10 이하, rescue score 0.78 이상인 매우 보수적 gate",
        probability_ceiling=0.10,
        score_threshold=0.78,
    ),
    RescuePolicy(
        policy_id="conservative_group2_prob030_score065",
        note="기본 후보. Stage1 정상 구간에서 stress group 2개 이상이 강한 KOSDAQ 제조업만 추가 검토",
        probability_ceiling=FN_RESCUE_DEFAULT_PROB_CEILING,
        score_threshold=FN_RESCUE_DEFAULT_SCORE_THRESHOLD,
    ),
    RescuePolicy(
        policy_id="moderate_group1_prob030_score055",
        note="Review load를 조금 더 허용하는 후보. stress group 1개 이상이면 추가 검토",
        probability_ceiling=0.30,
        score_threshold=0.55,
        min_group_count=1,
    ),
    RescuePolicy(
        policy_id="recall_prob030_score045",
        note="Recall 우선 참고 후보. FP 비용이 커 운영 기본값으로는 부적합할 수 있음",
        probability_ceiling=0.30,
        score_threshold=0.45,
        min_group_count=0,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic KOSDAQ manufacturing FN-rescue Stage 2 gates "
            "on top of the official credit_46_features XGBoost model."
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


def read_raw_rescue_sources(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"stock_code": str},
        usecols=[*MERGE_KEYS, *FN_RESCUE_RAW_COLUMNS],
    )
    raw["stock_code"] = raw["stock_code"].astype("string").str.zfill(6)
    duplicates = int(raw.duplicated(MERGE_KEYS).sum())
    if duplicates:
        raw = raw.sort_values(MERGE_KEYS).drop_duplicates(MERGE_KEYS, keep="last")
    for column in FN_RESCUE_RAW_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    return raw


def build_experiment_frame(master: pd.DataFrame, raw_ts2000_path: Path) -> pd.DataFrame:
    raw = read_raw_rescue_sources(raw_ts2000_path)
    frame = master.merge(raw, on=MERGE_KEYS, how="left", validate="one_to_one")
    return add_manufacturing_fn_rescue_scores(frame)


def evaluate_stage1_split(
    *,
    train: pd.DataFrame,
    policy: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model = train_stage1_xgboost(
        train=train,
        policy=policy,
        columns=feature_columns,
        seed=seed,
    )
    metrics, probabilities = evaluate_calibrated_stage1_split(
        model=model,
        policy=policy,
        evaluation=evaluation,
        columns=feature_columns,
    )
    scored = evaluation.loc[:, [*ID_COLUMNS, "is_speculative", *FN_RESCUE_SCORE_COLUMNS]].copy()
    scored["prob_speculative"] = probabilities
    scored["pred_label_tuned"] = (
        scored["prob_speculative"].astype(float) >= float(metrics["threshold_tuned"])
    ).astype(int)
    return metrics, scored


def gate_metrics(scored: pd.DataFrame, policy: RescuePolicy) -> dict[str, Any]:
    y_true = scored["is_speculative"].astype(int)
    stage1_predictions = scored["pred_label_tuned"].astype(int)
    if policy.policy_id == "baseline_stage1_only":
        rescue_trigger = pd.Series(False, index=scored.index)
    else:
        rescue_trigger = build_manufacturing_fn_rescue_gate(
            scored,
            probability_ceiling=policy.probability_ceiling,
            score_threshold=policy.score_threshold,
            min_group_count=policy.min_group_count,
        )
    combined_predictions = stage1_predictions.astype(bool) | rescue_trigger.astype(bool)
    target_segment = scored.eval(TARGET_SEGMENT_QUERY)
    target_y = y_true.loc[target_segment]
    target_stage1 = stage1_predictions.loc[target_segment].astype(bool)
    target_combined = combined_predictions.loc[target_segment]
    base = classification_metrics(y_true, stage1_predictions)
    combined = classification_metrics(y_true, combined_predictions)
    target_base = classification_metrics(target_y, target_stage1)
    target_rescue = classification_metrics(target_y, target_combined)
    extra_true_risk = int((rescue_trigger & y_true.eq(1)).sum())
    extra_normal = int((rescue_trigger & y_true.eq(0)).sum())
    return {
        **asdict(policy),
        "stage1_precision": base["precision"],
        "stage1_recall": base["recall"],
        "stage1_f1": base["f1"],
        "stage1_false_positive": base["false_positive"],
        "stage1_false_negative": base["false_negative"],
        "trigger_precision": combined["precision"],
        "trigger_recall": combined["recall"],
        "trigger_f1": combined["f1"],
        "trigger_false_positive": combined["false_positive"],
        "trigger_false_negative": combined["false_negative"],
        "rescue_trigger_count": int(rescue_trigger.sum()),
        "rescue_true_risk_count": extra_true_risk,
        "rescue_normal_count": extra_normal,
        "rescue_precision": extra_true_risk / int(rescue_trigger.sum())
        if int(rescue_trigger.sum())
        else 0.0,
        "target_stage1_recall": target_base["recall"],
        "target_stage1_false_negative": target_base["false_negative"],
        "target_trigger_recall": target_rescue["recall"],
        "target_trigger_false_negative": target_rescue["false_negative"],
    }


def evaluate_fold(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    eval_year: int,
    seed: int,
) -> list[dict[str, Any]]:
    policy_year = eval_year - 1
    train = frame.loc[frame["fiscal_year"] < policy_year].copy()
    policy = frame.loc[frame["fiscal_year"] == policy_year].copy()
    evaluation = frame.loc[frame["fiscal_year"] == eval_year].copy()
    stage1_metrics, scored = evaluate_stage1_split(
        train=train,
        policy=policy,
        evaluation=evaluation,
        feature_columns=feature_columns,
        seed=seed,
    )
    rows = []
    for rescue_policy in RESCUE_POLICIES:
        rows.append(
            {
                "eval_year": eval_year,
                "policy_year": policy_year,
                "train_rows": len(train),
                "policy_rows": len(policy),
                "eval_rows": len(evaluation),
                "eval_positive_rate": float(evaluation["is_speculative"].mean()),
                "threshold_tuned": float(stage1_metrics["threshold_tuned"]),
                **gate_metrics(scored, rescue_policy),
            }
        )
    return rows


def evaluate_final_test(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    train = frame.loc[frame["fiscal_year"] <= 2021].copy()
    policy = frame.loc[frame["fiscal_year"] == 2022].copy()
    evaluation = frame.loc[frame["fiscal_year"] >= 2023].copy()
    stage1_metrics, scored = evaluate_stage1_split(
        train=train,
        policy=policy,
        evaluation=evaluation,
        feature_columns=feature_columns,
        seed=seed,
    )
    rows = []
    for rescue_policy in RESCUE_POLICIES:
        rows.append(
            {
                "train_rows": len(train),
                "policy_rows": len(policy),
                "eval_rows": len(evaluation),
                "eval_positive_rate": float(evaluation["is_speculative"].mean()),
                "threshold_tuned": float(stage1_metrics["threshold_tuned"]),
                **gate_metrics(scored, rescue_policy),
            }
        )
    return rows


def summarize_rolling(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "trigger_precision",
        "trigger_recall",
        "trigger_f1",
        "target_trigger_recall",
    ]
    rows: list[dict[str, Any]] = []
    for policy_id, group in fold_metrics.groupby("policy_id", sort=False):
        row: dict[str, Any] = {
            "policy_id": policy_id,
            "note": group["note"].iloc[0],
            "probability_ceiling": float(group["probability_ceiling"].iloc[0]),
            "score_threshold": float(group["score_threshold"].iloc[0]),
            "min_group_count": int(group["min_group_count"].iloc[0]),
            "folds": len(group),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
        for column in [
            "trigger_false_positive",
            "trigger_false_negative",
            "rescue_trigger_count",
            "rescue_true_risk_count",
            "rescue_normal_count",
            "target_trigger_false_negative",
        ]:
            row[f"{column}_sum"] = int(group[column].sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["policy_id"].eq("baseline_stage1_only")].iloc[0]
    summary["rolling_f1_delta_vs_baseline"] = summary["trigger_f1_mean"] - float(
        baseline["trigger_f1_mean"]
    )
    summary["rolling_recall_delta_vs_baseline"] = summary["trigger_recall_mean"] - float(
        baseline["trigger_recall_mean"]
    )
    summary["rolling_fp_delta_vs_baseline"] = summary["trigger_false_positive_sum"] - int(
        baseline["trigger_false_positive_sum"]
    )
    summary["rolling_fn_delta_vs_baseline"] = summary["trigger_false_negative_sum"] - int(
        baseline["trigger_false_negative_sum"]
    )
    summary["rolling_target_fn_delta_vs_baseline"] = summary[
        "target_trigger_false_negative_sum"
    ] - int(baseline["target_trigger_false_negative_sum"])
    return summary.sort_values(
        [
            "rolling_fn_delta_vs_baseline",
            "rolling_f1_delta_vs_baseline",
            "rescue_normal_count_sum",
        ],
        ascending=[True, False, True],
    )


def merge_final(summary: pd.DataFrame, final_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "policy_id",
        "trigger_precision",
        "trigger_recall",
        "trigger_f1",
        "trigger_false_positive",
        "trigger_false_negative",
        "rescue_trigger_count",
        "rescue_true_risk_count",
        "rescue_normal_count",
        "target_trigger_recall",
        "target_trigger_false_negative",
    ]
    final = final_metrics.loc[:, columns].rename(
        columns={column: f"final_{column}" for column in columns if column != "policy_id"}
    )
    output = summary.merge(final, on="policy_id", how="left")
    baseline = output.loc[output["policy_id"].eq("baseline_stage1_only")].iloc[0]
    output["final_f1_delta_vs_baseline"] = output["final_trigger_f1"] - float(
        baseline["final_trigger_f1"]
    )
    output["final_recall_delta_vs_baseline"] = output["final_trigger_recall"] - float(
        baseline["final_trigger_recall"]
    )
    output["final_fp_delta_vs_baseline"] = output["final_trigger_false_positive"] - int(
        baseline["final_trigger_false_positive"]
    )
    output["final_fn_delta_vs_baseline"] = output["final_trigger_false_negative"] - int(
        baseline["final_trigger_false_negative"]
    )
    output["final_target_fn_delta_vs_baseline"] = output[
        "final_target_trigger_false_negative"
    ] - int(baseline["final_target_trigger_false_negative"])
    return output


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


def build_report(
    *,
    fold_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    eval_years: list[int],
) -> str:
    baseline = summary.loc[summary["policy_id"].eq("baseline_stage1_only")].iloc[0]
    best = summary.iloc[0]
    recommended = summary.loc[summary["policy_id"].eq("conservative_group2_prob030_score065")].iloc[
        0
    ]
    baseline_rows = fold_metrics.loc[fold_metrics["policy_id"].eq("baseline_stage1_only")]
    best_rows = fold_metrics.loc[fold_metrics["policy_id"].eq(str(best["policy_id"]))]
    final_ranked = final_metrics.sort_values(
        ["trigger_false_negative", "trigger_f1", "rescue_normal_count"],
        ascending=[True, False, True],
    )
    return "\n".join(
        [
            "# 46-Feature Manufacturing/KOSDAQ FN Rescue Gate Experiments",
            "",
            "공식 `feature_46_xgboost` Stage1 판단은 유지하고, 낮은 Stage1 확률로 놓치는 "
            "KOSDAQ 제조업 후보를 Stage2 에이전트 검토 대상으로 올리는 deterministic gate를 비교했습니다.",
            "",
            f"Rolling 평가연도는 `{', '.join(str(year) for year in eval_years)}`이고, "
            "Final Test는 공식 test split인 2023~2024 구간입니다.",
            "",
            "## 1. 결론",
            "",
            f"- Baseline rolling F1/Recall: `{format_metric(baseline['trigger_f1_mean'])}` / "
            f"`{format_metric(baseline['trigger_recall_mean'])}`",
            f"- Rolling 기준 FN 최소 후보: `{best['policy_id']}`",
            f"- Rolling FN 변화: `{format_int(best['rolling_fn_delta_vs_baseline'])}`",
            f"- Rolling FP 변화: `{format_int(best['rolling_fp_delta_vs_baseline'])}`",
            f"- Final Test FN 변화: `{format_int(best['final_fn_delta_vs_baseline'])}`",
            f"- 운영 기본 후보: `{recommended['policy_id']}` "
            f"(Rolling FN `{format_int(recommended['rolling_fn_delta_vs_baseline'])}`, "
            f"FP `{format_int(recommended['rolling_fp_delta_vs_baseline'])}` / "
            f"Final FN `{format_int(recommended['final_fn_delta_vs_baseline'])}`, "
            f"FP `{format_int(recommended['final_fp_delta_vs_baseline'])}`)",
            "",
            "## 2. 후보별 Rolling + Final Test 비교",
            "",
            markdown_table(
                summary,
                [
                    ("Policy", "policy_id", "text"),
                    ("Prob <= ", "probability_ceiling", "metric"),
                    ("Score >=", "score_threshold", "metric"),
                    ("Groups", "min_group_count", "int"),
                    ("Roll P", "trigger_precision_mean", "metric"),
                    ("Roll R", "trigger_recall_mean", "metric"),
                    ("Roll F1", "trigger_f1_mean", "metric"),
                    ("Roll FP", "trigger_false_positive_sum", "int"),
                    ("Roll FN", "trigger_false_negative_sum", "int"),
                    ("Roll dFN", "rolling_fn_delta_vs_baseline", "int"),
                    ("Roll extra TP", "rescue_true_risk_count_sum", "int"),
                    ("Roll extra FP", "rescue_normal_count_sum", "int"),
                    ("Final P", "final_trigger_precision", "metric"),
                    ("Final R", "final_trigger_recall", "metric"),
                    ("Final F1", "final_trigger_f1", "metric"),
                    ("Final FP", "final_trigger_false_positive", "int"),
                    ("Final FN", "final_trigger_false_negative", "int"),
                    ("Final dFN", "final_fn_delta_vs_baseline", "int"),
                    ("Final extra TP", "final_rescue_true_risk_count", "int"),
                    ("Final extra FP", "final_rescue_normal_count", "int"),
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
                    ("Precision", "trigger_precision", "metric"),
                    ("Recall", "trigger_recall", "metric"),
                    ("F1", "trigger_f1", "metric"),
                    ("FP", "trigger_false_positive", "int"),
                    ("FN", "trigger_false_negative", "int"),
                    ("Target FN", "target_trigger_false_negative", "int"),
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
                    ("Precision", "trigger_precision", "metric"),
                    ("Recall", "trigger_recall", "metric"),
                    ("F1", "trigger_f1", "metric"),
                    ("FP", "trigger_false_positive", "int"),
                    ("FN", "trigger_false_negative", "int"),
                    ("Extra TP", "rescue_true_risk_count", "int"),
                    ("Extra FP", "rescue_normal_count", "int"),
                    ("Target FN", "target_trigger_false_negative", "int"),
                ],
            ),
            "",
            "## 5. 참고용 Final Test 순위",
            "",
            markdown_table(
                final_ranked,
                [
                    ("Policy", "policy_id", "text"),
                    ("Precision", "trigger_precision", "metric"),
                    ("Recall", "trigger_recall", "metric"),
                    ("F1", "trigger_f1", "metric"),
                    ("FP", "trigger_false_positive", "int"),
                    ("FN", "trigger_false_negative", "int"),
                    ("Extra TP", "rescue_true_risk_count", "int"),
                    ("Extra FP", "rescue_normal_count", "int"),
                    ("Target Recall", "target_trigger_recall", "metric"),
                    ("Target FN", "target_trigger_false_negative", "int"),
                ],
            ),
            "",
            "## 6. Gate 정의",
            "",
            f"- Policy name: `{FN_RESCUE_POLICY_NAME}`",
            "- 대상: `market == KOSDAQ` and `industry_macro_category == manufacturing`",
            "- 공식 Stage1 예측은 정상이고, Stage1 확률이 policy ceiling 이하인 회사만 검토",
            "- Rescue score는 working capital stress, cashflow turn stress, borrowing pressure score를 조합",
            "- 이 gate는 공식 Stage1 판정을 바꾸지 않고 Stage2 에이전트 검토 큐에만 추가합니다.",
        ]
    )


def write_outputs(
    *,
    fold_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    eval_years: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fold_metrics": output_dir / "manufacturing_fn_rescue_gate_rolling_fold_metrics.csv",
        "final_test": output_dir / "manufacturing_fn_rescue_gate_final_test.csv",
        "summary": output_dir / "manufacturing_fn_rescue_gate_summary.csv",
        "report": output_dir / "manufacturing_fn_rescue_gate_report.md",
        "metadata": output_dir / "manufacturing_fn_rescue_gate_summary.json",
    }
    fold_metrics.to_csv(paths["fold_metrics"], index=False, encoding="utf-8-sig")
    final_metrics.to_csv(paths["final_test"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(
        build_report(
            fold_metrics=fold_metrics,
            final_metrics=final_metrics,
            summary=summary,
            eval_years=eval_years,
        ),
        encoding="utf-8",
    )
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": "feature_46_xgboost",
        "dataset": "credit_46_features",
        "policy_name": FN_RESCUE_POLICY_NAME,
        "eval_years": eval_years,
        "fold_policy": "train fiscal_year < eval_year-1, tune on eval_year-1, evaluate eval_year",
        "final_test_policy": "train fiscal_year <= 2021, tune on 2022, evaluate 2023-2024",
        "rescue_policies": [asdict(policy) for policy in RESCUE_POLICIES],
        "score_columns": FN_RESCUE_SCORE_COLUMNS,
        "raw_columns": FN_RESCUE_RAW_COLUMNS,
        "best_by_rolling_fn": summary.iloc[0].to_dict(),
        "recommended_operating_policy": summary.loc[
            summary["policy_id"].eq("conservative_group2_prob030_score065")
        ]
        .iloc[0]
        .to_dict(),
        "output_files": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    master = read_master(args.master_path)
    frame = build_experiment_frame(master, args.raw_ts2000_path)
    feature_columns = read_stage1_feature_columns(args.feature_list_path, frame)

    fold_rows: list[dict[str, Any]] = []
    for eval_year in args.eval_years:
        fold_rows.extend(
            evaluate_fold(
                frame=frame,
                feature_columns=feature_columns,
                eval_year=eval_year,
                seed=args.seed,
            )
        )
    fold_metrics = pd.DataFrame(fold_rows)
    final_metrics = pd.DataFrame(
        evaluate_final_test(frame=frame, feature_columns=feature_columns, seed=args.seed)
    )
    summary = merge_final(summarize_rolling(fold_metrics), final_metrics)
    write_outputs(
        fold_metrics=fold_metrics,
        final_metrics=final_metrics,
        summary=summary,
        output_dir=args.output_dir,
        eval_years=args.eval_years,
    )
    best = summary.iloc[0]
    print(
        json.dumps(
            {
                "best_policy": best["policy_id"],
                "rolling_fn_delta_vs_baseline": int(best["rolling_fn_delta_vs_baseline"]),
                "rolling_fp_delta_vs_baseline": int(best["rolling_fp_delta_vs_baseline"]),
                "final_fn_delta_vs_baseline": int(best["final_fn_delta_vs_baseline"]),
                "final_fp_delta_vs_baseline": int(best["final_fp_delta_vs_baseline"]),
                "report": str(
                    (args.output_dir / "manufacturing_fn_rescue_gate_report.md").relative_to(ROOT)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
