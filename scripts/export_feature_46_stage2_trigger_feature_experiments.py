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

from cas.modeling.calibration import (  # noqa: E402
    apply_probability_calibration,
    choose_max_precision_threshold_at_recall,
    choose_tuned_threshold,
    fit_platt_calibration,
)
from cas.modeling.fn_rescue import (  # noqa: E402
    FN_RESCUE_RAW_COLUMNS,
    FN_RESCUE_SCORE_COLUMNS,
    add_manufacturing_fn_rescue_scores,
)
from cas.modeling.stage1_xgboost import (  # noqa: E402
    DEFAULT_ROLLING_EVAL_YEARS,
    DEFAULT_STAGE1_RANDOM_STATE,
    classification_metrics,
    probability_metrics,
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
STAGE2_IT_SERVICES_RECALL_FLOOR = 0.90
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

STAGE2_AUX_BASE_COLUMNS = ["delta_accruals_ratio", "is_3y_consecutive_operating_loss"]
WORKING_CAPITAL_COLUMNS = [
    "accounts_receivable_ratio",
    "inventory_ratio",
    "contract_assets_ratio",
    "ar_days_diff",
    "inventory_days_diff",
    "ap_days_diff",
]
CASHFLOW_TURN_COLUMNS = [
    "is_ocf_turn_negative",
    "is_operating_income_turn_negative",
    "ocf_to_total_borrowings_diff",
    "ocf_to_total_liabilities_diff",
    "cash_ratio_diff",
]
BORROWING_PRESSURE_COLUMNS = [
    "delta_st_borrowings_share",
    "total_borrowings_growth",
    "current_ratio_diff",
    "capital_impairment_diff",
]
MACRO_REGIME_COLUMNS = [
    "market_spread_diff",
    "spec_spread_diff",
    "base_rate_diff",
    "treasury_3y_diff",
    "usd_krw_diff",
]

RAW_CANDIDATE_COLUMNS = sorted(
    {
        *STAGE2_AUX_BASE_COLUMNS,
        *WORKING_CAPITAL_COLUMNS,
        *CASHFLOW_TURN_COLUMNS,
        *BORROWING_PRESSURE_COLUMNS,
        *MACRO_REGIME_COLUMNS,
        *FN_RESCUE_RAW_COLUMNS,
    }
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    note: str
    added_columns: tuple[str, ...]


CANDIDATES = [
    CandidateSpec(
        candidate_id="stage2_aux_48_baseline",
        note="현재 운영 기준 Stage2 aux feature set",
        added_columns=tuple(STAGE2_AUX_BASE_COLUMNS),
    ),
    CandidateSpec(
        candidate_id="stage2_aux_plus_fn_rescue_scores_53",
        note="기존 aux에 제조업 FN rescue composite score 5개 추가",
        added_columns=tuple([*STAGE2_AUX_BASE_COLUMNS, *FN_RESCUE_SCORE_COLUMNS]),
    ),
    CandidateSpec(
        candidate_id="working_capital_trigger_54",
        note="매출채권, 재고, 운전자본 shock 원천 변수 추가",
        added_columns=tuple([*STAGE2_AUX_BASE_COLUMNS, *WORKING_CAPITAL_COLUMNS]),
    ),
    CandidateSpec(
        candidate_id="cashflow_turn_trigger_53",
        note="OCF/영업손익 turn negative와 현금흐름 악화 diff 추가",
        added_columns=tuple([*STAGE2_AUX_BASE_COLUMNS, *CASHFLOW_TURN_COLUMNS]),
    ),
    CandidateSpec(
        candidate_id="borrowing_pressure_trigger_52",
        note="단기차입, 총차입 증가, 유동성/자본잠식 악화 diff 추가",
        added_columns=tuple([*STAGE2_AUX_BASE_COLUMNS, *BORROWING_PRESSURE_COLUMNS]),
    ),
    CandidateSpec(
        candidate_id="fn_rescue_raw_trigger_64",
        note="제조업 FN rescue 원천 변수 전체 추가",
        added_columns=tuple([*STAGE2_AUX_BASE_COLUMNS, *FN_RESCUE_RAW_COLUMNS]),
    ),
    CandidateSpec(
        candidate_id="macro_regime_trigger_53",
        note="금리, 스프레드, 환율 변화량을 보조 트리거 모델에 추가",
        added_columns=tuple([*STAGE2_AUX_BASE_COLUMNS, *MACRO_REGIME_COLUMNS]),
    ),
    CandidateSpec(
        candidate_id="full_review_trigger_73",
        note="aux, FN rescue score, raw rescue, macro regime 후보를 모두 추가",
        added_columns=tuple(
            [
                *STAGE2_AUX_BASE_COLUMNS,
                *FN_RESCUE_SCORE_COLUMNS,
                *FN_RESCUE_RAW_COLUMNS,
                *MACRO_REGIME_COLUMNS,
            ]
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Stage2 review trigger feature sets against stage2_aux_48."
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


def read_raw_candidates(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"stock_code": str},
        usecols=[*MERGE_KEYS, *RAW_CANDIDATE_COLUMNS],
    )
    raw["stock_code"] = raw["stock_code"].astype("string").str.zfill(6)
    duplicates = int(raw.duplicated(MERGE_KEYS).sum())
    if duplicates:
        raw = raw.sort_values(MERGE_KEYS).drop_duplicates(MERGE_KEYS, keep="last")
    for column in RAW_CANDIDATE_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    return raw


def build_feature_frame(master: pd.DataFrame, raw_ts2000_path: Path) -> pd.DataFrame:
    raw = read_raw_candidates(raw_ts2000_path)
    frame = master.merge(raw, on=MERGE_KEYS, how="left", validate="one_to_one")
    return add_manufacturing_fn_rescue_scores(frame)


def candidate_feature_columns(base_columns: list[str], spec: CandidateSpec) -> list[str]:
    columns = list(base_columns)
    seen = set(columns)
    for column in spec.added_columns:
        if column in seen:
            continue
        columns.append(column)
        seen.add(column)
    return columns


def unique_added_columns(base_columns: list[str], spec: CandidateSpec) -> list[str]:
    seen = set(base_columns)
    added: list[str] = []
    for column in spec.added_columns:
        if column in seen:
            continue
        added.append(column)
        seen.add(column)
    return added


def score_model_split(
    *,
    train: pd.DataFrame,
    policy: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: list[str],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model = train_stage1_xgboost(
        train=train,
        policy=policy,
        columns=columns,
        seed=seed,
    )
    y_policy = policy["is_speculative"].astype(int)
    y_eval = evaluation["is_speculative"].astype(int)
    policy_raw = model.predict_proba(policy.loc[:, columns])[:, 1]
    eval_raw = model.predict_proba(evaluation.loc[:, columns])[:, 1]
    calibration = fit_platt_calibration(y_policy, policy_raw)
    policy_prob = apply_probability_calibration(policy_raw, calibration)
    eval_prob = apply_probability_calibration(eval_raw, calibration)
    threshold = choose_tuned_threshold(y_policy, policy_prob)
    policy_ids = policy.reset_index(drop=True)
    it_mask = policy_ids["industry_macro_category"].astype(str).eq("it_services").to_numpy()
    if it_mask.any():
        it_threshold = choose_max_precision_threshold_at_recall(
            y_policy.loc[it_mask],
            policy_prob[it_mask],
            STAGE2_IT_SERVICES_RECALL_FLOOR,
        )
    else:
        it_threshold = threshold
    metrics = {
        "threshold": float(threshold),
        "it_services_threshold": float(it_threshold),
        **{f"eval_{key}": value for key, value in probability_metrics(y_eval, eval_prob).items()},
    }
    return metrics, policy_prob, eval_prob


def evaluate_trigger_metrics(
    *,
    evaluation: pd.DataFrame,
    stage1_probabilities: np.ndarray,
    stage1_threshold: float,
    aux_probabilities: np.ndarray,
    aux_threshold: float,
    aux_it_services_threshold: float,
) -> dict[str, Any]:
    y_true = evaluation["is_speculative"].astype(int)
    stage1_risk = pd.Series(stage1_probabilities >= stage1_threshold, index=evaluation.index)
    aux_risk = pd.Series(aux_probabilities >= aux_threshold, index=evaluation.index)
    it_services_review = evaluation["industry_macro_category"].astype(str).eq(
        "it_services"
    ) & pd.Series(aux_probabilities >= aux_it_services_threshold, index=evaluation.index)
    secondary_trigger = (~stage1_risk) & (aux_risk | it_services_review)
    trigger = stage1_risk | secondary_trigger
    stage1_metrics = classification_metrics(y_true, stage1_risk)
    trigger_metrics = classification_metrics(y_true, trigger)
    secondary_true = int((secondary_trigger & y_true.eq(1)).sum())
    secondary_normal = int((secondary_trigger & y_true.eq(0)).sum())
    return {
        "stage1_precision": stage1_metrics["precision"],
        "stage1_recall": stage1_metrics["recall"],
        "stage1_f1": stage1_metrics["f1"],
        "stage1_false_positive": stage1_metrics["false_positive"],
        "stage1_false_negative": stage1_metrics["false_negative"],
        "trigger_precision": trigger_metrics["precision"],
        "trigger_recall": trigger_metrics["recall"],
        "trigger_f1": trigger_metrics["f1"],
        "trigger_false_positive": trigger_metrics["false_positive"],
        "trigger_false_negative": trigger_metrics["false_negative"],
        "trigger_true_positive": trigger_metrics["true_positive"],
        "trigger_true_negative": trigger_metrics["true_negative"],
        "stage2_secondary_trigger_count": int(secondary_trigger.sum()),
        "stage2_secondary_true_risk_count": secondary_true,
        "stage2_secondary_normal_count": secondary_normal,
        "stage2_secondary_precision": secondary_true / int(secondary_trigger.sum())
        if int(secondary_trigger.sum())
        else 0.0,
    }


def evaluate_candidate_fold(
    *,
    frame: pd.DataFrame,
    base_columns: list[str],
    spec: CandidateSpec,
    eval_year: int,
    seed: int,
) -> dict[str, Any]:
    policy_year = eval_year - 1
    train = frame.loc[frame["fiscal_year"] < policy_year].copy()
    policy = frame.loc[frame["fiscal_year"] == policy_year].copy()
    evaluation = frame.loc[frame["fiscal_year"] == eval_year].copy()
    stage1_metrics, _, stage1_prob = score_model_split(
        train=train,
        policy=policy,
        evaluation=evaluation,
        columns=base_columns,
        seed=seed,
    )
    columns = candidate_feature_columns(base_columns, spec)
    aux_metrics, _, aux_prob = score_model_split(
        train=train,
        policy=policy,
        evaluation=evaluation,
        columns=columns,
        seed=seed,
    )
    return {
        "candidate_id": spec.candidate_id,
        "note": spec.note,
        "added_features": ", ".join(spec.added_columns),
        "added_feature_count": len(unique_added_columns(base_columns, spec)),
        "feature_count": len(columns),
        "eval_year": eval_year,
        "policy_year": policy_year,
        "train_rows": len(train),
        "policy_rows": len(policy),
        "eval_rows": len(evaluation),
        "eval_positive_rate": float(evaluation["is_speculative"].mean()),
        "stage1_threshold": stage1_metrics["threshold"],
        "stage2_aux_threshold": aux_metrics["threshold"],
        "stage2_aux_it_services_threshold": aux_metrics["it_services_threshold"],
        "stage2_aux_pr_auc": aux_metrics["eval_pr_auc"],
        **evaluate_trigger_metrics(
            evaluation=evaluation,
            stage1_probabilities=stage1_prob,
            stage1_threshold=float(stage1_metrics["threshold"]),
            aux_probabilities=aux_prob,
            aux_threshold=float(aux_metrics["threshold"]),
            aux_it_services_threshold=float(aux_metrics["it_services_threshold"]),
        ),
    }


def evaluate_candidate_final_test(
    *,
    frame: pd.DataFrame,
    base_columns: list[str],
    spec: CandidateSpec,
    seed: int,
) -> dict[str, Any]:
    train = frame.loc[frame["fiscal_year"] <= 2021].copy()
    policy = frame.loc[frame["fiscal_year"] == 2022].copy()
    evaluation = frame.loc[frame["fiscal_year"] >= 2023].copy()
    stage1_metrics, _, stage1_prob = score_model_split(
        train=train,
        policy=policy,
        evaluation=evaluation,
        columns=base_columns,
        seed=seed,
    )
    columns = candidate_feature_columns(base_columns, spec)
    aux_metrics, _, aux_prob = score_model_split(
        train=train,
        policy=policy,
        evaluation=evaluation,
        columns=columns,
        seed=seed,
    )
    return {
        "candidate_id": spec.candidate_id,
        "note": spec.note,
        "added_features": ", ".join(spec.added_columns),
        "added_feature_count": len(unique_added_columns(base_columns, spec)),
        "feature_count": len(columns),
        "train_rows": len(train),
        "policy_rows": len(policy),
        "eval_rows": len(evaluation),
        "eval_positive_rate": float(evaluation["is_speculative"].mean()),
        "stage1_threshold": stage1_metrics["threshold"],
        "stage2_aux_threshold": aux_metrics["threshold"],
        "stage2_aux_it_services_threshold": aux_metrics["it_services_threshold"],
        "stage2_aux_pr_auc": aux_metrics["eval_pr_auc"],
        **evaluate_trigger_metrics(
            evaluation=evaluation,
            stage1_probabilities=stage1_prob,
            stage1_threshold=float(stage1_metrics["threshold"]),
            aux_probabilities=aux_prob,
            aux_threshold=float(aux_metrics["threshold"]),
            aux_it_services_threshold=float(aux_metrics["it_services_threshold"]),
        ),
    }


def summarize_rolling(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "stage2_aux_pr_auc",
        "trigger_precision",
        "trigger_recall",
        "trigger_f1",
        "stage2_secondary_precision",
    ]
    rows: list[dict[str, Any]] = []
    for candidate_id, group in fold_metrics.groupby("candidate_id", sort=False):
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "note": group["note"].iloc[0],
            "added_features": group["added_features"].iloc[0],
            "added_feature_count": int(group["added_feature_count"].iloc[0]),
            "feature_count": int(group["feature_count"].iloc[0]),
            "folds": int(len(group)),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
        for column in [
            "trigger_false_positive",
            "trigger_false_negative",
            "stage2_secondary_trigger_count",
            "stage2_secondary_true_risk_count",
            "stage2_secondary_normal_count",
        ]:
            row[f"{column}_sum"] = int(group[column].sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["candidate_id"].eq("stage2_aux_48_baseline")].iloc[0]
    summary["rolling_recall_delta_vs_stage2_aux_48"] = (
        summary["trigger_recall_mean"] - float(baseline["trigger_recall_mean"])
    )
    summary["rolling_f1_delta_vs_stage2_aux_48"] = (
        summary["trigger_f1_mean"] - float(baseline["trigger_f1_mean"])
    )
    summary["rolling_fp_delta_vs_stage2_aux_48"] = (
        summary["trigger_false_positive_sum"] - int(baseline["trigger_false_positive_sum"])
    )
    summary["rolling_fn_delta_vs_stage2_aux_48"] = (
        summary["trigger_false_negative_sum"] - int(baseline["trigger_false_negative_sum"])
    )
    return summary.sort_values(
        [
            "rolling_recall_delta_vs_stage2_aux_48",
            "rolling_f1_delta_vs_stage2_aux_48",
            "rolling_fp_delta_vs_stage2_aux_48",
        ],
        ascending=[False, False, True],
    )


def merge_final_metrics(summary: pd.DataFrame, final_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "stage2_aux_pr_auc",
        "trigger_precision",
        "trigger_recall",
        "trigger_f1",
        "trigger_false_positive",
        "trigger_false_negative",
        "stage2_secondary_trigger_count",
        "stage2_secondary_true_risk_count",
        "stage2_secondary_normal_count",
        "stage2_secondary_precision",
    ]
    final = final_metrics.loc[:, columns].rename(
        columns={column: f"final_{column}" for column in columns if column != "candidate_id"}
    )
    output = summary.merge(final, on="candidate_id", how="left")
    baseline = output.loc[output["candidate_id"].eq("stage2_aux_48_baseline")].iloc[0]
    output["final_recall_delta_vs_stage2_aux_48"] = (
        output["final_trigger_recall"] - float(baseline["final_trigger_recall"])
    )
    output["final_f1_delta_vs_stage2_aux_48"] = (
        output["final_trigger_f1"] - float(baseline["final_trigger_f1"])
    )
    output["final_fp_delta_vs_stage2_aux_48"] = (
        output["final_trigger_false_positive"] - int(baseline["final_trigger_false_positive"])
    )
    output["final_fn_delta_vs_stage2_aux_48"] = (
        output["final_trigger_false_negative"] - int(baseline["final_trigger_false_negative"])
    )
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
    baseline = summary.loc[summary["candidate_id"].eq("stage2_aux_48_baseline")].iloc[0]
    best_recall = summary.iloc[0]
    final_ranked = final_metrics.sort_values(
        ["trigger_recall", "trigger_f1", "trigger_false_positive"],
        ascending=[False, False, True],
    )
    return "\n".join(
        [
            "# 46-Feature Stage2 Trigger Feature Set Experiments",
            "",
            "공식 Stage1 `feature_46_xgboost` 판단은 유지하고, Stage2 review aux 모델의 "
            "후보 feature set을 바꿔 combined Stage2 trigger 성능을 비교했습니다.",
            "",
            f"Rolling 평가연도는 `{', '.join(str(year) for year in eval_years)}`이고, "
            "Final Test는 공식 test split인 2023~2024 구간입니다.",
            "각 fold는 `과거 연도 학습 -> 직전 1년 Platt calibration/threshold 선택 -> 다음 1년 평가` 구조입니다.",
            "",
            "## 1. 결론",
            "",
            f"- 기준선 `stage2_aux_48_baseline` rolling Recall/F1: "
            f"`{format_metric(baseline['trigger_recall_mean'])}` / "
            f"`{format_metric(baseline['trigger_f1_mean'])}`",
            f"- Rolling Recall 최상위 후보: `{best_recall['candidate_id']}` "
            f"(Recall delta `{format_signed(best_recall['rolling_recall_delta_vs_stage2_aux_48'])}`, "
            f"FN delta `{format_int(best_recall['rolling_fn_delta_vs_stage2_aux_48'])}`)",
            "",
            "## 2. 후보별 Rolling + Final Test 비교",
            "",
            markdown_table(
                summary,
                [
                    ("Candidate", "candidate_id", "text"),
                    ("Features", "feature_count", "int"),
                    ("Roll Aux PR", "stage2_aux_pr_auc_mean", "metric"),
                    ("Roll P", "trigger_precision_mean", "metric"),
                    ("Roll R", "trigger_recall_mean", "metric"),
                    ("Roll F1", "trigger_f1_mean", "metric"),
                    ("Roll dR", "rolling_recall_delta_vs_stage2_aux_48", "signed"),
                    ("Roll FP", "trigger_false_positive_sum", "int"),
                    ("Roll FN", "trigger_false_negative_sum", "int"),
                    ("Roll dFN", "rolling_fn_delta_vs_stage2_aux_48", "int"),
                    ("Roll Extra TP", "stage2_secondary_true_risk_count_sum", "int"),
                    ("Roll Extra FP", "stage2_secondary_normal_count_sum", "int"),
                    ("Final Aux PR", "final_stage2_aux_pr_auc", "metric"),
                    ("Final P", "final_trigger_precision", "metric"),
                    ("Final R", "final_trigger_recall", "metric"),
                    ("Final F1", "final_trigger_f1", "metric"),
                    ("Final dR", "final_recall_delta_vs_stage2_aux_48", "signed"),
                    ("Final FP", "final_trigger_false_positive", "int"),
                    ("Final FN", "final_trigger_false_negative", "int"),
                    ("Final dFN", "final_fn_delta_vs_stage2_aux_48", "int"),
                    ("Final Extra TP", "final_stage2_secondary_true_risk_count", "int"),
                    ("Final Extra FP", "final_stage2_secondary_normal_count", "int"),
                ],
            ),
            "",
            "## 3. 참고용 Final Test Recall 순위",
            "",
            markdown_table(
                final_ranked,
                [
                    ("Candidate", "candidate_id", "text"),
                    ("Aux PR-AUC", "stage2_aux_pr_auc", "metric"),
                    ("Precision", "trigger_precision", "metric"),
                    ("Recall", "trigger_recall", "metric"),
                    ("F1", "trigger_f1", "metric"),
                    ("FP", "trigger_false_positive", "int"),
                    ("FN", "trigger_false_negative", "int"),
                    ("Extra TP", "stage2_secondary_true_risk_count", "int"),
                    ("Extra FP", "stage2_secondary_normal_count", "int"),
                ],
            ),
            "",
            "## 4. 해석 주의",
            "",
            "- 이 실험은 공식 Stage1 판정을 덮어쓰지 않는 Stage2 review trigger 후보 비교입니다.",
            "- Recall이 올라가도 review load와 FP가 크게 늘면 운영 기본값으로는 부적합할 수 있습니다.",
            "- 후보 선택은 rolling OOT를 우선 기준으로 보고, Final Test는 사후 확인용입니다.",
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
        "fold_metrics": output_dir / "stage2_trigger_feature_set_rolling_fold_metrics.csv",
        "final_test": output_dir / "stage2_trigger_feature_set_final_test.csv",
        "summary": output_dir / "stage2_trigger_feature_set_summary.csv",
        "report": output_dir / "stage2_trigger_feature_set_report.md",
        "metadata": output_dir / "stage2_trigger_feature_set_summary.json",
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
        "baseline_candidate": "stage2_aux_48_baseline",
        "eval_years": eval_years,
        "it_services_recall_floor": STAGE2_IT_SERVICES_RECALL_FLOOR,
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "note": candidate.note,
                "added_columns": list(candidate.added_columns),
            }
            for candidate in CANDIDATES
        ],
        "best_by_rolling_recall": summary.iloc[0].to_dict(),
        "output_files": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    master = read_master(args.master_path)
    frame = build_feature_frame(master, args.raw_ts2000_path)
    base_columns = read_stage1_feature_columns(args.feature_list_path, frame)

    fold_rows = []
    final_rows = []
    for candidate in CANDIDATES:
        for eval_year in args.eval_years:
            fold_rows.append(
                evaluate_candidate_fold(
                    frame=frame,
                    base_columns=base_columns,
                    spec=candidate,
                    eval_year=eval_year,
                    seed=args.seed,
                )
            )
        final_rows.append(
            evaluate_candidate_final_test(
                frame=frame,
                base_columns=base_columns,
                spec=candidate,
                seed=args.seed,
            )
        )

    fold_metrics = pd.DataFrame(fold_rows)
    final_metrics = pd.DataFrame(final_rows)
    summary = merge_final_metrics(summarize_rolling(fold_metrics), final_metrics)
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
                "best_by_rolling_recall": best["candidate_id"],
                "rolling_recall_delta_vs_stage2_aux_48": float(
                    best["rolling_recall_delta_vs_stage2_aux_48"]
                ),
                "rolling_fn_delta_vs_stage2_aux_48": int(
                    best["rolling_fn_delta_vs_stage2_aux_48"]
                ),
                "final_recall_delta_vs_stage2_aux_48": float(
                    best["final_recall_delta_vs_stage2_aux_48"]
                ),
                "final_fn_delta_vs_stage2_aux_48": int(best["final_fn_delta_vs_stage2_aux_48"]),
                "report": str(
                    (args.output_dir / "stage2_trigger_feature_set_report.md").relative_to(ROOT)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
