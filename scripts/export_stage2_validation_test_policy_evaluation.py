"""Evaluate Stage 2 review policies on validation/test splits only.

This diagnostic deliberately avoids the 2026 external validation labels while
selecting candidate Stage 2 policies. The 2026 labels should remain a final
external check, not a tuning target.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cas.agents.nodes import committee_node, rule_engine_node
from cas.agents.stage2_review_signals import (
    LEGACY_STAGE2_REVIEW_AUX_IT_THRESHOLD_COLUMN,
    LEGACY_STAGE2_REVIEW_AUX_PROB_COLUMN,
    LEGACY_STAGE2_REVIEW_AUX_THRESHOLD_COLUMN,
    STAGE2_REVIEW_AUX_IT_THRESHOLD_COLUMN,
    STAGE2_REVIEW_AUX_PROB_COLUMN,
    STAGE2_REVIEW_AUX_THRESHOLD_COLUMN,
)

ROOT = Path(__file__).resolve().parents[1]
PREDICTION_SCORES_PATH = ROOT / "data/outputs/dashboard/feature_46_mvp/prediction_scores.csv"
FEATURE_MASTER_PATH = ROOT / "data/input/credit_46_features/feature_46_master.csv"
OUTPUT_DIR = ROOT / "data/outputs/modeling/feature_46_xgboost/diagnostics/stage2_agents"

KEY_COLUMNS = ["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]
EVALUATION_SPLITS = ("valid", "test")
VALIDATION_SPLIT = "valid"
TEST_SPLIT = "test"
RECALL_FLOOR = 0.88
TRACE_GATES = (
    ("stage1_model_view", "1차 모델 원판단"),
    ("veto_rule", "강제 경고 게이트"),
    ("hidden_tail_risk", "숨은 꼬리위험 점검"),
    ("secondary_review_trigger", "2차 보조 레이더"),
    ("boundary_rating_review", "경계등급 점검"),
    ("prior_hard_distress_context", "기준일 이전 severe 등급 컨텍스트"),
    ("overwarning_mitigation", "과민경고 완화 점검"),
    ("mitigation_residual_risk", "과민경고 완화 잔여위험"),
    ("risk_label_recall_adjustment", "위험 라벨 리콜 보정"),
    ("reject_confirmation", "부적격 확정 게이트"),
    ("final_committee_decision", "최종 위원회 판단"),
)
CONTRIBUTION_TRACE_GATES = tuple(
    item for item in TRACE_GATES if item[0] not in {"stage1_model_view", "final_committee_decision"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-scores", type=Path, default=PREDICTION_SCORES_PATH)
    parser.add_argument("--feature-master", type=Path, default=FEATURE_MASTER_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--skip-current-committee",
        action="store_true",
        help="Skip the deterministic committee replay and evaluate score-based policy columns only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_evaluation_frame(args.prediction_scores, args.feature_master)
    frame = add_policy_flags(frame)
    if not args.skip_current_committee:
        frame = add_current_committee_replay(frame)

    metrics = build_policy_metrics(frame)
    selections = select_validation_policies(metrics)
    segment_metrics = build_segment_metrics(frame, selections)
    trace_gate_contribution = build_trace_gate_contribution(frame)
    report = build_report(
        metrics=metrics,
        selections=selections,
        segment_metrics=segment_metrics,
        trace_gate_contribution=trace_gate_contribution,
    )

    scores_path = args.output_dir / "stage2_validation_test_policy_scores.csv"
    metrics_path = args.output_dir / "stage2_validation_test_policy_metrics.csv"
    segment_path = args.output_dir / "stage2_validation_test_segment_metrics.csv"
    trace_gate_path = args.output_dir / "stage2_validation_test_trace_gate_contribution.csv"
    summary_path = args.output_dir / "stage2_validation_test_policy_summary.json"
    report_path = args.output_dir / "stage2_validation_test_policy_report.md"

    frame.to_csv(scores_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    segment_metrics.to_csv(segment_path, index=False, encoding="utf-8-sig")
    trace_gate_contribution.to_csv(trace_gate_path, index=False, encoding="utf-8-sig")
    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "selection_rule": "Use validation only; test and 2026 external labels are confirmation sets.",
        "recall_floor": RECALL_FLOOR,
        "selected_policies": selections,
        "outputs": {
            "scores": str(scores_path.relative_to(ROOT)),
            "metrics": str(metrics_path.relative_to(ROOT)),
            "segment_metrics": str(segment_path.relative_to(ROOT)),
            "trace_gate_contribution": str(trace_gate_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "scores": str(scores_path.relative_to(ROOT)),
                "metrics": str(metrics_path.relative_to(ROOT)),
                "segment_metrics": str(segment_path.relative_to(ROOT)),
                "trace_gate_contribution": str(trace_gate_path.relative_to(ROOT)),
                "summary": str(summary_path.relative_to(ROOT)),
                "report": str(report_path.relative_to(ROOT)),
                "selected_policies": selections,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def read_evaluation_frame(prediction_scores_path: Path, feature_master_path: Path) -> pd.DataFrame:
    scores = pd.read_csv(prediction_scores_path, encoding="utf-8-sig", dtype={"stock_code": str})
    features = pd.read_csv(feature_master_path, encoding="utf-8-sig", dtype={"stock_code": str})
    scores["stock_code"] = scores["stock_code"].map(normalize_stock_code)
    features["stock_code"] = features["stock_code"].map(normalize_stock_code)
    scores = scores.loc[scores["split"].isin(EVALUATION_SPLITS)].copy()
    merged = scores.merge(
        features,
        on=KEY_COLUMNS,
        how="left",
        suffixes=("", "_feature"),
        validate="one_to_one",
    )
    if merged["is_speculative"].isna().any():
        raise ValueError("prediction_scores.csv contains rows without is_speculative labels.")
    return merged


def normalize_stock_code(value: object) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


def add_policy_flags(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    stage1 = output["pred_label_tuned"].astype(int).eq(1)
    aux_probability = numeric_first_column(
        output,
        STAGE2_REVIEW_AUX_PROB_COLUMN,
        LEGACY_STAGE2_REVIEW_AUX_PROB_COLUMN,
    )
    aux_threshold = numeric_first_column(
        output,
        STAGE2_REVIEW_AUX_THRESHOLD_COLUMN,
        LEGACY_STAGE2_REVIEW_AUX_THRESHOLD_COLUMN,
    )
    aux_it_threshold = numeric_first_column(
        output,
        STAGE2_REVIEW_AUX_IT_THRESHOLD_COLUMN,
        LEGACY_STAGE2_REVIEW_AUX_IT_THRESHOLD_COLUMN,
    )
    aux_risk = aux_probability.ge(aux_threshold)
    it_services = output["industry_macro_category"].astype(str).eq("it_services")
    it_low_threshold = it_services & aux_probability.ge(aux_it_threshold)
    high_margin_aux = aux_risk & aux_probability.ge(aux_threshold + 0.05)
    overwarning_candidate = bool_series(output["stage2_overwarning_filter_candidate"])

    output["policy_stage1_model"] = stage1
    output["policy_stage1_or_stage2_review_aux"] = stage1 | ((~stage1) & aux_risk)
    output["policy_stage1_or_it_low_threshold"] = stage1 | ((~stage1) & it_low_threshold)
    output["policy_stage1_or_stage2_review_aux_or_it_low_threshold"] = output[
        "stage2_review_trigger"
    ].astype(bool)
    output["policy_stage1_or_stage2_review_aux_no_it_low_threshold"] = stage1 | (
        (~stage1) & aux_risk & (~it_low_threshold)
    )
    output["policy_stage1_or_stage2_review_aux_high_margin"] = stage1 | (
        (~stage1) & high_margin_aux
    )
    output["policy_stage1_minus_overwarning_candidate"] = stage1 & (~overwarning_candidate)
    return output


def numeric_first_column(frame: pd.DataFrame, *columns: str) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def add_current_committee_replay(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        state = build_committee_state(row)
        state.update(rule_engine_node.run(state))
        state.update(committee_node.run(state))
        committee_view = dict(state.get("committee_view") or {})
        decision_trace = _decision_trace_items(committee_view)
        trace_by_gate = _decision_trace_by_gate(decision_trace)
        replay_row: dict[str, Any] = {
            "market": row["market"],
            "stock_code": row["stock_code"],
            "corp_name": row["corp_name"],
            "fiscal_year": row["fiscal_year"],
            "eval_year": row["eval_year"],
            "current_committee_label": committee_view.get("final_committee_label", ""),
            "current_committee_decision_type": committee_view.get("committee_decision_type", ""),
            "current_committee_decision_type_label": committee_view.get(
                "committee_decision_type_label",
                "",
            ),
            "current_committee_risk_signal": bool(
                committee_view.get("committee_risk_signal", False)
            ),
            "current_committee_veto_triggered": bool(committee_view.get("veto_triggered", False)),
            "current_committee_hidden_tail_risk_flag": bool(
                committee_view.get("hidden_tail_risk_flag", False)
            ),
            "current_committee_conflict_resolution": committee_view.get("conflict_resolution", ""),
            "current_committee_decision_trace": json.dumps(
                decision_trace,
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        for gate, _label in TRACE_GATES:
            trace_item = trace_by_gate.get(gate, {})
            replay_row[f"trace_{gate}_triggered"] = bool(trace_item.get("triggered", False))
            replay_row[f"trace_{gate}_severity"] = str(trace_item.get("severity") or "")
        rows.append(replay_row)
    committee = pd.DataFrame(rows)
    output = frame.merge(committee, on=KEY_COLUMNS, how="left", validate="one_to_one")
    output["policy_current_committee_hold_or_reject"] = output["current_committee_label"].isin(
        {"보류", "부적격"}
    )
    output["policy_current_committee_reject_only"] = output["current_committee_label"].eq("부적격")
    return output


def build_committee_state(row: pd.Series) -> dict[str, Any]:
    row_dict = {
        str(key): clean_scalar(value)
        for key, value in row.to_dict().items()
        if not str(key).endswith("_feature")
    }
    model_view = {
        "model_name": "credit_46_features",
        "model_version": "feature_46_xgboost",
        "probability_speculative": float(row["prob_speculative"]),
        "prediction_label": "부적격" if int(row["pred_label_tuned"]) == 1 else "투자적격",
        "risk_band": str(row["risk_band"]),
        "threshold": float(row["threshold"]),
        "stage2_review_trigger": bool(row.get("stage2_review_trigger", False)),
        "stage2_secondary_trigger": bool(row.get("stage2_secondary_trigger", False)),
        "stage2_review_priority": str(row.get("stage2_review_priority") or "none"),
        "trigger_reason": str(row.get("trigger_reason") or ""),
        "stage2_overwarning_filter_candidate": bool(
            row.get("stage2_overwarning_filter_candidate", False)
        ),
        "overwarning_filter_reason": str(row.get("overwarning_filter_reason") or ""),
        "top_drivers": [],
    }
    return {
        "company_id": f"{row['market']}-{row['stock_code']}-{int(row['fiscal_year'])}",
        "company_name": str(row["corp_name"]),
        "market": str(row["market"]),
        "analysis_year": int(row["fiscal_year"]),
        "company_profile": {
            "company_id": str(row["stock_code"]),
            "company_name": str(row["corp_name"]),
            "market": str(row["market"]),
        },
        "source_feature_row": row_dict,
        "model_view": model_view,
        "xgboost_result": dict(model_view),
        "news_cache_snapshot": {
            "status": "disabled",
            "enabled": False,
            "items": [],
            "as_of_date": f"{int(row['fiscal_year'])}-12-31",
            "message": "External evidence is disabled for validation/test policy diagnostics.",
        },
    }


def clean_scalar(value: object) -> object:
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _decision_trace_items(committee_view: dict[str, Any]) -> list[dict[str, Any]]:
    raw_trace = committee_view.get("decision_trace")
    if not isinstance(raw_trace, list):
        return []
    items: list[dict[str, Any]] = []
    for item in raw_trace:
        if isinstance(item, dict):
            items.append(dict(item))
    return items


def _decision_trace_by_gate(trace: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in trace:
        gate = str(item.get("gate") or "").strip()
        if gate:
            output[gate] = item
    return output


def build_policy_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    policy_columns = [
        column for column in frame.columns if column.startswith("policy_") and column != "policy_"
    ]
    stage1_metrics = {}
    rows: list[dict[str, Any]] = []
    for split in EVALUATION_SPLITS:
        split_frame = frame.loc[frame["split"].eq(split)].copy()
        y_true = split_frame["is_speculative"].astype(int)
        stage1_pred = split_frame["policy_stage1_model"].astype(bool).astype(int)
        stage1_metrics[split] = metrics_at_threshold(y_true, stage1_pred)
        for policy in policy_columns:
            y_pred = bool_series(split_frame[policy]).astype(int)
            metrics = metrics_at_threshold(y_true, y_pred)
            rows.append(
                {
                    "split": split,
                    "policy": policy.removeprefix("policy_"),
                    **metrics,
                    "predicted_count": int(y_pred.sum()),
                    "delta_fp_vs_stage1": metrics["fp"] - stage1_metrics[split]["fp"],
                    "delta_fn_vs_stage1": metrics["fn"] - stage1_metrics[split]["fn"],
                    "delta_recall_vs_stage1": metrics["recall"] - stage1_metrics[split]["recall"],
                    "delta_precision_vs_stage1": metrics["precision"]
                    - stage1_metrics[split]["precision"],
                    "delta_f1_vs_stage1": metrics["f1"] - stage1_metrics[split]["f1"],
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "policy"]).reset_index(drop=True)


def metrics_at_threshold(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    y_true_int = y_true.astype(int)
    y_pred_int = y_pred.astype(int)
    tp = int(y_true_int.eq(1).mul(y_pred_int.eq(1)).sum())
    fp = int(y_true_int.eq(0).mul(y_pred_int.eq(1)).sum())
    fn = int(y_true_int.eq(1).mul(y_pred_int.eq(0)).sum())
    tn = int(y_true_int.eq(0).mul(y_pred_int.eq(0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def select_validation_policies(metrics: pd.DataFrame) -> dict[str, dict[str, Any]]:
    valid = metrics.loc[metrics["split"].eq(VALIDATION_SPLIT)].copy()
    selectable = valid.loc[
        ~valid["policy"].isin(
            {
                "current_committee_hold_or_reject",
                "current_committee_reject_only",
            }
        )
    ].copy()
    selected: dict[str, dict[str, Any]] = {}
    selected["max_validation_f1"] = select_row(selectable, ["f1", "recall", "precision"])
    recall_pool = selectable.loc[selectable["recall"].ge(RECALL_FLOOR)]
    if not recall_pool.empty:
        selected["recall_floor_max_precision"] = select_row(
            recall_pool,
            ["precision", "f1", "recall"],
        )
    selected["official_stage1_baseline"] = select_specific_policy(valid, "stage1_model")
    selected["current_aux_review_trigger"] = select_specific_policy(
        valid,
        "stage1_or_stage2_review_aux_or_it_low_threshold",
    )
    return selected


def select_row(frame: pd.DataFrame, sort_columns: list[str]) -> dict[str, Any]:
    row = frame.sort_values(sort_columns, ascending=False).iloc[0]
    return serializable_row(row)


def select_specific_policy(frame: pd.DataFrame, policy: str) -> dict[str, Any]:
    row = frame.loc[frame["policy"].eq(policy)].iloc[0]
    return serializable_row(row)


def serializable_row(row: pd.Series) -> dict[str, Any]:
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in row.to_dict().items()
    }


def build_segment_metrics(
    frame: pd.DataFrame,
    selections: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    policy_names = {
        "stage1_model",
        str(selections["recall_floor_max_precision"]["policy"])
        if "recall_floor_max_precision" in selections
        else "",
        str(selections["max_validation_f1"]["policy"]),
        "current_committee_hold_or_reject"
        if "policy_current_committee_hold_or_reject" in frame.columns
        else "",
    }
    policy_names = {name for name in policy_names if name}
    dimensions = ["market", "industry_macro_category", "firm_size_group", "fiscal_year"]
    rows: list[dict[str, Any]] = []
    for split in EVALUATION_SPLITS:
        split_frame = frame.loc[frame["split"].eq(split)].copy()
        for dimension in dimensions:
            for value, group in split_frame.groupby(dimension, dropna=False):
                y_true = group["is_speculative"].astype(int)
                for policy in policy_names:
                    policy_column = f"policy_{policy}"
                    if policy_column not in group.columns:
                        continue
                    y_pred = bool_series(group[policy_column]).astype(int)
                    metrics = metrics_at_threshold(y_true, y_pred)
                    rows.append(
                        {
                            "split": split,
                            "dimension": dimension,
                            "segment": "" if pd.isna(value) else str(value),
                            "policy": policy,
                            "rows": len(group),
                            "positive_count": int(y_true.sum()),
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def build_trace_gate_contribution(frame: pd.DataFrame) -> pd.DataFrame:
    if "current_committee_label" not in frame.columns:
        return _empty_trace_gate_contribution()
    rows: list[dict[str, Any]] = []
    for split in EVALUATION_SPLITS:
        split_frame = frame.loc[frame["split"].eq(split)].copy()
        if split_frame.empty:
            continue
        actual_positive = split_frame["is_speculative"].astype(int).eq(1)
        stage1_positive = bool_series(split_frame["policy_stage1_model"])
        committee_review = (
            split_frame["current_committee_label"].astype(str).isin({"보류", "부적격"})
        )
        committee_softened = _committee_softened_stage1_fp(split_frame)
        stage1_fn = actual_positive & (~stage1_positive)
        stage1_fp = (~actual_positive) & stage1_positive
        for gate, gate_label in CONTRIBUTION_TRACE_GATES:
            column = f"trace_{gate}_triggered"
            gate_triggered = (
                bool_series(split_frame[column])
                if column in split_frame.columns
                else pd.Series(False, index=split_frame.index)
            )
            fn_escalated = stage1_fn & gate_triggered & committee_review
            fp_softened = stage1_fp & gate_triggered & committee_softened
            fp_unsoftened = stage1_fp & gate_triggered & (~committee_softened)
            rows.append(
                {
                    "split": split,
                    "gate": gate,
                    "gate_label": gate_label,
                    "triggered_count": int(gate_triggered.sum()),
                    "triggered_positive_count": int((gate_triggered & actual_positive).sum()),
                    "triggered_negative_count": int((gate_triggered & ~actual_positive).sum()),
                    "stage1_fn_total": int(stage1_fn.sum()),
                    "fn_escalated_count": int(fn_escalated.sum()),
                    "fn_escalation_share": _safe_divide(
                        int(fn_escalated.sum()), int(stage1_fn.sum())
                    ),
                    "stage1_fp_total": int(stage1_fp.sum()),
                    "fp_softened_count": int(fp_softened.sum()),
                    "fp_unsoftened_count": int(fp_unsoftened.sum()),
                    "fp_softening_share": _safe_divide(
                        int(fp_softened.sum()),
                        int(stage1_fp.sum()),
                    ),
                    "net_help_count": int(fn_escalated.sum() + fp_softened.sum()),
                    "dominant_effect": _dominant_gate_effect(
                        fn_escalated_count=int(fn_escalated.sum()),
                        fp_softened_count=int(fp_softened.sum()),
                    ),
                }
            )
    if not rows:
        return _empty_trace_gate_contribution()
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["split", "net_help_count", "fn_escalated_count"], ascending=[True, False, False]
        )
        .reset_index(drop=True)
    )


def _committee_softened_stage1_fp(frame: pd.DataFrame) -> pd.Series:
    final_label = frame["current_committee_label"].astype(str)
    decision_type = frame.get("current_committee_decision_type", pd.Series("", index=frame.index))
    risk_signal = (
        bool_series(frame["current_committee_risk_signal"])
        if "current_committee_risk_signal" in frame.columns
        else final_label.isin({"보류", "부적격"})
    )
    return final_label.eq("적격") | decision_type.astype(str).eq("mitigation_hold") | (~risk_signal)


def _dominant_gate_effect(*, fn_escalated_count: int, fp_softened_count: int) -> str:
    if fn_escalated_count and fp_softened_count:
        return "fn_and_fp"
    if fn_escalated_count:
        return "fn_escalation"
    if fp_softened_count:
        return "fp_softening"
    return "none"


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _empty_trace_gate_contribution() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "split",
            "gate",
            "gate_label",
            "triggered_count",
            "triggered_positive_count",
            "triggered_negative_count",
            "stage1_fn_total",
            "fn_escalated_count",
            "fn_escalation_share",
            "stage1_fp_total",
            "fp_softened_count",
            "fp_unsoftened_count",
            "fp_softening_share",
            "net_help_count",
            "dominant_effect",
        ]
    )


def build_report(
    *,
    metrics: pd.DataFrame,
    selections: dict[str, dict[str, Any]],
    segment_metrics: pd.DataFrame,
    trace_gate_contribution: pd.DataFrame,
) -> str:
    lines = [
        "# Stage 2 Validation/Test Policy Evaluation",
        "",
        "## 원칙",
        "",
        "- Stage 2 보류/검토 정책은 validation 기준으로만 비교합니다.",
        "- test는 validation에서 고른 후보가 유지되는지 확인하는 holdout 용도입니다.",
        "- 2026 신용평가 공시 라벨은 외부검증셋이므로 이 리포트의 선택 과정에는 사용하지 않습니다.",
        "- `보류`는 최종 부적격 확정이 아니라 추가 검토 대상으로 해석합니다.",
        "",
        "## Validation 기준 선택 결과",
        "",
        selection_line("공식 1차 모델 기준", selections["official_stage1_baseline"]),
        selection_line("Validation F1 최대 후보", selections["max_validation_f1"]),
    ]
    if "recall_floor_max_precision" in selections:
        lines.append(
            selection_line(
                f"Recall {RECALL_FLOOR:.2f} 이상 중 precision 최대 후보",
                selections["recall_floor_max_precision"],
            )
        )
    lines.extend(
        [
            selection_line(
                "현재 보조 review trigger",
                selections["current_aux_review_trigger"],
            ),
            "",
            "## Policy Metrics",
            "",
            "| Split | Policy | Precision | Recall | F1 | TP | FP | FN | TN | Count | ΔFP | ΔFN |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ordered_metrics = metrics.assign(
        _split_order=metrics["split"].map({VALIDATION_SPLIT: 0, TEST_SPLIT: 1}).fillna(9)
    ).sort_values(["_split_order", "f1"], ascending=[True, False])
    for _, row in ordered_metrics.iterrows():
        lines.append(metric_row(row))
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 현재 deterministic committee의 `보류/부적격` 전체를 위험 판단처럼 사용하면 recall은 높지만 FP와 검토량이 과도합니다.",
            "- 따라서 모델 성능표에는 1차 모델과 validation-selected review trigger를 분리해서 보여주는 편이 안전합니다.",
            "- 2차 위원회는 `부적격 확정기`가 아니라, 1차 모델 경고와 보조 trigger가 잡은 기업을 추가 검토하는 설명/검증 단계로 두는 것이 적절합니다.",
            "- test 결과는 후보 선택에 쓰지 않고, validation에서 고른 정책의 일반화 확인용으로만 기록합니다.",
            "",
            "## Segment Diagnostics",
            "",
            "아래는 validation-selected 후보와 stage1 기준의 주요 취약 구간을 보기 위한 상세 CSV입니다.",
            "",
            "- `stage2_validation_test_segment_metrics.csv`",
            "",
            "## Decision Trace Gate Contribution",
            "",
            "아래 표는 `decision_trace` 게이트가 1차 모델의 FN/FP를 보완한 사례 수를 집계한 결과입니다.",
            "한 기업에서 여러 게이트가 동시에 켜질 수 있으므로, 게이트별 건수는 서로 배타적이지 않습니다.",
            "",
            "| Split | Gate | Triggered | FN escalated | FN share | FP softened | FP share | Effect |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in trace_gate_contribution.iterrows():
        lines.append(trace_gate_row(row))
    lines.extend([""])
    top_segments = (
        segment_metrics.loc[
            segment_metrics["policy"].eq(str(selections["recall_floor_max_precision"]["policy"]))
        ]
        if "recall_floor_max_precision" in selections
        else pd.DataFrame()
    )
    if not top_segments.empty:
        lines.extend(
            [
                "### Recall-floor 후보의 test FP 상위 세그먼트",
                "",
                "| Dimension | Segment | Rows | Positives | Precision | Recall | FP | FN |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        view = top_segments.loc[top_segments["split"].eq(TEST_SPLIT)].sort_values(
            ["fp", "rows"], ascending=False
        )
        for _, row in view.head(10).iterrows():
            lines.append(segment_row(row))
    return "\n".join(lines)


def selection_line(label: str, row: dict[str, Any]) -> str:
    return (
        f"- {label}: `{row['policy']}` "
        f"(valid precision `{row['precision']:.4f}`, recall `{row['recall']:.4f}`, "
        f"F1 `{row['f1']:.4f}`, FP `{row['fp']}`, FN `{row['fn']}`)"
    )


def metric_row(row: pd.Series) -> str:
    return (
        f"| {row['split']} | {row['policy']} | {row['precision']:.4f} | "
        f"{row['recall']:.4f} | {row['f1']:.4f} | {int(row['tp'])} | "
        f"{int(row['fp'])} | {int(row['fn'])} | {int(row['tn'])} | "
        f"{int(row['predicted_count'])} | {int(row['delta_fp_vs_stage1'])} | "
        f"{int(row['delta_fn_vs_stage1'])} |"
    )


def segment_row(row: pd.Series) -> str:
    return (
        f"| {row['dimension']} | {row['segment']} | {int(row['rows'])} | "
        f"{int(row['positive_count'])} | {row['precision']:.4f} | "
        f"{row['recall']:.4f} | {int(row['fp'])} | {int(row['fn'])} |"
    )


def trace_gate_row(row: pd.Series) -> str:
    return (
        f"| {row['split']} | {row['gate_label']} | {int(row['triggered_count'])} | "
        f"{int(row['fn_escalated_count'])} | {float(row['fn_escalation_share']):.4f} | "
        f"{int(row['fp_softened_count'])} | {float(row['fp_softening_share']):.4f} | "
        f"{row['dominant_effect']} |"
    )


if __name__ == "__main__":
    main()
