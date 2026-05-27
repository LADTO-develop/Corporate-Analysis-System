"""Optimize Stage 2 policy thresholds on rolling-validation committee replay."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from cas.agents.stage2_policy import load_stage2_policy
from cas.agents.stage2_policy_optimizer import (
    DEFAULT_SEARCH_SPACE,
    OBJECTIVES,
    ObjectiveName,
    PolicyOptimizationResult,
    optimize_policy_thresholds,
    selected_policy_overrides,
)
from cas.utils.io import write_json, write_yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLING_SCORES_PATH = (
    ROOT / "data/outputs/modeling/feature_46_xgboost/diagnostics/stage2_agents/"
    "feature46_full_review_trigger_73_harness/rolling_samples/stage2_rolling_validation_scores.csv"
)
DEFAULT_FEATURE_MASTER_PATH = ROOT / "data/input/credit_46_features/feature_46_master.csv"
DEFAULT_OUTPUT_DIR = (
    ROOT / "data/outputs/modeling/feature_46_xgboost/diagnostics/stage2_agents/policy_optimizer"
)
MERGE_KEYS = ["market", "stock_code", "fiscal_year"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-scores", type=Path, default=DEFAULT_ROLLING_SCORES_PATH)
    parser.add_argument("--feature-master", type=Path, default=DEFAULT_FEATURE_MASTER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=OBJECTIVES,
        default=list(OBJECTIVES),
        help="Optimization objectives to tune independently.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Coordinate-search passes per objective.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional deterministic sample size for fast local smoke runs. 0 uses all rows.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = read_optimizer_frame(args.rolling_scores, args.feature_master)
    if args.max_rows and len(frame) > args.max_rows:
        frame = frame.sample(n=args.max_rows, random_state=args.random_state).sort_index()
    objectives = cast(tuple[ObjectiveName, ...], tuple(args.objectives))
    result = optimize_policy_thresholds(
        frame,
        objectives=objectives,
        max_iterations=max(1, int(args.max_iterations)),
    )
    write_outputs(result=result, frame=frame, output_dir=args.output_dir)


def read_optimizer_frame(rolling_scores_path: Path, feature_master_path: Path) -> pd.DataFrame:
    scores = pd.read_csv(rolling_scores_path, encoding="utf-8-sig", dtype={"stock_code": str})
    if not feature_master_path.exists():
        return _normalize_frame(scores)
    features = pd.read_csv(feature_master_path, encoding="utf-8-sig", dtype={"stock_code": str})
    scores = _normalize_frame(scores)
    features = _normalize_frame(features)
    feature_columns = [
        column
        for column in features.columns
        if column in MERGE_KEYS or column not in scores.columns
    ]
    feature_columns = list(dict.fromkeys(feature_columns))
    merged = scores.merge(
        features.loc[:, feature_columns].drop_duplicates(MERGE_KEYS),
        on=MERGE_KEYS,
        how="left",
        validate="many_to_one",
    )
    if "is_speculative" not in merged.columns:
        raise ValueError("Rolling scores must include is_speculative labels.")
    return merged


def write_outputs(
    *,
    result: PolicyOptimizationResult,
    frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "stage2_policy_optimizer_candidate_metrics.csv"
    selected_path = output_dir / "stage2_policy_optimizer_selected.json"
    recommendations_path = output_dir / "stage2_policy_optimizer_recommended_overrides.yaml"
    report_path = output_dir / "stage2_policy_optimizer_report.md"

    result.candidate_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    selected_summary = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_policy_version": load_stage2_policy().policy_version,
        "rows": len(frame),
        "rolling_eval_years": _unique_ints(frame, "rolling_eval_year"),
        "objectives": result.selected,
        "baseline": result.baseline,
        "outputs": {
            "candidate_metrics": _display_path(metrics_path),
            "recommended_overrides": _display_path(recommendations_path),
            "report": _display_path(report_path),
        },
    }
    write_json(selected_summary, selected_path)
    write_yaml(_recommendation_yaml(result), recommendations_path)
    report_path.write_text(build_report(result=result, rows=len(frame)), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_metrics": _display_path(metrics_path),
                "selected": _display_path(selected_path),
                "recommended_overrides": _display_path(recommendations_path),
                "report": _display_path(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_report(*, result: PolicyOptimizationResult, rows: int) -> str:
    lines = [
        "# Stage 2 Policy Threshold Optimizer",
        "",
        "rolling validation replay로 `stage2_policy.yaml` 후보 threshold를 목적함수별로 탐색했습니다.",
        "",
        "## Objective 정의",
        "",
        "- `review_safe`: F1, recall, precision을 함께 보되 전체 review rate를 벌점 처리합니다.",
        "- `strict`: FN 방지를 우선해 recall과 Stage1 FN rescue 비율을 크게 봅니다.",
        "- `tn_overhold`: 실제 정상기업을 보류하는 overhold를 줄이는 방향으로 TN/precision을 봅니다.",
        "- `fn_rescue`: Stage1 false negative를 Stage2가 보류/부적격으로 끌어올리는 비율을 봅니다.",
        "",
        f"- Replay rows: `{rows}`",
        f"- Candidate evaluations: `{len(result.candidate_metrics)}`",
        "",
        "## Selected Candidates",
        "",
        "| Objective | Candidate | Score | Precision | Recall | F1 | FP | FN | TN overhold | FN rescue | Updates |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for objective, payload in result.selected.items():
        updates = ", ".join(
            f"`{path}={value}`" for path, value in payload["threshold_updates"].items()
        )
        lines.append(
            f"| {objective} | {payload['candidate_id']} | {payload['score']:.4f} | "
            f"{payload['precision']:.4f} | {payload['recall']:.4f} | "
            f"{payload['f1']:.4f} | {int(payload['fp'])} | {int(payload['fn'])} | "
            f"{payload['tn_overhold_rate']:.4f} | {payload['fn_rescue_rate']:.4f} | "
            f"{updates or '`baseline`'} |"
        )
    lines.extend(
        [
            "",
            "## Baseline",
            "",
            _metric_sentence(result.baseline),
            "",
            "## Top Candidate Preview",
            "",
            "| Candidate | Review-safe | Strict | TN overhold | FN rescue | Precision | Recall | F1 | Updates |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    preview_columns = [
        "candidate_id",
        "review_safe_score",
        "strict_score",
        "tn_overhold_score",
        "fn_rescue_score",
        "precision",
        "recall",
        "f1",
        "threshold_updates_json",
    ]
    for _, row in result.candidate_metrics.loc[:, preview_columns].head(15).iterrows():
        lines.append(_candidate_row(row))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Optimizer는 `stage2_policy.yaml`을 직접 수정하지 않고 추천 override YAML만 생성합니다.",
            "- ReviewQA/RiskRecallQA threshold는 replay에서 관측 가능한 결정 변화가 있을 때만 선택됩니다.",
            "- test holdout과 2026 외부검증 라벨은 별도 확인용으로 남겨두는 전제를 유지합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "stock_code" in output.columns:
        output["stock_code"] = output["stock_code"].map(_normalize_stock_code)
    return output


def _normalize_stock_code(value: object) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


def _recommendation_yaml(result: PolicyOptimizationResult) -> dict[str, Any]:
    overrides = selected_policy_overrides(result.selected)
    return {
        "policy_version": "stage2_policy_optimizer_recommendations_v1",
        "source_policy_version": load_stage2_policy().policy_version,
        "selection_rule": "rolling_validation_coordinate_search",
        "objectives": {
            objective: {
                "candidate_id": payload["candidate_id"],
                "score": payload["score"],
                "threshold_updates": _dot_paths_to_nested(payload["threshold_updates"]),
            }
            for objective, payload in overrides.items()
        },
        "search_space": [
            {
                "path": spec.path,
                "values": list(spec.values),
                "objectives": list(spec.objectives),
                "description": spec.description,
            }
            for spec in DEFAULT_SEARCH_SPACE
        ],
    }


def _dot_paths_to_nested(updates: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for path, value in updates.items():
        current = output
        parts = [part for part in path.split(".") if part]
        for part in parts[:-1]:
            next_value = current.setdefault(part, {})
            if not isinstance(next_value, dict):
                next_value = {}
                current[part] = next_value
            current = next_value
        if parts:
            current[parts[-1]] = value
    return output


def _unique_ints(frame: pd.DataFrame, column: str) -> list[int]:
    if column not in frame.columns:
        return []
    values = pd.to_numeric(frame[column], errors="coerce").dropna().astype(int).unique()
    return [int(value) for value in sorted(values)]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _metric_sentence(metrics: dict[str, Any]) -> str:
    return (
        f"`{metrics['candidate_id']}` precision `{metrics['precision']:.4f}`, "
        f"recall `{metrics['recall']:.4f}`, F1 `{metrics['f1']:.4f}`, "
        f"FP `{int(metrics['fp'])}`, FN `{int(metrics['fn'])}`."
    )


def _candidate_row(row: pd.Series) -> str:
    updates = str(row["threshold_updates_json"])
    if updates == "{}":
        updates = "baseline"
    return (
        f"| {row['candidate_id']} | {float(row['review_safe_score']):.4f} | "
        f"{float(row['strict_score']):.4f} | {float(row['tn_overhold_score']):.4f} | "
        f"{float(row['fn_rescue_score']):.4f} | {float(row['precision']):.4f} | "
        f"{float(row['recall']):.4f} | {float(row['f1']):.4f} | "
        f"`{updates.replace('|', '/')}` |"
    )


if __name__ == "__main__":
    main()
