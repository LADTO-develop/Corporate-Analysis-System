"""Stage 1 XGBoost improvement report builders."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd


@dataclass(frozen=True)
class PromotionGates:
    """Minimum evidence required before a candidate feature set is promoted."""

    min_rolling_f1_delta: float = 0.005
    min_rolling_pr_auc_delta: float = 0.0
    min_final_test_f1_delta: float = 0.0
    max_final_test_fn_delta: int = 0
    max_final_test_fp_delta: int = 0


@dataclass(frozen=True)
class Stage1ImprovementTables:
    """Experiment tables used to build the consolidated Stage 1 report."""

    candidate_pack_metrics: pd.DataFrame
    rolling_validation_summary: pd.DataFrame
    rolling_selection_comparison: pd.DataFrame


DEFAULT_PROMOTION_GATES = PromotionGates()
BASELINE_VARIANT = "baseline_43_native"
REPORT_FILENAME = "stage1_xgboost_improvement_report.md"
SUMMARY_FILENAME = "stage1_xgboost_improvement_summary.json"
PROMOTION_CANDIDATES_FILENAME = "stage1_xgboost_promotion_candidates.csv"

ColumnSpec = tuple[str, str, str]


def load_stage1_improvement_tables(diagnostics_dir: Path) -> Stage1ImprovementTables:
    """Load generated Stage 1 experiment outputs from the diagnostics directory."""
    return Stage1ImprovementTables(
        candidate_pack_metrics=pd.read_csv(
            diagnostics_dir / "candidate_feature_pack_metrics.csv",
        ),
        rolling_validation_summary=pd.read_csv(
            diagnostics_dir / "rolling_validation_summary.csv",
        ),
        rolling_selection_comparison=pd.read_csv(
            diagnostics_dir / "rolling_selection_test_comparison.csv",
        ),
    )


def select_promotion_candidates(
    selection_table: pd.DataFrame,
    gates: PromotionGates = DEFAULT_PROMOTION_GATES,
) -> pd.DataFrame:
    """Return candidates that pass the strict promotion gate."""
    table = _coerce_numeric(
        selection_table.copy(),
        [
            "rolling_f1_delta_vs_baseline",
            "rolling_pr_auc_delta_vs_baseline",
            "test_f1_delta_vs_baseline",
            "test_fn_delta_vs_baseline",
            "test_fp_delta_vs_baseline",
            "eval_f1_mean",
            "eval_pr_auc_mean",
            "test_f1_at_threshold",
            "test_pr_auc",
        ],
    )
    mask = (
        table["variant"].ne(BASELINE_VARIANT)
        & table["rolling_f1_delta_vs_baseline"].ge(gates.min_rolling_f1_delta)
        & table["rolling_pr_auc_delta_vs_baseline"].ge(gates.min_rolling_pr_auc_delta)
        & table["test_f1_delta_vs_baseline"].ge(gates.min_final_test_f1_delta)
        & table["test_fn_delta_vs_baseline"].le(gates.max_final_test_fn_delta)
        & table["test_fp_delta_vs_baseline"].le(gates.max_final_test_fp_delta)
    )
    return table.loc[mask].sort_values(
        [
            "rolling_f1_delta_vs_baseline",
            "test_f1_delta_vs_baseline",
            "test_fn_delta_vs_baseline",
            "test_fp_delta_vs_baseline",
        ],
        ascending=[False, False, True, True],
    )


def build_candidate_feature_set(
    promotion_candidates: pd.DataFrame,
    *,
    base_feature_count: int = 43,
) -> dict[str, object]:
    """Build the recommended next feature-set payload."""
    if promotion_candidates.empty:
        return {
            "status": "hold",
            "name": None,
            "base_model": "feature_46_xgboost",
            "base_feature_count": base_feature_count,
            "added_features": [],
            "feature_count": base_feature_count,
            "rationale": (
                "No candidate passed the strict rolling/test promotion gate. "
                "Keep the active feature_43 model."
            ),
        }

    selected = cast(pd.Series, promotion_candidates.iloc[0])
    added_features = _split_features(selected.get("added_features"))
    feature_count = base_feature_count + len(added_features)
    return {
        "status": "candidate",
        "name": f"feature_{feature_count}_robust_candidate",
        "base_model": "feature_46_xgboost",
        "base_feature_count": base_feature_count,
        "added_features": added_features,
        "feature_count": feature_count,
        "source_variant": str(selected["variant"]),
        "rolling_f1_delta_vs_baseline": float(selected["rolling_f1_delta_vs_baseline"]),
        "rolling_pr_auc_delta_vs_baseline": float(selected["rolling_pr_auc_delta_vs_baseline"]),
        "final_test_f1_delta_vs_baseline": float(selected["test_f1_delta_vs_baseline"]),
        "final_test_fn_delta_vs_baseline": int(selected["test_fn_delta_vs_baseline"]),
        "final_test_fp_delta_vs_baseline": int(selected["test_fp_delta_vs_baseline"]),
        "rationale": (
            "This compact feature set improves rolling F1, does not reduce final test F1, "
            "and does not increase final test FP/FN under the strict promotion gate."
        ),
    }


def build_stage1_improvement_summary(
    tables: Stage1ImprovementTables,
    *,
    gates: PromotionGates = DEFAULT_PROMOTION_GATES,
) -> dict[str, object]:
    """Build a JSON-serializable Stage 1 improvement summary."""
    pack_metrics = enrich_candidate_pack_metrics(tables.candidate_pack_metrics)
    selection_table = _coerce_selection_metrics(tables.rolling_selection_comparison)
    rolling_summary = _coerce_numeric(
        tables.rolling_validation_summary.copy(),
        ["eval_f1_mean", "eval_pr_auc_mean", "total_false_positive", "total_false_negative"],
    )
    promotion_candidates = select_promotion_candidates(selection_table, gates)
    candidate_feature_set = build_candidate_feature_set(promotion_candidates)

    baseline_selection = _row_by_variant(selection_table, BASELINE_VARIANT)
    pack_baseline = _row_by_variant(pack_metrics, BASELINE_VARIANT)
    pack_best_valid = _top_row(pack_metrics, ["valid_f1_at_threshold", "valid_pr_auc"])
    pack_best_test = _top_row(pack_metrics, ["test_f1_at_threshold", "test_pr_auc"])
    rolling_best_f1 = _top_row(rolling_summary, ["eval_f1_mean", "eval_pr_auc_mean"])
    selection_best_f1 = _top_row(selection_table, ["eval_f1_mean", "eval_pr_auc_mean"])
    selection_best_pr_auc = _top_row(selection_table, ["eval_pr_auc_mean", "eval_f1_mean"])

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "active_model_decision": (
            "keep_feature_43_active_until_candidate_artifact_and_external_validation_pass"
        ),
        "promotion_gates": asdict(gates),
        "baseline": _compact_selection_row(baseline_selection),
        "candidate_feature_set": candidate_feature_set,
        "candidate_pack": {
            "baseline": _compact_pack_row(pack_baseline),
            "best_by_validation": _compact_pack_row(pack_best_valid),
            "best_by_test_reference_only": _compact_pack_row(pack_best_test),
        },
        "rolling_pack": {
            "best_by_rolling_f1": _compact_rolling_row(rolling_best_f1),
        },
        "rolling_selection": {
            "best_by_rolling_f1": _compact_selection_row(selection_best_f1),
            "best_by_rolling_pr_auc": _compact_selection_row(selection_best_pr_auc),
            "promotion_candidate_count": len(promotion_candidates),
        },
        "output_files": {
            "report": REPORT_FILENAME,
            "summary": SUMMARY_FILENAME,
            "promotion_candidates": PROMOTION_CANDIDATES_FILENAME,
        },
    }


def build_stage1_improvement_report(
    tables: Stage1ImprovementTables,
    summary: dict[str, object],
    *,
    gates: PromotionGates = DEFAULT_PROMOTION_GATES,
    promotion_candidate_limit: int = 12,
) -> str:
    """Render the consolidated Stage 1 improvement report as Markdown."""
    pack_metrics = enrich_candidate_pack_metrics(tables.candidate_pack_metrics)
    rolling_summary = _coerce_numeric(
        tables.rolling_validation_summary.copy(),
        ["eval_f1_mean", "eval_pr_auc_mean", "total_false_positive", "total_false_negative"],
    )
    selection_table = _coerce_selection_metrics(tables.rolling_selection_comparison)
    promotion_candidates = select_promotion_candidates(selection_table, gates)
    candidate_set = cast(dict[str, object], summary["candidate_feature_set"])

    pack_baseline = _row_by_variant(pack_metrics, BASELINE_VARIANT)
    pack_best_valid = _top_row(pack_metrics, ["valid_f1_at_threshold", "valid_pr_auc"])
    rolling_best = _top_row(rolling_summary, ["eval_f1_mean", "eval_pr_auc_mean"])
    selection_best = _top_row(selection_table, ["eval_f1_mean", "eval_pr_auc_mean"])
    selection_pr_best = _top_row(selection_table, ["eval_pr_auc_mean", "eval_f1_mean"])

    if candidate_set["status"] == "candidate":
        decision = (
            f"- 승격 후보: `{candidate_set['name']}` = feature_43 + "
            f"`{', '.join(cast(list[str], candidate_set['added_features']))}`"
        )
        next_action = (
            "- 이 후보는 active production으로 바로 교체하지 않고, 별도 artifact로 학습한 뒤 "
            "2026 external validation과 segment diagnostics를 통과하면 승격합니다."
        )
    else:
        decision = "- 승격 후보 없음: active Stage 1 모델은 `feature_46_xgboost`를 유지합니다."
        next_action = "- 후보 변수는 추가 OOT 기간이나 외부검증 표본이 늘어난 뒤 다시 평가합니다."

    return "\n".join(
        [
            "# Stage 1 XGBoost Improvement Report",
            "",
            "공식 Stage 1 XGBoost를 더 키울지 판단하기 위해 후보 feature pack, rolling OOT, "
            "rolling-selected final test 확인 결과를 하나로 묶었습니다.",
            "",
            "## 1. 결론",
            "",
            "- 현재 active 모델은 `feature_46_xgboost`로 유지합니다.",
            decision,
            next_action,
            "",
            "## 2. 왜 바로 교체하지 않는가",
            "",
            (
                f"- Feature pack validation 최고 후보 `{pack_best_valid['variant']}`는 "
                f"valid F1을 `{_signed(pack_best_valid['valid_f1_delta_vs_baseline'])}` 올렸지만 "
                f"test F1은 `{_signed(pack_best_valid['test_f1_delta_vs_baseline'])}` 하락했습니다."
            ),
            (
                f"- Rolling pack 최고 후보 `{rolling_best['variant']}`는 rolling F1을 "
                f"`{_format_metric(rolling_best['eval_f1_mean'])}`까지 올렸지만, "
                "후보 선택 근거로만 쓰고 final test/오류 비용을 별도로 봐야 합니다."
            ),
            (
                f"- Rolling selection 최고 F1 후보 `{selection_best['variant']}`는 rolling F1은 "
                f"`{_format_metric(selection_best['eval_f1_mean'])}`이지만 final test F1 delta가 "
                f"`{_signed(selection_best['test_f1_delta_vs_baseline'])}`입니다."
            ),
            (
                f"- PR-AUC 최고 후보 `{selection_pr_best['variant']}`는 ranking 성능은 좋지만 "
                f"final test F1 delta가 `{_signed(selection_pr_best['test_f1_delta_vs_baseline'])}`라 "
                "운영 threshold 성능과는 분리해서 봅니다."
            ),
            "",
            "## 3. Strict Promotion Gate",
            "",
            f"- rolling F1 delta >= `{gates.min_rolling_f1_delta:.4f}`",
            f"- rolling PR-AUC delta >= `{gates.min_rolling_pr_auc_delta:.4f}`",
            f"- final test F1 delta >= `{gates.min_final_test_f1_delta:.4f}`",
            f"- final test FN delta <= `{gates.max_final_test_fn_delta}`",
            f"- final test FP delta <= `{gates.max_final_test_fp_delta}`",
            "",
            (
                "통과 후보가 없습니다."
                if promotion_candidates.empty
                else _markdown_table(
                    promotion_candidates.head(promotion_candidate_limit),
                    [
                        ("Variant", "variant", "text"),
                        ("Features", "added_features", "text"),
                        ("Roll F1", "eval_f1_mean", "metric"),
                        ("Roll ΔF1", "rolling_f1_delta_vs_baseline", "metric"),
                        ("Roll ΔPR", "rolling_pr_auc_delta_vs_baseline", "metric"),
                        ("Test F1", "test_f1_at_threshold", "metric"),
                        ("Test ΔF1", "test_f1_delta_vs_baseline", "metric"),
                        ("Test ΔFN", "test_fn_delta_vs_baseline", "int"),
                        ("Test ΔFP", "test_fp_delta_vs_baseline", "int"),
                    ],
                )
            ),
            "",
            "## 4. Candidate Feature Set",
            "",
            _candidate_feature_set_markdown(candidate_set),
            "",
            "## 5. Feature Pack Result",
            "",
            _markdown_table(
                pack_metrics.sort_values(
                    ["valid_f1_at_threshold", "valid_pr_auc"],
                    ascending=False,
                ).head(10),
                [
                    ("Variant", "variant", "text"),
                    ("Added", "added_feature_count", "int"),
                    ("Valid F1", "valid_f1_at_threshold", "metric"),
                    ("Valid ΔF1", "valid_f1_delta_vs_baseline", "metric"),
                    ("Test F1", "test_f1_at_threshold", "metric"),
                    ("Test ΔF1", "test_f1_delta_vs_baseline", "metric"),
                    ("Test FP", "test_false_positive_at_threshold", "int"),
                    ("Test FN", "test_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 6. Rolling OOT Pack Result",
            "",
            _markdown_table(
                rolling_summary.sort_values(
                    ["eval_f1_mean", "eval_pr_auc_mean"],
                    ascending=False,
                ),
                [
                    ("Variant", "variant", "text"),
                    ("Features", "added_features", "text"),
                    ("Folds", "folds", "int"),
                    ("PR-AUC", "eval_pr_auc_mean", "metric"),
                    ("F1", "eval_f1_mean", "metric"),
                    ("F1 min", "eval_f1_min", "metric"),
                    ("Total FP", "total_false_positive", "int"),
                    ("Total FN", "total_false_negative", "int"),
                ],
            ),
            "",
            "## 7. 재생성 명령",
            "",
            "```bash",
            "/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_candidate_feature_pack_experiments.py",
            "/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_rolling_validation_experiments.py",
            "/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_rolling_selection_test_experiments.py",
            "/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_stage1_improvement_report.py",
            "```",
            "",
            "긴 rolling selection은 XGBoost를 여러 번 재학습하므로 release 전 재생성용으로 봅니다.",
            f"Baseline feature pack test F1은 `{_format_metric(pack_baseline['test_f1_at_threshold'])}`입니다.",
        ]
    )


def write_stage1_improvement_outputs(
    diagnostics_dir: Path,
    *,
    gates: PromotionGates = DEFAULT_PROMOTION_GATES,
) -> dict[str, object]:
    """Write the consolidated Stage 1 improvement outputs."""
    tables = load_stage1_improvement_tables(diagnostics_dir)
    summary = build_stage1_improvement_summary(tables, gates=gates)
    report = build_stage1_improvement_report(tables, summary, gates=gates)
    promotion_candidates = select_promotion_candidates(
        _coerce_selection_metrics(tables.rolling_selection_comparison),
        gates,
    )

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    promotion_candidates.to_csv(
        diagnostics_dir / PROMOTION_CANDIDATES_FILENAME,
        index=False,
        encoding="utf-8-sig",
    )
    (diagnostics_dir / REPORT_FILENAME).write_text(report, encoding="utf-8")
    (diagnostics_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def enrich_candidate_pack_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add baseline deltas to candidate feature-pack metrics."""
    output = _coerce_numeric(
        metrics.copy(),
        [
            "valid_f1_at_threshold",
            "valid_pr_auc",
            "test_f1_at_threshold",
            "test_pr_auc",
            "test_false_positive_at_threshold",
            "test_false_negative_at_threshold",
        ],
    )
    baseline = _row_by_variant(output, BASELINE_VARIANT)
    output["valid_f1_delta_vs_baseline"] = output["valid_f1_at_threshold"] - float(
        baseline["valid_f1_at_threshold"]
    )
    output["test_f1_delta_vs_baseline"] = output["test_f1_at_threshold"] - float(
        baseline["test_f1_at_threshold"]
    )
    output["test_fp_delta_vs_baseline"] = output["test_false_positive_at_threshold"] - int(
        baseline["test_false_positive_at_threshold"]
    )
    output["test_fn_delta_vs_baseline"] = output["test_false_negative_at_threshold"] - int(
        baseline["test_false_negative_at_threshold"]
    )
    return output


def _coerce_selection_metrics(selection_table: pd.DataFrame) -> pd.DataFrame:
    return _coerce_numeric(
        selection_table.copy(),
        [
            "eval_f1_mean",
            "eval_pr_auc_mean",
            "eval_f1_min",
            "rolling_f1_delta_vs_baseline",
            "rolling_pr_auc_delta_vs_baseline",
            "test_f1_at_threshold",
            "test_pr_auc",
            "test_f1_delta_vs_baseline",
            "test_fn_delta_vs_baseline",
            "test_fp_delta_vs_baseline",
        ],
    )


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _row_by_variant(frame: pd.DataFrame, variant: str) -> pd.Series:
    rows = frame.loc[frame["variant"].eq(variant)]
    if rows.empty:
        raise ValueError(f"Missing required variant: {variant}")
    return cast(pd.Series, rows.iloc[0])


def _top_row(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return cast(pd.Series, frame.sort_values(columns, ascending=False).iloc[0])


def _compact_pack_row(row: pd.Series) -> dict[str, object]:
    return {
        "variant": str(row["variant"]),
        "added_features": _split_features(row.get("added_features")),
        "valid_f1": _optional_float(row.get("valid_f1_at_threshold")),
        "test_f1": _optional_float(row.get("test_f1_at_threshold")),
        "test_pr_auc": _optional_float(row.get("test_pr_auc")),
        "valid_f1_delta_vs_baseline": _optional_float(row.get("valid_f1_delta_vs_baseline")),
        "test_f1_delta_vs_baseline": _optional_float(row.get("test_f1_delta_vs_baseline")),
        "test_fp_delta_vs_baseline": _optional_int(row.get("test_fp_delta_vs_baseline")),
        "test_fn_delta_vs_baseline": _optional_int(row.get("test_fn_delta_vs_baseline")),
    }


def _compact_rolling_row(row: pd.Series) -> dict[str, object]:
    return {
        "variant": str(row["variant"]),
        "added_features": _split_features(row.get("added_features")),
        "rolling_f1": _optional_float(row.get("eval_f1_mean")),
        "rolling_pr_auc": _optional_float(row.get("eval_pr_auc_mean")),
        "rolling_f1_min": _optional_float(row.get("eval_f1_min")),
        "total_false_positive": _optional_int(row.get("total_false_positive")),
        "total_false_negative": _optional_int(row.get("total_false_negative")),
    }


def _compact_selection_row(row: pd.Series) -> dict[str, object]:
    return {
        "variant": str(row["variant"]),
        "selection_stage": str(row.get("selection_stage", "")),
        "added_features": _split_features(row.get("added_features")),
        "rolling_f1": _optional_float(row.get("eval_f1_mean")),
        "rolling_pr_auc": _optional_float(row.get("eval_pr_auc_mean")),
        "final_test_f1": _optional_float(row.get("test_f1_at_threshold")),
        "final_test_pr_auc": _optional_float(row.get("test_pr_auc")),
        "rolling_f1_delta_vs_baseline": _optional_float(row.get("rolling_f1_delta_vs_baseline")),
        "rolling_pr_auc_delta_vs_baseline": _optional_float(
            row.get("rolling_pr_auc_delta_vs_baseline")
        ),
        "final_test_f1_delta_vs_baseline": _optional_float(row.get("test_f1_delta_vs_baseline")),
        "final_test_fp_delta_vs_baseline": _optional_int(row.get("test_fp_delta_vs_baseline")),
        "final_test_fn_delta_vs_baseline": _optional_int(row.get("test_fn_delta_vs_baseline")),
    }


def _candidate_feature_set_markdown(candidate_set: dict[str, object]) -> str:
    if candidate_set["status"] != "candidate":
        return "- 이번 run에서는 승격 후보가 없습니다."
    features = cast(list[str], candidate_set["added_features"])
    return "\n".join(
        [
            f"- 후보명: `{candidate_set['name']}`",
            f"- 기준 모델: `{candidate_set['base_model']}`",
            f"- 추가 변수: `{', '.join(features)}`",
            f"- 입력 변수 수: `{candidate_set['base_feature_count']}` -> `{candidate_set['feature_count']}`",
            (
                f"- rolling F1 delta: `{_format_metric(candidate_set['rolling_f1_delta_vs_baseline'])}`, "
                f"final test F1 delta: `{_format_metric(candidate_set['final_test_f1_delta_vs_baseline'])}`"
            ),
            (
                f"- final test FN/FP delta: `{candidate_set['final_test_fn_delta_vs_baseline']}` / "
                f"`{candidate_set['final_test_fp_delta_vs_baseline']}`"
            ),
        ]
    )


def _split_features(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [piece.strip() for piece in text.split(",") if piece.strip()]


def _format_metric(value: object) -> str:
    number = _optional_float(value)
    if number is None:
        return "-"
    return f"{number:.4f}"


def _format_int(value: object) -> str:
    number = _optional_int(value)
    if number is None:
        return "-"
    return f"{number:,}"


def _signed(value: object) -> str:
    number = _optional_float(value)
    if number is None:
        return "-"
    return f"{number:+.4f}"


def _optional_float(value: object) -> float | None:
    try:
        number = float(cast(float, value))
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number)


def _markdown_table(frame: pd.DataFrame, columns: list[ColumnSpec]) -> str:
    header = "| " + " | ".join(label for label, _, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, divider]
    for _, row in frame.iterrows():
        series = cast(pd.Series, row)
        rows.append("| " + " | ".join(_markdown_cell(series, spec) for spec in columns) + " |")
    return "\n".join(rows)


def _markdown_cell(row: pd.Series, spec: ColumnSpec) -> str:
    _, column, kind = spec
    value = row.get(column)
    if kind == "metric":
        return _format_metric(value)
    if kind == "int":
        return _format_int(value)
    text = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
    return text.replace("|", "\\|")
