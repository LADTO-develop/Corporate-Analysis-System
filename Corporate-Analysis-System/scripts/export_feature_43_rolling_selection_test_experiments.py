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
    JOIN_KEYS,
    OUTPUT_DIR,
    RAW_PATH,
    RECALL_FLOOR,
    attach_candidate_columns,
    evaluate_variant,
    format_metric,
    markdown_table,
    read_id_frames,
    read_raw_features,
    read_split_frames,
    unique_preserve_order,
)
from export_feature_43_forward_selection_experiments import CANDIDATE_COLUMNS
from export_feature_43_rolling_validation_experiments import (
    ID_COLUMNS,
    MASTER_PATH,
    ROLLING_EVAL_YEARS,
    evaluate_fold,
    read_master,
    summarize_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIR_POOL_SIZE = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select feature candidates with rolling OOT validation, then inspect their "
            "final 2023-2024 test performance without using test for selection."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--master-path", type=Path, default=MASTER_PATH)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--eval-years", type=int, nargs="+", default=ROLLING_EVAL_YEARS)
    parser.add_argument(
        "--pair-pool-size",
        type=int,
        default=DEFAULT_PAIR_POOL_SIZE,
        help=(
            "Number of top rolling single features, by F1 and by PR-AUC respectively, "
            "used to form two-feature pair candidates."
        ),
    )
    return parser.parse_args()


def base_feature_columns(master: pd.DataFrame) -> list[str]:
    excluded = {*ID_COLUMNS, "label_eval_year", "is_speculative"}
    return [column for column in master.columns if column not in excluded]


def attach_raw_columns(master: pd.DataFrame, raw: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return master.copy()
    missing_columns = [column for column in columns if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Candidate columns are missing from raw Model V1: {missing_columns}")

    raw_subset = raw.loc[:, [*JOIN_KEYS, *columns]].copy()
    for column in columns:
        raw_subset[column] = pd.to_numeric(raw_subset[column], errors="coerce")
    joined = master.merge(raw_subset, on=JOIN_KEYS, how="left", indicator=True)
    unmatched = int(joined["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"feature_43_master has unmatched raw rows: {unmatched}")
    return joined.drop(columns=["_merge"])


def variant_name(prefix: str, columns: list[str]) -> str:
    if not columns:
        return "baseline_43_native"
    return f"{prefix}__{'__'.join(columns)}"


def variant_specs_from_columns(
    *,
    prefix: str,
    column_sets: list[list[str]],
    note_prefix: str,
) -> list[dict[str, Any]]:
    specs = []
    for columns in column_sets:
        specs.append(
            {
                "variant": variant_name(prefix, columns),
                "note": f"{note_prefix}: {', '.join(columns)}",
                "columns": columns,
                "selection_stage": prefix,
            }
        )
    return specs


def run_rolling_for_specs(
    *,
    frame: pd.DataFrame,
    base_features: list[str],
    specs: list[dict[str, Any]],
    eval_years: list[int],
) -> pd.DataFrame:
    rows = []
    for eval_year in eval_years:
        for spec in specs:
            row = evaluate_fold(
                variant=spec["variant"],
                note=spec["note"],
                columns=spec["columns"],
                frame=frame,
                base_features=base_features,
                eval_year=eval_year,
            )
            row["selection_stage"] = spec["selection_stage"]
            rows.append(row)
    return pd.DataFrame(rows)


def top_single_pair_pool(
    single_summary: pd.DataFrame,
    pair_pool_size: int,
) -> list[str]:
    singles = single_summary.loc[~single_summary["variant"].eq("baseline_43_native")].copy()
    by_f1 = (
        singles.sort_values(["eval_f1_mean", "eval_pr_auc_mean"], ascending=False)
        .head(pair_pool_size)["added_features"]
        .tolist()
    )
    by_pr_auc = (
        singles.sort_values(["eval_pr_auc_mean", "eval_f1_mean"], ascending=False)
        .head(pair_pool_size)["added_features"]
        .tolist()
    )
    return unique_preserve_order([*by_f1, *by_pr_auc])


def rank_by_rolling(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        ["eval_f1_mean", "eval_pr_auc_mean", "eval_recall_mean"],
        ascending=False,
    )


def rank_by_rolling_pr_auc(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        ["eval_pr_auc_mean", "eval_f1_mean", "eval_recall_mean"],
        ascending=False,
    )


def evaluate_final_test_for_specs(
    *,
    input_dir: Path,
    raw: pd.DataFrame,
    candidate_columns: list[str],
    specs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, valid, test = read_split_frames(input_dir)
    id_frames = read_id_frames(input_dir)
    base_features = [column for column in train.columns if column != "is_speculative"]
    base_frames = {"train": train, "valid": valid, "test": test}
    candidate_frames = attach_candidate_columns(
        frames=base_frames,
        id_frames=id_frames,
        raw=raw,
        candidate_columns=candidate_columns,
    )

    metric_rows = []
    segment_rows = []
    for spec in specs:
        frames = base_frames if not spec["columns"] else candidate_frames
        feature_columns = unique_preserve_order([*base_features, *spec["columns"]])
        row, segments = evaluate_variant(
            variant=spec["variant"],
            note=spec["note"],
            frames=frames,
            id_frames=id_frames,
            feature_columns=feature_columns,
            added_columns=spec["columns"],
        )
        row["selection_stage"] = spec["selection_stage"]
        metric_rows.append(row)
        for segment in segments:
            segment["selection_stage"] = spec["selection_stage"]
        segment_rows.extend(segments)
    return pd.DataFrame(metric_rows), pd.DataFrame(segment_rows)


def run_experiments(
    *,
    master: pd.DataFrame,
    raw: pd.DataFrame,
    input_dir: Path,
    eval_years: list[int],
    pair_pool_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_features = base_feature_columns(master)
    candidate_columns = [
        column for column in unique_preserve_order(CANDIDATE_COLUMNS) if column not in base_features
    ]
    master_with_candidates = attach_raw_columns(master, raw, candidate_columns)

    baseline_spec = {
        "variant": "baseline_43_native",
        "note": "현재 43개 변수 baseline",
        "columns": [],
        "selection_stage": "baseline",
    }
    single_specs = variant_specs_from_columns(
        prefix="single",
        column_sets=[[column] for column in candidate_columns],
        note_prefix="단일 후보 변수",
    )
    single_fold_metrics = run_rolling_for_specs(
        frame=master_with_candidates,
        base_features=base_features,
        specs=[baseline_spec, *single_specs],
        eval_years=eval_years,
    )
    single_summary = summarize_metrics(single_fold_metrics)
    single_summary["selection_stage"] = (
        single_summary["variant"].map({"baseline_43_native": "baseline"}).fillna("single")
    )

    pair_pool = top_single_pair_pool(single_summary, pair_pool_size)
    pair_specs = variant_specs_from_columns(
        prefix="pair_rolling_pool",
        column_sets=[list(pair) for pair in itertools.combinations(pair_pool, 2)],
        note_prefix="rolling 상위 단일 후보 2개 조합",
    )
    pair_fold_metrics = run_rolling_for_specs(
        frame=master_with_candidates,
        base_features=base_features,
        specs=pair_specs,
        eval_years=eval_years,
    )
    all_fold_metrics = pd.concat([single_fold_metrics, pair_fold_metrics], ignore_index=True)
    rolling_summary = summarize_metrics(all_fold_metrics)
    stage_map = {
        **{spec["variant"]: spec["selection_stage"] for spec in [baseline_spec, *single_specs]},
        **{spec["variant"]: spec["selection_stage"] for spec in pair_specs},
    }
    rolling_summary["selection_stage"] = rolling_summary["variant"].map(stage_map)

    all_specs = [baseline_spec, *single_specs, *pair_specs]
    final_metrics, final_segments = evaluate_final_test_for_specs(
        input_dir=input_dir,
        raw=raw,
        candidate_columns=candidate_columns,
        specs=all_specs,
    )
    return all_fold_metrics, rolling_summary, final_metrics, final_segments


def merge_selection_and_test(
    rolling_summary: pd.DataFrame,
    final_metrics: pd.DataFrame,
) -> pd.DataFrame:
    final_columns = [
        "variant",
        "test_pr_auc",
        "test_roc_auc",
        "test_precision_at_threshold",
        "test_recall_at_threshold",
        "test_f1_at_threshold",
        "test_false_positive_at_threshold",
        "test_false_negative_at_threshold",
        "threshold_tuned",
    ]
    merged = rolling_summary.merge(
        final_metrics.loc[:, final_columns],
        on="variant",
        how="left",
    )
    baseline = merged.loc[merged["variant"].eq("baseline_43_native")].iloc[0]
    merged["rolling_f1_delta_vs_baseline"] = merged["eval_f1_mean"] - baseline["eval_f1_mean"]
    merged["rolling_pr_auc_delta_vs_baseline"] = (
        merged["eval_pr_auc_mean"] - baseline["eval_pr_auc_mean"]
    )
    merged["test_f1_delta_vs_baseline"] = (
        merged["test_f1_at_threshold"] - baseline["test_f1_at_threshold"]
    )
    merged["test_pr_auc_delta_vs_baseline"] = merged["test_pr_auc"] - baseline["test_pr_auc"]
    merged["test_fn_delta_vs_baseline"] = (
        merged["test_false_negative_at_threshold"] - baseline["test_false_negative_at_threshold"]
    )
    merged["test_fp_delta_vs_baseline"] = (
        merged["test_false_positive_at_threshold"] - baseline["test_false_positive_at_threshold"]
    )
    return rank_by_rolling(merged)


def build_report(selection_table: pd.DataFrame, fold_metrics: pd.DataFrame) -> str:
    baseline = selection_table.loc[selection_table["variant"].eq("baseline_43_native")].iloc[0]
    best_f1 = rank_by_rolling(selection_table).iloc[0]
    best_pr = rank_by_rolling_pr_auc(selection_table).iloc[0]
    test_helpful = selection_table.loc[
        (selection_table["rolling_f1_delta_vs_baseline"] > 0)
        & (selection_table["test_f1_delta_vs_baseline"] > 0)
    ].sort_values(
        ["rolling_f1_delta_vs_baseline", "test_f1_delta_vs_baseline"],
        ascending=False,
    )
    best_test_aligned = None if test_helpful.empty else test_helpful.iloc[0]

    if best_test_aligned is None:
        recommendation = (
            "- Rolling 기준으로 좋아지는 후보는 있지만 final test F1까지 동시에 좋아지는 후보는 없습니다. "
            "feature 반영은 보류하고 threshold 정책/추가 OOT 검증을 먼저 보는 편이 안전합니다."
        )
    else:
        recommendation = (
            f"- `{best_test_aligned['variant']}`는 rolling과 final test F1이 모두 baseline보다 높습니다. "
            "다음 후보 모델로 검토할 가치가 있습니다."
        )

    return "\n".join(
        [
            "# Rolling-Selected Candidate Test Experiments",
            "",
            "전체 단일 후보 변수를 rolling OOT validation으로 평가하고, rolling 상위 단일 후보의 2개 조합까지 비교한 뒤 final test 성능을 확인했습니다.",
            "final test는 후보 선택에 사용하지 않고, rolling 기준으로 고른 후보가 마지막 구간에서 어떤지 확인하는 용도로만 사용합니다.",
            "",
            "## 1. 결론",
            "",
            f"- Baseline rolling mean F1/PR-AUC: `{format_metric(baseline['eval_f1_mean'])}` / "
            f"`{format_metric(baseline['eval_pr_auc_mean'])}`",
            f"- Baseline final test F1/PR-AUC: `{format_metric(baseline['test_f1_at_threshold'])}` / "
            f"`{format_metric(baseline['test_pr_auc'])}`",
            f"- Rolling F1 기준 최상위: `{best_f1['variant']}` "
            f"(rolling F1 `{format_metric(best_f1['eval_f1_mean'])}`, "
            f"final test F1 `{format_metric(best_f1['test_f1_at_threshold'])}`)",
            f"- Rolling PR-AUC 기준 최상위: `{best_pr['variant']}` "
            f"(rolling PR-AUC `{format_metric(best_pr['eval_pr_auc_mean'])}`, "
            f"final test PR-AUC `{format_metric(best_pr['test_pr_auc'])}`)",
            recommendation,
            "",
            "## 2. Rolling F1 기준 상위 후보와 Final Test 확인",
            "",
            markdown_table(
                rank_by_rolling(selection_table).head(20),
                [
                    ("Variant", "variant", "text"),
                    ("Stage", "selection_stage", "text"),
                    ("Features", "added_features", "text"),
                    ("Roll PR", "eval_pr_auc_mean", "metric"),
                    ("Roll P", "eval_precision_mean", "metric"),
                    ("Roll R", "eval_recall_mean", "metric"),
                    ("Roll F1", "eval_f1_mean", "metric"),
                    ("Roll ΔF1", "rolling_f1_delta_vs_baseline", "metric"),
                    ("Test PR", "test_pr_auc", "metric"),
                    ("Test P", "test_precision_at_threshold", "metric"),
                    ("Test R", "test_recall_at_threshold", "metric"),
                    ("Test F1", "test_f1_at_threshold", "metric"),
                    ("Test ΔF1", "test_f1_delta_vs_baseline", "metric"),
                    ("Test ΔFN", "test_fn_delta_vs_baseline", "int"),
                    ("Test ΔFP", "test_fp_delta_vs_baseline", "int"),
                ],
            ),
            "",
            "## 3. Rolling PR-AUC 기준 상위 후보",
            "",
            markdown_table(
                rank_by_rolling_pr_auc(selection_table).head(20),
                [
                    ("Variant", "variant", "text"),
                    ("Stage", "selection_stage", "text"),
                    ("Features", "added_features", "text"),
                    ("Roll PR", "eval_pr_auc_mean", "metric"),
                    ("Roll F1", "eval_f1_mean", "metric"),
                    ("Test PR", "test_pr_auc", "metric"),
                    ("Test ROC", "test_roc_auc", "metric"),
                    ("Test P", "test_precision_at_threshold", "metric"),
                    ("Test R", "test_recall_at_threshold", "metric"),
                    ("Test F1", "test_f1_at_threshold", "metric"),
                    ("Test ΔPR", "test_pr_auc_delta_vs_baseline", "metric"),
                ],
            ),
            "",
            "## 4. Rolling과 Final Test가 같이 좋아진 후보",
            "",
            (
                "해당 후보가 없습니다."
                if test_helpful.empty
                else markdown_table(
                    test_helpful.head(20),
                    [
                        ("Variant", "variant", "text"),
                        ("Features", "added_features", "text"),
                        ("Roll ΔF1", "rolling_f1_delta_vs_baseline", "metric"),
                        ("Test ΔF1", "test_f1_delta_vs_baseline", "metric"),
                        ("Test ΔFN", "test_fn_delta_vs_baseline", "int"),
                        ("Test ΔFP", "test_fp_delta_vs_baseline", "int"),
                    ],
                )
            ),
            "",
            "## 5. 해석 기준",
            "",
            "- 변수 선택은 rolling validation 기준으로만 판단합니다.",
            "- threshold는 별도 정책으로 조정 가능하므로 PR-AUC/ROC-AUC 같은 ranking 지표도 함께 봅니다.",
            "- 다만 최종 서비스에 표시되는 Precision/Recall/F1은 threshold 정책의 영향을 받으므로, 후보 모델 확정 전 threshold 재탐색이 필요합니다.",
            f"- rolling fold rows: `{len(fold_metrics)}`",
        ]
    )


def write_outputs(
    *,
    fold_metrics: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    final_metrics: pd.DataFrame,
    final_segments: pd.DataFrame,
    output_dir: Path,
    pair_pool_size: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_table = merge_selection_and_test(rolling_summary, final_metrics)

    fold_metrics_path = output_dir / "rolling_selection_fold_metrics.csv"
    rolling_summary_path = output_dir / "rolling_selection_summary.csv"
    final_metrics_path = output_dir / "rolling_selection_final_test_metrics.csv"
    final_segments_path = output_dir / "rolling_selection_final_test_segments.csv"
    selection_table_path = output_dir / "rolling_selection_test_comparison.csv"
    report_path = output_dir / "rolling_selection_test_report.md"
    meta_path = output_dir / "rolling_selection_test_summary.json"

    fold_metrics.to_csv(fold_metrics_path, index=False, encoding="utf-8-sig")
    rolling_summary.to_csv(rolling_summary_path, index=False, encoding="utf-8-sig")
    final_metrics.to_csv(final_metrics_path, index=False, encoding="utf-8-sig")
    final_segments.to_csv(final_segments_path, index=False, encoding="utf-8-sig")
    selection_table.to_csv(selection_table_path, index=False, encoding="utf-8-sig")
    report_path.write_text(build_report(selection_table, fold_metrics), encoding="utf-8")

    baseline = selection_table.loc[selection_table["variant"].eq("baseline_43_native")].iloc[0]
    best_f1 = rank_by_rolling(selection_table).iloc[0]
    best_pr = rank_by_rolling_pr_auc(selection_table).iloc[0]
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "pair_pool_size": pair_pool_size,
        "threshold_policy": f"max precision with validation/policy recall >= {RECALL_FLOOR:.2f}",
        "baseline": baseline.to_dict(),
        "best_by_rolling_f1": best_f1.to_dict(),
        "best_by_rolling_pr_auc": best_pr.to_dict(),
        "output_files": {
            "fold_metrics": str(fold_metrics_path.relative_to(ROOT)),
            "rolling_summary": str(rolling_summary_path.relative_to(ROOT)),
            "final_test_metrics": str(final_metrics_path.relative_to(ROOT)),
            "selection_test_comparison": str(selection_table_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    }
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return selection_table


def main() -> None:
    args = parse_args()
    master = read_master(args.master_path)
    raw = read_raw_features(args.raw_path)
    fold_metrics, rolling_summary, final_metrics, final_segments = run_experiments(
        master=master,
        raw=raw,
        input_dir=args.input_dir,
        eval_years=args.eval_years,
        pair_pool_size=args.pair_pool_size,
    )
    selection_table = write_outputs(
        fold_metrics=fold_metrics,
        rolling_summary=rolling_summary,
        final_metrics=final_metrics,
        final_segments=final_segments,
        output_dir=args.output_dir,
        pair_pool_size=args.pair_pool_size,
    )
    baseline = selection_table.loc[selection_table["variant"].eq("baseline_43_native")].iloc[0]
    best_f1 = rank_by_rolling(selection_table).iloc[0]
    best_pr = rank_by_rolling_pr_auc(selection_table).iloc[0]
    print(
        json.dumps(
            {
                "best_by_rolling_f1": best_f1["variant"],
                "best_rolling_f1": float(best_f1["eval_f1_mean"]),
                "best_final_test_f1": float(best_f1["test_f1_at_threshold"]),
                "best_by_rolling_pr_auc": best_pr["variant"],
                "best_rolling_pr_auc": float(best_pr["eval_pr_auc_mean"]),
                "best_final_test_pr_auc": float(best_pr["test_pr_auc"]),
                "baseline_rolling_f1": float(baseline["eval_f1_mean"]),
                "baseline_final_test_f1": float(baseline["test_f1_at_threshold"]),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
