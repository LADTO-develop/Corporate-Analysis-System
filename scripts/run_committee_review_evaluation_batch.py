"""Run a small committee-review evaluation batch from historical replay samples."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from cas.agents.graph import run_once

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES_PATH = (
    ROOT
    / "data/outputs/modeling/feature_43_xgboost/diagnostics/"
    / "committee_review_historical_test_replay_samples.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/outputs/modeling/feature_43_xgboost/diagnostics"
DEFAULT_POLICY = "balanced_current_45_or_near_threshold_0_10"
CATEGORY_ORDER = [
    "fn_caught_by_stage2_review",
    "fp_needing_committee_mitigation",
    "bbb_minus_bb_plus_boundary",
    "true_positive_risk_explanation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--live-external-evidence", action="store_true")
    parser.add_argument(
        "--stage2-runner",
        choices=["deterministic", "agno"],
        default="deterministic",
        help="Use deterministic runner for fast, reproducible pilots; agno calls LLMs.",
    )
    return parser.parse_args()


def read_samples(path: Path) -> pd.DataFrame:
    samples = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    samples = samples.copy()
    samples["stock_code"] = samples["stock_code"].astype(str).str.zfill(6)
    for column in ["fiscal_year", "eval_year", "prob_speculative", "threshold"]:
        if column in samples.columns:
            samples[column] = pd.to_numeric(samples[column], errors="coerce")
    return samples


def select_batch(
    samples: pd.DataFrame, *, policy: str, per_category: int, max_cases: int
) -> pd.DataFrame:
    scoped = samples.loc[samples["committee_policy"].astype(str).eq(policy)].copy()
    if scoped.empty:
        raise ValueError(f"No samples found for committee_policy={policy!r}.")
    frames: list[pd.DataFrame] = []
    for category in CATEGORY_ORDER:
        subset = scoped.loc[scoped["sample_category"].astype(str).eq(category)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["model_error_type", "prob_speculative", "stock_code"],
            ascending=[True, False, True],
        )
        frames.append(subset.head(per_category))
    if not frames:
        raise ValueError("No rows selected for committee review batch.")
    batch = pd.concat(frames, ignore_index=True)
    batch = batch.drop_duplicates(["market", "stock_code", "fiscal_year", "eval_year"])
    return batch.head(max_cases).reset_index(drop=True)


def configure_runtime(*, live_external_evidence: bool, stage2_runner: str) -> None:
    load_dotenv(ROOT / ".env")
    os.environ["CAS_STAGE2_RUNNER"] = stage2_runner
    os.environ.setdefault("CAS_STAGE2_FALLBACK_ON_ERROR", "1")
    os.environ.setdefault(
        "CAS_OPENDART_CORP_CODE_CACHE_PATH", "/private/tmp/cas_opendart_corp_codes.csv"
    )
    if live_external_evidence:
        os.environ["CAS_ENABLE_EXTERNAL_EVIDENCE"] = "1"
    else:
        os.environ.pop("CAS_ENABLE_EXTERNAL_EVIDENCE", None)


def run_batch(batch: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, sample in batch.iterrows():
        row = sample.to_dict()
        print(
            "[Run] "
            f"{index + 1}/{len(batch)} "
            f"{row.get('corp_name')} "
            f"{row.get('fiscal_year')} "
            f"{row.get('sample_category')}"
        )
        try:
            selection = json.loads(str(row["company_selection_json"]))
            state = run_once(company_selection=selection)
            rows.append(_result_row(row, state=state, error_message=""))
        except Exception as error:  # pragma: no cover - operational guard
            rows.append(_result_row(row, state={}, error_message=str(error)))
    return pd.DataFrame(rows)


def _result_row(
    sample: dict[str, Any], *, state: dict[str, Any], error_message: str
) -> dict[str, Any]:
    committee_view = _dict_value(state.get("committee_view"))
    evidence = _dict_value(state.get("news_cache_snapshot"))
    xgboost_result = _dict_value(state.get("xgboost_result"))
    final_label = str(committee_view.get("final_committee_label") or "")
    actual_label = str(sample.get("actual_label_name") or "")
    sample_model_error_type = str(sample.get("model_error_type") or "")
    graph_model_label = str(xgboost_result.get("prediction_label") or "")
    graph_model_error_type = _model_error_type(
        actual_label=actual_label,
        model_label=graph_model_label,
    )
    model_error_type = graph_model_error_type or sample_model_error_type
    success, effect = _committee_success(model_error_type=model_error_type, final_label=final_label)
    provider_statuses = _provider_statuses(evidence.get("providers"))
    evidence_titles = _evidence_titles(evidence.get("items", []))
    return {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "evaluation_mode": sample.get("evaluation_mode"),
        "committee_policy": sample.get("committee_policy"),
        "sample_category": sample.get("sample_category"),
        "market": sample.get("market"),
        "stock_code": str(sample.get("stock_code") or "").zfill(6),
        "corp_name": sample.get("corp_name"),
        "fiscal_year": sample.get("fiscal_year"),
        "eval_year": sample.get("eval_year"),
        "as_of_date": sample.get("as_of_date"),
        "actual_label_name": actual_label,
        "model_predicted_label_name": sample.get("model_predicted_label_name"),
        "sample_model_error_type": sample_model_error_type,
        "model_error_type": model_error_type,
        "credit_rating": sample.get("credit_rating"),
        "rating_boundary_group": sample.get("rating_boundary_group"),
        "industry_macro_category": sample.get("industry_macro_category"),
        "firm_size_group": sample.get("firm_size_group"),
        "sample_prob_speculative": sample.get("prob_speculative"),
        "sample_threshold": sample.get("threshold"),
        "graph_model_label": graph_model_label,
        "graph_prob_speculative": xgboost_result.get("probability_speculative"),
        "final_committee_label": final_label,
        "committee_success": success,
        "committee_effect": effect,
        "veto_triggered": bool(committee_view.get("veto_triggered", False)),
        "hidden_tail_risk_flag": bool(committee_view.get("hidden_tail_risk_flag", False)),
        "evidence_status": evidence.get("status"),
        "evidence_as_of_date": evidence.get("as_of_date"),
        "evidence_items": len(evidence.get("items", []) or []),
        "direct_match_count": evidence.get("direct_match_count"),
        "verified_item_count": evidence.get("verified_item_count"),
        "veto_candidate_count": evidence.get("veto_candidate_count"),
        "high_confidence_critical_count": evidence.get("high_confidence_critical_count"),
        "provider_statuses": json.dumps(provider_statuses, ensure_ascii=False, sort_keys=True),
        "top_evidence_titles": " / ".join(evidence_titles),
        "conflict_resolution": committee_view.get("conflict_resolution"),
        "final_review_memo": committee_view.get("final_review_memo"),
        "error_message": error_message,
    }


def _committee_success(*, model_error_type: str, final_label: str) -> tuple[bool, str]:
    if not final_label:
        return False, "run_failed"
    if model_error_type == "false_negative":
        success = final_label in {"보류", "부적격"}
        return success, "fn_escalated" if success else "fn_not_escalated"
    if model_error_type == "false_positive":
        success = final_label in {"적격", "보류"}
        return success, "fp_mitigated" if success else "fp_not_mitigated"
    if model_error_type == "true_positive":
        success = final_label in {"보류", "부적격"}
        return success, "tp_risk_supported" if success else "tp_softened"
    if model_error_type == "true_negative":
        success = final_label == "적격"
        return success, "tn_kept_eligible" if success else "tn_escalated"
    return False, "unknown_case_type"


def _model_error_type(*, actual_label: str, model_label: str) -> str:
    actual = _normalize_binary_label(actual_label)
    model = _normalize_binary_label(model_label)
    if actual is None or model is None:
        return ""
    if actual == "speculative" and model == "investment":
        return "false_negative"
    if actual == "investment" and model == "speculative":
        return "false_positive"
    if actual == "speculative" and model == "speculative":
        return "true_positive"
    if actual == "investment" and model == "investment":
        return "true_negative"
    return ""


def _normalize_binary_label(label: str) -> str | None:
    normalized = str(label or "").strip()
    if normalized in {"투기등급", "부적격"}:
        return "speculative"
    if normalized in {"투자적격", "적격"}:
        return "investment"
    return None


def _dict_value(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _provider_statuses(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {
        str(name): provider.get("status")
        for name, provider in value.items()
        if isinstance(provider, dict)
    }


def _evidence_titles(items: object, *, limit: int = 3) -> Iterable[str]:
    if not isinstance(items, list):
        return []
    titles = []
    for item in items[:limit]:
        if isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            if title:
                titles.append(title)
    return titles


def write_outputs(results: pd.DataFrame, *, output_dir: Path) -> None:
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "committee_review_batch_results.csv"
    summary_path = output_dir / "committee_review_batch_summary.json"
    report_path = output_dir / "committee_review_batch_report.md"
    results.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary = _summary(results)
    summary["paths"] = {
        "details": str(detail_path.relative_to(ROOT)),
        "summary": str(summary_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_report(results, summary), encoding="utf-8")
    print(f"[Saved] {detail_path}")
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {report_path}")


def _summary(results: pd.DataFrame) -> dict[str, Any]:
    success_rate = float(results["committee_success"].mean()) if len(results) else 0.0
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(results),
        "success_rate": success_rate,
        "by_error_type": _group_counts(results, ["model_error_type", "committee_effect"]),
        "by_final_label": _group_counts(results, ["final_committee_label"]),
        "by_evidence_status": _group_counts(results, ["evidence_status"]),
    }


def _group_counts(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.groupby(columns, dropna=False).size().reset_index(name="rows").to_dict("records")


def _report(results: pd.DataFrame, summary: dict[str, Any]) -> str:
    preview_columns = [
        "sample_category",
        "corp_name",
        "actual_label_name",
        "model_predicted_label_name",
        "final_committee_label",
        "committee_effect",
        "evidence_status",
        "evidence_items",
    ]
    preview = results.loc[:, [column for column in preview_columns if column in results.columns]]
    return "\n".join(
        [
            "# Committee Review Batch Results",
            "",
            f"- Rows: {summary['rows']}",
            f"- Committee success rate: {summary['success_rate']:.1%}",
            "",
            "## Result Preview",
            "",
            _markdown_table(preview),
            "",
        ]
    )


def _markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    preview = frame.head(max_rows).copy()
    columns = [str(column) for column in preview.columns]
    rows = preview.astype(object).where(pd.notna(preview), "").astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(value.replace("|", "/") for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def main() -> None:
    args = parse_args()
    configure_runtime(
        live_external_evidence=args.live_external_evidence,
        stage2_runner=args.stage2_runner,
    )
    samples = read_samples(args.samples)
    batch = select_batch(
        samples,
        policy=args.policy,
        per_category=args.per_category,
        max_cases=args.max_cases,
    )
    results = run_batch(batch)
    write_outputs(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
