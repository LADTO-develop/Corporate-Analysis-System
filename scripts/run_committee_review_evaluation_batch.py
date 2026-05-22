"""Run a small committee-review evaluation batch from validation samples."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from cas.agents.contracts import build_agent_state_seed
from cas.agents.graph import run_once
from cas.agents.nodes import (
    base_prediction_node,
    committee_node,
    data_node,
    feature_node,
    news_overlay_node,
    rule_engine_node,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES_PATH = (
    ROOT
    / "data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/"
    / "committee_review_rolling_validation_tuning_samples.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/"
    / "committee_review_rolling_validation_batch"
)
DEFAULT_POLICY = "rolling_stage1_or_near_threshold_0_10"
CATEGORY_ORDER = [
    "fn_caught_by_stage2_review",
    "fp_needing_committee_mitigation",
    "bbb_minus_bb_plus_boundary",
    "true_positive_risk_explanation",
    "true_negative_overescalation_guardrail",
]
TRACE_GATES = (
    "veto_rule",
    "hidden_tail_risk",
    "secondary_review_trigger",
    "boundary_rating_review",
    "overwarning_mitigation",
    "reject_confirmation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--live-external-evidence", action="store_true")
    parser.add_argument(
        "--use-sample-model-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replay committee judgment with the sample's stored OOT model probability, "
            "threshold, and label. Keep this on for rolling validation."
        ),
    )
    parser.add_argument(
        "--stage2-runner",
        choices=["deterministic", "agno"],
        default="deterministic",
        help="Use deterministic runner for fast, reproducible pilots; agno calls LLMs.",
    )
    parser.add_argument(
        "--stage2-agno-mode",
        choices=["single", "multi", "multi_llm", "multi_llm_committee"],
        default=os.environ.get("CAS_STAGE2_AGNO_MODE", "single"),
        help=(
            "Agno routing mode for --stage2-runner agno. Default is single so "
            "OpenAI-only API runs do not require Claude/Gemini credentials."
        ),
    )
    parser.add_argument(
        "--stage2-model-provider",
        default=os.environ.get("CAS_STAGE2_MODEL_PROVIDER", "openai"),
        help="Provider for single-model Agno mode: anthropic/claude, openai/gpt, or google/gemini.",
    )
    parser.add_argument(
        "--stage2-model",
        default=os.environ.get("CAS_STAGE2_MODEL", "gpt-4.1-mini"),
        help="Model id for single-model Agno mode.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help=(
            "Number of companies to process concurrently. Use 1 for strict sequential "
            "runs or 3-5 for faster live Agno batches when API rate limits allow it."
        ),
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


def configure_runtime(
    *,
    live_external_evidence: bool,
    stage2_runner: str,
    stage2_agno_mode: str = "single",
    stage2_model_provider: str = "openai",
    stage2_model: str = "gpt-4.1-mini",
) -> None:
    load_dotenv(ROOT / ".env")
    os.environ["CAS_STAGE2_RUNNER"] = stage2_runner
    if stage2_runner == "agno":
        os.environ["CAS_STAGE2_AGNO_MODE"] = stage2_agno_mode
        os.environ["CAS_STAGE2_MODEL_PROVIDER"] = stage2_model_provider
        os.environ["CAS_STAGE2_MODEL"] = stage2_model
    os.environ.setdefault("CAS_STAGE2_FALLBACK_ON_ERROR", "1")
    os.environ.setdefault(
        "CAS_OPENDART_CORP_CODE_CACHE_PATH", "/private/tmp/cas_opendart_corp_codes.csv"
    )
    if live_external_evidence:
        os.environ["CAS_ENABLE_EXTERNAL_EVIDENCE"] = "1"
    else:
        os.environ.pop("CAS_ENABLE_EXTERNAL_EVIDENCE", None)


def run_batch(
    batch: pd.DataFrame,
    *,
    use_sample_model_view: bool,
    workers: int = 1,
) -> pd.DataFrame:
    records = batch.to_dict(orient="records")
    if not records:
        return pd.DataFrame()

    batch_started_at = time.perf_counter()
    worker_count = _bounded_worker_count(workers, len(records))
    if worker_count == 1:
        result = pd.DataFrame(
            [
                _run_batch_case(
                    index=index,
                    total=len(records),
                    row=row,
                    use_sample_model_view=use_sample_model_view,
                )
                for index, row in enumerate(records)
            ]
        )
        result["batch_wall_time_seconds"] = round(time.perf_counter() - batch_started_at, 4)
        return result

    rows: list[dict[str, Any] | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _run_batch_case,
                index=index,
                total=len(records),
                row=row,
                use_sample_model_view=use_sample_model_view,
            ): index
            for index, row in enumerate(records)
        }
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
    result = pd.DataFrame([row for row in rows if row is not None])
    result["batch_wall_time_seconds"] = round(time.perf_counter() - batch_started_at, 4)
    return result


def _run_batch_case(
    *,
    index: int,
    total: int,
    row: dict[str, Any],
    use_sample_model_view: bool,
) -> dict[str, Any]:
    case_started_at = time.perf_counter()
    print(
        "[Run] "
        f"{index + 1}/{total} "
        f"{row.get('corp_name')} "
        f"{row.get('fiscal_year')} "
        f"{row.get('sample_category')}",
        flush=True,
    )
    try:
        selection = json.loads(str(row["company_selection_json"]))
        if use_sample_model_view:
            state = _run_graph_until_rule_engine(company_selection=selection)
            state = _rerun_committee_with_sample_model_view(state, row)
        else:
            state = run_once(company_selection=selection)
        result = _result_row(row, state=state, error_message="")
    except Exception as error:  # pragma: no cover - operational guard
        result = _result_row(row, state={}, error_message=str(error))
    result["case_elapsed_seconds"] = round(time.perf_counter() - case_started_at, 4)
    return result


def _run_graph_until_rule_engine(*, company_selection: dict[str, Any]) -> dict[str, Any]:
    """Run the graph only up to rule_engine, avoiding pre-replay Stage 2 LLM calls."""
    state = dict(build_agent_state_seed(company_selection))
    state = _merge_state(state, data_node.run(state))
    if data_node.has_enough_data(state) != "enough":
        return state
    for node in (
        feature_node.run,
        news_overlay_node.run,
        base_prediction_node.run,
        rule_engine_node.run,
    ):
        state = _merge_state(state, node(state))
    return state


def _rerun_committee_with_sample_model_view(
    state: dict[str, Any], sample: dict[str, Any]
) -> dict[str, Any]:
    """Re-evaluate downstream rules using the OOT model view stored in the sample row."""
    updated = _merge_state(state, _sample_model_view_updates(state, sample))
    updated = _merge_state(updated, rule_engine_node.run(updated))
    updated = _merge_state(updated, committee_node.run(updated))
    return updated


def _sample_model_view_updates(state: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    probability = _safe_float(sample.get("prob_speculative"), default=0.0)
    threshold = _safe_float(sample.get("threshold"), default=0.315)
    prediction_label = str(sample.get("model_predicted_label_name") or "").strip()
    if prediction_label not in {"투자적격", "부적격"}:
        prediction_label = "부적격" if probability >= threshold else "투자적격"
    risk_band = _risk_band_from_probability(probability, threshold=threshold)
    existing_xgb = _dict_value(state.get("xgboost_result"))
    top_drivers = existing_xgb.get("top_drivers", [])
    stage2_signals = _sample_stage2_signals(sample, probability=probability, threshold=threshold)
    model_view = {
        "probability_speculative": probability,
        "prediction_label": prediction_label,
        "risk_band": risk_band,
        "threshold": threshold,
        "top_drivers": top_drivers,
        "stage2_signal_source": "committee_review_sample_replay",
        **stage2_signals,
    }
    xgboost_result = {
        "model_name": "feature_43_xgboost_rolling_validation_replay",
        "model_version": "rolling_validation_replay",
        "probability_speculative": probability,
        "prediction_label": prediction_label,
        "risk_band": risk_band,
        "threshold": threshold,
        "top_drivers": top_drivers,
        **stage2_signals,
    }
    return {
        "model_view": model_view,
        "xgboost_result": xgboost_result,
        "model_registry_ref": {
            "registry_name": "rolling_validation_replay",
            "active_model": "feature_43_xgboost_rolling_validation_replay",
            "model_version": "rolling_validation_replay",
            "threshold": threshold,
            "artifact_path": str(sample.get("evaluation_mode") or ""),
        },
    }


def _sample_stage2_signals(
    sample: dict[str, Any], *, probability: float, threshold: float
) -> dict[str, Any]:
    prediction_label = str(sample.get("model_predicted_label_name") or "").strip()
    committee_policy = str(sample.get("committee_policy") or "")
    margin = probability - threshold
    near_threshold = abs(margin) <= 0.10
    eligible_near_threshold = prediction_label == "투자적격" and near_threshold
    risky_near_threshold = prediction_label == "부적격" and near_threshold
    priority = "none"
    if prediction_label == "부적격":
        priority = "high" if probability >= threshold + 0.10 else "medium"
    elif eligible_near_threshold:
        priority = "high" if probability >= threshold - 0.05 else "medium"
    if "recall_first" in committee_policy and prediction_label == "투자적격":
        priority = "high" if priority == "medium" else priority
    trigger_reason = str(sample.get("trigger_reason") or "").strip()
    return {
        "stage2_review_trigger": prediction_label == "부적격" or near_threshold,
        "stage2_secondary_trigger": eligible_near_threshold,
        "stage2_review_priority": priority,
        "trigger_reason_code": str(sample.get("trigger_reason_code") or ""),
        "trigger_reason": trigger_reason,
        "stage2_overwarning_filter_candidate": risky_near_threshold,
        "overwarning_filter_reason": (
            "rolling validation 기준 1차 모델은 부적격이지만 확률이 기준선 근처라 "
            "과민 경고 여부를 위원회에서 재점검합니다."
            if risky_near_threshold
            else ""
        ),
        "stage2_probability_margin": margin,
    }


def _risk_band_from_probability(probability: float, *, threshold: float) -> str:
    if probability >= max(0.65, threshold + 0.25):
        return "high_risk"
    if probability >= max(0.40, threshold):
        return "watch"
    return "stable"


def _merge_state(current: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in updates.items():
        if key in {"audit", "committee_reviews", "agent_outputs"}:
            merged[key] = [*(merged.get(key) or []), *(value or [])]
        elif key in {"base_assessments", "artifacts", "agent_summary"}:
            existing = dict(merged.get(key) or {})
            existing.update(value or {})
            merged[key] = existing
        else:
            merged[key] = value
    return merged


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
    review_safe_success, review_safe_effect = _committee_review_safe_success(
        model_error_type=model_error_type,
        final_label=final_label,
    )
    provider_statuses = _provider_statuses(evidence.get("providers"))
    evidence_titles = _evidence_titles(evidence.get("items", []))
    decision_trace = _decision_trace_items(committee_view)
    trace_by_gate = _decision_trace_by_gate(decision_trace)
    result = {
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
        "prior_credit_rating": sample.get("prior_credit_rating"),
        "prior_rating_date": sample.get("prior_rating_date"),
        "prior_rating_age_days": sample.get("prior_rating_age_days"),
        "prior_rating_agency": sample.get("prior_rating_agency"),
        "industry_macro_category": sample.get("industry_macro_category"),
        "firm_size_group": sample.get("firm_size_group"),
        "sample_prob_speculative": sample.get("prob_speculative"),
        "sample_threshold": sample.get("threshold"),
        "graph_model_label": graph_model_label,
        "graph_prob_speculative": xgboost_result.get("probability_speculative"),
        "final_committee_label": final_label,
        "committee_decision_type": committee_view.get("committee_decision_type"),
        "committee_decision_type_label": committee_view.get("committee_decision_type_label"),
        "committee_risk_signal": bool(committee_view.get("committee_risk_signal", False)),
        "decision_trace": json.dumps(decision_trace, ensure_ascii=False, sort_keys=True),
        "committee_success": success,
        "committee_effect": effect,
        "committee_review_safe_success": review_safe_success,
        "committee_review_safe_effect": review_safe_effect,
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
    for gate in TRACE_GATES:
        trace_item = trace_by_gate.get(gate, {})
        result[f"trace_{gate}_triggered"] = bool(trace_item.get("triggered", False))
        result[f"trace_{gate}_severity"] = str(trace_item.get("severity") or "")
    return result


def _decision_trace_items(committee_view: dict[str, Any]) -> list[dict[str, Any]]:
    raw_trace = committee_view.get("decision_trace")
    if not isinstance(raw_trace, list):
        return []
    return [dict(item) for item in raw_trace if isinstance(item, dict)]


def _decision_trace_by_gate(trace: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in trace:
        gate = str(item.get("gate") or "").strip()
        if gate:
            output[gate] = item
    return output


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


def _committee_review_safe_success(*, model_error_type: str, final_label: str) -> tuple[bool, str]:
    """Evaluate Stage 2 as a review triage, where hold is acceptable for normal firms."""
    if not final_label:
        return False, "run_failed"
    if model_error_type == "false_negative":
        success = final_label in {"보류", "부적격"}
        return success, "review_safe_fn_escalated" if success else "review_safe_fn_missed"
    if model_error_type == "false_positive":
        success = final_label in {"적격", "보류"}
        return success, "review_safe_fp_not_rejected" if success else "review_safe_fp_rejected"
    if model_error_type == "true_positive":
        success = final_label in {"보류", "부적격"}
        return success, "review_safe_tp_supported" if success else "review_safe_tp_softened"
    if model_error_type == "true_negative":
        success = final_label in {"적격", "보류"}
        return success, "review_safe_tn_not_rejected" if success else "review_safe_tn_rejected"
    return False, "review_safe_unknown_case_type"


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


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if not isinstance(value, int | float | str):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _default_workers() -> int:
    raw_value = os.environ.get("CAS_COMMITTEE_BATCH_WORKERS", "1").strip()
    try:
        workers = int(raw_value)
    except ValueError:
        return 1
    return max(workers, 1)


def _bounded_worker_count(workers: int, row_count: int) -> int:
    if row_count <= 0:
        return 1
    return min(max(workers, 1), row_count)


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
    review_safe_success_rate = (
        float(results["committee_review_safe_success"].mean())
        if "committee_review_safe_success" in results.columns and len(results)
        else 0.0
    )
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(results),
        "success_rate": success_rate,
        "review_safe_success_rate": review_safe_success_rate,
        "by_error_type": _group_counts(results, ["model_error_type", "committee_effect"]),
        "by_review_safe_effect": _group_counts(
            results,
            ["model_error_type", "committee_review_safe_effect"],
        ),
        "by_final_label": _group_counts(results, ["final_committee_label"]),
        "by_evidence_status": _group_counts(results, ["evidence_status"]),
    }
    if "case_elapsed_seconds" in results.columns and len(results):
        case_elapsed = pd.to_numeric(results["case_elapsed_seconds"], errors="coerce")
        batch_wall = pd.to_numeric(
            results.get("batch_wall_time_seconds", pd.Series(dtype=float)),
            errors="coerce",
        )
        wall_seconds = (
            float(batch_wall.dropna().max())
            if not batch_wall.dropna().empty
            else float(case_elapsed.sum())
        )
        summary["speed"] = {
            "batch_wall_time_seconds": round(wall_seconds, 4),
            "case_elapsed_seconds_sum": round(float(case_elapsed.sum()), 4),
            "case_elapsed_seconds_mean": round(float(case_elapsed.mean()), 4),
            "case_elapsed_seconds_median": round(float(case_elapsed.median()), 4),
            "case_elapsed_seconds_max": round(float(case_elapsed.max()), 4),
            "throughput_cases_per_minute": (
                round(len(results) / wall_seconds * 60.0, 4) if wall_seconds > 0 else None
            ),
        }
    return summary


def _group_counts(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.groupby(columns, dropna=False).size().reset_index(name="rows").to_dict("records")


def _report(results: pd.DataFrame, summary: dict[str, Any]) -> str:
    preview_columns = [
        "sample_category",
        "corp_name",
        "prior_credit_rating",
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
            f"- Strict committee success rate: {summary['success_rate']:.1%}",
            f"- Review-safe success rate: {summary['review_safe_success_rate']:.1%}",
            _speed_report_line(summary),
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


def _speed_report_line(summary: dict[str, Any]) -> str:
    speed = summary.get("speed")
    if not isinstance(speed, dict):
        return "- Speed: not measured"
    return (
        "- Speed: "
        f"wall `{speed.get('batch_wall_time_seconds')}` sec, "
        f"mean case `{speed.get('case_elapsed_seconds_mean')}` sec, "
        f"throughput `{speed.get('throughput_cases_per_minute')}` cases/min"
    )


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    configure_runtime(
        live_external_evidence=args.live_external_evidence,
        stage2_runner=args.stage2_runner,
        stage2_agno_mode=args.stage2_agno_mode,
        stage2_model_provider=args.stage2_model_provider,
        stage2_model=args.stage2_model,
    )
    samples = read_samples(args.samples)
    batch = select_batch(
        samples,
        policy=args.policy,
        per_category=args.per_category,
        max_cases=args.max_cases,
    )
    results = run_batch(
        batch,
        use_sample_model_view=args.use_sample_model_view,
        workers=args.workers,
    )
    write_outputs(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
