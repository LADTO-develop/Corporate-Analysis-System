"""Run a small committee-review evaluation batch from validation samples."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import dotenv_values, load_dotenv

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
from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
from cas.llm.model_catalog import (
    DEFAULT_STAGE2_AGNO_MODE,
    DEFAULT_STAGE2_RUNNER,
    stage2_single_model_default,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES_PATH = (
    ROOT
    / "data/outputs/modeling/feature_46_xgboost/diagnostics/stage2_agents/"
    / "committee_review_rolling_validation_tuning_samples.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data/outputs/modeling/feature_46_xgboost/diagnostics/stage2_agents/"
    / "committee_review_rolling_validation_batch"
)
DEFAULT_POLICY = "feature46_full_review_trigger_73"
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
    "risk_hold_reason_tagging",
)


def parse_args() -> argparse.Namespace:
    stage2_default = stage2_single_model_default()
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
        default=DEFAULT_STAGE2_RUNNER,
        help="Use deterministic runner for fast, reproducible pilots; agno calls LLMs.",
    )
    parser.add_argument(
        "--stage2-agno-mode",
        choices=[
            "single",
            "multi",
            "multi_llm",
            "multi_llm_committee",
        ],
        default=os.environ.get("CAS_STAGE2_AGNO_MODE", DEFAULT_STAGE2_AGNO_MODE),
        help=(
            "Agno routing mode for --stage2-runner agno. Default single uses one "
            "provider across the three role agents. OpenAI-only API runs do not require "
            "Claude/Gemini credentials."
        ),
    )
    parser.add_argument(
        "--stage2-model-provider",
        default=os.environ.get("CAS_STAGE2_MODEL_PROVIDER", stage2_default.provider),
        help="Provider for single-model Agno mode: anthropic/claude, openai/gpt, or google/gemini.",
    )
    parser.add_argument(
        "--stage2-model",
        default=os.environ.get("CAS_STAGE2_MODEL", stage2_default.model),
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
    parser.add_argument(
        "--stage2-llm-cache",
        action=argparse.BooleanOptionalAction,
        default=_default_stage2_llm_cache(),
        help=(
            "Use cached Stage 2 LLM responses. Disable with --no-stage2-llm-cache "
            "when measuring true live API latency or prompt changes."
        ),
    )
    parser.add_argument(
        "--retry-failed-attempts",
        type=int,
        default=_default_retry_failed_attempts(),
        help=(
            "Retry only rows with operational failures, such as Stage 2 error messages, "
            "empty final labels, or failed evidence collection. Default 0 keeps the "
            "historical single-pass behavior."
        ),
    )
    parser.add_argument(
        "--retry-failed-workers",
        type=int,
        default=_default_retry_failed_workers(),
        help="Worker count for retry passes. Use 1 to avoid API TPM bursts.",
    )
    parser.add_argument(
        "--retry-failed-delay-seconds",
        type=float,
        default=_default_retry_failed_delay_seconds(),
        help="Sleep before each retry pass to let provider rate limits cool down.",
    )
    parser.add_argument(
        "--retry-failed-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Write retry sample/result CSVs under output_dir/retry_artifacts for audit."),
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
    stage2_agno_mode: str = DEFAULT_STAGE2_AGNO_MODE,
    stage2_model_provider: str | None = None,
    stage2_model: str | None = None,
    stage2_llm_cache: bool = True,
) -> Stage2RuntimeConfig:
    dotenv_env = _load_dotenv_preserving_stage2_env()
    stage2_default = stage2_single_model_default()
    runtime_env = {**dotenv_env, **dict(os.environ)}
    runtime_env["CAS_STAGE2_RUNNER"] = stage2_runner
    if stage2_runner == "agno":
        runtime_env["CAS_STAGE2_AGNO_MODE"] = stage2_agno_mode
        runtime_env["CAS_STAGE2_MODEL_PROVIDER"] = stage2_model_provider or stage2_default.provider
        runtime_env["CAS_STAGE2_MODEL"] = stage2_model or stage2_default.model
        runtime_env["CAS_STAGE2_LLM_CACHE_ENABLED"] = "1" if stage2_llm_cache else "0"
    runtime_env.setdefault("CAS_STAGE2_FALLBACK_ON_ERROR", "1")
    os.environ.setdefault(
        "CAS_OPENDART_CORP_CODE_CACHE_PATH", "/private/tmp/cas_opendart_corp_codes.csv"
    )
    if live_external_evidence:
        os.environ["CAS_ENABLE_EXTERNAL_EVIDENCE"] = "1"
    else:
        os.environ.pop("CAS_ENABLE_EXTERNAL_EVIDENCE", None)
    return Stage2RuntimeConfig.from_env(runtime_env)


def _load_dotenv_preserving_stage2_env() -> dict[str, str]:
    """Load .env for provider credentials without mutating Stage 2 runtime knobs."""
    env_path = ROOT / ".env"
    dotenv_env = {
        str(key): str(value) for key, value in dotenv_values(env_path).items() if value is not None
    }
    stage2_keys = {
        key
        for key in (*os.environ.keys(), *dotenv_env.keys())
        if str(key).startswith("CAS_STAGE2_")
    }
    snapshot = {key: os.environ.get(key) for key in stage2_keys}
    load_dotenv(env_path)
    for key in stage2_keys:
        original = snapshot.get(key)
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original
    return dotenv_env


def run_batch(
    batch: pd.DataFrame,
    *,
    use_sample_model_view: bool,
    workers: int = 1,
    runtime_config: Stage2RuntimeConfig | None = None,
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
                    runtime_config=runtime_config,
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
                runtime_config=runtime_config,
            ): index
            for index, row in enumerate(records)
        }
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
    result = pd.DataFrame([row for row in rows if row is not None])
    result["batch_wall_time_seconds"] = round(time.perf_counter() - batch_started_at, 4)
    return result


def retry_failed_cases(
    batch: pd.DataFrame,
    results: pd.DataFrame,
    *,
    use_sample_model_view: bool,
    attempts: int,
    workers: int = 1,
    delay_seconds: float = 0.0,
    output_dir: Path | None = None,
    write_artifacts: bool = True,
    runtime_config: Stage2RuntimeConfig | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Retry operationally failed rows and merge successful retry outputs by row position."""
    retry_attempts = max(int(attempts), 0)
    if retry_attempts <= 0 or batch.empty or results.empty:
        return results, []

    combined = results.reset_index(drop=True).copy()
    original_batch = batch.reset_index(drop=True).copy()
    reports: list[dict[str, Any]] = []

    for attempt in range(1, retry_attempts + 1):
        failed_positions = _failed_result_positions(combined)
        if not failed_positions:
            break
        retry_batch = original_batch.iloc[failed_positions].reset_index(drop=True)
        print(
            "[Retry] "
            f"attempt {attempt}/{retry_attempts}: "
            f"rerunning {len(retry_batch)} failed row(s) with workers={workers}",
            flush=True,
        )
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        retry_results = run_batch(
            retry_batch,
            use_sample_model_view=use_sample_model_view,
            workers=workers,
            runtime_config=runtime_config,
        ).reset_index(drop=True)
        artifacts = _write_retry_artifacts(
            output_dir=output_dir,
            attempt=attempt,
            retry_batch=retry_batch,
            retry_results=retry_results,
            enabled=write_artifacts,
        )

        recovered_rows = 0
        retry_failed_positions = set(_failed_result_positions(retry_results))
        for retry_index, original_position in enumerate(failed_positions):
            if retry_index >= len(retry_results):
                continue
            retry_row = retry_results.iloc[retry_index].copy()
            retry_row["retry_attempt"] = attempt
            combined.loc[original_position, retry_row.index] = retry_row
            if retry_index not in retry_failed_positions:
                recovered_rows += 1

        remaining_failed = _failed_result_positions(combined)
        reports.append(
            {
                "attempt": attempt,
                "failed_rows_before": len(failed_positions),
                "retried_rows": len(retry_batch),
                "recovered_rows": recovered_rows,
                "remaining_failed_rows": len(remaining_failed),
                "workers": _bounded_worker_count(workers, len(retry_batch)),
                "artifact_paths": artifacts,
            }
        )
        if not remaining_failed:
            break

    return combined, reports


def _failed_result_positions(results: pd.DataFrame) -> list[int]:
    if results.empty:
        return []
    return [
        index
        for index, row in enumerate(results.to_dict(orient="records"))
        if _result_needs_retry(row)
    ]


def _result_needs_retry(row: dict[str, Any]) -> bool:
    if _non_empty(row.get("error_message")):
        return True
    if _non_empty(row.get("stage2_error_message")):
        return True
    if not _non_empty(row.get("final_committee_label")):
        return True
    if str(row.get("committee_effect") or "").strip() == "run_failed":
        return True
    if str(row.get("committee_review_safe_effect") or "").strip() == "run_failed":
        return True
    evidence_status = str(row.get("evidence_status") or "").strip().lower()
    return evidence_status in {"error", "failed"}


def _write_retry_artifacts(
    *,
    output_dir: Path | None,
    attempt: int,
    retry_batch: pd.DataFrame,
    retry_results: pd.DataFrame,
    enabled: bool,
) -> dict[str, str]:
    if not enabled or output_dir is None:
        return {}
    target_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    artifact_dir = target_dir / "retry_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    samples_path = artifact_dir / f"retry_attempt_{attempt}_samples.csv"
    results_path = artifact_dir / f"retry_attempt_{attempt}_results.csv"
    retry_batch.to_csv(samples_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    retry_results.to_csv(results_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return {
        "samples": _path_text(samples_path),
        "results": _path_text(results_path),
    }


def _run_batch_case(
    *,
    index: int,
    total: int,
    row: dict[str, Any],
    use_sample_model_view: bool,
    runtime_config: Stage2RuntimeConfig | None = None,
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
        with _stage2_runtime_context(runtime_config):
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


@contextmanager
def _stage2_runtime_context(runtime_config: Stage2RuntimeConfig | None) -> Iterator[None]:
    if runtime_config is None:
        yield
        return
    with committee_node.stage2_runtime_config_override(runtime_config):
        yield


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
        "model_name": "feature_46_xgboost_rolling_validation_replay",
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
            "active_model": "feature_46_xgboost_rolling_validation_replay",
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
    sample_trigger = _optional_bool(sample.get("stage2_review_trigger"))
    sample_secondary_trigger = _optional_bool(sample.get("stage2_secondary_trigger"))
    sample_aux_secondary_trigger = _optional_bool(sample.get("stage2_review_aux_secondary_trigger"))
    sample_fn_rescue_trigger = _optional_bool(sample.get("stage2_fn_rescue_trigger"))
    stage2_review_trigger = (
        sample_trigger
        if sample_trigger is not None
        else prediction_label == "부적격" or near_threshold
    )
    stage2_secondary_trigger = (
        sample_secondary_trigger
        if sample_secondary_trigger is not None
        else eligible_near_threshold
    )
    priority = str(sample.get("stage2_review_priority") or "").strip().lower()
    if priority not in {"none", "low", "medium", "high"}:
        priority = "none"
        if prediction_label == "부적격":
            priority = "high" if probability >= threshold + 0.10 else "medium"
        elif eligible_near_threshold or stage2_secondary_trigger:
            priority = "high" if probability >= threshold - 0.05 else "medium"
        if "recall_first" in committee_policy and prediction_label == "투자적격":
            priority = "high" if priority == "medium" else priority
    trigger_reason = str(sample.get("trigger_reason") or "").strip()
    signals = {
        "stage2_review_trigger": stage2_review_trigger,
        "stage2_secondary_trigger": stage2_secondary_trigger,
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
    if sample_aux_secondary_trigger is not None:
        signals["stage2_review_aux_secondary_trigger"] = sample_aux_secondary_trigger
    if sample_fn_rescue_trigger is not None:
        signals["stage2_fn_rescue_trigger"] = sample_fn_rescue_trigger
    for sample_column, signal_column in {
        "prob_speculative_stage2_review_aux": "probability_stage2_review_aux",
        "threshold_stage2_review_aux": "threshold_stage2_review_aux",
        "threshold_stage2_review_aux_it_services_review": (
            "threshold_stage2_review_aux_it_services_review"
        ),
        "fn_rescue_score": "fn_rescue_score",
        "fn_rescue_group_count": "fn_rescue_group_count",
    }.items():
        value = _optional_float(sample.get(sample_column))
        if value is not None:
            signals[signal_column] = value
    return signals


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    cleaned = str(value).strip().lower()
    if cleaned in {"", "nan", "none", "null"}:
        return None
    if cleaned in {"1", "true", "yes", "y", "on"}:
        return True
    if cleaned in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    stage2_runtime = _dict_value(state.get("stage2_runtime_diagnostics"))
    stage2_agent_timings = _dict_value(stage2_runtime.get("agent_elapsed_seconds"))
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
    materiality_summary = _materiality_summary(evidence.get("items", []))
    decision_trace = _decision_trace_items(committee_view)
    trace_by_gate = _decision_trace_by_gate(decision_trace)
    evidence_audit_structured = _evidence_audit_structured_fields(state)
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
        "risk_hold_reason_tags": " / ".join(
            str(item) for item in committee_view.get("risk_hold_reason_tags", []) or []
        ),
        "risk_hold_reason_labels": " / ".join(
            str(item) for item in committee_view.get("risk_hold_reason_labels", []) or []
        ),
        "risk_hold_reason_summary": committee_view.get("risk_hold_reason_summary", ""),
        "agent_disagreement_score": committee_view.get("agent_disagreement_score"),
        "agent_disagreement_level": committee_view.get("agent_disagreement_level", ""),
        "agent_disagreement_reasons": " / ".join(
            str(item) for item in committee_view.get("agent_disagreement_reasons", []) or []
        ),
        "agent_disagreement_summary": committee_view.get("agent_disagreement_summary", ""),
        "decision_trace": json.dumps(decision_trace, ensure_ascii=False, sort_keys=True),
        "committee_success": success,
        "committee_effect": effect,
        "committee_review_safe_success": review_safe_success,
        "committee_review_safe_effect": review_safe_effect,
        "stage2_backend_name": stage2_runtime.get("backend_name"),
        "stage2_llm_cache_hit": bool(stage2_runtime.get("cache_hit", False)),
        "stage2_total_elapsed_seconds": stage2_runtime.get("stage2_total_elapsed_seconds"),
        "stage2_agent_elapsed_seconds_sum": stage2_runtime.get("agent_elapsed_seconds_sum"),
        "stage2_quant_credit_elapsed_seconds": stage2_agent_timings.get("quant_credit"),
        "stage2_evidence_audit_elapsed_seconds": stage2_agent_timings.get("evidence_audit"),
        "stage2_chair_report_elapsed_seconds": stage2_agent_timings.get("chair_report"),
        "stage2_review_qa_elapsed_seconds": stage2_agent_timings.get("review_qa"),
        "stage2_risk_recall_qa_elapsed_seconds": stage2_agent_timings.get("risk_recall_qa"),
        "stage2_parallel_independent_agents": bool(
            stage2_runtime.get("parallel_independent_agents", False)
        ),
        **evidence_audit_structured,
        "stage2_review_qa_triggered": bool(stage2_runtime.get("review_qa_triggered", False)),
        "stage2_review_qa_cache_hit": bool(stage2_runtime.get("review_qa_cache_hit", False)),
        "stage2_review_qa_trigger_reasons": " / ".join(
            str(item) for item in stage2_runtime.get("review_qa_trigger_reasons", []) or []
        ),
        "stage2_review_qa_recommended_action": stage2_runtime.get(
            "review_qa_recommended_action", ""
        ),
        "stage2_review_qa_advisory_applied": bool(
            stage2_runtime.get("review_qa_advisory_applied", False)
        ),
        "stage2_review_qa_adjusted_decision_type": stage2_runtime.get(
            "review_qa_adjusted_decision_type", ""
        ),
        "stage2_review_qa_advisory_apply_reason": stage2_runtime.get(
            "review_qa_advisory_apply_reason", ""
        ),
        "stage2_risk_recall_qa_triggered": bool(
            stage2_runtime.get("risk_recall_qa_triggered", False)
        ),
        "stage2_risk_recall_qa_cache_hit": bool(
            stage2_runtime.get("risk_recall_qa_cache_hit", False)
        ),
        "stage2_risk_recall_qa_trigger_reasons": " / ".join(
            str(item) for item in stage2_runtime.get("risk_recall_qa_trigger_reasons", []) or []
        ),
        "stage2_risk_recall_qa_recommended_action": stage2_runtime.get(
            "risk_recall_qa_recommended_action", ""
        ),
        "stage2_risk_recall_qa_advisory_applied": bool(
            stage2_runtime.get("risk_recall_qa_advisory_applied", False)
        ),
        "stage2_risk_recall_qa_adjusted_decision_type": stage2_runtime.get(
            "risk_recall_qa_adjusted_decision_type", ""
        ),
        "stage2_risk_recall_qa_advisory_apply_reason": stage2_runtime.get(
            "risk_recall_qa_advisory_apply_reason", ""
        ),
        "stage2_error_message": stage2_runtime.get("error_message", ""),
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
        "materiality_event_count": materiality_summary["event_count"],
        "materiality_substantive_count": materiality_summary["substantive_count"],
        "materiality_watch_count": materiality_summary["watch_count"],
        "materiality_max_ratio": materiality_summary["max_ratio"],
        "materiality_top_basis": materiality_summary["top_basis"],
        "materiality_event_classes": materiality_summary["event_classes"],
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


def _evidence_audit_structured_fields(state: dict[str, Any]) -> dict[str, object]:
    findings = _agent_findings(state, "evidence_audit")
    structured_line = next(
        (
            finding
            for finding in findings
            if "recommended_evidence_treatment=" in finding
            or "Structured evidence treatment:" in finding
        ),
        "",
    )
    return {
        "evidence_audit_structured_found": bool(structured_line),
        "evidence_audit_critical_evidence_count": _extract_int_marker(
            structured_line,
            ("critical_evidence_count", "critical"),
        ),
        "evidence_audit_watch_context_count": _extract_int_marker(
            structured_line,
            ("watch_context_count", "watch"),
        ),
        "evidence_audit_hard_distress_detected": _extract_bool_marker(
            structured_line,
            "hard_distress_detected",
        ),
        "evidence_audit_recommended_evidence_treatment": _extract_text_marker(
            structured_line,
            "recommended_evidence_treatment",
        ),
        "evidence_audit_top_materiality_basis": _extract_text_marker(
            structured_line,
            "top_materiality_basis",
        ),
    }


def _agent_findings(state: dict[str, Any], role: str) -> list[str]:
    agent_summary = _dict_value(state.get("agent_summary"))
    agents = _dict_value(agent_summary.get("agents"))
    agent = _dict_value(agents.get(role))
    raw_findings = agent.get("findings", [])
    if isinstance(raw_findings, list):
        return [str(item) for item in raw_findings if str(item).strip()]
    raw_agent_outputs = state.get("agent_outputs")
    if not isinstance(raw_agent_outputs, list):
        return []
    for output in raw_agent_outputs:
        output_dict = _model_or_dict_value(output)
        if str(output_dict.get("role") or "") != role:
            continue
        raw_output_findings = output_dict.get("findings", [])
        if isinstance(raw_output_findings, list):
            return [str(item) for item in raw_output_findings if str(item).strip()]
    return []


def _extract_int_marker(text: str, marker_names: tuple[str, ...]) -> int:
    for marker_name in marker_names:
        match = re.search(rf"{re.escape(marker_name)}=(\d+)", text)
        if match:
            return int(match.group(1))
    return 0


def _extract_bool_marker(text: str, marker_name: str) -> bool:
    match = re.search(rf"{re.escape(marker_name)}=(true|false)", text, flags=re.IGNORECASE)
    return bool(match and match.group(1).lower() == "true")


def _extract_text_marker(text: str, marker_name: str) -> str:
    match = re.search(rf"{re.escape(marker_name)}=([^;]+)", text)
    if not match:
        return ""
    return match.group(1).strip()


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


def _default_stage2_llm_cache() -> bool:
    raw_value = os.environ.get("CAS_STAGE2_LLM_CACHE_ENABLED", "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def _default_retry_failed_attempts() -> int:
    raw_value = os.environ.get("CAS_COMMITTEE_BATCH_RETRY_FAILED_ATTEMPTS", "0").strip()
    try:
        attempts = int(raw_value)
    except ValueError:
        return 0
    return min(max(attempts, 0), 5)


def _default_retry_failed_workers() -> int:
    raw_value = os.environ.get("CAS_COMMITTEE_BATCH_RETRY_FAILED_WORKERS", "1").strip()
    try:
        workers = int(raw_value)
    except ValueError:
        return 1
    return max(workers, 1)


def _default_retry_failed_delay_seconds() -> float:
    raw_value = os.environ.get("CAS_COMMITTEE_BATCH_RETRY_FAILED_DELAY_SECONDS", "1.0").strip()
    try:
        delay = float(raw_value)
    except ValueError:
        return 1.0
    return min(max(delay, 0.0), 120.0)


def _bounded_worker_count(workers: int, row_count: int) -> int:
    if row_count <= 0:
        return 1
    return min(max(workers, 1), row_count)


def _dict_value(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _model_or_dict_value(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    return {}


def _non_empty(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


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


def _materiality_summary(items: object) -> dict[str, object]:
    if not isinstance(items, list):
        return {
            "event_count": 0,
            "substantive_count": 0,
            "watch_count": 0,
            "max_ratio": "",
            "top_basis": "",
            "event_classes": "",
        }

    materiality_items: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict) or item.get("company_match") is not True:
            continue
        if not (
            item.get("materiality_ratio") not in {None, ""}
            or item.get("materiality_basis")
            or item.get("disclosure_materiality")
        ):
            continue
        materiality_items.append(item)

    ratios: list[tuple[float, str]] = []
    event_classes: list[str] = []
    substantive_count = 0
    watch_count = 0
    for item in materiality_items:
        event_class = str(item.get("disclosure_event_class") or "").strip()
        if event_class and event_class not in event_classes:
            event_classes.append(event_class)
        materiality = str(item.get("disclosure_materiality") or "").strip().lower()
        severity = str(item.get("disclosure_severity") or "").strip().lower()
        if materiality == "substantive_adverse" or severity == "adverse":
            substantive_count += 1
        elif materiality in {"watch_context", "procedural_or_one_off"} or severity == "caution":
            watch_count += 1
        ratio = _safe_float(item.get("materiality_ratio"), default=-1.0)
        if ratio >= 0:
            ratios.append((ratio, str(item.get("materiality_basis") or "").strip()))

    max_ratio = ""
    top_basis = ""
    if ratios:
        ratio, basis = max(ratios, key=lambda pair: pair[0])
        max_ratio = round(ratio, 4)
        top_basis = basis

    return {
        "event_count": len(materiality_items),
        "substantive_count": substantive_count,
        "watch_count": watch_count,
        "max_ratio": max_ratio,
        "top_basis": top_basis,
        "event_classes": " / ".join(event_classes[:6]),
    }


def write_outputs(
    results: pd.DataFrame,
    *,
    output_dir: Path,
    retry_reports: list[dict[str, Any]] | None = None,
) -> None:
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "committee_review_batch_results.csv"
    summary_path = output_dir / "committee_review_batch_summary.json"
    report_path = output_dir / "committee_review_batch_report.md"
    results.to_csv(detail_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    summary = _summary(results)
    if retry_reports:
        summary["retry"] = _retry_summary(results, retry_reports)
    summary["paths"] = {
        "details": _path_text(detail_path),
        "summary": _path_text(summary_path),
        "report": _path_text(report_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_report(results, summary), encoding="utf-8")
    print(f"[Saved] {detail_path}")
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {report_path}")


def _retry_summary(results: pd.DataFrame, retry_reports: list[dict[str, Any]]) -> dict[str, Any]:
    final_failed_rows = len(_failed_result_positions(results))
    return {
        "attempts_run": len(retry_reports),
        "initial_failed_rows": retry_reports[0]["failed_rows_before"] if retry_reports else 0,
        "final_failed_rows": final_failed_rows,
        "total_recovered_rows": sum(
            int(report.get("recovered_rows", 0)) for report in retry_reports
        ),
        "attempts": retry_reports,
    }


def _path_text(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
    if "stage2_total_elapsed_seconds" in results.columns and len(results):
        stage2_elapsed = pd.to_numeric(results["stage2_total_elapsed_seconds"], errors="coerce")
        if not stage2_elapsed.dropna().empty:
            summary["stage2_speed"] = {
                "stage2_total_elapsed_seconds_sum": round(float(stage2_elapsed.sum()), 4),
                "stage2_total_elapsed_seconds_mean": round(float(stage2_elapsed.mean()), 4),
                "stage2_total_elapsed_seconds_median": round(float(stage2_elapsed.median()), 4),
                "stage2_total_elapsed_seconds_max": round(float(stage2_elapsed.max()), 4),
                "stage2_llm_cache_hit_rows": int(
                    results.get("stage2_llm_cache_hit", pd.Series(dtype=bool)).fillna(False).sum()
                ),
            }
    if "materiality_event_count" in results.columns and len(results):
        event_counts = pd.to_numeric(results["materiality_event_count"], errors="coerce").fillna(0)
        max_ratios = pd.to_numeric(results.get("materiality_max_ratio"), errors="coerce")
        summary["materiality"] = {
            "rows_with_materiality_events": int((event_counts > 0).sum()),
            "materiality_event_count_sum": int(event_counts.sum()),
            "materiality_max_ratio": (
                round(float(max_ratios.max()), 4) if not max_ratios.dropna().empty else None
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
        "materiality_event_count",
        "materiality_max_ratio",
        "materiality_top_basis",
    ]
    preview = results.loc[:, [column for column in preview_columns if column in results.columns]]
    return "\n".join(
        [
            "# Committee Review Batch Results",
            "",
            f"- Rows: {summary['rows']}",
            f"- Strict committee success rate: {summary['success_rate']:.1%}",
            f"- Review-safe success rate: {summary['review_safe_success_rate']:.1%}",
            _retry_report_line(summary),
            _speed_report_line(summary),
            _stage2_speed_report_line(summary),
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


def _stage2_speed_report_line(summary: dict[str, Any]) -> str:
    speed = summary.get("stage2_speed")
    if not isinstance(speed, dict):
        return "- Stage 2 LLM speed: not measured"
    return (
        "- Stage 2 LLM speed: "
        f"mean `{speed.get('stage2_total_elapsed_seconds_mean')}` sec, "
        f"max `{speed.get('stage2_total_elapsed_seconds_max')}` sec, "
        f"cache hits `{speed.get('stage2_llm_cache_hit_rows')}`"
    )


def _retry_report_line(summary: dict[str, Any]) -> str:
    retry = summary.get("retry")
    if not isinstance(retry, dict):
        return "- Retry: not enabled"
    return (
        "- Retry: "
        f"attempts `{retry.get('attempts_run')}`, "
        f"initial failed rows `{retry.get('initial_failed_rows')}`, "
        f"recovered `{retry.get('total_recovered_rows')}`, "
        f"final failed rows `{retry.get('final_failed_rows')}`"
    )


def main() -> None:
    args = parse_args()
    runtime_config = configure_runtime(
        live_external_evidence=args.live_external_evidence,
        stage2_runner=args.stage2_runner,
        stage2_agno_mode=args.stage2_agno_mode,
        stage2_model_provider=args.stage2_model_provider,
        stage2_model=args.stage2_model,
        stage2_llm_cache=args.stage2_llm_cache,
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
        runtime_config=runtime_config,
    )
    results, retry_reports = retry_failed_cases(
        batch,
        results,
        use_sample_model_view=args.use_sample_model_view,
        attempts=args.retry_failed_attempts,
        workers=args.retry_failed_workers,
        delay_seconds=args.retry_failed_delay_seconds,
        output_dir=args.output_dir,
        write_artifacts=args.retry_failed_artifacts,
        runtime_config=runtime_config,
    )
    write_outputs(results, output_dir=args.output_dir, retry_reports=retry_reports)


if __name__ == "__main__":
    main()
