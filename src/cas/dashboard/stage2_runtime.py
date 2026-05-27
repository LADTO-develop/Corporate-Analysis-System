"""Stage 2 dashboard runtime, cache, and live-review helpers."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
from cas.dashboard.settings import (
    DASHBOARD_BASE_STAGE2_RUNNER,
    DASHBOARD_COMMITTEE_CONTEXT_CACHE_VERSION,
    DASHBOARD_LIVE_STAGE2_RUNNER,
)
from cas.evidence import collect_external_evidence
from cas.utils.live_cache import live_cache_dir, read_json_cache, stable_cache_key, write_json_cache

LOGGER = logging.getLogger(__name__)


class DashboardCommitteeContextBuilder(Protocol):
    """Callable contract for building a committee context from dashboard rows."""

    def __call__(
        self,
        *,
        selected_row: pd.Series,
        prediction_row: pd.Series | None,
        local_shap: pd.DataFrame,
        peer_slice: pd.DataFrame,
        external_evidence_snapshot: dict[str, object] | None = None,
        stage2_runner: str | None = None,
        stage2_runtime_config: Stage2RuntimeConfig | None = None,
    ) -> dict[str, object] | None:
        """Build and return a dashboard committee context."""


def empty_dashboard_evidence_snapshot() -> dict[str, object]:
    """Return a friendly empty evidence snapshot for dashboard-first review."""
    return {
        "status": "not_requested",
        "source": "external_evidence",
        "enabled": False,
        "items": [],
        "providers": {},
        "has_critical_risk": False,
        "critical_terms": [],
        "message": "대시보드에서 아직 실시간 뉴스/웹/OpenDART 외부 근거를 수집하지 않았습니다.",
    }


def dashboard_evidence_key(selected_row: pd.Series) -> str:
    """Build a Streamlit session key for cached dashboard evidence."""
    return (
        f"external_evidence:v2:{_stock_code_text(selected_row.get('stock_code'))}:"
        f"{_optional_int(selected_row.get('fiscal_year')) or 'latest'}"
    )


def dashboard_cache_read(namespace: str, key: str) -> dict[str, object] | None:
    """Read a dashboard JSON cache that can survive browser refreshes."""
    return cast(
        "dict[str, object] | None",
        read_json_cache(
            namespace,
            key,
            env_var="CAS_DASHBOARD_CACHE_ENABLED",
            default=True,
        ),
    )


def dashboard_cache_write(
    namespace: str,
    key: str,
    payload: dict[str, object],
) -> None:
    """Persist a dashboard JSON cache without exposing this concern to render code."""
    write_json_cache(
        namespace,
        key,
        payload,
        env_var="CAS_DASHBOARD_CACHE_ENABLED",
        default=True,
    )


def dashboard_cache_file_path(namespace: str, key: str) -> Path:
    """Return the dashboard cache file path for metadata display."""
    safe_namespace = namespace.replace("/", "_").replace("..", "_")
    return Path(live_cache_dir()) / safe_namespace / f"{key}.json"


def dashboard_cache_saved_at_label(namespace: str, key: str) -> str | None:
    """Return a user-facing saved timestamp for an existing dashboard cache file."""
    path = dashboard_cache_file_path(namespace, key)
    if not path.exists():
        return None
    try:
        saved_at = datetime.fromtimestamp(path.stat().st_mtime, tz=ZoneInfo("Asia/Seoul"))
    except OSError:
        return None
    return saved_at.strftime("%Y-%m-%d %H:%M")


def dashboard_stage2_async_workers() -> int:
    """Return the small worker pool size for live Stage 2 dashboard jobs."""
    try:
        raw_value = int(os.environ.get("CAS_DASHBOARD_STAGE2_ASYNC_WORKERS", "2"))
    except ValueError:
        raw_value = 2
    return max(1, min(raw_value, 4))


_stage2_executor_cache = cast(
    Callable[[Callable[[int], ThreadPoolExecutor]], Callable[[int], ThreadPoolExecutor]],
    st.cache_resource(show_spinner=False),
)


@_stage2_executor_cache
def dashboard_stage2_executor(max_workers: int) -> ThreadPoolExecutor:
    """Share a bounded executor across Streamlit reruns."""
    return ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="cas-dashboard-stage2",
    )


def dashboard_get_stage2_executor() -> ThreadPoolExecutor:
    """Return the shared dashboard Stage 2 executor."""
    return dashboard_stage2_executor(dashboard_stage2_async_workers())


def dashboard_evidence_cache_key(selected_row: pd.Series) -> str:
    """Build a stable file-cache key for dashboard external evidence."""
    return cast(
        str,
        stable_cache_key(
            {
                "cache_version": "dashboard_external_evidence_v1",
                "stock_code": _stock_code_text(selected_row.get("stock_code")),
                "corp_name": str(selected_row.get("corp_name") or ""),
                "corp_code": _optional_text(selected_row.get("corp_code")),
                "fiscal_year": _optional_int(selected_row.get("fiscal_year")),
                "eval_year": _optional_int(selected_row.get("eval_year")),
                "as_of_date": dashboard_evidence_as_of_date(selected_row),
            }
        ),
    )


def dashboard_stage2_runtime_config(
    stage2_runner: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Stage2RuntimeConfig:
    """Capture dashboard Stage 2 settings before cache lookup or async execution."""
    source_env = os.environ if env is None else env
    runner = (
        stage2_runner
        or source_env.get("CAS_DASHBOARD_STAGE2_RUNNER")
        or source_env.get("CAS_STAGE2_RUNNER")
        or DASHBOARD_BASE_STAGE2_RUNNER
    )
    return Stage2RuntimeConfig.from_env(source_env, runner=runner)


def dashboard_stage2_runner_name(
    stage2_runner: str | None = None,
    *,
    runtime_config: Stage2RuntimeConfig | None = None,
) -> str:
    """Return the dashboard Stage 2 runner name used for cache separation."""
    config = runtime_config or dashboard_stage2_runtime_config(stage2_runner)
    return str(config.runner).strip().lower() or DASHBOARD_BASE_STAGE2_RUNNER


def dashboard_stage2_cache_config(
    stage2_runner: str | None = None,
    *,
    runtime_config: Stage2RuntimeConfig | None = None,
) -> dict[str, object]:
    """Return the Stage 2 prompt/model knobs that should invalidate dashboard cache."""
    config = runtime_config or dashboard_stage2_runtime_config(stage2_runner)
    return cast(
        dict[str, object],
        config.cache_payload(cache_version=DASHBOARD_COMMITTEE_CONTEXT_CACHE_VERSION),
    )


def dashboard_committee_cache_key(
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    external_evidence_snapshot: dict[str, object],
    *,
    stage2_runner: str | None = None,
    runtime_config: Stage2RuntimeConfig | None = None,
) -> str | None:
    """Build a stable cache key for the rendered committee decision context."""
    if prediction_row is None:
        return None
    return cast(
        str,
        stable_cache_key(
            {
                "cache_version": DASHBOARD_COMMITTEE_CONTEXT_CACHE_VERSION,
                "stage2_cache_config": dashboard_stage2_cache_config(
                    stage2_runner,
                    runtime_config=runtime_config,
                ),
                "stock_code": _stock_code_text(selected_row.get("stock_code")),
                "corp_name": str(selected_row.get("corp_name") or ""),
                "fiscal_year": _optional_int(selected_row.get("fiscal_year")),
                "eval_year": _optional_int(selected_row.get("eval_year")),
                "probability_speculative": _optional_float(prediction_row.get("prob_speculative")),
                "threshold": _optional_float(prediction_row.get("threshold")),
                "predicted_label": _clean_dashboard_value(prediction_row.get("predicted_label")),
                "risk_band": _clean_dashboard_value(prediction_row.get("risk_band")),
                "stage2_review_priority": _clean_dashboard_value(
                    prediction_row.get("stage2_review_priority")
                ),
                "stage2_review_trigger": _optional_bool(
                    prediction_row.get("stage2_review_trigger")
                ),
                "stage2_secondary_trigger": _optional_bool(
                    prediction_row.get("stage2_secondary_trigger")
                ),
                "overwarning_filter_candidate": _optional_bool(
                    prediction_row.get("stage2_overwarning_filter_candidate")
                ),
                "external_evidence_key": cast(str, stable_cache_key(external_evidence_snapshot)),
            }
        ),
    )


def dashboard_stage2_job_key(
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    *,
    runtime_config: Stage2RuntimeConfig | None = None,
) -> str | None:
    """Build a session key for one in-flight live Agno dashboard job."""
    if prediction_row is None:
        return None
    return "dashboard_stage2_live_job:" + cast(
        str,
        stable_cache_key(
            {
                "cache_version": "dashboard_stage2_live_job_v1",
                "stage2_cache_config": dashboard_stage2_cache_config(
                    DASHBOARD_LIVE_STAGE2_RUNNER,
                    runtime_config=runtime_config,
                ),
                "stock_code": _stock_code_text(selected_row.get("stock_code")),
                "corp_name": str(selected_row.get("corp_name") or ""),
                "fiscal_year": _optional_int(selected_row.get("fiscal_year")),
                "eval_year": _optional_int(selected_row.get("eval_year")),
                "probability_speculative": _optional_float(prediction_row.get("prob_speculative")),
                "threshold": _optional_float(prediction_row.get("threshold")),
                "predicted_label": _clean_dashboard_value(prediction_row.get("predicted_label")),
                "risk_band": _clean_dashboard_value(prediction_row.get("risk_band")),
            }
        ),
    )


def dashboard_stage2_header_request_key(selected_row: pd.Series) -> str:
    """Return the key used when the selected-company header requests precise review."""
    fiscal_year = _optional_int(selected_row.get("fiscal_year"))
    fiscal_year_text = (
        str(fiscal_year) if fiscal_year is not None else str(selected_row.get("fiscal_year"))
    )
    return f"{_stock_code_text(selected_row.get('stock_code'))}:{fiscal_year_text}"


def persist_dashboard_committee_context(
    *,
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    external_evidence_snapshot: dict[str, object],
    committee_context: dict[str, object] | None,
    stage2_runner: str | None = None,
    runtime_config: Stage2RuntimeConfig | None = None,
) -> None:
    """Persist a dashboard committee context in session and disk cache."""
    if committee_context is None:
        return
    cache_key = dashboard_committee_cache_key(
        selected_row,
        prediction_row,
        external_evidence_snapshot,
        stage2_runner=stage2_runner,
        runtime_config=runtime_config,
    )
    if cache_key:
        st.session_state[cache_key] = committee_context
        dashboard_cache_write("dashboard_committee_context", cache_key, committee_context)


def resolve_dashboard_committee_context(
    *,
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    local_shap: pd.DataFrame,
    peer_slice: pd.DataFrame,
    external_evidence_snapshot: dict[str, object],
    build_committee_context: DashboardCommitteeContextBuilder,
    stage2_runner: str | None = None,
    runtime_config: Stage2RuntimeConfig | None = None,
    build_if_missing: bool = True,
) -> tuple[dict[str, object] | None, bool]:
    """Return the dashboard committee context, reusing it within the current session."""
    resolved_config = runtime_config or dashboard_stage2_runtime_config(stage2_runner)
    cache_key = dashboard_committee_cache_key(
        selected_row,
        prediction_row,
        external_evidence_snapshot,
        runtime_config=resolved_config,
    )
    if cache_key:
        cached = st.session_state.get(cache_key)
        if isinstance(cached, dict):
            return cast(dict[str, object], cached), True
        disk_cached = dashboard_cache_read("dashboard_committee_context", cache_key)
        if isinstance(disk_cached, dict):
            st.session_state[cache_key] = disk_cached
            return disk_cached, True
    if not build_if_missing:
        return None, False
    committee_context = build_committee_context(
        selected_row=selected_row,
        prediction_row=prediction_row,
        local_shap=local_shap,
        peer_slice=peer_slice,
        external_evidence_snapshot=external_evidence_snapshot,
        stage2_runner=resolved_config.runner,
        stage2_runtime_config=resolved_config,
    )
    persist_dashboard_committee_context(
        selected_row=selected_row,
        prediction_row=prediction_row,
        external_evidence_snapshot=external_evidence_snapshot,
        committee_context=committee_context,
        runtime_config=resolved_config,
    )
    return committee_context, False


def dashboard_evidence_as_of_date(selected_row: pd.Series) -> str:
    """Return the date cut-off for dashboard evidence collection."""
    fiscal_year = _optional_int(selected_row.get("fiscal_year"))
    if fiscal_year is not None:
        return min(date(fiscal_year, 12, 31), date.today()).isoformat()
    eval_year = _optional_int(selected_row.get("eval_year"))
    if eval_year is None:
        return date.today().isoformat()
    return min(date(eval_year, 12, 31), date.today()).isoformat()


def collect_dashboard_external_evidence(selected_row: pd.Series) -> dict[str, object]:
    """Collect live external evidence for the selected dashboard company on demand."""
    env = dict(os.environ)
    env["CAS_ENABLE_EXTERNAL_EVIDENCE"] = "1"
    env.setdefault("CAS_OPENDART_CORP_CODE_CACHE_PATH", "/private/tmp/cas_opendart_corp_codes.csv")
    return cast(
        dict[str, object],
        collect_external_evidence(
            company_name=str(selected_row.get("corp_name") or ""),
            stock_code=_stock_code_text(selected_row.get("stock_code")),
            corp_code=_optional_text(selected_row.get("corp_code")),
            as_of_date=dashboard_evidence_as_of_date(selected_row),
            env=env,
        ),
    )


def persist_dashboard_external_evidence(
    selected_row: pd.Series,
    snapshot: dict[str, object],
) -> None:
    """Persist a successful external-evidence snapshot in session and disk cache."""
    evidence_key = dashboard_evidence_key(selected_row)
    st.session_state[evidence_key] = snapshot
    if snapshot.get("status") != "error":
        dashboard_cache_write(
            "dashboard_external_evidence",
            dashboard_evidence_cache_key(selected_row),
            snapshot,
        )


def resolve_dashboard_external_evidence_cached(selected_row: pd.Series) -> dict[str, object]:
    """Return cached dashboard evidence without triggering network calls."""
    evidence_key = dashboard_evidence_key(selected_row)
    cached = st.session_state.get(evidence_key)
    if isinstance(cached, dict):
        return cast(dict[str, object], cached)
    disk_cache_key = dashboard_evidence_cache_key(selected_row)
    disk_cached = dashboard_cache_read("dashboard_external_evidence", disk_cache_key)
    if isinstance(disk_cached, dict):
        st.session_state[evidence_key] = disk_cached
        return disk_cached
    return empty_dashboard_evidence_snapshot()


def resolve_dashboard_external_evidence(selected_row: pd.Series) -> dict[str, object]:
    """Return cached live evidence, collecting it automatically on first tab render."""
    cached = resolve_dashboard_external_evidence_cached(selected_row)
    if cached.get("status") != "not_requested":
        return cached
    try:
        with st.spinner("외부 근거를 자동 수집하고 2차 위원회 판단에 반영하는 중입니다..."):
            snapshot = collect_dashboard_external_evidence(selected_row)
    except Exception as error:  # pragma: no cover - runtime/network dependent
        snapshot = {
            "status": "error",
            "source": "external_evidence",
            "enabled": True,
            "items": [],
            "providers": {},
            "has_critical_risk": False,
            "critical_terms": [],
            "message": str(error),
        }
    persist_dashboard_external_evidence(selected_row, snapshot)
    return snapshot


def dashboard_needs_live_stage2_from_views(
    model_view: dict[str, object],
    evidence_snapshot: dict[str, object],
) -> bool:
    """Return whether the dashboard should suggest live Agno review."""
    if bool(model_view.get("stage2_review_trigger")):
        return True
    if bool(model_view.get("stage2_secondary_trigger")):
        return True
    if str(model_view.get("stage2_review_priority") or "").strip().lower() in {"medium", "high"}:
        return True

    return (
        bool(evidence_snapshot.get("has_critical_risk"))
        or (_optional_int(evidence_snapshot.get("veto_candidate_count")) or 0) > 0
        or (_optional_int(evidence_snapshot.get("high_confidence_critical_count")) or 0) > 0
    )


def run_dashboard_live_stage2_job(
    *,
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    local_shap: pd.DataFrame,
    peer_slice: pd.DataFrame,
    build_committee_context: DashboardCommitteeContextBuilder,
    runtime_config: Stage2RuntimeConfig,
) -> dict[str, object]:
    """Run network-backed evidence collection and Agno Stage 2 off the Streamlit thread."""
    evidence_snapshot = collect_dashboard_external_evidence(selected_row)
    committee_context = build_committee_context(
        selected_row=selected_row,
        prediction_row=prediction_row,
        local_shap=local_shap,
        peer_slice=peer_slice,
        external_evidence_snapshot=evidence_snapshot,
        stage2_runner=runtime_config.runner,
        stage2_runtime_config=runtime_config,
    )
    return {
        "evidence_snapshot": evidence_snapshot,
        "committee_context": committee_context or {},
    }


def start_dashboard_live_stage2_job(
    *,
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    local_shap: pd.DataFrame,
    peer_slice: pd.DataFrame,
    build_committee_context: DashboardCommitteeContextBuilder,
    runtime_config: Stage2RuntimeConfig | None = None,
) -> Future[dict[str, object]] | None:
    """Start one in-flight live Agno job for the selected dashboard company."""
    resolved_config = runtime_config or dashboard_stage2_runtime_config(
        DASHBOARD_LIVE_STAGE2_RUNNER
    )
    job_key = dashboard_stage2_job_key(
        selected_row,
        prediction_row,
        runtime_config=resolved_config,
    )
    if job_key is None:
        return None
    existing = st.session_state.get(job_key)
    if isinstance(existing, Future) and not existing.done():
        return cast(Future[dict[str, object]], existing)
    future = dashboard_get_stage2_executor().submit(
        run_dashboard_live_stage2_job,
        selected_row=selected_row.copy(deep=True),
        prediction_row=prediction_row.copy(deep=True) if prediction_row is not None else None,
        local_shap=local_shap.copy(deep=True),
        peer_slice=peer_slice.copy(deep=True),
        build_committee_context=build_committee_context,
        runtime_config=resolved_config,
    )
    st.session_state[job_key] = future
    return future


def consume_dashboard_live_stage2_job(
    *,
    selected_row: pd.Series,
    prediction_row: pd.Series | None,
    format_error_detail: Callable[[Exception], str],
    runtime_config: Stage2RuntimeConfig | None = None,
) -> tuple[dict[str, object] | None, dict[str, object] | None, str | None]:
    """Resolve a completed live Agno job and persist its caches."""
    resolved_config = runtime_config or dashboard_stage2_runtime_config(
        DASHBOARD_LIVE_STAGE2_RUNNER
    )
    job_key = dashboard_stage2_job_key(
        selected_row,
        prediction_row,
        runtime_config=resolved_config,
    )
    if job_key is None:
        return None, None, None
    future = st.session_state.get(job_key)
    if not isinstance(future, Future):
        return None, None, None
    if not future.done():
        return None, None, "running"
    st.session_state.pop(job_key, None)
    try:
        result = future.result()
    except Exception as error:  # pragma: no cover - runtime/network dependent
        LOGGER.exception("dashboard_live_stage2_job_failed")
        return None, None, f"error:{format_error_detail(error)}"

    evidence_snapshot = cast(
        dict[str, object],
        result.get("evidence_snapshot") or empty_dashboard_evidence_snapshot(),
    )
    raw_committee_context = result.get("committee_context")
    committee_context = (
        cast(dict[str, object], raw_committee_context)
        if isinstance(raw_committee_context, dict) and raw_committee_context
        else None
    )
    persist_dashboard_external_evidence(selected_row, evidence_snapshot)
    persist_dashboard_committee_context(
        selected_row=selected_row,
        prediction_row=prediction_row,
        external_evidence_snapshot=evidence_snapshot,
        committee_context=committee_context,
        runtime_config=resolved_config,
    )
    return committee_context, evidence_snapshot, "completed"


def _clean_dashboard_value(value: object) -> object:
    """Convert pandas/numpy scalars into plain values for Stage 2 payloads."""
    if isinstance(value, dict):
        return {str(key): _clean_dashboard_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean_dashboard_value(item) for item in value]
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, str):
        try:
            return _clean_dashboard_value(value.item())
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return value


def _optional_float(value: object, *, default: float = 0.0) -> float:
    """Return a safe float from dashboard artifacts."""
    cleaned = _clean_dashboard_value(value)
    if cleaned is None:
        return default
    try:
        return float(str(cleaned))
    except (TypeError, ValueError):
        return default


def _optional_int(value: object) -> int | None:
    """Return a safe int from dashboard artifacts."""
    cleaned = _clean_dashboard_value(value)
    if cleaned is None:
        return None
    try:
        return int(float(str(cleaned)))
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object, *, default: bool = False) -> bool:
    """Return a safe bool from dashboard artifacts."""
    cleaned = _clean_dashboard_value(value)
    if cleaned is None:
        return default
    if isinstance(cleaned, bool):
        return cleaned
    if isinstance(cleaned, int | float):
        return bool(cleaned)
    text = str(cleaned).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _stock_code_text(value: object) -> str:
    """Normalize stock code text while preserving leading zeroes."""
    cleaned = _clean_dashboard_value(value)
    text = str(cleaned or "").strip()
    if text.endswith(".0"):
        text = text.removesuffix(".0")
    return text.zfill(6) if text.isdigit() else text


def _optional_text(value: object) -> str | None:
    """Return a clean optional text value."""
    cleaned = _clean_dashboard_value(value)
    if cleaned is None:
        return None
    text = str(cleaned).strip()
    return text or None
