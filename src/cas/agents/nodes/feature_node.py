"""Load the selected company's feature-store snapshot."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cas.agents.state import AgentState, AuditEntry
from cas.utils.io import read_json, read_yaml

_FEATURE_LIST_PATH = Path("data/input/credit_46_features/feature_46_list.json")


def run(state: AgentState) -> dict[str, Any]:
    """Compute a local feature-store snapshot for downstream realtime inference."""
    source_row = dict(state.get("source_feature_row") or {})
    if source_row:
        # 현재 프로젝트의 주 흐름은 정형화된 43변수 입력셋을 그대로 쓰는 것이다.
        # 이 경우 feature node는 재계산보다 "모델 입력 벡터 구성"에 집중한다.
        return _run_dataset_backed_feature_store(state, source_row)

    cfg = read_yaml("configs/runtime/analysis.yaml")
    financials = state.get("raw_financials") or {}
    profile = state.get("company_profile") or {}
    qualitative = profile.get("qualitative") or {}
    market_context = profile.get("market_context") or {}

    if not financials:
        audit = AuditEntry(
            node="feature_store",
            timestamp=_now(),
            summary="No financial inputs in state; skipping feature computation.",
        )
        return {"audit": [audit]}

    ranges = cfg["feature_ranges"]
    features = {
        "revenue_growth_score": _score(
            financials.get("revenue_growth_pct"), ranges["revenue_growth_pct"]
        ),
        "profitability_score": _score(
            financials.get("operating_margin_pct"), ranges["operating_margin_pct"]
        ),
        "leverage_health_score": _score(financials.get("debt_to_equity"), ranges["debt_to_equity"]),
        "liquidity_score": _score(financials.get("current_ratio"), ranges["current_ratio"]),
        "cash_generation_score": _score(
            financials.get("free_cash_flow_margin_pct"), ranges["free_cash_flow_margin_pct"]
        ),
        "interest_coverage_score": _score(
            financials.get("interest_coverage"),
            ranges["interest_coverage"],
        ),
        "governance_score": _score(qualitative.get("governance_score"), ranges["governance_score"]),
        "product_momentum_score": _score(
            qualitative.get("product_momentum_score"), ranges["product_momentum_score"]
        ),
        "concentration_health_score": _score(
            qualitative.get("customer_concentration_pct", 0.0),
            ranges["customer_concentration_pct"],
        ),
        "industry_position_score": _score(
            market_context.get("industry_growth_score", 0.5),
            ranges["industry_growth_score"],
        ),
        "controversy_penalty": _controversy_penalty(
            str(qualitative.get("controversy_level", "low"))
        ),
    }

    audit = AuditEntry(
        node="feature_store",
        timestamp=_now(),
        summary=f"Loaded feature-store snapshot with {len(features)} normalized features",
        metrics={"n_features": float(len(features))},
    )
    model_registry = cfg.get("model_registry", {})
    return {
        "normalized_features": features,
        "feature_store_snapshot": {
            "store_name": "local_feature_store",
            "company_id": state.get("company_id"),
            "analysis_year": state.get("analysis_year", 0),
            "features": features,
            "source": state.get("processed_company_list_ref", "data/input/companies"),
        },
        "model_registry_ref": {
            "registry_name": model_registry.get("registry_name", "local_model_registry"),
            "active_model": model_registry.get("active_model", "xgboost_realtime"),
            "model_version": model_registry.get("model_version", "local-deterministic"),
        },
        "audit": [audit],
    }


def _run_dataset_backed_feature_store(
    state: AgentState,
    source_row: dict[str, Any],
) -> dict[str, Any]:
    feature_spec = read_json(_FEATURE_LIST_PATH)
    model_feature_names = [str(name) for name in feature_spec["model_features"]]
    model_features: dict[str, float] = {}
    for name in model_feature_names:
        value = _numeric_or_none(source_row.get(name))
        if value is not None:
            model_features[name] = value
    normalized_features = {
        "revenue_growth_score": _scale(source_row.get("total_assets_growth"), -0.3, 0.3),
        "profitability_score": _scale(source_row.get("net_margin"), -0.3, 0.3),
        "leverage_health_score": _scale(
            source_row.get("debt_ratio"),
            0.0,
            2.0,
            higher_is_better=False,
        ),
        "liquidity_score": _scale(source_row.get("current_ratio"), 0.5, 5.0),
        "cash_generation_score": _scale(source_row.get("cashflow_coverage_ratio"), -5.0, 20.0),
        "interest_coverage_score": _scale(
            source_row.get("interest_coverage_ratio"),
            -10.0,
            20.0,
        ),
        "governance_score": 0.5,
        "product_momentum_score": _scale(source_row.get("market_to_book"), 0.0, 10.0),
        "concentration_health_score": 0.5,
        "industry_position_score": 0.5,
        "controversy_penalty": 0.45,
    }
    audit = AuditEntry(
        node="feature_store",
        timestamp=_now(),
        summary=(
            f"Loaded dataset-backed feature snapshot with {len(model_features)} model features "
            f"for {state.get('company_name', state.get('company_id', 'unknown'))}"
        ),
        metrics={"n_model_features": float(len(model_features))},
    )
    return {
        "normalized_features": normalized_features,
        "model_features": model_features,
        "feature_store_snapshot": {
            "store_name": "credit_43_feature_store",
            "company_id": state.get("company_id"),
            "analysis_year": state.get("analysis_year", 0),
            "source_path": str(_FEATURE_LIST_PATH),
            "normalized_features": normalized_features,
            "model_features": model_features,
        },
        "audit": [audit],
    }


def _score(value: Any, spec: dict[str, Any]) -> float:
    if value is None:
        return 0.5
    raw = float(value)
    lower = float(spec["min"])
    upper = float(spec["max"])
    if upper <= lower:
        return 0.5
    clipped = min(max(raw, lower), upper)
    ratio = (clipped - lower) / (upper - lower)
    if not bool(spec.get("higher_is_better", True)):
        ratio = 1.0 - ratio
    return round(ratio, 4)


def _controversy_penalty(level: str) -> float:
    return {"low": 0.9, "moderate": 0.45, "high": 0.1}.get(level.lower(), 0.45)


def _scale(
    value: Any,
    lower: float,
    upper: float,
    *,
    higher_is_better: bool = True,
) -> float:
    if value is None:
        return 0.5
    numeric = float(value)
    if numeric != numeric:
        return 0.5
    clipped = min(max(numeric, lower), upper)
    ratio = (clipped - lower) / (upper - lower)
    if not higher_is_better:
        ratio = 1.0 - ratio
    return round(ratio, 4)


def _numeric_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
        if numeric != numeric:
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
