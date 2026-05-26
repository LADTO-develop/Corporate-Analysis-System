"""Build provider-neutral input bundles for Stage 2 agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from cas.agents.state import AgentState

PromptPayloadRole = Literal[
    "stage2",
    "quant_credit",
    "evidence_audit",
    "chair_report",
    "review_qa",
    "risk_recall_qa",
]

_FINANCIAL_PROMPT_KEYS = (
    "stock_code",
    "corp_name",
    "company_name",
    "market",
    "fiscal_year",
    "eval_year",
    "industry_macro_category",
    "firm_size_group",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "revenue",
    "sales",
    "operating_income",
    "net_income",
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "cashflow_coverage_ratio",
    "ocf_to_total_liabilities",
    "ocf_to_sales",
    "interest_coverage_ratio",
    "icr_under_1",
    "net_margin",
    "operating_margin",
    "roa",
    "debt_ratio",
    "equity_ratio",
    "total_borrowings_ratio",
    "short_term_borrowings_share",
    "capital_impairment_ratio",
    "is_2y_consecutive_operating_loss",
    "is_2y_consecutive_ocf_deficit",
    "financial_statement_missing",
)
_MODEL_PROMPT_KEYS = (
    "prediction_label",
    "probability_speculative",
    "y_proba",
    "threshold",
    "risk_band",
    "stage2_review_trigger",
    "stage2_secondary_trigger",
    "stage2_review_priority",
    "trigger_reason_code",
    "trigger_reason",
    "stage2_probability_margin",
    "stage2_overwarning_filter_candidate",
    "overwarning_filter_reason",
)
_PRIOR_RATING_PROMPT_KEYS = (
    "has_prior_rating",
    "prior_credit_rating",
    "credit_rating",
    "prior_credit_rating_rank",
    "prior_rating_boundary_group",
    "prior_rating_date",
    "prior_rating_age_days",
    "prior_rating_agency",
    "as_of_date",
)
_EVIDENCE_ITEM_PROMPT_KEYS = (
    "source",
    "title",
    "summary",
    "url",
    "published_at",
    "receipt_no",
    "company_match",
    "evidence_quality",
    "evidence_score",
    "provider_relevance",
    "disclosure_severity",
    "disclosure_event_class",
    "disclosure_materiality",
    "materiality_ratio",
    "materiality_basis",
    "dilution_basis",
    "critical_context_confirmed",
    "veto_candidate",
    "critical_terms",
)
_CREDIT_POLICY_PROMPT_KEYS = (
    "policy_version",
    "label_override_allowed",
    "risk_signal_count",
    "mitigating_signal_count",
    "critical_signal_count",
)


@dataclass(frozen=True)
class Stage2InputBundle:
    """Normalized context passed to every Stage 2 agent.

    The bundle keeps Agno/Claude-facing inputs explicit while preserving the
    current deterministic scaffold. Agent implementations should read from this
    object rather than reaching into the full LangGraph state directly.
    """

    company_id: str
    company_name: str
    market: str
    analysis_year: int | None
    company_profile: dict[str, Any]
    model_view: dict[str, Any]
    xgboost_result: dict[str, Any]
    rule_result: dict[str, Any]
    source_feature_row: dict[str, Any]
    peer_comparison_rows: tuple[dict[str, Any], ...]
    news_cache_snapshot: dict[str, Any]
    prior_rating_reference: dict[str, Any] = field(default_factory=dict)
    credit_policy_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def prediction_label(self) -> str:
        """Return the read-only Stage 1 model label used by Stage 2."""
        return str(
            self.xgboost_result.get("prediction_label")
            or self.model_view.get("prediction_label")
            or "unknown"
        )

    @property
    def probability_speculative(self) -> float:
        """Return the speculative-grade probability with a safe default."""
        return _safe_float(
            self.xgboost_result.get("probability_speculative")
            or self.model_view.get("probability_speculative")
            or self.model_view.get("y_proba")
        )

    @property
    def threshold(self) -> float:
        """Return the Stage 1 decision threshold with a safe default."""
        return _safe_float(self.xgboost_result.get("threshold") or self.model_view.get("threshold"))

    @property
    def news_status(self) -> str:
        """Return the external evidence collection status."""
        return str(self.news_cache_snapshot.get("status", "not_implemented"))

    @property
    def peer_rows_by_feature(self) -> dict[str, dict[str, Any]]:
        """Index peer comparison rows by feature name for SHAP explanations."""
        return {
            str(row.get("feature")): row
            for row in self.peer_comparison_rows
            if isinstance(row.get("feature"), str)
        }

    def to_prompt_payload(self) -> dict[str, Any]:
        """Return the full prompt/debug payload preserved for cache compatibility."""
        return {
            "company": {
                "company_id": self.company_id,
                "company_name": self.company_name,
                "market": self.market,
                "analysis_year": self.analysis_year,
                "profile": self.company_profile,
            },
            "model_view": self.model_view,
            "xgboost_result": self.xgboost_result,
            "rule_result": self.rule_result,
            "source_feature_row": self.source_feature_row,
            "peer_comparison_rows": list(self.peer_comparison_rows),
            "news_cache_snapshot": self.news_cache_snapshot,
            "prior_rating_reference": self.prior_rating_reference,
            "credit_policy_snapshot": self.credit_policy_snapshot,
        }

    def to_compact_prompt_payload(self, *, role: PromptPayloadRole = "stage2") -> dict[str, Any]:
        """Return a role-scoped prompt payload for live Agno calls."""
        payload: dict[str, Any] = {
            "prompt_context_version": "stage2_compact_prompt_context_v1",
            "role": role,
            "company": {
                "company_id": self.company_id,
                "company_name": self.company_name,
                "market": self.market,
                "analysis_year": self.analysis_year,
                "profile": _compact_mapping(
                    self.company_profile,
                    keys=("company_id", "company_name", "market", "industry", "sector"),
                ),
            },
            "stage1_model": {
                **_compact_mapping(self.model_view, keys=_MODEL_PROMPT_KEYS),
                **_compact_mapping(self.xgboost_result, keys=_MODEL_PROMPT_KEYS),
                "prediction_label": self.prediction_label,
                "probability_speculative": self.probability_speculative,
                "threshold": self.threshold,
                "top_drivers": _compact_top_drivers(
                    self.xgboost_result.get("top_drivers")
                    or self.model_view.get("top_drivers")
                    or []
                ),
            },
            "prior_rating_reference": _compact_mapping(
                self.prior_rating_reference,
                keys=_PRIOR_RATING_PROMPT_KEYS,
            ),
            "financial_metrics": _compact_mapping(
                self.source_feature_row,
                keys=_FINANCIAL_PROMPT_KEYS,
            ),
            "materiality_summary": _materiality_prompt_summary(
                self.news_cache_snapshot,
                source_feature_row=self.source_feature_row,
            ),
        }
        if role in {"stage2", "quant_credit"}:
            payload["peer_comparison_rows"] = _compact_peer_rows(self.peer_comparison_rows)
        if role in {"stage2", "quant_credit", "chair_report"}:
            payload["credit_policy_snapshot"] = _compact_credit_policy(self.credit_policy_snapshot)
        if role in {"stage2", "evidence_audit", "review_qa", "risk_recall_qa"}:
            payload["news_cache_snapshot"] = _compact_news_cache(
                self.news_cache_snapshot,
                source_feature_row=self.source_feature_row,
            )
        return payload


def build_stage2_input_bundle(state: AgentState) -> Stage2InputBundle:
    """Normalize LangGraph state into the Stage 2 agent input contract."""
    company_profile = _as_dict(state.get("company_profile"))
    source_feature_row = _as_dict(state.get("source_feature_row"))
    prior_rating_reference = _as_dict(state.get("prior_rating_reference"))
    if not prior_rating_reference:
        prior_rating_reference = _as_dict(company_profile.get("prior_rating_reference"))
    if not prior_rating_reference:
        prior_rating_reference = _as_dict(
            _as_dict(state.get("model_view")).get("prior_rating_reference")
        )
    company_id = str(
        state.get("company_id")
        or company_profile.get("company_id")
        or source_feature_row.get("company_id")
        or "unknown"
    )
    company_name = str(
        state.get("company_name")
        or company_profile.get("company_name")
        or source_feature_row.get("company_name")
        or company_id
    )
    market = str(
        state.get("market")
        or source_feature_row.get("market")
        or company_profile.get("market")
        or "UNKNOWN"
    )

    return Stage2InputBundle(
        company_id=company_id,
        company_name=company_name,
        market=market,
        analysis_year=_optional_int(state.get("analysis_year")),
        company_profile=company_profile,
        model_view=_as_dict(state.get("model_view")),
        xgboost_result=_as_dict(state.get("xgboost_result")),
        rule_result=_as_dict(state.get("rule_result")),
        source_feature_row=source_feature_row,
        peer_comparison_rows=_as_dict_tuple(state.get("peer_comparison_rows")),
        news_cache_snapshot=_as_dict(state.get("news_cache_snapshot")),
        prior_rating_reference=prior_rating_reference,
        credit_policy_snapshot=_as_dict(state.get("credit_policy_snapshot")),
    )


def _as_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return cast(dict[str, Any], value.model_dump())
    return {}


def _as_dict_tuple(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _optional_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        if not isinstance(value, int | float | str):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        if not isinstance(value, int | float | str):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_optional_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        if not isinstance(value, int | float | str):
            return None
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _compact_mapping(
    value: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in keys:
        raw_value = value.get(key)
        if raw_value is None or raw_value == "":
            continue
        output[key] = raw_value
    return output


def _compact_top_drivers(value: object, *, limit: int = 8) -> list[dict[str, Any] | str]:
    if not isinstance(value, list | tuple):
        return []
    output: list[dict[str, Any] | str] = []
    for item in value[:limit]:
        if isinstance(item, dict):
            compact = _compact_mapping(
                item,
                keys=(
                    "feature",
                    "name",
                    "label",
                    "value",
                    "raw_value",
                    "shap_value",
                    "impact",
                    "direction",
                    "description",
                    "peer_percentile",
                    "industry_percentile",
                ),
            )
            if compact:
                output.append(compact)
        elif str(item).strip():
            output.append(str(item))
    return output


def _compact_peer_rows(
    rows: tuple[dict[str, Any], ...],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        compact = _compact_mapping(
            row,
            keys=(
                "feature",
                "feature_label",
                "company_value",
                "value",
                "industry_median",
                "market_median",
                "industry_percentile",
                "market_percentile",
                "peer_percentile",
                "direction",
            ),
        )
        if compact:
            output.append(compact)
        if len(output) >= limit:
            break
    return output


def _compact_credit_policy(snapshot: dict[str, Any]) -> dict[str, Any]:
    output = _compact_mapping(snapshot, keys=_CREDIT_POLICY_PROMPT_KEYS)
    signals = snapshot.get("signals")
    if isinstance(signals, list | tuple):
        output["signals"] = [
            _compact_mapping(
                signal,
                keys=("name", "feature", "severity", "direction", "value", "threshold", "summary"),
            )
            for signal in signals[:8]
            if isinstance(signal, dict)
        ]
    return output


def _compact_news_cache(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any],
    limit: int = 12,
) -> dict[str, Any]:
    raw_items = news_cache.get("items", [])
    items = raw_items if isinstance(raw_items, list) else []
    return {
        **_compact_mapping(
            news_cache,
            keys=(
                "status",
                "as_of_date",
                "query",
                "direct_match_count",
                "verified_item_count",
                "weak_evidence_count",
                "veto_candidate_count",
                "high_confidence_critical_count",
                "critical_terms",
            ),
        ),
        "providers": _compact_providers(news_cache.get("providers")),
        "items": [
            _compact_evidence_item(item, source_feature_row=source_feature_row)
            for item in items[:limit]
            if isinstance(item, dict)
        ],
        "omitted_item_count": max(len(items) - limit, 0),
    }


def _compact_providers(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, Any] = {}
    for name, provider in value.items():
        if isinstance(provider, dict):
            output[str(name)] = _compact_mapping(
                provider,
                keys=("status", "items", "error", "start_date", "end_date"),
            )
        else:
            output[str(name)] = provider
    return output


def _compact_evidence_item(
    item: dict[str, Any],
    *,
    source_feature_row: dict[str, Any],
) -> dict[str, Any]:
    from cas.agents.signals.materiality_signals import (
        confirmed_hard_distress_item,
        substantive_external_risk_item,
    )

    compact = _compact_mapping(item, keys=_EVIDENCE_ITEM_PROMPT_KEYS)
    summary = compact.get("summary")
    if isinstance(summary, str) and len(summary) > 360:
        compact["summary"] = f"{summary[:357]}..."
    compact["is_substantive_external_risk"] = substantive_external_risk_item(
        item,
        source_feature_row=source_feature_row,
    )
    compact["has_hard_distress_terms"] = confirmed_hard_distress_item(item)
    return compact


def _materiality_prompt_summary(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any],
) -> dict[str, Any]:
    from cas.agents.signals.materiality_signals import (
        confirmed_hard_distress_item,
        financing_evidence_items,
        has_substantive_external_risk,
        high_risk_financing_evidence_count,
        material_financing_evidence_blocks_tn_hold,
        substantive_external_risk_item,
    )

    raw_items = news_cache.get("items", [])
    items = (
        [item for item in raw_items if isinstance(item, dict)]
        if isinstance(raw_items, list)
        else []
    )
    materiality_items = [
        item for item in items if _safe_optional_float(item.get("materiality_ratio")) is not None
    ]
    top_item = max(
        materiality_items,
        key=lambda item: _safe_optional_float(item.get("materiality_ratio")) or -1.0,
        default={},
    )
    event_classes = sorted(
        {
            str(item.get("disclosure_event_class") or "").strip()
            for item in items
            if str(item.get("disclosure_event_class") or "").strip()
        }
    )
    materiality_classes = sorted(
        {
            str(item.get("disclosure_materiality") or "").strip()
            for item in items
            if str(item.get("disclosure_materiality") or "").strip()
        }
    )
    substantive_count = sum(
        1
        for item in items
        if substantive_external_risk_item(item, source_feature_row=source_feature_row)
    )
    hard_distress_count = sum(1 for item in items if confirmed_hard_distress_item(item))
    return {
        "item_count": len(items),
        "materiality_event_count": len(materiality_items),
        "substantive_external_risk_count": substantive_count,
        "has_substantive_external_risk": has_substantive_external_risk(
            news_cache,
            source_feature_row=source_feature_row,
        ),
        "financing_evidence_count": len(financing_evidence_items(news_cache)),
        "high_risk_financing_evidence_count": high_risk_financing_evidence_count(
            news_cache,
            source_feature_row=source_feature_row,
        ),
        "material_financing_blocks_tn_hold": material_financing_evidence_blocks_tn_hold(
            news_cache,
            source_feature_row=source_feature_row,
        ),
        "hard_distress_item_count": hard_distress_count,
        "max_materiality_ratio": _safe_optional_float(top_item.get("materiality_ratio")),
        "top_materiality_basis": str(top_item.get("materiality_basis") or ""),
        "top_materiality_title": str(top_item.get("title") or ""),
        "event_classes": event_classes[:12],
        "materiality_classes": materiality_classes[:12],
    }


__all__ = ["PromptPayloadRole", "Stage2InputBundle", "build_stage2_input_bundle"]
