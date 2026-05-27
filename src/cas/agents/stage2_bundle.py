"""Build provider-neutral input bundles for Stage 2 agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

from cas.agents.stage2_policy import load_stage2_policy
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
    "event_id",
    "source",
    "source_evidence_type",
    "source_reliability",
    "title",
    "summary",
    "url",
    "published_at",
    "receipt_no",
    "rcept_no",
    "company_match",
    "company_match_type",
    "company_disambiguation",
    "temporal_status",
    "as_of_date_violation",
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
    "verification_flags",
    "duplicate_count",
    "duplicate_sources",
)
_CREDIT_POLICY_PROMPT_KEYS = (
    "policy_version",
    "label_override_allowed",
    "risk_signal_count",
    "mitigating_signal_count",
    "critical_signal_count",
)
_SPECULATIVE_PRIOR_RATINGS = frozenset(
    {"BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"}
)
_HARD_DISTRESS_PRIOR_RATINGS = frozenset({"CCC+", "CCC", "CCC-", "CC", "C", "D"})


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

    @property
    def normalized_signal_summary(self) -> dict[str, Any]:
        """Return precomputed Stage 2 signals so LLMs do not reinterpret raw values."""
        return _normalized_signal_summary(
            model_view=self.model_view,
            xgboost_result=self.xgboost_result,
            source_feature_row=self.source_feature_row,
            news_cache_snapshot=self.news_cache_snapshot,
            prior_rating_reference=self.prior_rating_reference,
            probability_speculative=self.probability_speculative,
            threshold=self.threshold,
            prediction_label=self.prediction_label,
        )

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
            "normalized_signal_summary": self.normalized_signal_summary,
        }

    def to_compact_prompt_payload(self, *, role: PromptPayloadRole = "stage2") -> dict[str, Any]:
        """Return a role-scoped prompt payload for live Agno calls."""
        normalized_signal_summary = self.normalized_signal_summary
        payload: dict[str, Any] = {
            "prompt_context_version": "stage2_compact_prompt_context_v2",
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
            "materiality_summary": normalized_signal_summary["materiality_profile"],
            "normalized_signal_summary": normalized_signal_summary,
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


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t", "on", "참", "예"}
    return False


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


def _first_present(
    *sources: dict[str, Any],
    key: str,
    default: object = None,
) -> object:
    for source in sources:
        value = source.get(key)
        if value is not None and value != "":
            return value
    return default


def _metric_value(row: dict[str, Any], key: str) -> float | None:
    return _safe_optional_float(row.get(key))


def _metric_below_value(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _metric_value(row, key)
    return value is not None and value < threshold


def _metric_above_value(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _metric_value(row, key)
    return value is not None and value > threshold


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


def _normalized_signal_summary(
    *,
    model_view: dict[str, Any],
    xgboost_result: dict[str, Any],
    source_feature_row: dict[str, Any],
    news_cache_snapshot: dict[str, Any],
    prior_rating_reference: dict[str, Any],
    probability_speculative: float,
    threshold: float,
    prediction_label: str,
) -> dict[str, Any]:
    materiality_profile = _materiality_prompt_summary(
        news_cache_snapshot,
        source_feature_row=source_feature_row,
    )
    evidence_treatment = _evidence_treatment_summary(
        news_cache_snapshot,
        source_feature_row=source_feature_row,
        materiality_profile=materiality_profile,
    )
    weak_axes = _weak_financial_axes(source_feature_row)
    return {
        "summary_version": "stage2_normalized_signals_v1",
        "weak_financial_axes": weak_axes,
        "weak_financial_axis_count": len(weak_axes),
        "materiality_profile": materiality_profile,
        "evidence_treatment": evidence_treatment,
        "boundary_context": _boundary_context(prior_rating_reference),
        "secondary_trigger_profile": _secondary_trigger_profile(
            model_view=model_view,
            xgboost_result=xgboost_result,
            probability_speculative=probability_speculative,
            threshold=threshold,
            prediction_label=prediction_label,
        ),
    }


def _weak_financial_axes(row: dict[str, Any]) -> list[str]:
    policy = load_stage2_policy()
    section = ("risk_recall_qa", "trigger")
    axes: list[str] = []
    if _metric_below_value(row, "current_ratio", policy.float(*section, "current_ratio_floor")):
        axes.append("low_current_ratio")
    if _metric_below_value(row, "cash_ratio", policy.float(*section, "cash_ratio_floor")):
        axes.append("low_cash_ratio")
    if (
        _metric_below_value(
            row,
            "cashflow_coverage_ratio",
            policy.float(*section, "cashflow_coverage_ratio_floor"),
        )
        or _metric_below_value(
            row,
            "ocf_to_total_liabilities",
            policy.float(*section, "ocf_to_total_liabilities_floor"),
        )
        or _metric_below_value(row, "ocf_to_sales", policy.float(*section, "ocf_to_sales_floor"))
        or _truthy(row.get("is_2y_consecutive_ocf_deficit"))
    ):
        axes.append("weak_cashflow")
    if _metric_below_value(
        row,
        "interest_coverage_ratio",
        policy.float(*section, "interest_coverage_ratio_floor"),
    ) or _truthy(row.get("icr_under_1")):
        axes.append("weak_interest_coverage")
    if _metric_above_value(row, "debt_ratio", policy.float(*section, "debt_ratio_floor")):
        axes.append("high_debt_ratio")
    if _metric_above_value(
        row,
        "total_borrowings_ratio",
        policy.float(*section, "total_borrowings_ratio_floor"),
    ) or _metric_above_value(
        row,
        "short_term_borrowings_share",
        policy.float(*section, "short_term_borrowings_share_floor"),
    ):
        axes.append("high_borrowing_pressure")
    return axes


def _evidence_treatment_summary(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any],
    materiality_profile: dict[str, Any],
) -> dict[str, Any]:
    from cas.agents.signals.evidence_treatment_signals import evaluate_evidence_treatment

    return cast(
        dict[str, Any],
        evaluate_evidence_treatment(
            news_cache,
            source_feature_row=source_feature_row,
            materiality_summary=materiality_profile,
        ).as_payload(),
    )


def _boundary_context(prior_rating_reference: dict[str, Any]) -> dict[str, Any]:
    policy = load_stage2_policy()
    rating = (
        str(
            prior_rating_reference.get("prior_credit_rating")
            or prior_rating_reference.get("credit_rating")
            or ""
        )
        .strip()
        .upper()
    )
    group = str(prior_rating_reference.get("prior_rating_boundary_group") or "").strip()
    group_normalized = group.lower()
    rank = _optional_int(prior_rating_reference.get("prior_credit_rating_rank"))
    speculative_min_rank = policy.int(
        "committee_guardrails", "prior_rating", "speculative_min_rank"
    )
    hard_distress_min_rank = policy.int(
        "committee_guardrails", "prior_rating", "hard_distress_min_rank"
    )
    stable_investment_max_rank = policy.int(
        "committee_guardrails",
        "prior_rating",
        "stable_investment_max_rank",
    )
    has_prior_rating = _truthy(prior_rating_reference.get("has_prior_rating"))
    exact_boundary = group == "exact_bbb_minus_bb_plus_boundary" or rating in {"BBB-", "BB+"}
    has_boundary_context = "boundary" in group_normalized or exact_boundary
    speculative_prior = bool(
        has_prior_rating
        and (
            (rank is not None and rank >= speculative_min_rank)
            or rating in _SPECULATIVE_PRIOR_RATINGS
        )
    )
    hard_distress_prior = bool(
        has_prior_rating
        and (
            (rank is not None and rank >= hard_distress_min_rank)
            or rating in _HARD_DISTRESS_PRIOR_RATINGS
        )
    )
    stable_investment_non_boundary = bool(
        has_prior_rating
        and group == "investment_grade_non_boundary"
        and (
            (rank is not None and rank <= stable_investment_max_rank)
            or rating in {"AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+"}
        )
    )
    return {
        "has_prior_rating": has_prior_rating,
        "prior_credit_rating": rating,
        "prior_credit_rating_rank": rank,
        "prior_rating_boundary_group": group,
        "has_rating_boundary_context": has_boundary_context,
        "is_exact_bbb_minus_bb_plus_boundary": exact_boundary,
        "is_speculative_prior_rating": speculative_prior,
        "has_prior_hard_distress_context": hard_distress_prior,
        "is_stable_investment_non_boundary": stable_investment_non_boundary,
        "prior_rating_age_days": _optional_int(prior_rating_reference.get("prior_rating_age_days")),
        "prior_rating_agency": str(prior_rating_reference.get("prior_rating_agency") or ""),
        "prior_rating_date": str(prior_rating_reference.get("prior_rating_date") or ""),
    }


def _secondary_trigger_profile(
    *,
    model_view: dict[str, Any],
    xgboost_result: dict[str, Any],
    probability_speculative: float,
    threshold: float,
    prediction_label: str,
) -> dict[str, Any]:
    policy = load_stage2_policy()
    near_threshold_margin = policy.float("risk_recall_qa", "trigger", "near_threshold_margin")
    margin_to_threshold = round(threshold - probability_speculative, 6)
    absolute_margin = round(abs(margin_to_threshold), 6)
    review_trigger = _truthy(
        _first_present(xgboost_result, model_view, key="stage2_review_trigger", default=False)
    )
    secondary_trigger = _truthy(
        _first_present(xgboost_result, model_view, key="stage2_secondary_trigger", default=False)
    )
    return {
        "stage1_risk_trigger": prediction_label == "부적격",
        "stage2_review_trigger": review_trigger,
        "stage2_secondary_trigger": secondary_trigger,
        "stage2_review_priority": str(
            _first_present(xgboost_result, model_view, key="stage2_review_priority", default="none")
            or "none"
        ),
        "trigger_reason_code": str(
            _first_present(xgboost_result, model_view, key="trigger_reason_code", default="") or ""
        ),
        "trigger_reason": str(
            _first_present(xgboost_result, model_view, key="trigger_reason", default="") or ""
        ),
        "risk_band": str(
            _first_present(xgboost_result, model_view, key="risk_band", default="") or ""
        ),
        "stage2_overwarning_filter_candidate": _truthy(
            _first_present(
                xgboost_result,
                model_view,
                key="stage2_overwarning_filter_candidate",
                default=False,
            )
        ),
        "overwarning_filter_reason": str(
            _first_present(xgboost_result, model_view, key="overwarning_filter_reason", default="")
            or ""
        ),
        "probability_speculative": round(probability_speculative, 6),
        "threshold": round(threshold, 6),
        "margin_to_threshold": margin_to_threshold,
        "absolute_margin_to_threshold": absolute_margin,
        "near_threshold_margin_policy": near_threshold_margin,
        "eligible_near_threshold": bool(
            prediction_label == "투자적격"
            and threshold > 0
            and 0.0 <= margin_to_threshold <= near_threshold_margin
        ),
        "absolute_near_threshold": bool(threshold > 0 and absolute_margin <= near_threshold_margin),
    }


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
                "verification_summary",
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
