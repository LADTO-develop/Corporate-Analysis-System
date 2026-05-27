"""Rolling-validation optimizer for Stage 2 policy thresholds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from cas.agents.nodes import committee_node, rule_engine_node
from cas.agents.stage2_policy import (
    Stage2Policy,
    load_stage2_policy,
    stage2_policy_override,
    stage2_policy_with_updates,
)

ObjectiveName = Literal["review_safe", "strict", "tn_overhold", "fn_rescue"]
OBJECTIVES: tuple[ObjectiveName, ...] = (
    "review_safe",
    "strict",
    "tn_overhold",
    "fn_rescue",
)
RISK_LABELS = {"보류", "부적격"}


@dataclass(frozen=True)
class ThresholdSpec:
    """One tunable Stage 2 policy threshold."""

    path: str
    values: tuple[float | int, ...]
    objectives: tuple[ObjectiveName, ...]
    description: str


@dataclass(frozen=True)
class PolicyOptimizationResult:
    """Optimizer output tables and selected objective winners."""

    candidate_metrics: pd.DataFrame
    selected: dict[str, dict[str, Any]]
    baseline: dict[str, Any]


DEFAULT_SEARCH_SPACE: tuple[ThresholdSpec, ...] = (
    ThresholdSpec(
        path="committee_guardrails.secondary_review.probability_floor_absolute",
        values=(0.24, 0.28, 0.32),
        objectives=("review_safe", "strict", "fn_rescue"),
        description="Absolute probability floor for secondary-review holds.",
    ),
    ThresholdSpec(
        path="committee_guardrails.secondary_review.threshold_buffer",
        values=(0.06, 0.10, 0.14),
        objectives=("review_safe", "strict", "tn_overhold", "fn_rescue"),
        description="Distance below the Stage 1 threshold still allowed to hold.",
    ),
    ThresholdSpec(
        path="committee_guardrails.secondary_review.risk_signal_threshold_buffer",
        values=(0.02, 0.04, 0.08),
        objectives=("review_safe", "strict", "fn_rescue"),
        description="Distance below threshold for risk-signal secondary review.",
    ),
    ThresholdSpec(
        path="committee_guardrails.secondary_review.rule_confidence_floor",
        values=(0.50, 0.60, 0.70),
        objectives=("review_safe", "strict", "fn_rescue"),
        description="Rule confidence floor for liquidity-watch secondary review.",
    ),
    ThresholdSpec(
        path="committee_guardrails.secondary_overhold_supports.min_required_supports",
        values=(1, 2, 3),
        objectives=("review_safe", "tn_overhold"),
        description="Minimum financial defense supports before TN overhold release.",
    ),
    ThresholdSpec(
        path="committee_guardrails.financial_resilience_overwarning.min_support_count",
        values=(6, 8, 10),
        objectives=("review_safe", "tn_overhold"),
        description="Broad support count required for financial-resilience overwarning relief.",
    ),
    ThresholdSpec(
        path="committee_guardrails.financial_resilience_overwarning.max_blocker_count",
        values=(0, 1),
        objectives=("tn_overhold",),
        description="Allowed blocker count for financial-resilience overwarning relief.",
    ),
    ThresholdSpec(
        path="committee_guardrails.risk_hold_financial_stress.min_financial_flags",
        values=(2, 3),
        objectives=("review_safe", "strict", "fn_rescue"),
        description="Financial stress flag count for risk-hold reason retention.",
    ),
    ThresholdSpec(
        path="committee_guardrails.mitigation_residual_risk.probability_floor",
        values=(0.90, 0.92, 0.95),
        objectives=("strict", "fn_rescue"),
        description="Probability floor for turning softened over-warning holds back into risk holds.",
    ),
    ThresholdSpec(
        path="committee_guardrails.severe_financial_watch.current_ratio_floor",
        values=(0.60, 0.70, 0.80),
        objectives=("strict", "fn_rescue"),
        description="Current-ratio floor for severe financial watch.",
    ),
    ThresholdSpec(
        path="committee_guardrails.severe_financial_watch.cash_ratio_floor",
        values=(0.03, 0.05, 0.08),
        objectives=("strict", "fn_rescue"),
        description="Cash-ratio floor for severe financial watch.",
    ),
    ThresholdSpec(
        path="risk_recall_qa.trigger.near_threshold_margin",
        values=(0.06, 0.10, 0.14),
        objectives=("strict", "fn_rescue"),
        description="Near-threshold margin used by RiskRecallQA trigger diagnostics.",
    ),
    ThresholdSpec(
        path="risk_recall_qa.advisory.near_threshold_min_weak_axes",
        values=(1, 2, 3),
        objectives=("strict", "fn_rescue"),
        description="Weak financial-axis count for near-threshold RiskRecallQA escalation.",
    ),
    ThresholdSpec(
        path="risk_recall_qa.advisory.near_threshold_risk_hold_min_weak_axes",
        values=(1, 2, 3),
        objectives=("strict", "fn_rescue"),
        description="Weak financial-axis count for near-threshold risk-label recall guardrail.",
    ),
    ThresholdSpec(
        path="review_qa.advisory.overstated_risk_hold_min_confidence",
        values=(0.50, 0.55, 0.65),
        objectives=("review_safe", "tn_overhold"),
        description="ReviewQA confidence floor for overstated risk-hold downgrade.",
    ),
)


def optimize_policy_thresholds(
    frame: pd.DataFrame,
    *,
    objectives: tuple[ObjectiveName, ...] = OBJECTIVES,
    search_space: tuple[ThresholdSpec, ...] = DEFAULT_SEARCH_SPACE,
    max_iterations: int = 2,
    base_policy: Stage2Policy | None = None,
) -> PolicyOptimizationResult:
    """Tune Stage 2 policy threshold candidates with coordinate search."""
    if frame.empty:
        raise ValueError("Policy optimizer requires at least one rolling-validation row.")
    policy = base_policy or load_stage2_policy()
    evaluation_cache: dict[tuple[tuple[str, float | int], ...], dict[str, Any]] = {}

    def evaluate(updates: dict[str, float | int]) -> dict[str, Any]:
        key = tuple(sorted(updates.items()))
        cached = evaluation_cache.get(key)
        if cached is not None:
            return cached
        candidate = evaluate_policy_candidate(frame, updates=updates, base_policy=policy)
        evaluation_cache[key] = candidate
        return candidate

    baseline = evaluate({})
    selected: dict[str, dict[str, Any]] = {}
    for objective in objectives:
        current_updates: dict[str, float | int] = {}
        best = baseline
        for iteration in range(1, max_iterations + 1):
            iteration_best = best
            iteration_updates = current_updates
            for spec in search_space:
                if objective not in spec.objectives:
                    continue
                for value in _candidate_values(policy, spec):
                    trial_updates = dict(current_updates)
                    trial_updates[spec.path] = value
                    trial = evaluate(trial_updates)
                    if objective_score(trial, objective) > objective_score(
                        iteration_best, objective
                    ):
                        iteration_best = trial | {
                            "objective_iteration": iteration,
                            "objective_last_changed_path": spec.path,
                        }
                        iteration_updates = trial_updates
            if objective_score(iteration_best, objective) <= objective_score(best, objective):
                break
            best = iteration_best
            current_updates = iteration_updates
        selected[objective] = _selected_payload(best, objective)

    candidate_metrics = pd.DataFrame(evaluation_cache.values()).sort_values(
        ["review_safe_score", "strict_score", "tn_overhold_score", "fn_rescue_score"],
        ascending=False,
    )
    return PolicyOptimizationResult(
        candidate_metrics=candidate_metrics.reset_index(drop=True),
        selected=selected,
        baseline=baseline,
    )


def evaluate_policy_candidate(
    frame: pd.DataFrame,
    *,
    updates: dict[str, float | int],
    base_policy: Stage2Policy | None = None,
) -> dict[str, Any]:
    """Replay deterministic committee decisions for one candidate policy."""
    base = base_policy or load_stage2_policy()
    candidate_policy = stage2_policy_with_updates(
        updates,
        base_policy=base,
        policy_version_suffix=_candidate_id(updates),
    )
    with stage2_policy_override(candidate_policy):
        replay = replay_committee_labels(frame)
    predicted = replay["final_committee_label"].astype(str).isin(RISK_LABELS)
    stage1_predicted = _stage1_prediction(frame)
    metrics = policy_metrics_from_predictions(
        frame,
        predicted_risk=predicted,
        stage1_predicted_risk=stage1_predicted,
    )
    updates_json = json.dumps(updates, ensure_ascii=False, sort_keys=True)
    output = {
        "candidate_id": _candidate_id(updates),
        "policy_version": candidate_policy.policy_version,
        "updated_threshold_count": len(updates),
        "threshold_updates_json": updates_json,
        **metrics,
    }
    for objective in OBJECTIVES:
        output[f"{objective}_score"] = objective_score(output, objective)
    return output


def replay_committee_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Run deterministic Stage 2 rule and committee nodes for optimizer rows."""
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        state = _build_optimizer_state(row)
        state.update(rule_engine_node.run(state))
        state.update(committee_node.run(state))
        committee_view = dict(state.get("committee_view") or {})
        rows.append(
            {
                "final_committee_label": str(committee_view.get("final_committee_label") or ""),
                "committee_decision_type": str(committee_view.get("committee_decision_type") or ""),
                "committee_risk_signal": bool(committee_view.get("committee_risk_signal", False)),
            }
        )
    return pd.DataFrame(rows)


def policy_metrics_from_predictions(
    frame: pd.DataFrame,
    *,
    predicted_risk: pd.Series,
    stage1_predicted_risk: pd.Series | None = None,
) -> dict[str, Any]:
    """Return regression metrics for policy candidate predictions."""
    actual = _actual_speculative(frame)
    predicted = predicted_risk.reindex(frame.index, fill_value=False).astype(bool)
    stage1_predicted = (
        stage1_predicted_risk.reindex(frame.index, fill_value=False).astype(bool)
        if stage1_predicted_risk is not None
        else _stage1_prediction(frame)
    )
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    stage1_tp = int((actual & stage1_predicted).sum())
    stage1_fp = int((~actual & stage1_predicted).sum())
    stage1_fn = int((actual & ~stage1_predicted).sum())
    stage1_tn = int((~actual & ~stage1_predicted).sum())
    stage1_fn_mask = actual & ~stage1_predicted
    stage1_fp_mask = ~actual & stage1_predicted
    fn_rescued = int((stage1_fn_mask & predicted).sum())
    fp_softened = int((stage1_fp_mask & ~predicted).sum())
    return {
        "rows": len(frame),
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
        "f1": _f1(tp, fp, fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_count": int(predicted.sum()),
        "review_rate": _safe_divide(int(predicted.sum()), len(frame)),
        "positive_count": int(actual.sum()),
        "negative_count": int((~actual).sum()),
        "fp_rate": _safe_divide(fp, int((~actual).sum())),
        "fn_rate": _safe_divide(fn, int(actual.sum())),
        "tn_rate": _safe_divide(tn, int((~actual).sum())),
        "stage1_precision": _safe_divide(stage1_tp, stage1_tp + stage1_fp),
        "stage1_recall": _safe_divide(stage1_tp, stage1_tp + stage1_fn),
        "stage1_f1": _f1(stage1_tp, stage1_fp, stage1_fn),
        "stage1_tp": stage1_tp,
        "stage1_fp": stage1_fp,
        "stage1_fn": stage1_fn,
        "stage1_tn": stage1_tn,
        "delta_fp_vs_stage1": fp - stage1_fp,
        "delta_fn_vs_stage1": fn - stage1_fn,
        "delta_recall_vs_stage1": _safe_divide(tp, tp + fn)
        - _safe_divide(stage1_tp, stage1_tp + stage1_fn),
        "delta_precision_vs_stage1": _safe_divide(tp, tp + fp)
        - _safe_divide(stage1_tp, stage1_tp + stage1_fp),
        "fn_rescued_count": fn_rescued,
        "fn_rescue_rate": _safe_divide(fn_rescued, stage1_fn),
        "fp_softened_count": fp_softened,
        "fp_softening_rate": _safe_divide(fp_softened, stage1_fp),
        "tn_overhold_count": fp,
        "tn_overhold_rate": _safe_divide(fp, int((~actual).sum())),
    }


def objective_score(metrics: dict[str, Any] | pd.Series, objective: ObjectiveName) -> float:
    """Return scalar score for one policy objective."""
    precision = _metric_float(metrics, "precision")
    recall = _metric_float(metrics, "recall")
    f1 = _metric_float(metrics, "f1")
    fp_rate = _metric_float(metrics, "fp_rate")
    fn_rate = _metric_float(metrics, "fn_rate")
    review_rate = _metric_float(metrics, "review_rate")
    fn_rescue_rate = _metric_float(metrics, "fn_rescue_rate")
    fp_softening_rate = _metric_float(metrics, "fp_softening_rate")
    tn_rate = _metric_float(metrics, "tn_rate")
    delta_fp = _metric_float(metrics, "delta_fp_vs_stage1")
    extra_fp_penalty = max(0.0, delta_fp) / max(1.0, _metric_float(metrics, "negative_count"))
    if objective == "review_safe":
        return (0.45 * f1) + (0.25 * recall) + (0.20 * precision) - (0.10 * review_rate)
    if objective == "strict":
        return (0.70 * recall) + (0.20 * fn_rescue_rate) - (0.10 * fp_rate)
    if objective == "tn_overhold":
        return (0.45 * tn_rate) + (0.25 * precision) + (0.20 * fp_softening_rate) - (0.10 * fn_rate)
    if objective == "fn_rescue":
        return (0.55 * fn_rescue_rate) + (0.30 * recall) - (0.15 * extra_fp_penalty)
    raise ValueError(f"Unknown objective: {objective}")


def selected_policy_overrides(selected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return selected threshold updates grouped by objective."""
    return {
        objective: {
            "candidate_id": payload["candidate_id"],
            "threshold_updates": payload["threshold_updates"],
            "score": payload["score"],
        }
        for objective, payload in selected.items()
    }


def _build_optimizer_state(row: pd.Series) -> dict[str, Any]:
    row_dict = {
        str(key): _clean_scalar(value)
        for key, value in row.to_dict().items()
        if not str(key).endswith("_feature")
    }
    probability = _first_float(row, "prob_speculative", "sample_prob_speculative")
    threshold = _first_float(row, "threshold", "sample_threshold")
    pred_label = _prediction_label(row, probability=probability, threshold=threshold)
    stock_code = str(row.get("stock_code") or "").zfill(6)
    fiscal_year = _safe_int(row.get("fiscal_year")) or 0
    model_view = {
        "model_name": "credit_46_features",
        "model_version": "feature_46_xgboost",
        "probability_speculative": probability,
        "prediction_label": pred_label,
        "risk_band": str(row.get("risk_band") or ""),
        "threshold": threshold,
        "stage2_review_trigger": _as_bool(row.get("stage2_review_trigger")),
        "stage2_secondary_trigger": _as_bool(row.get("stage2_secondary_trigger")),
        "stage2_review_priority": str(row.get("stage2_review_priority") or "none"),
        "trigger_reason": str(row.get("trigger_reason") or ""),
        "stage2_overwarning_filter_candidate": _as_bool(
            row.get("stage2_overwarning_filter_candidate")
        ),
        "overwarning_filter_reason": str(row.get("overwarning_filter_reason") or ""),
        "top_drivers": [],
    }
    return {
        "company_id": f"{row.get('market', '')}-{stock_code}-{fiscal_year}",
        "company_name": str(row.get("corp_name") or ""),
        "market": str(row.get("market") or ""),
        "analysis_year": fiscal_year,
        "company_profile": {
            "company_id": stock_code,
            "company_name": str(row.get("corp_name") or ""),
            "market": str(row.get("market") or ""),
        },
        "source_feature_row": row_dict,
        "normalized_features": row_dict,
        "model_view": model_view,
        "xgboost_result": dict(model_view),
        "news_cache_snapshot": {
            "status": "disabled",
            "enabled": False,
            "items": [],
            "as_of_date": f"{fiscal_year}-12-31" if fiscal_year else "",
            "message": "External evidence is disabled for rolling policy optimization.",
        },
    }


def _candidate_values(policy: Stage2Policy, spec: ThresholdSpec) -> tuple[float | int, ...]:
    current = _policy_scalar(policy, spec.path)
    values = [*spec.values]
    if current is not None and current not in values:
        values.append(current)
    return tuple(dict.fromkeys(values))


def _policy_scalar(policy: Stage2Policy, dot_path: str) -> float | int | None:
    try:
        value = policy.value(*dot_path.split("."))
    except KeyError:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    return None


def _candidate_id(updates: dict[str, float | int]) -> str:
    if not updates:
        return "baseline"
    digest = hashlib.sha1(
        json.dumps(updates, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return f"policy_{digest}"


def _selected_payload(metrics: dict[str, Any], objective: ObjectiveName) -> dict[str, Any]:
    updates = json.loads(str(metrics["threshold_updates_json"]))
    return {
        "objective": objective,
        "candidate_id": metrics["candidate_id"],
        "score": objective_score(metrics, objective),
        "threshold_updates": updates,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn_overhold_rate": metrics["tn_overhold_rate"],
        "fn_rescue_rate": metrics["fn_rescue_rate"],
    }


def _actual_speculative(frame: pd.DataFrame) -> pd.Series:
    if "is_speculative" not in frame.columns:
        raise ValueError("Policy optimizer input requires an is_speculative column.")
    return pd.to_numeric(frame["is_speculative"], errors="coerce").fillna(0).astype(int).eq(1)


def _stage1_prediction(frame: pd.DataFrame) -> pd.Series:
    if "pred_label_tuned" in frame.columns:
        return pd.to_numeric(frame["pred_label_tuned"], errors="coerce").fillna(0).astype(int).eq(1)
    if {"prob_speculative", "threshold"}.issubset(frame.columns):
        probability = pd.to_numeric(frame["prob_speculative"], errors="coerce")
        threshold = pd.to_numeric(frame["threshold"], errors="coerce")
        return probability.ge(threshold)
    if "model_predicted_label_name" in frame.columns:
        return frame["model_predicted_label_name"].astype(str).isin({"투기등급", "부적격"})
    raise ValueError("Policy optimizer input requires pred_label_tuned or probability columns.")


def _prediction_label(row: pd.Series, *, probability: float, threshold: float) -> str:
    if "pred_label_tuned" in row.index:
        return "부적격" if _safe_int(row.get("pred_label_tuned")) == 1 else "투자적격"
    if "model_predicted_label_name" in row.index:
        label = str(row.get("model_predicted_label_name") or "")
        if label in {"투기등급", "부적격"}:
            return "부적격"
        if label in {"투자적격", "적격"}:
            return "투자적격"
    return "부적격" if probability >= threshold else "투자적격"


def _first_float(row: pd.Series, *columns: str) -> float:
    for column in columns:
        if column in row.index:
            value = _safe_float(row.get(column))
            if value is not None:
                return value
    return 0.0


def _metric_float(metrics: dict[str, Any] | pd.Series, key: str) -> float:
    value = metrics[key] if isinstance(metrics, pd.Series) else metrics.get(key)
    return float(value or 0.0)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float | str):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _safe_int(value: object) -> int | None:
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "보류", "부적격"}


def _clean_scalar(value: object) -> object:
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


__all__ = [
    "DEFAULT_SEARCH_SPACE",
    "OBJECTIVES",
    "ObjectiveName",
    "PolicyOptimizationResult",
    "ThresholdSpec",
    "evaluate_policy_candidate",
    "objective_score",
    "optimize_policy_thresholds",
    "policy_metrics_from_predictions",
    "replay_committee_labels",
    "selected_policy_overrides",
]
