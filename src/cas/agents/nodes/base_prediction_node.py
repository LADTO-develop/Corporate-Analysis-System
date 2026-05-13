"""Run Stage 1 quantitative prediction using the saved XGBoost artifact."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from cas.agents.state import AgentState, AuditEntry, BaseAssessment, ModelResult, RiskBand
from cas.utils.io import read_json, read_yaml

if TYPE_CHECKING:
    from xgboost import Booster

_MODEL_ARTIFACT_DIR = Path("data/outputs/modeling/feature_43_xgboost")
_MODEL_ARTIFACT_PATH = _MODEL_ARTIFACT_DIR / "xgboost_model.json"
_MODEL_METADATA_PATH = _MODEL_ARTIFACT_DIR / "model_artifact_metadata.json"
_FEATURE_LIST_PATH = Path("data/input/credit_43_features/feature_43_list.json")


def run(state: AgentState) -> dict[str, Any]:
    """Create the Stage 1 model_view from the saved XGBoost artifact."""
    model_features = dict(state.get("model_features") or {})
    normalized_features = dict(state.get("normalized_features") or {})

    if not model_features:
        # feature_node가 정형 입력 벡터를 만들지 못한 경우에는
        # 예전 deterministic scoring 경로로 내려가 파이프라인을 끊지 않는다.
        return _run_fallback_prediction(state, normalized_features)

    cfg = read_yaml("configs/runtime/analysis.yaml")
    lens_scores = _lens_scores(normalized_features, cfg)
    overall_score = _weighted_score(
        {name: assessment.score for name, assessment in lens_scores.items()},
        cfg["overall_weights"],
    )

    try:
        # CI/로컬 모두 같은 Stage 1 결과를 쓰도록 repo에 추적되는 model artifact를 우선 로드한다.
        bundle = _load_model_bundle()
    except Exception as error:
        if not _is_missing_model_artifact_error(error):
            raise
        return _run_fallback_prediction(
            state,
            normalized_features,
            fallback_reason=(
                "Saved XGBoost artifact was not found; "
                "falling back to deterministic Stage 1 scoring."
            ),
        )
    import xgboost as xgb

    frame = _build_model_frame(model_features, bundle["feature_columns"], bundle["fill_values"])
    model = bundle["model"]
    probability_speculative = round(float(model.predict(xgb.DMatrix(frame))[0]), 4)

    model_registry = dict(cfg.get("model_registry", {}))
    threshold = float(bundle.get("threshold_default", model_registry.get("threshold", 0.5)))
    watch_threshold = float(model_registry.get("watch_threshold", 0.4))
    high_risk_threshold = float(model_registry.get("high_risk_threshold", 0.65))
    prediction_label = "부적격" if probability_speculative >= threshold else "투자적격"
    risk_band = _risk_band(
        probability_speculative,
        watch_threshold=watch_threshold,
        high_risk_threshold=high_risk_threshold,
    )
    top_drivers = _top_risk_drivers(model, frame)

    xgboost_result = ModelResult(
        model_name=str(bundle.get("dataset_name", "feature_43_xgboost")),
        model_version=str(model_registry.get("model_version", "feature_43_xgboost")),
        probability_speculative=probability_speculative,
        prediction_label=prediction_label,
        risk_band=risk_band,
        threshold=threshold,
        top_drivers=top_drivers,
    )
    # model_view는 이후 에이전트들이 "해석만" 하도록 남겨 두는 원본 정량 판단이다.
    model_view = {
        "probability_speculative": probability_speculative,
        "prediction_label": prediction_label,
        "risk_band": risk_band,
        "threshold": threshold,
        "top_drivers": [{"name": name, "value": value} for name, value in top_drivers],
    }
    audit = AuditEntry(
        node="xgboost_inference",
        timestamp=_now(),
        summary=(
            "Stage 1 XGBoost inference completed: "
            f"probability_speculative={probability_speculative:.3f}, "
            f"prediction_label={prediction_label}, risk_band={risk_band}"
        ),
        metrics={
            "probability_speculative": probability_speculative,
            "overall_score": overall_score,
            "n_top_drivers": float(len(top_drivers)),
        },
    )
    return {
        "base_assessments": lens_scores,
        "overall_score": overall_score,
        # model_view는 화면/에이전트가 공통으로 읽는 가벼운 표현이고,
        # xgboost_result는 schema/export 쪽에서 쓰는 구조화 결과다.
        "model_view": model_view,
        "xgboost_result": xgboost_result.model_dump(),
        # model_registry_ref는 "어떤 artifact와 threshold로 이 판단이 나왔는지"를 남기는 추적 정보다.
        "model_registry_ref": {
            "registry_name": model_registry.get("registry_name", "local_model_registry"),
            "active_model": xgboost_result.model_name,
            "model_version": xgboost_result.model_version,
            "threshold": threshold,
            "watch_threshold": watch_threshold,
            "high_risk_threshold": high_risk_threshold,
            "artifact_path": str(_MODEL_ARTIFACT_PATH),
        },
        "audit": [audit],
    }


def _run_fallback_prediction(
    state: AgentState,
    normalized_features: dict[str, float],
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Preserve the legacy deterministic facade when model features are unavailable."""
    cfg = read_yaml("configs/runtime/analysis.yaml")
    if not normalized_features:
        audit = AuditEntry(
            node="xgboost_inference",
            timestamp=_now(),
            summary="No feature-store snapshot available; skipping realtime inference.",
        )
        return {"audit": [audit]}

    lens_scores = _lens_scores(normalized_features, cfg)
    overall_score = _weighted_score(
        {name: assessment.score for name, assessment in lens_scores.items()},
        cfg["overall_weights"],
    )
    model_registry = dict(cfg.get("model_registry", {}))
    threshold = float(model_registry.get("threshold", 0.5))
    watch_threshold = float(model_registry.get("watch_threshold", 0.4))
    high_risk_threshold = float(model_registry.get("high_risk_threshold", 0.65))
    probability_speculative = round(1.0 - overall_score, 4)
    prediction_label = "부적격" if probability_speculative >= threshold else "투자적격"
    risk_band = _risk_band(
        probability_speculative,
        watch_threshold=watch_threshold,
        high_risk_threshold=high_risk_threshold,
    )
    # artifact가 없거나 최소 기능만 필요한 환경에서도 파이프라인 전체는 끝까지 돌도록 둔다.
    top_drivers = _top_risk_drivers_from_scores(normalized_features)
    fallback_model_name = (
        "credit_43_features"
        if state.get("source_feature_row")
        else str(model_registry.get("active_model", "xgboost_realtime"))
    )
    xgboost_result = ModelResult(
        model_name=fallback_model_name,
        model_version=str(model_registry.get("model_version", "local-deterministic")),
        probability_speculative=probability_speculative,
        prediction_label=prediction_label,
        risk_band=risk_band,
        threshold=threshold,
        top_drivers=top_drivers,
    )
    audit = AuditEntry(
        node="xgboost_inference",
        timestamp=_now(),
        summary=(
            (f"{fallback_reason} " if fallback_reason else "")
            + "Fallback Stage 1 inference completed from normalized snapshot: "
            f"probability_speculative={probability_speculative:.3f}, "
            f"risk_band={risk_band}"
        ),
        metrics={f"score_{k}": v.score for k, v in lens_scores.items()}
        | {
            "overall_score": overall_score,
            "probability_speculative": probability_speculative,
        },
    )
    return {
        "base_assessments": lens_scores,
        "overall_score": overall_score,
        # fallback도 반환 shape는 Stage 1 정상 경로와 맞춰 둬야
        # downstream node가 분기 없이 동일한 state key를 읽을 수 있다.
        "model_view": {
            "probability_speculative": probability_speculative,
            "prediction_label": prediction_label,
            "risk_band": risk_band,
            "threshold": threshold,
            "top_drivers": [{"name": name, "value": value} for name, value in top_drivers],
        },
        "xgboost_result": xgboost_result.model_dump(),
        "model_registry_ref": {
            "registry_name": model_registry.get("registry_name", "local_model_registry"),
            "active_model": xgboost_result.model_name,
            "model_version": xgboost_result.model_version,
            "threshold": threshold,
            "watch_threshold": watch_threshold,
            "high_risk_threshold": high_risk_threshold,
        },
        "audit": [audit],
    }


def _is_missing_model_artifact_error(error: Exception) -> bool:
    if isinstance(error, ModuleNotFoundError) and error.name == "xgboost":
        return True
    if isinstance(error, FileNotFoundError):
        return True
    return error.__class__.__name__ == "XGBoostError" and "No such file or directory" in str(error)


def _lens_scores(
    normalized_features: dict[str, float],
    cfg: dict[str, Any],
) -> dict[str, BaseAssessment]:
    lens_scores: dict[str, BaseAssessment] = {}
    for lens_name, weights in cfg["lenses"].items():
        score = _weighted_score(normalized_features, weights)
        drivers = sorted(
            ((metric, float(normalized_features.get(metric, 0.5))) for metric in weights),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        lens_scores[lens_name] = BaseAssessment(
            lens_name=lens_name,
            score=score,
            summary=_lens_summary(lens_name, score),
            drivers=drivers,
        )
    return lens_scores


@lru_cache(maxsize=1)
def _load_model_bundle() -> dict[str, Any]:
    import xgboost as xgb

    metadata = cast(dict[str, Any], read_json(_MODEL_METADATA_PATH))
    booster = xgb.Booster()
    booster.load_model(_MODEL_ARTIFACT_PATH)
    # json artifact와 metadata를 한 번에 묶어 두면
    # 추론 시 feature column, 결측치 대치값, threshold를 같은 버전 기준으로 재사용할 수 있다.
    return {
        "dataset_name": metadata.get("dataset_name", "credit_43_features"),
        "model_type": metadata.get("model_type", "xgboost_booster_json"),
        "feature_columns": list(metadata.get("feature_columns", [])),
        "source_features": list(metadata.get("source_features", [])),
        "fill_values": dict(metadata.get("fill_values", {})),
        "threshold_default": metadata.get("threshold_default", 0.5),
        "threshold_tuned": metadata.get("threshold_tuned", 0.5),
        "model": booster,
    }


def _build_model_frame(
    model_features: dict[str, float],
    feature_columns: list[str],
    fill_values: dict[str, float],
) -> pd.DataFrame:
    row = {
        column: _to_float(model_features.get(column, fill_values.get(column, 0.0)))
        for column in feature_columns
    }
    return pd.DataFrame([row], columns=feature_columns)


def _top_risk_drivers(
    model: Booster,
    frame: pd.DataFrame,
) -> list[tuple[str, float]]:
    import xgboost as xgb

    feature_spec = read_json(_FEATURE_LIST_PATH)
    contribs = model.predict(xgb.DMatrix(frame), pred_contribs=True)
    contributions = contribs[0][:-1]
    feature_columns = list(frame.columns)
    feature_index = {name: idx for idx, name in enumerate(feature_columns)}

    grouped_scores: list[tuple[str, float]] = []
    for source_feature in feature_spec["selected_source_features"]:
        model_columns = _model_columns_for_source(feature_spec, str(source_feature))
        score = sum(
            contributions[feature_index[column]]
            for column in model_columns
            if column in feature_index
        )
        grouped_scores.append((str(source_feature), round(float(score), 4)))
    return sorted(grouped_scores, key=lambda item: abs(item[1]), reverse=True)[:5]


def _model_columns_for_source(feature_spec: dict[str, Any], source_feature: str) -> list[str]:
    for item in feature_spec.get("feature_metadata", []):
        if str(item.get("source_feature")) == source_feature:
            return [str(column) for column in item.get("model_features", [])]
    return [source_feature]


def _weighted_score(values: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(float(weight) for weight in weights.values())
    if total_weight <= 0:
        return 0.0
    total = 0.0
    for key, weight in weights.items():
        total += float(values.get(key, 0.5)) * float(weight)
    return round(total / total_weight, 4)


def _lens_summary(lens_name: str, score: float) -> str:
    if score >= 0.75:
        return f"{lens_name} is a clear strength."
    if score >= 0.55:
        return f"{lens_name} is acceptable with room to improve."
    return f"{lens_name} needs closer review."


def _risk_band(
    probability_speculative: float,
    *,
    watch_threshold: float,
    high_risk_threshold: float,
) -> RiskBand:
    if probability_speculative >= high_risk_threshold:
        return "high_risk"
    if probability_speculative >= watch_threshold:
        return "watch"
    return "stable"


def _top_risk_drivers_from_scores(features: dict[str, float]) -> list[tuple[str, float]]:
    driver_scores = [
        (name, round(1.0 - float(value), 4))
        for name, value in features.items()
        if name != "controversy_penalty"
    ]
    driver_scores.append(
        (
            "controversy_penalty",
            round(1.0 - float(features.get("controversy_penalty", 0.5)), 4),
        )
    )
    return sorted(driver_scores, key=lambda item: item[1], reverse=True)[:5]


def _to_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        if not isinstance(value, int | float | str):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
