from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from export_feature_43_dashboard_artifacts import (
    build_company_latest,
    build_global_shap_reference,
    build_industry_latest_summary,
    build_industry_shap_summary,
    build_industry_year_summary,
    build_peer_percentiles,
    risk_band,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input" / "credit_43_features"
INFERENCE_PATH = INPUT_DIR / "feature_43_inference_2026.csv"
FEATURE_LIST_PATH = INPUT_DIR / "feature_43_list.json"
MODEL_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_43_xgboost"
MODEL_PATH = MODEL_DIR / "xgboost_model.json"
MODEL_METADATA_PATH = MODEL_DIR / "model_artifact_metadata.json"
SOURCE_DASHBOARD_DIR = ROOT / "data" / "outputs" / "dashboard" / "feature_43_mvp"
OUTPUT_DIR = ROOT / "data" / "outputs" / "dashboard" / "feature_43_inference_2026"
VALIDATION_SCORES_PATH = MODEL_DIR / "diagnostics" / "external_validation_2026_scores.csv"
TOP_K_SHAP = 10


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def apply_platt_calibration(
    raw_probabilities: np.ndarray, calibration: dict[str, Any]
) -> np.ndarray:
    epsilon = float(calibration.get("clip_epsilon", 1e-6))
    clipped = np.clip(raw_probabilities, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped))
    calibrated_logits = float(calibration["coef"]) * logits + float(calibration["intercept"])
    return 1.0 / (1.0 + np.exp(-calibrated_logits))


def source_feature_mapping(model_features: list[str], source_features: list[str]) -> dict[str, str]:
    """Map one-hot model columns back to dashboard-facing source features."""
    mapping: dict[str, str] = {}
    source_set = set(source_features)
    for feature in model_features:
        if feature in source_set:
            mapping[feature] = feature
        elif feature.startswith("market_"):
            mapping[feature] = "market"
        elif feature.startswith("firm_size_group_"):
            mapping[feature] = "firm_size_group"
        elif feature.startswith("industry_macro_category_"):
            mapping[feature] = "industry_macro_category"
        else:
            mapping[feature] = feature
    return mapping


def build_prediction_scores(
    inference: pd.DataFrame,
    raw_probabilities: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    calibration_method: str,
) -> pd.DataFrame:
    id_columns = [
        "market",
        "stock_code",
        "corp_name",
        "fiscal_year",
        "eval_year",
        "firm_size_group",
        "industry_macro_category",
    ]
    scored = inference.loc[:, id_columns].copy()
    scored["label_eval_year"] = scored["eval_year"]
    scored["split"] = "inference_2026"
    scored["is_speculative"] = np.nan
    scored["prob_speculative_raw"] = raw_probabilities
    scored["probability_calibration_method"] = calibration_method
    scored["prob_speculative"] = probabilities
    scored["pred_label_0_5"] = (scored["prob_speculative"] >= 0.5).astype(int)
    scored["pred_label_tuned"] = (scored["prob_speculative"] >= threshold).astype(int)
    scored["predicted_label"] = scored["pred_label_tuned"]
    scored["threshold_default"] = 0.5
    scored["threshold_tuned"] = threshold
    scored["threshold"] = threshold
    scored["risk_band"] = scored["prob_speculative"].map(risk_band)

    stage1_risk = scored["pred_label_tuned"].astype(int).eq(1)
    near_threshold = (~stage1_risk) & scored["prob_speculative"].ge(max(threshold - 0.08, 0.0))
    scored["prob_speculative_45"] = np.nan
    scored["prob_speculative_45_raw"] = np.nan
    scored["probability_45_calibration_method"] = "not_available_for_inference_2026"
    scored["threshold_45"] = np.nan
    scored["threshold_45_it_services_review"] = np.nan
    scored["pred_label_45_tuned"] = np.nan
    scored["stage2_review_trigger"] = stage1_risk | near_threshold
    scored["stage2_secondary_trigger"] = near_threshold
    scored["stage2_review_priority"] = np.select(
        [stage1_risk, near_threshold],
        ["high", "watch"],
        default="none",
    )
    scored["trigger_reason_code"] = np.select(
        [stage1_risk, near_threshold],
        ["stage1_model_risk", "near_stage1_threshold"],
        default="none",
    )
    scored["trigger_reason"] = np.select(
        [stage1_risk, near_threshold],
        [
            "1차 모델이 위험 기준선을 넘어 위원회 검토 대상으로 분류했습니다.",
            "1차 모델 기준선에 가까워 위원회가 한 번 더 확인할 대상으로 분류했습니다.",
        ],
        default="추가 위원회 검토 트리거 없음",
    )

    scored["prob_speculative_overwarning_filter"] = np.nan
    scored["prob_speculative_overwarning_filter_raw"] = np.nan
    scored["probability_overwarning_filter_calibration_method"] = "not_available_for_inference_2026"
    scored["threshold_overwarning_filter"] = np.nan
    scored["pred_label_overwarning_filter_tuned"] = np.nan
    scored["stage2_overwarning_filter_candidate"] = False
    scored["overwarning_filter_reason_code"] = "none"
    scored["overwarning_filter_reason"] = (
        "과민 경고 보조필터는 2026 추론 산출물에서 별도 계산하지 않았습니다."
    )
    scored["overwarning_filter_policy"] = "not_available_for_inference_2026"
    return scored


def build_local_shap(
    inference: pd.DataFrame,
    prediction_scores: pd.DataFrame,
    shap_contribs: np.ndarray,
    model_features: list[str],
    source_features: list[str],
) -> pd.DataFrame:
    mapping = source_feature_mapping(model_features, source_features)
    grouped_indices: dict[str, list[int]] = {feature: [] for feature in source_features}
    for index, feature in enumerate(model_features):
        grouped = mapping.get(feature, feature)
        grouped_indices.setdefault(grouped, []).append(index)

    grouped_shap = np.zeros((len(inference), len(source_features)), dtype=float)
    for feature_index, feature in enumerate(source_features):
        indices = grouped_indices.get(feature, [])
        if indices:
            grouped_shap[:, feature_index] = shap_contribs[:, indices].sum(axis=1)

    rows: list[dict[str, Any]] = []
    for row_index, row_values in enumerate(grouped_shap):
        source_row = inference.iloc[row_index]
        score_row = prediction_scores.iloc[row_index]
        top_indices = np.argsort(np.abs(row_values))[::-1][:TOP_K_SHAP]
        for rank, feature_index in enumerate(top_indices, start=1):
            feature = source_features[feature_index]
            shap_value = float(row_values[feature_index])
            rows.append(
                {
                    "market": source_row["market"],
                    "stock_code": source_row["stock_code"],
                    "corp_name": source_row["corp_name"],
                    "fiscal_year": source_row["fiscal_year"],
                    "eval_year": source_row["eval_year"],
                    "industry_macro_category": source_row["industry_macro_category"],
                    "firm_size_group": source_row["firm_size_group"],
                    "split": "inference_2026",
                    "is_speculative": np.nan,
                    "prob_speculative": float(score_row["prob_speculative"]),
                    "feature": feature,
                    "rank": rank,
                    "shap_value": shap_value,
                    "abs_shap": abs(shap_value),
                    "direction": "increase_risk" if shap_value > 0 else "decrease_risk",
                    "feature_value": source_row.get(feature),
                }
            )
    return pd.DataFrame(rows)


def attach_validation_recommendation_flags(prediction_scores: pd.DataFrame) -> pd.DataFrame:
    if not VALIDATION_SCORES_PATH.exists():
        prediction_scores["external_validation_stage2_effect"] = ""
        prediction_scores["external_validation_actual_label"] = ""
        prediction_scores["external_validation_credit_rating"] = ""
        prediction_scores["landing_recommendation_bucket"] = ""
        return prediction_scores

    validation = pd.read_csv(
        VALIDATION_SCORES_PATH,
        encoding="utf-8-sig",
        dtype={"stock_code": str},
    )
    validation["stock_code"] = validation["stock_code"].astype(str).str.zfill(6)
    keep_columns = [
        "stock_code",
        "fiscal_year",
        "stage2_effect",
        "actual_label",
        "credit_rating",
        "stage1_error_type",
        "final_committee_label",
    ]
    output = prediction_scores.copy()
    output["stock_code"] = output["stock_code"].astype(str).str.zfill(6)
    output = output.merge(
        validation.loc[:, keep_columns],
        on=["stock_code", "fiscal_year"],
        how="left",
    )
    output = output.rename(
        columns={
            "stage2_effect": "external_validation_stage2_effect",
            "actual_label": "external_validation_actual_label",
            "credit_rating": "external_validation_credit_rating",
            "stage1_error_type": "external_validation_stage1_error_type",
            "final_committee_label": "external_validation_committee_label",
        }
    )
    bucket_map = {
        "tp_risk_preserved": "risk_detected",
        "tn_kept_eligible": "stable_confirmed",
        "fn_caught_as_review_or_reject": "committee_caught",
        "fp_softened_to_eligible_or_hold": "overwarning_softened",
    }
    output["landing_recommendation_bucket"] = (
        output["external_validation_stage2_effect"].map(bucket_map).fillna("")
    )
    return output


def copy_static_artifacts(output_dir: Path) -> None:
    static_files = [
        "feature_dictionary.csv",
        "scenario_presets.json",
        "llm_payload_template.json",
    ]
    for filename in static_files:
        shutil.copy2(SOURCE_DASHBOARD_DIR / filename, output_dir / filename)


def build_model_summary(
    metadata: dict[str, Any], prediction_scores: pd.DataFrame
) -> dict[str, Any]:
    return {
        "dashboard_dataset": "feature_43_inference_2026",
        "source_inference_file": str(INFERENCE_PATH.relative_to(ROOT)),
        "model_artifact": str(MODEL_PATH.relative_to(ROOT)),
        "metadata_file": str(MODEL_METADATA_PATH.relative_to(ROOT)),
        "row_count": len(prediction_scores),
        "fiscal_year": 2025,
        "eval_year": 2026,
        "feature_count": int(metadata.get("feature_count", len(metadata["feature_columns"]))),
        "threshold_tuned": float(metadata["threshold_tuned"]),
        "probability_calibration": metadata.get("probability_calibration", {}),
        "prediction_label_counts": (
            prediction_scores["predicted_label"].value_counts().sort_index().to_dict()
        ),
        "risk_band_counts": prediction_scores["risk_band"].value_counts().to_dict(),
        "stage2_review_trigger_count": int(prediction_scores["stage2_review_trigger"].sum()),
        "external_validation_recommendation_count": int(
            prediction_scores["landing_recommendation_bucket"].astype(str).ne("").sum()
        ),
    }


def main() -> None:
    from xgboost import Booster, DMatrix

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = read_json(MODEL_METADATA_PATH)
    feature_json = read_json(FEATURE_LIST_PATH)
    model_features = list(metadata["feature_columns"])
    source_features = list(metadata["source_features"])
    categorical_source_features = set(feature_json.get("categorical_one_hot_columns", []))
    numeric_source_features = [
        feature for feature in source_features if feature not in categorical_source_features
    ]

    inference = pd.read_csv(INFERENCE_PATH, encoding="utf-8-sig", dtype={"stock_code": str})
    inference["stock_code"] = inference["stock_code"].astype(str).str.zfill(6)
    missing_features = [feature for feature in model_features if feature not in inference.columns]
    if missing_features:
        raise ValueError(f"Missing model features in inference input: {missing_features}")

    model = Booster()
    model.load_model(MODEL_PATH)
    matrix = DMatrix(inference.loc[:, model_features], feature_names=model_features)
    raw_probabilities = model.predict(matrix)
    calibration = metadata.get("probability_calibration") or {}
    probabilities = apply_platt_calibration(raw_probabilities, calibration)
    threshold = float(metadata["threshold_tuned"])
    prediction_scores = build_prediction_scores(
        inference,
        raw_probabilities,
        probabilities,
        threshold,
        str(calibration.get("method", "none")),
    )
    prediction_scores = attach_validation_recommendation_flags(prediction_scores)

    shap_contribs = model.predict(matrix, pred_contribs=True)[:, :-1]
    local_shap = build_local_shap(
        inference,
        prediction_scores,
        shap_contribs,
        model_features,
        source_features,
    )
    feature_dictionary = pd.read_csv(
        SOURCE_DASHBOARD_DIR / "feature_dictionary.csv",
        encoding="utf-8-sig",
    )
    global_shap_reference = build_global_shap_reference(local_shap, feature_dictionary)

    company_universe_columns = [
        "market",
        "stock_code",
        "corp_name",
        "fiscal_year",
        "eval_year",
        "listed_year",
        "firm_size_group",
        "industry_macro_category",
        *source_features,
    ]
    company_universe = inference.loc[
        :,
        [
            column
            for column in dict.fromkeys(company_universe_columns)
            if column in inference.columns
        ],
    ].copy()
    company_latest = build_company_latest(inference, source_features)
    peer_percentiles = build_peer_percentiles(inference, numeric_source_features)
    industry_year_summary = build_industry_year_summary(prediction_scores)
    industry_latest_summary = build_industry_latest_summary(prediction_scores)
    industry_shap_summary = build_industry_shap_summary(local_shap)

    copy_static_artifacts(OUTPUT_DIR)
    company_universe.to_csv(OUTPUT_DIR / "company_universe.csv", index=False, encoding="utf-8-sig")
    company_latest.to_csv(OUTPUT_DIR / "company_latest.csv", index=False, encoding="utf-8-sig")
    peer_percentiles.to_csv(OUTPUT_DIR / "peer_percentiles.csv", index=False, encoding="utf-8-sig")
    prediction_scores.to_csv(
        OUTPUT_DIR / "prediction_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_scores.to_csv(
        OUTPUT_DIR / "stage2_review_signals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    local_shap.to_csv(OUTPUT_DIR / "local_shap.csv", index=False, encoding="utf-8-sig")
    global_shap_reference.to_csv(
        OUTPUT_DIR / "global_shap_reference.csv",
        index=False,
        encoding="utf-8-sig",
    )
    industry_year_summary.to_csv(
        OUTPUT_DIR / "industry_year_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    industry_latest_summary.to_csv(
        OUTPUT_DIR / "industry_latest_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    industry_shap_summary.to_csv(
        OUTPUT_DIR / "industry_shap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_summary = build_model_summary(metadata, prediction_scores)
    write_json(OUTPUT_DIR / "model_summary.json", model_summary)
    write_json(
        OUTPUT_DIR / "dashboard_export_manifest.json",
        {
            "dashboard_dataset": "feature_43_inference_2026",
            "created_at": datetime.now(UTC).isoformat(),
            "source_files": {
                "inference": str(INFERENCE_PATH.relative_to(ROOT)),
                "model": str(MODEL_PATH.relative_to(ROOT)),
                "metadata": str(MODEL_METADATA_PATH.relative_to(ROOT)),
                "external_validation_scores": str(VALIDATION_SCORES_PATH.relative_to(ROOT))
                if VALIDATION_SCORES_PATH.exists()
                else None,
            },
            "outputs": [
                "company_universe.csv",
                "company_latest.csv",
                "peer_percentiles.csv",
                "prediction_scores.csv",
                "stage2_review_signals.csv",
                "local_shap.csv",
                "global_shap_reference.csv",
                "industry_year_summary.csv",
                "industry_latest_summary.csv",
                "industry_shap_summary.csv",
            ],
        },
    )
    readme = """# feature_43_inference_2026 Dashboard Artifacts

이 폴더는 2025 회계연도 입력값으로 2026년 신용위험을 예측하는 대시보드 산출물입니다.

- `company_latest.csv`: 2026 예측 대상 기업 목록
- `prediction_scores.csv`: 43개 XGBoost 모델의 1차 예측확률, 위험 구간, 위원회 검토 트리거
- `local_shap.csv`: 기업별 주요 SHAP 영향 요인
- `peer_percentiles.csv`: 시장/산업 내 백분위 비교
- `stage2_review_signals.csv`: 대시보드와 에이전트 위원회가 참고하는 2차 검토 트리거

참고: 2026년 공시 외부검증과 매칭된 기업은 `landing_recommendation_bucket`으로 표시됩니다.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"Exported 2026 dashboard artifacts to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
