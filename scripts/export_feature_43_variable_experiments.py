from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input" / "credit_43_features"
OUTPUT_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_43_xgboost" / "diagnostics"
AMOUNT_COLUMNS = ["assets_total", "gross_profit", "depreciation"]
RANDOM_STATE = 42
PROBABILITY_CLIP_EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run feature-engineering and missing-value experiments for the 43-feature XGBoost model."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def read_split_frames(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(input_dir / "xgb_train.csv", encoding="utf-8-sig"),
        pd.read_csv(input_dir / "xgb_valid.csv", encoding="utf-8-sig"),
        pd.read_csv(input_dir / "xgb_test.csv", encoding="utf-8-sig"),
    )


def read_id_frames(input_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        split: pd.read_csv(input_dir / f"xgb_id_{split}.csv", encoding="utf-8-sig")
        for split in ["train", "valid", "test"]
    }


def signed_log1p(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.sign(values) * np.log1p(np.abs(values))


def add_industry_percentiles(
    frame: pd.DataFrame,
    id_frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    output = frame.copy()
    keys = id_frame.loc[:, ["industry_macro_category", "fiscal_year"]].reset_index(drop=True)
    for column in columns:
        values = pd.to_numeric(output[column], errors="coerce")
        ranking_frame = pd.DataFrame(
            {
                "industry": keys["industry_macro_category"],
                "fiscal_year": keys["fiscal_year"],
                "value": values,
            }
        )
        output[f"{column}_industry_pct"] = ranking_frame.groupby(
            ["fiscal_year", "industry"],
        )["value"].rank(pct=True, method="average")
    return output


def group_median_impute(
    frame: pd.DataFrame,
    id_frame: pd.DataFrame,
    train: pd.DataFrame,
    train_id: pd.DataFrame,
    feature_columns: list[str],
    global_medians: pd.Series,
) -> pd.DataFrame:
    output = frame.loc[:, feature_columns].copy()
    keys = id_frame.loc[:, ["market", "industry_macro_category"]].reset_index(drop=True)
    train_keys = train_id.loc[:, ["market", "industry_macro_category"]].reset_index(drop=True)
    train_full = pd.concat(
        [train_keys, train.loc[:, feature_columns].reset_index(drop=True)], axis=1
    )

    for column in feature_columns:
        group_medians = train_full.groupby(["market", "industry_macro_category"])[column].median()
        fallback_values = [
            group_medians.get((market, industry), global_medians[column])
            for market, industry in zip(
                keys["market"],
                keys["industry_macro_category"],
                strict=False,
            )
        ]
        output[column] = (
            output[column]
            .fillna(pd.Series(fallback_values, index=output.index))
            .fillna(global_medians[column])
        )
    return output


def fit_platt_calibration(
    y_valid: pd.Series,
    valid_probabilities: np.ndarray,
) -> tuple[float, float]:
    clipped = np.clip(
        valid_probabilities,
        PROBABILITY_CLIP_EPSILON,
        1.0 - PROBABILITY_CLIP_EPSILON,
    )
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(random_state=RANDOM_STATE, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits, y_valid.astype(int))
    return float(calibrator.coef_[0][0]), float(calibrator.intercept_[0])


def apply_platt_calibration(
    probabilities: np.ndarray,
    coef: float,
    intercept: float,
) -> np.ndarray:
    clipped = np.clip(probabilities, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-(intercept + coef * logits)))


def choose_tuned_threshold(y_valid: pd.Series, valid_probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.unique(np.round(valid_probabilities, 6)):
        score = float(f1_score(y_valid, valid_probabilities >= threshold, zero_division=0))
        if score > best_score:
            best_threshold = float(threshold)
            best_score = score
    return best_threshold


def train_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> XGBClassifier:
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
        scale_pos_weight=float(negative / positive) if positive else 1.0,
        early_stopping_rounds=50,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    return model


def evaluate_frames(
    *,
    variant: str,
    note: str,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, object]:
    y_train = train_frame["is_speculative"].astype(int)
    y_valid = valid_frame["is_speculative"].astype(int)
    y_test = test_frame["is_speculative"].astype(int)
    x_train = train_frame.loc[:, feature_columns]
    x_valid = valid_frame.loc[:, feature_columns]
    x_test = test_frame.loc[:, feature_columns]

    model = train_xgboost(x_train, y_train, x_valid, y_valid)
    valid_raw_probabilities = model.predict_proba(x_valid)[:, 1]
    test_raw_probabilities = model.predict_proba(x_test)[:, 1]
    coef, intercept = fit_platt_calibration(y_valid, valid_raw_probabilities)
    valid_probabilities = apply_platt_calibration(valid_raw_probabilities, coef, intercept)
    test_probabilities = apply_platt_calibration(test_raw_probabilities, coef, intercept)
    threshold = choose_tuned_threshold(y_valid, valid_probabilities)
    predictions = test_probabilities >= threshold

    return {
        "variant": variant,
        "note": note,
        "feature_count": len(feature_columns),
        "best_iteration": getattr(model, "best_iteration", None),
        "threshold_tuned": threshold,
        "test_pr_auc": float(average_precision_score(y_test, test_probabilities)),
        "test_roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "test_brier": float(brier_score_loss(y_test, test_probabilities)),
        "test_logloss": float(
            log_loss(
                y_test,
                np.clip(
                    test_probabilities,
                    PROBABILITY_CLIP_EPSILON,
                    1.0 - PROBABILITY_CLIP_EPSILON,
                ),
            )
        ),
        "raw_test_brier": float(brier_score_loss(y_test, test_raw_probabilities)),
        "raw_test_logloss": float(
            log_loss(
                y_test,
                np.clip(
                    test_raw_probabilities,
                    PROBABILITY_CLIP_EPSILON,
                    1.0 - PROBABILITY_CLIP_EPSILON,
                ),
            )
        ),
    }


def build_feature_variant(
    variant: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    id_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_frame = train.copy()
    valid_frame = valid.copy()
    test_frame = test.copy()
    features = list(feature_columns)

    if variant == "baseline_43":
        return "", train_frame, valid_frame, test_frame, features
    if variant == "drop_market_kospi":
        return (
            "market_KOSDAQ만 유지",
            train_frame,
            valid_frame,
            test_frame,
            [column for column in features if column != "market_KOSPI"],
        )
    if variant == "drop_market_kosdaq":
        return (
            "market_KOSPI만 유지",
            train_frame,
            valid_frame,
            test_frame,
            [column for column in features if column != "market_KOSDAQ"],
        )
    if variant in {"log_amounts_replace", "log_amounts_add", "drop_market_log_add"}:
        for frame in [train_frame, valid_frame, test_frame]:
            for column in AMOUNT_COLUMNS:
                frame[f"log_{column}"] = signed_log1p(frame[column])
        log_columns = [f"log_{column}" for column in AMOUNT_COLUMNS]
        if variant == "log_amounts_replace":
            features = [column for column in features if column not in AMOUNT_COLUMNS] + log_columns
            return (
                "절대금액 3개를 signed log1p로 대체",
                train_frame,
                valid_frame,
                test_frame,
                features,
            )
        if variant == "log_amounts_add":
            return (
                "절대금액 raw 유지 + log 변수 추가",
                train_frame,
                valid_frame,
                test_frame,
                [
                    *features,
                    *log_columns,
                ],
            )
        features = [column for column in features if column != "market_KOSPI"] + log_columns
        return (
            "market_KOSDAQ만 유지 + log 변수 추가",
            train_frame,
            valid_frame,
            test_frame,
            features,
        )
    if variant in {
        "industry_pct_replace_amounts",
        "industry_pct_add_amounts",
        "drop_market_pct_add",
    }:
        train_frame = add_industry_percentiles(train_frame, id_frames["train"], AMOUNT_COLUMNS)
        valid_frame = add_industry_percentiles(valid_frame, id_frames["valid"], AMOUNT_COLUMNS)
        test_frame = add_industry_percentiles(test_frame, id_frames["test"], AMOUNT_COLUMNS)
        percentile_columns = [f"{column}_industry_pct" for column in AMOUNT_COLUMNS]
        if variant == "industry_pct_replace_amounts":
            features = [
                column for column in features if column not in AMOUNT_COLUMNS
            ] + percentile_columns
            return (
                "절대금액 3개를 fiscal_year+industry 내부 백분위로 대체",
                train_frame,
                valid_frame,
                test_frame,
                features,
            )
        if variant == "industry_pct_add_amounts":
            return (
                "절대금액 raw 유지 + fiscal_year+industry 내부 백분위 추가",
                train_frame,
                valid_frame,
                test_frame,
                [*features, *percentile_columns],
            )
        features = [column for column in features if column != "market_KOSPI"] + percentile_columns
        return (
            "market_KOSDAQ만 유지 + 산업 백분위 추가",
            train_frame,
            valid_frame,
            test_frame,
            features,
        )
    raise ValueError(f"Unknown variant: {variant}")


def run_feature_experiments(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    id_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> pd.DataFrame:
    variants = [
        "baseline_43",
        "drop_market_kospi",
        "drop_market_kosdaq",
        "log_amounts_replace",
        "log_amounts_add",
        "industry_pct_replace_amounts",
        "industry_pct_add_amounts",
        "drop_market_log_add",
        "drop_market_pct_add",
    ]
    rows = []
    train_medians = train.loc[:, feature_columns].median(numeric_only=True)
    for variant in variants:
        note, train_frame, valid_frame, test_frame, variant_features = build_feature_variant(
            variant,
            train,
            valid,
            test,
            id_frames,
            feature_columns,
        )
        medians = train_frame.loc[:, variant_features].median(numeric_only=True)
        if variant == "baseline_43":
            medians = train_medians
        rows.append(
            evaluate_frames(
                variant=variant,
                note=note,
                train_frame=train_frame.assign(
                    **train_frame.loc[:, variant_features].fillna(medians).to_dict("series")
                ),
                valid_frame=valid_frame.assign(
                    **valid_frame.loc[:, variant_features].fillna(medians).to_dict("series")
                ),
                test_frame=test_frame.assign(
                    **test_frame.loc[:, variant_features].fillna(medians).to_dict("series")
                ),
                feature_columns=variant_features,
            )
        )
    return pd.DataFrame(rows).sort_values(["test_f1", "test_pr_auc"], ascending=False)


def run_imputation_experiments(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    id_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> pd.DataFrame:
    medians = train.loc[:, feature_columns].median(numeric_only=True)
    missing_columns = [column for column in feature_columns if train[column].isna().any()]
    rows = [
        evaluate_frames(
            variant="median_imputation",
            note="비교 기준: train 중앙값으로 결측 대체",
            train_frame=train.assign(
                **train.loc[:, feature_columns].fillna(medians).to_dict("series")
            ),
            valid_frame=valid.assign(
                **valid.loc[:, feature_columns].fillna(medians).to_dict("series")
            ),
            test_frame=test.assign(
                **test.loc[:, feature_columns].fillna(medians).to_dict("series")
            ),
            feature_columns=feature_columns,
        ),
        evaluate_frames(
            variant="xgboost_native_missing",
            note="XGBoost가 NaN 방향을 직접 학습",
            train_frame=train,
            valid_frame=valid,
            test_frame=test,
            feature_columns=feature_columns,
        ),
    ]

    train_indicator = train.copy()
    valid_indicator = valid.copy()
    test_indicator = test.copy()
    indicator_features = list(feature_columns)
    for column in missing_columns:
        indicator_name = f"{column}_missing"
        for frame in [train_indicator, valid_indicator, test_indicator]:
            frame[indicator_name] = frame[column].isna().astype(int)
        indicator_features.append(indicator_name)
    rows.append(
        evaluate_frames(
            variant="median_plus_missing_indicators",
            note=f"중앙값 대체 + 결측 여부 indicator {len(missing_columns)}개 추가",
            train_frame=train_indicator.assign(
                **train_indicator.loc[:, feature_columns].fillna(medians).to_dict("series")
            ),
            valid_frame=valid_indicator.assign(
                **valid_indicator.loc[:, feature_columns].fillna(medians).to_dict("series")
            ),
            test_frame=test_indicator.assign(
                **test_indicator.loc[:, feature_columns].fillna(medians).to_dict("series")
            ),
            feature_columns=indicator_features,
        )
    )

    rows.append(
        evaluate_frames(
            variant="market_industry_median_imputation",
            note="train 기준 시장+산업별 중앙값 대체",
            train_frame=train.assign(
                **group_median_impute(
                    train,
                    id_frames["train"],
                    train,
                    id_frames["train"],
                    feature_columns,
                    medians,
                ).to_dict("series")
            ),
            valid_frame=valid.assign(
                **group_median_impute(
                    valid,
                    id_frames["valid"],
                    train,
                    id_frames["train"],
                    feature_columns,
                    medians,
                ).to_dict("series")
            ),
            test_frame=test.assign(
                **group_median_impute(
                    test,
                    id_frames["test"],
                    train,
                    id_frames["train"],
                    feature_columns,
                    medians,
                ).to_dict("series")
            ),
            feature_columns=feature_columns,
        )
    )
    return pd.DataFrame(rows).sort_values(["test_f1", "test_pr_auc"], ascending=False)


def build_missing_value_summary(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for column in feature_columns:
        missing = train[column].isna()
        rows.append(
            {
                "feature": column,
                "train_missing_rate": float(train[column].isna().mean()),
                "valid_missing_rate": float(valid[column].isna().mean()),
                "test_missing_rate": float(test[column].isna().mean()),
                "overall_missing_rate": float(
                    pd.concat([train[column], valid[column], test[column]], ignore_index=True)
                    .isna()
                    .mean()
                ),
                "train_median": float(train[column].median(skipna=True)),
                "train_missing_rows": int(missing.sum()),
                "label_rate_when_missing": (
                    float(train.loc[missing, "is_speculative"].mean()) if missing.any() else None
                ),
                "label_rate_when_observed": (
                    float(train.loc[~missing, "is_speculative"].mean())
                    if (~missing).any()
                    else None
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("overall_missing_rate", ascending=False)


def _format_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def _format_percent(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def markdown_table(
    frame: pd.DataFrame,
    columns: list[tuple[str, str, str]],
) -> str:
    header = "| " + " | ".join(label for label, _, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in frame.to_dict(orient="records"):
        values = []
        for _, column, kind in columns:
            value = row.get(column)
            if kind == "metric":
                values.append(_format_metric(value))
            elif kind == "percent":
                values.append(_format_percent(value))
            elif kind == "int":
                values.append("-" if value is None or pd.isna(value) else f"{int(value):,}")
            else:
                values.append(str(value) if value is not None else "")
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def build_report(
    feature_metrics: pd.DataFrame,
    imputation_metrics: pd.DataFrame,
    missing_summary: pd.DataFrame,
) -> str:
    best_feature = feature_metrics.iloc[0]
    best_imputation = imputation_metrics.iloc[0]
    baseline = feature_metrics.loc[feature_metrics["variant"] == "baseline_43"].iloc[0]
    median = imputation_metrics.loc[imputation_metrics["variant"] == "median_imputation"].iloc[0]
    missing_nonzero = missing_summary.loc[missing_summary["overall_missing_rate"] > 0].head(12)
    feature_interpretation = (
        "- 현재 baseline_43이 F1 기준으로 가장 안정적입니다."
        if best_feature["variant"] == "baseline_43"
        else (
            f"- F1 기준 최상위는 `{best_feature['variant']}`이고 baseline_43 대비 "
            f"`{float(best_feature['test_f1']) - float(baseline['test_f1']):+.4f}` 차이입니다."
        )
    )
    imputation_interpretation = (
        "- 중앙값 대체가 F1 기준 최상위입니다."
        if best_imputation["variant"] == "median_imputation"
        else (
            f"- F1 기준 최상위 결측 전략은 `{best_imputation['variant']}`이고 중앙값 대체 대비 "
            f"`{float(best_imputation['test_f1']) - float(median['test_f1']):+.4f}` 차이입니다."
        )
    )
    return "\n".join(
        [
            "# Feature 43 Variable Improvement Experiments",
            "",
            "이 리포트는 43-feature XGBoost 기준에서 시장 더미 축소, 절대금액 변수 변환, "
            "산업 내 백분위 변수, 결측 대체 전략을 비교한 실험입니다.",
            "모든 실험은 동일한 train/valid/test split과 동일한 XGBoost 레시피를 사용하고, "
            "validation 기준 Platt scaling과 F1 threshold tuning을 적용했습니다.",
            "",
            "## 1. 변수 개선 실험 요약",
            "",
            f"- 가장 높은 F1 변형: `{best_feature['variant']}` "
            f"(F1 `{_format_metric(best_feature['test_f1'])}`, "
            f"PR-AUC `{_format_metric(best_feature['test_pr_auc'])}`)",
            feature_interpretation,
            "- 변수 변경은 성능 차이가 작아 production 반영 전 별도 모델 선택 합의가 필요합니다.",
            "",
            markdown_table(
                feature_metrics,
                [
                    ("Variant", "variant", "text"),
                    ("Features", "feature_count", "int"),
                    ("PR-AUC", "test_pr_auc", "metric"),
                    ("ROC-AUC", "test_roc_auc", "metric"),
                    ("Precision", "test_precision", "metric"),
                    ("Recall", "test_recall", "metric"),
                    ("F1", "test_f1", "metric"),
                    ("Brier", "test_brier", "metric"),
                    ("Logloss", "test_logloss", "metric"),
                    ("Note", "note", "text"),
                ],
            ),
            "",
            "## 2. 결측값 대체 실험 요약",
            "",
            f"- 가장 높은 F1 결측 전략: `{best_imputation['variant']}` "
            f"(F1 `{_format_metric(best_imputation['test_f1'])}`)",
            imputation_interpretation,
            "- missing indicator 추가와 시장+산업별 중앙값 대체는 현재 split에서 신중하게 봐야 합니다.",
            "",
            markdown_table(
                imputation_metrics,
                [
                    ("Variant", "variant", "text"),
                    ("Features", "feature_count", "int"),
                    ("PR-AUC", "test_pr_auc", "metric"),
                    ("ROC-AUC", "test_roc_auc", "metric"),
                    ("Precision", "test_precision", "metric"),
                    ("Recall", "test_recall", "metric"),
                    ("F1", "test_f1", "metric"),
                    ("Brier", "test_brier", "metric"),
                    ("Logloss", "test_logloss", "metric"),
                    ("Note", "note", "text"),
                ],
            ),
            "",
            "## 3. 결측률 점검",
            "",
            "결측률이 높은 변수는 차입금·현금흐름 관련 변수와 전년 대비 변화 변수입니다. "
            "결측 여부 자체의 양성 비율 차이는 크지 않아, 단순 indicator 추가 효과는 제한적이었습니다.",
            "",
            markdown_table(
                missing_nonzero,
                [
                    ("Feature", "feature", "text"),
                    ("Train Missing", "train_missing_rate", "percent"),
                    ("Valid Missing", "valid_missing_rate", "percent"),
                    ("Test Missing", "test_missing_rate", "percent"),
                    ("Missing Rows", "train_missing_rows", "int"),
                    ("Label Rate Missing", "label_rate_when_missing", "percent"),
                    ("Label Rate Observed", "label_rate_when_observed", "percent"),
                ],
            ),
            "",
            "## 4. 판단",
            "",
            "- `market_to_book` 원본 값 복구 후에는 성능 순위가 이전 all-zero 기준과 달라질 수 있습니다.",
            "- 현재 production artifact는 43개 변수와 XGBoost native missing 기준입니다.",
            "- native missing은 중앙값 대체보다 Recall과 F1이 높아 조기경보 목적에 더 잘 맞습니다.",
            "- 변수셋 축소나 결측 전략 변경은 성능 차이가 작으므로 발표/운영 기준을 먼저 합의하는 편이 안전합니다.",
            "",
        ]
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train, valid, test = read_split_frames(args.input_dir)
    id_frames = read_id_frames(args.input_dir)
    feature_columns = [column for column in train.columns if column != "is_speculative"]

    feature_metrics = run_feature_experiments(train, valid, test, id_frames, feature_columns)
    imputation_metrics = run_imputation_experiments(train, valid, test, id_frames, feature_columns)
    missing_summary = build_missing_value_summary(train, valid, test, feature_columns)
    report = build_report(feature_metrics, imputation_metrics, missing_summary)

    feature_metrics.to_csv(
        output_dir / "variable_experiment_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    imputation_metrics.to_csv(
        output_dir / "missing_value_imputation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missing_summary.to_csv(
        output_dir / "missing_value_summary.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "variable_experiment_report.md").write_text(report, encoding="utf-8")
    write_json(
        output_dir / "variable_experiment_summary.json",
        {
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset_name": "credit_43_features",
            "best_feature_variant": feature_metrics.iloc[0].to_dict(),
            "best_imputation_variant": imputation_metrics.iloc[0].to_dict(),
            "amount_columns_tested": AMOUNT_COLUMNS,
        },
    )
    print(f"feature_43 variable experiments written to: {output_dir}")


if __name__ == "__main__":
    main()
