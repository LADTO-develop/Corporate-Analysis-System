"""Shared builders for dashboard export artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def read_json(path: Path) -> object:
    """Read a UTF-8 JSON file."""
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    return payload


def write_json(path: Path, payload: object) -> None:
    """Write a UTF-8 JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_company_universe(master: pd.DataFrame, source_features: list[str]) -> pd.DataFrame:
    """Build the dashboard company-year universe table."""
    keep_columns = list(
        dict.fromkeys(
            [
                "market",
                "stock_code",
                "corp_name",
                "fiscal_year",
                "eval_year",
                "listed_year",
                "firm_size_group",
                "industry_macro_category",
                "is_speculative",
                *source_features,
            ]
        )
    )
    available = [column for column in keep_columns if column in master.columns]
    return (
        master.loc[:, available]
        .sort_values(["market", "corp_name", "stock_code", "fiscal_year", "eval_year"])
        .reset_index(drop=True)
    )


def build_company_latest(master: pd.DataFrame, source_features: list[str]) -> pd.DataFrame:
    """Build the latest company snapshot used by the dashboard landing view."""
    latest = (
        master.sort_values(["fiscal_year", "eval_year"])
        .groupby(["market", "stock_code", "corp_name"], as_index=False)
        .tail(1)
    )
    keep_columns = list(
        dict.fromkeys(
            [
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
        )
    )
    available = [column for column in keep_columns if column in latest.columns]
    return (
        latest.loc[:, available]
        .sort_values(["market", "corp_name", "stock_code"])
        .reset_index(drop=True)
    )


def build_peer_percentiles(master: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    """Build per-company peer percentile rows for dashboard comparison charts."""
    chunks: list[pd.DataFrame] = []
    base_columns = [
        "stock_code",
        "corp_name",
        "fiscal_year",
        "eval_year",
        "market",
        "industry_macro_category",
    ]
    for feature in numeric_features:
        values = pd.to_numeric(master[feature], errors="coerce")
        chunk = master.loc[:, base_columns].copy()
        chunk["feature"] = feature
        chunk["value"] = values
        chunk["overall_percentile"] = values.rank(method="average", pct=True) * 100.0
        chunk["market_percentile"] = (
            master.groupby("market")[feature].rank(method="average", pct=True) * 100.0
        )
        chunk["industry_percentile"] = (
            master.groupby("industry_macro_category")[feature].rank(
                method="average",
                pct=True,
            )
            * 100.0
        )
        chunk["overall_median"] = values.median(skipna=True)
        chunk["market_median"] = master.groupby("market")[feature].transform("median")
        chunk["industry_median"] = master.groupby("industry_macro_category")[feature].transform(
            "median"
        )
        chunks.append(chunk)
    return (
        pd.concat(chunks, ignore_index=True)
        .sort_values(["stock_code", "fiscal_year", "feature"])
        .reset_index(drop=True)
    )


def build_feature_dictionary(
    metadata_columns: list[dict[str, object]],
    feature_json: dict[str, object],
) -> pd.DataFrame:
    """Build the dashboard feature dictionary from source metadata."""
    metadata_lookup = {
        str(column["variable_name"]): column
        for column in metadata_columns
        if "variable_name" in column
    }

    raw_feature_metadata = feature_json.get("feature_metadata", [])
    feature_metadata = raw_feature_metadata if isinstance(raw_feature_metadata, list) else []
    feature_group_lookup: dict[str, str] = {}
    for raw_item in feature_metadata:
        if not isinstance(raw_item, Mapping) or "source_feature" not in raw_item:
            continue
        feature_group_lookup[str(raw_item["source_feature"])] = str(
            raw_item.get("feature_group", "unknown")
        )

    raw_selected_features = feature_json.get("selected_source_features", [])
    selected_features = raw_selected_features if isinstance(raw_selected_features, list) else []

    rows: list[dict[str, object]] = []
    for raw_feature in selected_features:
        feature = str(raw_feature)
        info = metadata_lookup.get(feature, {})
        rows.append(
            {
                "feature": feature,
                "feature_group": feature_group_lookup.get(feature, "unknown"),
                "korean_name": info.get("korean_name", feature),
                "description": info.get("description", ""),
                "formula_or_logic": info.get("formula_or_logic", ""),
                "unit": info.get("unit", ""),
                "source": info.get("source", "credit_46_features"),
                "note": info.get("note", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_group", "feature"]).reset_index(drop=True)


def sanitize_feature_name(name: str, mapping: dict[str, str]) -> str:
    """Map a model feature name back to the dashboard-facing source feature."""
    return mapping.get(name, name)


def risk_band(probability: float) -> str:
    """Return the dashboard risk band label for a speculative probability."""
    if probability < 0.35:
        return "안정"
    if probability < 0.65:
        return "관찰"
    return "고위험"


def build_global_shap_reference(
    local_shap: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """Build global SHAP reference rows from local SHAP exports."""
    grouped = (
        local_shap.groupby("feature", as_index=False)
        .agg(mean_abs_shap=("abs_shap", "mean"))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    grouped["rank"] = grouped.index + 1
    merged = grouped.merge(feature_dictionary, how="left", on="feature")
    return merged.loc[
        :,
        [
            "rank",
            "feature",
            "feature_group",
            "mean_abs_shap",
            "korean_name",
            "description",
            "unit",
            "note",
        ],
    ]


def build_industry_year_summary(prediction_scores: pd.DataFrame) -> pd.DataFrame:
    """Build market/industry/year aggregate prediction summaries."""
    return (
        prediction_scores.groupby(
            ["market", "industry_macro_category", "fiscal_year", "split"],
            dropna=False,
        )
        .agg(
            rows=("stock_code", "size"),
            companies=("stock_code", "nunique"),
            positive_rows=("is_speculative", "sum"),
            positive_rate=("is_speculative", "mean"),
            mean_prob_speculative=("prob_speculative", "mean"),
            median_prob_speculative=("prob_speculative", "median"),
            pred_share_0_5=("pred_label_0_5", "mean"),
            pred_share_tuned=("pred_label_tuned", "mean"),
        )
        .reset_index()
        .sort_values(["market", "industry_macro_category", "fiscal_year"])
    )


def build_industry_latest_summary(prediction_scores: pd.DataFrame) -> pd.DataFrame:
    """Build latest market/industry aggregate prediction summaries."""
    latest = (
        prediction_scores.sort_values(["fiscal_year", "eval_year"])
        .groupby(["market", "stock_code", "corp_name"], as_index=False)
        .tail(1)
    )
    return (
        latest.groupby(["market", "industry_macro_category"], dropna=False)
        .agg(
            companies=("stock_code", "nunique"),
            positive_companies=("is_speculative", "sum"),
            positive_rate=("is_speculative", "mean"),
            mean_prob_speculative=("prob_speculative", "mean"),
            median_prob_speculative=("prob_speculative", "median"),
            pred_share_0_5=("pred_label_0_5", "mean"),
            pred_share_tuned=("pred_label_tuned", "mean"),
        )
        .reset_index()
        .sort_values(["market", "industry_macro_category"])
    )


def build_industry_shap_summary(local_shap: pd.DataFrame) -> pd.DataFrame:
    """Build market/industry SHAP aggregate summaries."""
    grouped = (
        local_shap.groupby(["market", "industry_macro_category", "split", "feature"], dropna=False)
        .agg(
            count=("feature", "size"),
            mean_abs_shap=("abs_shap", "mean"),
            mean_signed_shap=("shap_value", "mean"),
        )
        .reset_index()
    )
    grouped["rank_within_group"] = (
        grouped.groupby(["market", "industry_macro_category", "split"])["mean_abs_shap"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return grouped.sort_values(
        ["market", "industry_macro_category", "split", "rank_within_group", "feature"]
    ).reset_index(drop=True)
