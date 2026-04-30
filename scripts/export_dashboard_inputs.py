from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TS2000_DIR = ROOT / "data" / "external" / "ts2000"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "outputs" / "dashboard" / "ts2000_core29_mvp"

MODEL_V1_DATASET_PATH = TS2000_DIR / "TS2000_Credit_Model_Dataset_Model_V1.csv"
CORE29_MANIFEST_PATH = TS2000_DIR / "TS2000_Model_Core29_Manifest.json"
MODEL_V1_MANIFEST_PATH = TS2000_DIR / "TS2000_Model_V1_Manifest.json"
METADATA_PATH = TS2000_DIR / "column_dictionary" / "ts2000_column_dictionary_metadata.json"
PERFORMANCE_SUMMARY_PATH = (
    TS2000_DIR / "model_results" / "round1_model_comparison" / "performance_summary.csv"
)
THRESHOLD_SUMMARY_PATH = (
    TS2000_DIR / "model_results" / "xgboost_threshold_shap" / "threshold_summary.csv"
)
SHAP_IMPORTANCE_PATH = (
    TS2000_DIR / "model_results" / "xgboost_threshold_shap" / "shap_importance_grouped.csv"
)

SCENARIO_PRESETS: dict[str, dict[str, float]] = {
    "base": {},
    "mild_stress": {
        "spec_spread": 0.50,
        "cash_ratio": -0.05,
        "net_margin": -0.01,
    },
    "severe_stress": {
        "spec_spread": 1.00,
        "cash_ratio": -0.10,
        "net_margin": -0.02,
        "short_term_borrowings_share": 0.05,
        "capital_impairment_ratio": 0.05,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TS2000 Core29 dashboard-ready input artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where dashboard artifacts will be written.",
    )
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reverse_feature_groups(feature_groups: dict[str, list[str]]) -> dict[str, str]:
    group_by_feature: dict[str, str] = {}
    for group_name, features in feature_groups.items():
        for feature in features:
            group_by_feature[feature] = group_name
    return group_by_feature


def build_company_latest_snapshot(
    dataset: pd.DataFrame,
    *,
    id_columns: list[str],
    time_columns: list[str],
    categorical_columns: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    display_columns = ["market", "corp_name", "stock_code", "fiscal_year", "eval_year"]
    optional_columns = ["listed_year", "firm_size_group", "industry_macro_category"]
    keep_columns = []
    seen: set[str] = set()
    for column in display_columns + optional_columns + categorical_columns + feature_columns:
        if column in dataset.columns and column not in seen:
            keep_columns.append(column)
            seen.add(column)

    latest = dataset.sort_values(time_columns).groupby(id_columns, as_index=False).tail(1)
    latest = latest.loc[:, keep_columns].sort_values(["market", "corp_name", "stock_code"])
    return latest.reset_index(drop=True)


def build_company_universe(
    dataset: pd.DataFrame,
    *,
    id_columns: list[str],
    time_columns: list[str],
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    keep_columns = list(
        dict.fromkeys(id_columns + time_columns + [target_column] + feature_columns)
    )
    universe = dataset.loc[
        :, [column for column in keep_columns if column in dataset.columns]
    ].copy()
    return universe.sort_values(id_columns + time_columns).reset_index(drop=True)


def build_peer_percentiles(
    dataset: pd.DataFrame,
    *,
    id_columns: list[str],
    time_columns: list[str],
    numeric_features: list[str],
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    base_columns = id_columns + time_columns + ["market", "industry_macro_category"]

    for feature in numeric_features:
        values = dataset[feature]
        chunk = dataset.loc[:, base_columns].copy()
        chunk["feature"] = feature
        chunk["value"] = values
        chunk["overall_percentile"] = values.rank(method="average", pct=True) * 100.0
        chunk["market_percentile"] = (
            dataset.groupby("market")[feature].rank(
                method="average",
                pct=True,
            )
            * 100.0
        )
        chunk["industry_percentile"] = (
            dataset.groupby("industry_macro_category")[feature].rank(
                method="average",
                pct=True,
            )
            * 100.0
        )
        chunk["overall_median"] = values.median(skipna=True)
        chunk["market_median"] = dataset.groupby("market")[feature].transform("median")
        chunk["industry_median"] = dataset.groupby("industry_macro_category")[feature].transform(
            "median"
        )
        chunks.append(chunk)

    peer_percentiles = pd.concat(chunks, ignore_index=True)
    return peer_percentiles.sort_values(id_columns + time_columns + ["feature"]).reset_index(
        drop=True
    )


def build_feature_dictionary(
    columns_meta: list[dict[str, object]],
    *,
    feature_columns: list[str],
    feature_group_map: dict[str, str],
) -> pd.DataFrame:
    rows = []
    feature_set = set(feature_columns)
    for column in columns_meta:
        variable_name = column["variable_name"]
        if variable_name not in feature_set:
            continue
        rows.append(
            {
                "feature": variable_name,
                "feature_group": feature_group_map.get(variable_name, "unknown"),
                "korean_name": column.get("korean_name", ""),
                "description": column.get("description", ""),
                "formula_or_logic": column.get("formula_or_logic", ""),
                "unit": column.get("unit", ""),
                "source": column.get("source", ""),
                "note": column.get("note", ""),
            }
        )

    feature_dictionary = pd.DataFrame(rows)
    return feature_dictionary.sort_values(["feature_group", "feature"]).reset_index(drop=True)


def build_global_shap_reference(
    shap_importance: pd.DataFrame,
    *,
    feature_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    merged = shap_importance.merge(
        feature_dictionary,
        how="left",
        left_on="feature_name",
        right_on="feature",
    )
    merged["rank"] = merged["mean_abs_shap"].rank(method="dense", ascending=False).astype(int)
    merged = merged.rename(columns={"feature_name": "feature"})
    columns = [
        "rank",
        "feature",
        "feature_group",
        "mean_abs_shap",
        "korean_name",
        "description",
        "unit",
        "note",
    ]
    return merged.loc[:, columns].sort_values("rank").reset_index(drop=True)


def build_model_summary(
    performance_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    *,
    selected_model: str,
) -> dict[str, object]:
    test_overall = performance_summary.query("split == 'test' and segment == 'overall'").copy()
    models = []
    for record in test_overall.to_dict(orient="records"):
        models.append(
            {
                "model": record["model"],
                "rows": int(record["rows"]),
                "positive_rows": int(record["positive_rows"]),
                "positive_rate": float(record["positive_rate"]),
                "pr_auc": float(record["pr_auc"]),
                "roc_auc": float(record["roc_auc"]),
                "precision_at_0_5": float(record["precision_at_0_5"]),
                "recall_at_0_5": float(record["recall_at_0_5"]),
            }
        )

    thresholds = []
    for record in threshold_summary.to_dict(orient="records"):
        thresholds.append(
            {
                "threshold_type": record["threshold_type"],
                "threshold": float(record["threshold"]),
                "selection_rule": record["selection_rule"],
                "test_precision": float(record["test_precision"]),
                "test_recall": float(record["test_recall"]),
                "test_f1": float(record["test_f1"]),
                "test_pr_auc": float(record["test_pr_auc"]),
                "test_roc_auc": float(record["test_roc_auc"]),
            }
        )

    return {
        "selected_model": selected_model,
        "test_overall_models": models,
        "xgboost_thresholds": thresholds,
        "prediction_artifacts_ready": False,
        "prediction_artifacts_note": (
            "Per-company prediction probabilities and local SHAP values are not bundled in the "
            "repository package. This export contains dashboard base tables and global model "
            "references only."
        ),
    }


def build_llm_payload_template() -> dict[str, object]:
    return {
        "company_profile": {
            "corp_name": "Example Corp",
            "stock_code": "000000",
            "market": "KOSDAQ",
            "industry_macro_category": "Manufacturing",
            "firm_size_group": "Small",
            "fiscal_year": 2024,
        },
        "model_output": {
            "prob_speculative": None,
            "predicted_label": None,
            "threshold": 0.5437,
            "risk_band": None,
        },
        "key_metrics": {
            "cash_ratio": None,
            "interest_coverage_ratio": None,
            "capital_impairment_ratio": None,
            "net_margin": None,
        },
        "top_shap": [
            {
                "feature": "gross_profit",
                "feature_group": "profitability_returns",
                "feature_value": None,
                "shap_value": None,
                "direction": "risk_down",
            }
        ],
        "peer_context": {
            "cash_ratio": {
                "industry_percentile": None,
                "industry_median": None,
                "market_percentile": None,
                "market_median": None,
            }
        },
        "scenario_result": None,
    }


def build_export_manifest(
    *,
    output_dir: Path,
    core29_manifest: dict[str, object],
    numeric_features: list[str],
) -> dict[str, object]:
    return {
        "dataset_file": str(MODEL_V1_DATASET_PATH.relative_to(ROOT)),
        "core29_manifest_file": str(CORE29_MANIFEST_PATH.relative_to(ROOT)),
        "model_v1_manifest_file": str(MODEL_V1_MANIFEST_PATH.relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "core29_feature_count": len(core29_manifest["core29_feature_columns"]),
        "numeric_core29_feature_count": len(numeric_features),
        "generated_files": [
            "company_universe_core29.csv",
            "company_latest_core29.csv",
            "peer_percentiles_core29.csv",
            "feature_dictionary_core29.csv",
            "global_shap_reference_core29.csv",
            "scenario_presets_core29.json",
            "llm_payload_template_core29.json",
            "model_summary_core29.json",
            "README.md",
        ],
        "prediction_artifacts_ready": False,
        "prediction_artifacts_note": (
            "Per-company prediction and local SHAP exports are not present in the current "
            "repository package."
        ),
    }


def build_readme(
    *,
    output_dir: Path,
    core29_feature_count: int,
    numeric_core29_feature_count: int,
) -> str:
    relative_dir = output_dir.relative_to(ROOT)
    return f"""# TS2000 Dashboard Input Export

This folder contains dashboard-ready artifacts exported from the official TS2000 `Model_V1` dataset and `Core29` definition.

## Included files

- `company_universe_core29.csv`
  - Company-year level table containing id/time columns, target, and official Core29 features.
- `company_latest_core29.csv`
  - Latest available company snapshot for search/select UI.
- `peer_percentiles_core29.csv`
  - Long-format percentile/median table by feature, market, and industry.
- `feature_dictionary_core29.csv`
  - Core29 feature dictionary for labels, tooltips, and explanations.
- `global_shap_reference_core29.csv`
  - Global SHAP importance reference for the official XGBoost model.
- `scenario_presets_core29.json`
  - What-if scenario presets for MVP stress testing.
- `llm_payload_template_core29.json`
  - Suggested payload schema for LLM explanation generation.
- `model_summary_core29.json`
  - Official model comparison and threshold summary.
- `dashboard_export_manifest.json`
  - Export metadata and completeness notes.

## Notes

- Official Core29 feature count: `{core29_feature_count}`
- Numeric Core29 feature count used for percentile export: `{numeric_core29_feature_count}`
- Per-company prediction probabilities and local SHAP values are not included in the current repository package.
- This export is intended to support the first dashboard MVP described in `docs/ts2000_dashboard_mvp_design.md`.

## Source

- dataset: `data/external/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv`
- feature definition: `data/external/ts2000/TS2000_Model_Core29_Manifest.json`
- output directory: `{relative_dir}`
"""


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    core29_manifest = load_json(CORE29_MANIFEST_PATH)
    metadata = load_json(METADATA_PATH)

    dataset = pd.read_csv(MODEL_V1_DATASET_PATH, encoding="utf-8-sig")
    performance_summary = pd.read_csv(PERFORMANCE_SUMMARY_PATH, encoding="utf-8-sig")
    threshold_summary = pd.read_csv(THRESHOLD_SUMMARY_PATH, encoding="utf-8-sig")
    shap_importance = pd.read_csv(SHAP_IMPORTANCE_PATH, encoding="utf-8-sig")

    id_columns = list(core29_manifest["id_columns"])
    time_columns = list(core29_manifest["time_columns"])
    target_column = str(core29_manifest["target_column"])
    categorical_columns = list(core29_manifest["recommended_categorical_columns"])
    feature_columns = list(core29_manifest["core29_feature_columns"])
    feature_group_map = reverse_feature_groups(core29_manifest["feature_groups"])

    numeric_features = [
        feature
        for feature in feature_columns
        if feature not in categorical_columns and pd.api.types.is_numeric_dtype(dataset[feature])
    ]

    company_universe = build_company_universe(
        dataset,
        id_columns=id_columns,
        time_columns=time_columns,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    company_latest = build_company_latest_snapshot(
        dataset,
        id_columns=id_columns,
        time_columns=time_columns,
        categorical_columns=categorical_columns,
        feature_columns=feature_columns,
    )
    peer_percentiles = build_peer_percentiles(
        dataset,
        id_columns=id_columns,
        time_columns=time_columns,
        numeric_features=numeric_features,
    )
    feature_dictionary = build_feature_dictionary(
        metadata["columns"],
        feature_columns=feature_columns,
        feature_group_map=feature_group_map,
    )
    global_shap_reference = build_global_shap_reference(
        shap_importance,
        feature_dictionary=feature_dictionary,
    )
    model_summary = build_model_summary(
        performance_summary,
        threshold_summary,
        selected_model="xgboost",
    )
    export_manifest = build_export_manifest(
        output_dir=output_dir,
        core29_manifest=core29_manifest,
        numeric_features=numeric_features,
    )

    company_universe.to_csv(
        output_dir / "company_universe_core29.csv", index=False, encoding="utf-8-sig"
    )
    company_latest.to_csv(
        output_dir / "company_latest_core29.csv", index=False, encoding="utf-8-sig"
    )
    peer_percentiles.to_csv(
        output_dir / "peer_percentiles_core29.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_dictionary.to_csv(
        output_dir / "feature_dictionary_core29.csv",
        index=False,
        encoding="utf-8-sig",
    )
    global_shap_reference.to_csv(
        output_dir / "global_shap_reference_core29.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_json(output_dir / "scenario_presets_core29.json", SCENARIO_PRESETS)
    write_json(output_dir / "llm_payload_template_core29.json", build_llm_payload_template())
    write_json(output_dir / "model_summary_core29.json", model_summary)
    write_json(output_dir / "dashboard_export_manifest.json", export_manifest)

    readme_text = build_readme(
        output_dir=output_dir,
        core29_feature_count=len(feature_columns),
        numeric_core29_feature_count=len(numeric_features),
    )
    (output_dir / "README.md").write_text(readme_text, encoding="utf-8")

    print(f"Exported dashboard inputs to: {output_dir}")
    print(f"Rows in company_universe_core29.csv: {len(company_universe):,}")
    print(f"Rows in company_latest_core29.csv: {len(company_latest):,}")
    print(f"Rows in peer_percentiles_core29.csv: {len(peer_percentiles):,}")
    print(f"Rows in feature_dictionary_core29.csv: {len(feature_dictionary):,}")
    print(f"Rows in global_shap_reference_core29.csv: {len(global_shap_reference):,}")
    print(f"Numeric Core29 feature count: {len(numeric_features):,}")
    print(
        "Per-company prediction probabilities are not exported because the repository package "
        "does not currently include prediction-level artifacts."
    )


if __name__ == "__main__":
    main()
