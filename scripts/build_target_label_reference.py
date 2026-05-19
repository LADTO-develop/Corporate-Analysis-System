"""Build a diagnostic target-label reference inside the CAS repository.

The reference keeps credit-rating strings and related audit fields out of the
model input tables while preserving them for model diagnostics, especially
BBB-/BB+ boundary analysis.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
MODEL_V1_PATH = ROOT / "data/raw/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv"
LEGACY_TARGET_AUDIT_PATH = PROJECT_ROOT / "02_Processed_Data/Target_Processed_audit.csv"
DISCLOSURE_2025_AUDIT_PATH = (
    PROJECT_ROOT / "02_Processed_Data/Target_Processed_2025_Disclosures_audit.csv"
)
OUTPUT_PATH = ROOT / "data/evaluation/target_label_reference.csv"
SUMMARY_PATH = ROOT / "data/evaluation/target_label_reference_summary.json"

KEY_COLUMNS = ["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]

RATING_RANK = {
    "AAA": 1,
    "AA+": 2,
    "AA": 3,
    "AA-": 4,
    "A+": 5,
    "A": 6,
    "A-": 7,
    "BBB+": 8,
    "BBB": 9,
    "BBB-": 10,
    "BB+": 11,
    "BB": 12,
    "BB-": 13,
    "B+": 14,
    "B": 15,
    "B-": 16,
    "CCC+": 17,
    "CCC": 18,
    "CCC-": 19,
    "CC": 20,
    "C": 21,
    "D": 22,
}

OUTPUT_COLUMNS = [
    "market",
    "stock_code",
    "corp_name",
    "fiscal_year",
    "eval_year",
    "is_speculative",
    "credit_rating",
    "credit_rating_rank",
    "rating_agency",
    "rating_agency_group",
    "rating_target",
    "rating_date",
    "selection_scope",
    "selection_rule",
    "candidate_count_in_year",
    "big3_candidate_count_in_year",
    "other_domestic_candidate_count_in_year",
    "foreign_candidate_count_in_year",
    "source_label_set",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-v1", type=Path, default=MODEL_V1_PATH)
    parser.add_argument("--legacy-target-audit", type=Path, default=LEGACY_TARGET_AUDIT_PATH)
    parser.add_argument("--disclosure-2025-audit", type=Path, default=DISCLOSURE_2025_AUDIT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def normalize_stock_code(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )


def read_model_keys(path: Path) -> pd.DataFrame:
    model = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    model = model.loc[:, KEY_COLUMNS].copy()
    model["stock_code"] = normalize_stock_code(model["stock_code"])
    for column in ["fiscal_year", "eval_year"]:
        model[column] = pd.to_numeric(model[column], errors="coerce").astype("Int64")
    return model.drop_duplicates(KEY_COLUMNS)


def normalize_label_frame(path: Path, source_label_set: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    labels = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    labels = labels.copy()
    labels["stock_code"] = normalize_stock_code(labels["stock_code"])
    for column in ["fiscal_year", "eval_year", "is_speculative"]:
        if column in labels.columns:
            labels[column] = pd.to_numeric(labels[column], errors="coerce").astype("Int64")

    if "credit_rating_rank" not in labels.columns:
        if "rating_rank" in labels.columns:
            labels = labels.rename(columns={"rating_rank": "credit_rating_rank"})
        else:
            labels["credit_rating_rank"] = labels["credit_rating"].map(RATING_RANK)
    labels["credit_rating"] = labels["credit_rating"].astype("string").str.strip()
    labels["credit_rating_rank"] = pd.to_numeric(
        labels["credit_rating_rank"], errors="coerce"
    ).astype("Int64")

    if "rating_target" not in labels.columns:
        labels["rating_target"] = labels.get("security_name", "")
    if "rating_date" not in labels.columns:
        labels["rating_date"] = labels.get("evaluation_date", "")

    labels["source_label_set"] = source_label_set

    for column in OUTPUT_COLUMNS:
        if column not in labels.columns:
            labels[column] = pd.NA

    return labels.loc[:, OUTPUT_COLUMNS].copy()


def choose_representative(group: pd.DataFrame) -> pd.Series:
    ranked = group.assign(
        rating_date_sort=pd.to_datetime(group["rating_date"], errors="coerce")
    ).sort_values(
        ["credit_rating_rank", "rating_date_sort", "source_label_set", "rating_agency"],
        ascending=[False, False, False, True],
    )
    selected = ranked.iloc[0].copy()
    selected["selection_rule"] = "worst_rating_then_latest_rating_date_for_diagnostics"
    return selected


def build_reference(
    model_keys: pd.DataFrame,
    legacy: pd.DataFrame,
    disclosure_2025: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat([legacy, disclosure_2025], ignore_index=True)
    combined = combined.dropna(subset=["credit_rating"])
    combined = combined.loc[combined["credit_rating"].ne("")]
    combined = combined.merge(model_keys, on=KEY_COLUMNS, how="inner", validate="many_to_one")

    selected = (
        combined.groupby(KEY_COLUMNS, group_keys=False, sort=False)
        .apply(choose_representative, include_groups=False)
        .reset_index()
    )
    selected = selected.loc[:, OUTPUT_COLUMNS].copy()
    selected = selected.sort_values(["eval_year", "market", "stock_code"]).reset_index(drop=True)
    return selected


def build_summary(reference: pd.DataFrame, model_keys: pd.DataFrame) -> dict[str, object]:
    model_by_year = model_keys.groupby("eval_year").size().astype(int).to_dict()
    labels_by_year = reference.groupby("eval_year").size().astype(int).to_dict()
    speculative_by_year = (
        reference.groupby("eval_year")["is_speculative"].sum().fillna(0).astype(int).to_dict()
    )
    missing_by_year = {
        int(year): int(model_by_year.get(year, 0) - labels_by_year.get(year, 0))
        for year in sorted(model_by_year)
    }
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "Diagnostic label reference for model evaluation. Credit-rating fields are "
            "not used as model inputs."
        ),
        "rows": len(reference),
        "model_v1_rows": len(model_keys),
        "rows_by_eval_year": {str(k): int(v) for k, v in labels_by_year.items()},
        "model_rows_by_eval_year": {str(k): int(v) for k, v in model_by_year.items()},
        "missing_label_rows_by_eval_year": {str(k): int(v) for k, v in missing_by_year.items()},
        "speculative_rows_by_eval_year": {str(k): int(v) for k, v in speculative_by_year.items()},
        "rating_agency_group_counts": reference["rating_agency_group"]
        .fillna("UNKNOWN")
        .value_counts()
        .astype(int)
        .to_dict(),
        "source_label_set_counts": reference["source_label_set"]
        .fillna("UNKNOWN")
        .value_counts()
        .astype(int)
        .to_dict(),
        "columns": OUTPUT_COLUMNS,
    }


def main() -> None:
    args = parse_args()
    model_keys = read_model_keys(args.model_v1)
    legacy = normalize_label_frame(args.legacy_target_audit, "legacy_target_processed_audit")
    disclosure_2025 = normalize_label_frame(
        args.disclosure_2025_audit,
        "recent_2025_disclosures",
    )
    reference = build_reference(model_keys, legacy, disclosure_2025)
    summary = build_summary(reference, model_keys)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(args.output, index=False, encoding="utf-8-sig")
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        {
            "output": str(args.output),
            "summary": str(args.summary),
            "rows": len(reference),
            "rows_by_eval_year": summary["rows_by_eval_year"],
            "missing_label_rows_by_eval_year": summary["missing_label_rows_by_eval_year"],
        }
    )


if __name__ == "__main__":
    main()
