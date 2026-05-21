from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from apply_opendart_financial_supplements import (
    BASE_STATEMENT_COLUMNS,
    build_supplement_frame,
    load_supplement_rows,
    missing_financial_statement_source,
    normalize_stock_code,
    recompute_derived_columns,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_V1_PATH = ROOT / "data" / "raw" / "ts2000" / "TS2000_Credit_Model_Dataset_Model_V1.csv"
INFERENCE_PATH = ROOT / "data" / "input" / "credit_43_features" / "feature_43_inference_2026.csv"
DEFAULT_RAW_SUPPLEMENT_PATH = (
    ROOT
    / "data"
    / "raw"
    / "opendart"
    / "financial_statements_inference_2025_cfs_with_ofs_fallback_raw.csv"
)
DEFAULT_AUDIT_PATH = (
    ROOT / "data" / "raw" / "opendart" / "inference_2026_opendart_supplement_audit.csv"
)
ID_COLUMNS = {
    "market",
    "stock_code",
    "corp_name",
    "fiscal_year",
    "eval_year",
    "firm_size_group",
    "industry_macro_category",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply OpenDART CFS/OFS fallback financial statement supplements to the "
            "2026 inference feature table and recompute financial ratios using the "
            "historical Model_V1 panel for lag features."
        )
    )
    parser.add_argument("--inference", type=Path, default=INFERENCE_PATH)
    parser.add_argument("--history", type=Path, default=MODEL_V1_PATH)
    parser.add_argument("--raw-supplement", type=Path, default=DEFAULT_RAW_SUPPLEMENT_PATH)
    parser.add_argument("--output", type=Path, default=INFERENCE_PATH)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_full_inference_frame(inference: pd.DataFrame, history_columns: list[str]) -> pd.DataFrame:
    full = pd.DataFrame(index=inference.index, columns=history_columns)
    for column in inference.columns:
        if column in full.columns:
            full[column] = inference[column]
    full["stock_code"] = full["stock_code"].map(normalize_stock_code)
    full["_inference_row_id"] = range(len(full))
    return full


def apply_supplements_to_inference(
    inference_full: pd.DataFrame,
    supplements: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = inference_full.copy()
    output["_stock_code_key"] = output["stock_code"].map(normalize_stock_code)
    output["fiscal_year"] = pd.to_numeric(output["fiscal_year"], errors="coerce").astype(int)
    for column in BASE_STATEMENT_COLUMNS:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").astype("float64")
    missing_mask = missing_financial_statement_source(output)

    supplements = supplements.copy()
    supplements["_stock_code_key"] = supplements["stock_code"].map(normalize_stock_code)
    supplement_lookup = supplements.set_index(["market", "_stock_code_key", "fiscal_year"], drop=False)

    audit_records: list[dict[str, object]] = []
    for index, row in output.loc[missing_mask].iterrows():
        key = (row["market"], row["_stock_code_key"], row["fiscal_year"])
        if key not in supplement_lookup.index:
            continue
        supplement = supplement_lookup.loc[key]
        if isinstance(supplement, pd.DataFrame):
            supplement = supplement.iloc[0]
        changed_columns: list[str] = []
        for column in BASE_STATEMENT_COLUMNS:
            value = supplement.get(column)
            if value is None or pd.isna(value) or column not in output.columns:
                continue
            old_value = output.at[index, column]
            output.at[index, column] = value
            if pd.isna(old_value) or float(old_value) != float(value):
                changed_columns.append(column)
        audit_records.append(
            {
                "market": row["market"],
                "stock_code": row["stock_code"],
                "corp_name": row["corp_name"],
                "fiscal_year": row["fiscal_year"],
                "eval_year": row.get("eval_year"),
                "fs_div_requested": supplement.get("fs_div_requested", ""),
                "fs_div_used": supplement.get("fs_div", ""),
                "raw_account_rows": supplement.get("raw_account_rows", 0),
                "changed_column_count": len(changed_columns),
                "changed_columns": "|".join(changed_columns),
            }
        )

    return output.drop(columns=["_stock_code_key"]), pd.DataFrame(audit_records)


def recompute_with_history(
    history: pd.DataFrame,
    inference_full: pd.DataFrame,
) -> pd.DataFrame:
    history_panel = history.copy()
    history_panel["stock_code"] = history_panel["stock_code"].map(normalize_stock_code)
    history_panel["_inference_row_id"] = pd.NA
    inference_panel = inference_full.dropna(axis=1, how="all")
    panel = pd.concat([history_panel, inference_panel], ignore_index=True, sort=False)
    panel = panel.sort_values(["market", "stock_code", "fiscal_year"]).reset_index(drop=True)
    recomputed = recompute_derived_columns(panel)
    return recomputed.loc[recomputed["_inference_row_id"].notna()].copy()


def main() -> None:
    args = parse_args()
    inference = pd.read_csv(args.inference, encoding="utf-8-sig", dtype={"stock_code": "string"})
    original_columns = inference.columns.tolist()
    original_stock_code = inference["stock_code"].copy()

    history = pd.read_csv(args.history, encoding="utf-8-sig", low_memory=False, dtype={"stock_code": "string"})
    raw = load_supplement_rows(args.raw_supplement)
    supplements = build_supplement_frame(raw)

    inference_full = build_full_inference_frame(inference, history.columns.tolist())
    supplemented, audit = apply_supplements_to_inference(inference_full, supplements)
    recomputed = recompute_with_history(history, supplemented)
    recomputed = recomputed.sort_values("_inference_row_id")

    updated = inference.copy()
    feature_columns = [column for column in original_columns if column not in ID_COLUMNS]
    for column in feature_columns:
        if column in recomputed.columns:
            updated[column] = recomputed[column].to_numpy()
    updated["stock_code"] = original_stock_code
    updated = updated.loc[:, original_columns]

    print(f"[Supplements] raw_rows={len(raw):,}, rows={len(supplements):,}")
    print(f"[Applied] inference_rows={len(audit):,}")
    if not audit.empty:
        print(
            audit[
                ["market", "stock_code", "corp_name", "fiscal_year", "fs_div_used", "changed_column_count"]
            ]
            .head(20)
            .to_string(index=False)
        )

    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(args.output, index=False, encoding="utf-8-sig")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.audit_output, index=False, encoding="utf-8-sig")
    print(f"[Saved] {args.output} ({len(updated):,} rows)")
    print(f"[Saved] {args.audit_output} ({len(audit):,} rows)")


if __name__ == "__main__":
    main()
