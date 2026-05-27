from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from apply_opendart_financial_supplements import (
    BASE_STATEMENT_COLUMNS,
    build_supplement_frame,
    load_supplement_rows,
    normalize_stock_code,
    recompute_derived_columns,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_V1_PATH = ROOT / "data" / "raw" / "ts2000" / "TS2000_Credit_Model_Dataset_Model_V1.csv"
INFERENCE_PATH = ROOT / "data" / "input" / "credit_46_features" / "feature_46_inference_2026.csv"
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
DEFAULT_LAG_AUDIT_PATH = (
    ROOT / "data" / "raw" / "opendart" / "inference_2026_opendart_lag_2024_audit.csv"
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
INFERENCE_SUPPLEMENT_MISSING_THRESHOLD = 8
INFERENCE_SUPPLEMENT_CRITICAL_FEATURES = [
    "current_ratio",
    "cash_ratio",
    "equity_ratio",
    "debt_ratio",
    "total_borrowings_ratio",
    "net_margin",
    "interest_coverage_ratio",
    "pretax_roa",
    "operating_roa",
    "pretax_roe",
    "ocf_to_total_liabilities",
    "ocf_to_total_borrowings",
    "ocf_to_sales",
    "cashflow_coverage_ratio",
    "accruals_ratio",
    "intangible_assets_ratio",
    "total_debt_turnover",
    "short_term_borrowings_share",
    "total_assets_growth",
    "net_margin_diff",
]


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
    parser.add_argument(
        "--lag-raw-supplement",
        type=Path,
        default=None,
        help=(
            "Optional OpenDART raw file for the previous fiscal year. "
            "When supplied, it is added as a lag panel before recomputing trend features."
        ),
    )
    parser.add_argument("--output", type=Path, default=INFERENCE_PATH)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--lag-audit-output", type=Path, default=DEFAULT_LAG_AUDIT_PATH)
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


def inference_financial_supplement_target(frame: pd.DataFrame) -> pd.Series:
    """Select inference rows whose critical Stage 1 features need recomputation."""
    available = [column for column in INFERENCE_SUPPLEMENT_CRITICAL_FEATURES if column in frame]
    if not available:
        return pd.Series(False, index=frame.index)
    critical_missing_count = (
        frame[available]
        .apply(lambda column: pd.to_numeric(column, errors="coerce").isna())
        .sum(axis=1)
    )
    return critical_missing_count.ge(INFERENCE_SUPPLEMENT_MISSING_THRESHOLD)


def apply_supplements_to_inference(
    inference_full: pd.DataFrame,
    supplements: pd.DataFrame,
    *,
    apply_all_available: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = inference_full.copy()
    output["_stock_code_key"] = output["stock_code"].map(normalize_stock_code)
    output["fiscal_year"] = pd.to_numeric(output["fiscal_year"], errors="coerce").astype(int)
    for column in BASE_STATEMENT_COLUMNS:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").astype("float64")
    missing_mask = inference_financial_supplement_target(output)
    target_mask = pd.Series(True, index=output.index) if apply_all_available else missing_mask

    supplements = supplements.copy()
    supplements["_stock_code_key"] = supplements["stock_code"].map(normalize_stock_code)
    supplement_lookup = supplements.set_index(
        ["market", "_stock_code_key", "fiscal_year"], drop=False
    )

    audit_records: list[dict[str, object]] = []
    for index, row in output.loc[target_mask].iterrows():
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
                "critical_feature_missing_threshold": INFERENCE_SUPPLEMENT_MISSING_THRESHOLD,
                "critical_missing_target": bool(missing_mask.at[index]),
                "changed_column_count": len(changed_columns),
                "changed_columns": "|".join(changed_columns),
            }
        )

    return output.drop(columns=["_stock_code_key"]), pd.DataFrame(audit_records)


def recompute_with_history(
    history: pd.DataFrame,
    inference_full: pd.DataFrame,
    lag_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    history_panel = history.copy()
    history_panel["stock_code"] = history_panel["stock_code"].map(normalize_stock_code)
    if lag_history is not None and not lag_history.empty:
        lag_panel = lag_history.copy()
        lag_panel["stock_code"] = lag_panel["stock_code"].map(normalize_stock_code)
        lag_panel = lag_panel.dropna(axis=1, how="all")
        history_panel = pd.concat([history_panel, lag_panel], ignore_index=True, sort=False)
    history_panel["_inference_row_id"] = pd.NA
    inference_panel = inference_full.dropna(axis=1, how="all")
    panel = pd.concat([history_panel, inference_panel], ignore_index=True, sort=False)
    panel = panel.sort_values(["market", "stock_code", "fiscal_year"]).reset_index(drop=True)
    recomputed = recompute_derived_columns(panel)
    return recomputed.loc[recomputed["_inference_row_id"].notna()].copy()


def build_lag_history_frame(
    lag_supplements: pd.DataFrame,
    history: pd.DataFrame,
    inference_full: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert previous-year OpenDART supplements into synthetic history rows."""
    if lag_supplements.empty:
        return pd.DataFrame(columns=history.columns), pd.DataFrame()

    existing_keys = set(
        zip(
            history["market"],
            history["stock_code"].map(normalize_stock_code),
            pd.to_numeric(history["fiscal_year"], errors="coerce").astype("Int64"),
            strict=False,
        )
    )

    inference_meta = inference_full.assign(
        _stock_code_key=inference_full["stock_code"].map(normalize_stock_code)
    ).set_index(["market", "_stock_code_key"], drop=False)

    records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []
    for row in lag_supplements.to_dict(orient="records"):
        market = row.get("market")
        stock_code = normalize_stock_code(row.get("stock_code"))
        fiscal_year = int(row.get("fiscal_year"))
        key = (market, stock_code, fiscal_year)
        if key in existing_keys:
            continue

        record = {column: pd.NA for column in history.columns}
        for column, value in row.items():
            if column in record:
                record[column] = value
        record["market"] = market
        record["stock_code"] = stock_code
        record["fiscal_year"] = fiscal_year
        record["eval_year"] = fiscal_year + 1

        meta_key = (market, stock_code)
        if meta_key in inference_meta.index:
            meta = inference_meta.loc[meta_key]
            if isinstance(meta, pd.DataFrame):
                meta = meta.iloc[0]
            for column in ["firm_size_group", "industry_macro_category"]:
                if column in record and column in meta:
                    record[column] = meta.get(column)
            if "corp_name" in record and not record.get("corp_name"):
                record["corp_name"] = meta.get("corp_name")

        records.append(record)
        base_values = [
            column
            for column in BASE_STATEMENT_COLUMNS
            if column in row and row.get(column) is not None and not pd.isna(row.get(column))
        ]
        audit_records.append(
            {
                "market": market,
                "stock_code": stock_code,
                "corp_name": row.get("corp_name"),
                "fiscal_year": fiscal_year,
                "eval_year": fiscal_year + 1,
                "fs_div_used": row.get("fs_div", ""),
                "raw_account_rows": row.get("raw_account_rows", 0),
                "base_statement_value_count": len(base_values),
                "base_statement_columns": "|".join(base_values),
            }
        )

    return pd.DataFrame(records, columns=history.columns), pd.DataFrame(audit_records)


def load_lag_supplements(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    raw = load_supplement_rows(path)
    if raw.empty:
        return raw
    if "opendart_bsns_year" in raw.columns:
        raw = raw.copy()
        raw["fiscal_year"] = pd.to_numeric(raw["opendart_bsns_year"], errors="coerce").astype(
            "Int64"
        )
    return build_supplement_frame(raw)


def main() -> None:
    args = parse_args()
    inference = pd.read_csv(args.inference, encoding="utf-8-sig", dtype={"stock_code": "string"})
    original_columns = inference.columns.tolist()
    original_stock_code = inference["stock_code"].copy()

    history = pd.read_csv(
        args.history, encoding="utf-8-sig", low_memory=False, dtype={"stock_code": "string"}
    )
    raw = load_supplement_rows(args.raw_supplement)
    supplements = build_supplement_frame(raw)

    inference_full = build_full_inference_frame(inference, history.columns.tolist())
    supplemented, audit = apply_supplements_to_inference(
        inference_full,
        supplements,
        apply_all_available=args.lag_raw_supplement is not None,
    )
    lag_supplements = load_lag_supplements(args.lag_raw_supplement)
    lag_history, lag_audit = build_lag_history_frame(lag_supplements, history, supplemented)
    recomputed = recompute_with_history(history, supplemented, lag_history)
    recomputed = recomputed.sort_values("_inference_row_id")

    updated = inference.copy()
    feature_columns = [column for column in original_columns if column not in ID_COLUMNS]
    for column in feature_columns:
        if column in recomputed.columns:
            recomputed_values = recomputed[column].reset_index(drop=True)
            has_recomputed_value = recomputed_values.notna()
            updated.loc[has_recomputed_value, column] = recomputed_values.loc[
                has_recomputed_value
            ].to_numpy()
    updated["stock_code"] = original_stock_code
    updated = updated.loc[:, original_columns]

    print(f"[Supplements] raw_rows={len(raw):,}, rows={len(supplements):,}")
    print(f"[Applied] inference_rows={len(audit):,}")
    if args.lag_raw_supplement is not None:
        print(
            f"[Lag supplements] rows={len(lag_supplements):,}, added_history_rows={len(lag_history):,}"
        )
    if not audit.empty:
        print(
            audit[
                [
                    "market",
                    "stock_code",
                    "corp_name",
                    "fiscal_year",
                    "fs_div_used",
                    "changed_column_count",
                ]
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
    if args.lag_raw_supplement is not None:
        args.lag_audit_output.parent.mkdir(parents=True, exist_ok=True)
        lag_audit.to_csv(args.lag_audit_output, index=False, encoding="utf-8-sig")
    print(f"[Saved] {args.output} ({len(updated):,} rows)")
    print(f"[Saved] {args.audit_output} ({len(audit):,} rows)")
    if args.lag_raw_supplement is not None:
        print(f"[Saved] {args.lag_audit_output} ({len(lag_audit):,} rows)")


if __name__ == "__main__":
    main()
