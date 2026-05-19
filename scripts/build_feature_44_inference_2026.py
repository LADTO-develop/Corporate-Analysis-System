from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input" / "credit_44_features"
FEATURE_MASTER_PATH = INPUT_DIR / "feature_44_master.csv"
FEATURE_SPEC_PATH = INPUT_DIR / "feature_44_list.json"
CANONICAL_INFERENCE_PATH = INPUT_DIR / "feature_44_inference_2026.csv"
# The auxiliary source is independent of the model feature count. Keep the
# existing raw filename so teams do not need to duplicate a large local helper CSV.
INFERENCE_AUX_PATH = ROOT / "data" / "raw" / "ts2000" / "feature_43_inference_2026_aux.csv"
DERIVED_PEER_PERCENTILE_FEATURES: dict[str, tuple[str, list[str]]] = {
    "industry_current_ratio_percentile": (
        "current_ratio",
        ["fiscal_year", "industry_macro_category"],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and materialize the CAS-internal 2026 inference table. "
            "This script intentionally reads only files inside this repository."
        )
    )
    parser.add_argument("--source", type=Path, default=CANONICAL_INFERENCE_PATH)
    parser.add_argument("--output", type=Path, default=CANONICAL_INFERENCE_PATH)
    parser.add_argument("--target-fiscal-year", type=int, default=2025)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def load_expected_columns() -> tuple[list[str], list[str]]:
    master_columns = pd.read_csv(
        FEATURE_MASTER_PATH,
        nrows=1,
        encoding="utf-8-sig",
    ).columns.tolist()
    expected_columns = [column for column in master_columns if column != "is_speculative"]

    feature_spec = json.loads(FEATURE_SPEC_PATH.read_text(encoding="utf-8-sig"))
    model_features = [str(name) for name in feature_spec["model_features"]]
    missing_model_features = [column for column in model_features if column not in expected_columns]
    if missing_model_features:
        raise ValueError(
            "feature_44_list.json contains model features missing from inference columns: "
            f"{missing_model_features}"
        )

    return expected_columns, model_features


def repair_inference_frame(
    frame: pd.DataFrame,
    *,
    model_features: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Repair placeholder inference fields from the CAS-internal 2025 auxiliary source."""
    repaired = frame.copy()
    stats = {
        "firm_size_rows_repaired": 0,
        "market_to_book_rows_repaired": 0,
        "market_to_book_rows_set_missing": 0,
    }
    if not INFERENCE_AUX_PATH.exists():
        return repaired, stats

    aux = pd.read_csv(INFERENCE_AUX_PATH, encoding="utf-8-sig", dtype={"stock_code": "string"})
    aux["stock_code"] = normalize_stock_code(aux["stock_code"])
    repaired["stock_code"] = normalize_stock_code(repaired["stock_code"])
    merge_columns = [
        "market",
        "stock_code",
        "fiscal_year",
        "firm_size_group_source",
        "market_to_book_source",
    ]
    merged = repaired.merge(
        aux[merge_columns],
        on=["market", "stock_code", "fiscal_year"],
        how="left",
    )

    source_size = merged["firm_size_group_source"].astype("string")
    valid_size = (source_size.notna() & source_size.ne("") & source_size.ne("<NA>")).fillna(False)
    before_size = repaired["firm_size_group"].astype("string")
    repaired.loc[valid_size, "firm_size_group"] = source_size.loc[valid_size]
    stats["firm_size_rows_repaired"] = int((valid_size & before_size.ne(source_size)).sum())
    for feature in [name for name in model_features if name.startswith("firm_size_group_")]:
        category = feature.removeprefix("firm_size_group_")
        repaired[feature] = (repaired["firm_size_group"].astype(str) == category).astype(int)

    source_market_to_book = pd.to_numeric(merged["market_to_book_source"], errors="coerce")
    current_market_to_book = pd.to_numeric(repaired["market_to_book"], errors="coerce")
    placeholder_all_zero = bool(
        current_market_to_book.notna().any() and current_market_to_book.fillna(0).eq(0).all()
    )
    if placeholder_all_zero:
        repaired["market_to_book"] = source_market_to_book
        stats["market_to_book_rows_repaired"] = int(source_market_to_book.notna().sum())
        stats["market_to_book_rows_set_missing"] = int(source_market_to_book.isna().sum())
    else:
        missing_market_to_book = current_market_to_book.isna()
        fill_mask = missing_market_to_book & source_market_to_book.notna()
        repaired.loc[fill_mask, "market_to_book"] = source_market_to_book.loc[fill_mask]
        stats["market_to_book_rows_repaired"] = int(fill_mask.sum())

    return repaired, stats


def normalize_stock_code(series: pd.Series) -> pd.Series:
    text = (
        series.astype("string")
        .fillna("")
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
    )
    return text.where(text == "", text.str.zfill(6))


def add_derived_features(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    for feature, (base_column, group_columns) in DERIVED_PEER_PERCENTILE_FEATURES.items():
        required_columns = [base_column, *group_columns]
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            raise KeyError(
                f"Missing required columns for derived feature {feature}: {missing_columns}"
            )
        groupers = [frame[column] for column in group_columns]
        frame[feature] = (
            pd.to_numeric(frame[base_column], errors="coerce")
            .groupby(groupers)
            .rank(method="average", pct=True)
        )
    return frame


def validate_inference_frame(
    frame: pd.DataFrame,
    *,
    expected_columns: list[str],
    model_features: list[str],
    target_fiscal_year: int,
) -> None:
    if "is_speculative" in frame.columns:
        raise ValueError("Inference table must not contain the labeled target column.")

    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Inference table is missing required columns: {missing_columns}")

    extra_columns = [column for column in frame.columns if column not in expected_columns]
    if extra_columns:
        raise ValueError(f"Inference table contains unexpected columns: {extra_columns}")

    if frame.empty:
        raise ValueError("Inference table is empty.")

    fiscal_year = pd.to_numeric(frame["fiscal_year"], errors="coerce")
    eval_year = pd.to_numeric(frame["eval_year"], errors="coerce")
    invalid_fiscal_year = fiscal_year.ne(target_fiscal_year) | fiscal_year.isna()
    invalid_eval_year = eval_year.ne(target_fiscal_year + 1) | eval_year.isna()
    if invalid_fiscal_year.any():
        bad_values = sorted(fiscal_year.loc[invalid_fiscal_year].dropna().astype(int).unique())
        raise ValueError(
            f"Inference table must contain only fiscal_year={target_fiscal_year}; "
            f"found {bad_values}"
        )
    if invalid_eval_year.any():
        bad_values = sorted(eval_year.loc[invalid_eval_year].dropna().astype(int).unique())
        raise ValueError(
            f"Inference table must contain only eval_year={target_fiscal_year + 1}; "
            f"found {bad_values}"
        )

    duplicate_count = int(frame.duplicated(["market", "stock_code", "fiscal_year"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Inference table has duplicate market/stock_code/fiscal_year rows: {duplicate_count}"
        )

    null_model_feature_counts = frame[model_features].isna().sum()
    all_null_features = [
        column for column, count in null_model_feature_counts.items() if int(count) == len(frame)
    ]
    if all_null_features:
        raise ValueError(f"Model features are entirely missing: {all_null_features}")


def main() -> None:
    args = parse_args()
    expected_columns, model_features = load_expected_columns()
    inference = pd.read_csv(args.source, encoding="utf-8-sig")
    inference, repair_stats = repair_inference_frame(
        inference,
        model_features=model_features,
    )
    inference = add_derived_features(inference)
    validate_inference_frame(
        inference,
        expected_columns=expected_columns,
        model_features=model_features,
        target_fiscal_year=args.target_fiscal_year,
    )

    ordered = inference.loc[:, expected_columns].copy()
    if not args.check_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ordered.to_csv(args.output, index=False, encoding="utf-8-sig")

    action = "Checked" if args.check_only else "Saved"
    print(f"[{action}] {args.output} ({len(ordered):,} rows)")
    if INFERENCE_AUX_PATH.exists():
        print(
            "[Repair] "
            f"firm_size={repair_stats['firm_size_rows_repaired']:,}, "
            f"market_to_book={repair_stats['market_to_book_rows_repaired']:,}, "
            f"market_to_book_missing={repair_stats['market_to_book_rows_set_missing']:,}"
        )
    print(
        ordered[["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
