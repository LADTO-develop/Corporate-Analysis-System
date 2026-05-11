from __future__ import annotations

import argparse
import json
from importlib import util
from pathlib import Path
from types import ModuleType

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
LEGACY_BUILDER_PATH = WORKSPACE_ROOT / "01_Raw_Data" / "build_ts2000_credit_model_dataset.py"
INPUT_DIR = ROOT / "data" / "input" / "credit_43_features"
FEATURE_MASTER_PATH = INPUT_DIR / "feature_43_master.csv"
FEATURE_SPEC_PATH = INPUT_DIR / "feature_43_list.json"
OUTPUT_PATH = INPUT_DIR / "feature_43_inference_2026.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unlabeled 2026 inference rows from 2025 fiscal-year raw data."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--target-fiscal-year", type=int, default=2025)
    return parser.parse_args()


def load_legacy_builder() -> ModuleType:
    spec = util.spec_from_file_location("legacy_ts2000_builder", LEGACY_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy builder: {LEGACY_BUILDER_PATH}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_market_div_with_latest_year(legacy: ModuleType, latest_year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for market, path in legacy.MARKET_DIV_FILE_CONFIG:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={legacy.MARKET_DIV_CODE_COL: str})
        df["market"] = market
        df["stock_code"] = legacy.normalize_stock_code(df[legacy.MARKET_DIV_CODE_COL])
        df["fiscal_year"] = legacy.parse_year(df[legacy.MARKET_DIV_FISCAL_COL])
        df["shares_outstanding_raw"] = legacy.parse_numeric(df[legacy.MARKET_DIV_SHARES_COL])
        df["close_price_raw"] = legacy.parse_numeric(df[legacy.MARKET_DIV_CLOSE_COL])
        df["cash_dividend_thousand_raw"] = legacy.parse_numeric(
            df[legacy.MARKET_DIV_DIVIDEND_COL]
        ).fillna(0.0)
        frames.append(
            df[
                [
                    *legacy.KEY_COLUMNS,
                    "shares_outstanding_raw",
                    "close_price_raw",
                    "cash_dividend_thousand_raw",
                ]
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["fiscal_year"].between(2014, latest_year, inclusive="both")].copy()
    aggregated = (
        combined.groupby(legacy.KEY_COLUMNS, as_index=False)
        .agg(
            shares_outstanding=("shares_outstanding_raw", "max"),
            close_price=("close_price_raw", "max"),
            cash_dividend_thousand=(
                "cash_dividend_thousand_raw",
                lambda s: s.loc[s.abs().idxmax()] if s.notna().any() else pd.NA,
            ),
            dividend_payer=(
                "cash_dividend_thousand_raw",
                lambda s: int(s.fillna(0.0).ne(0.0).any()),
            ),
        )
        .sort_values(legacy.KEY_COLUMNS)
        .reset_index(drop=True)
    )
    aggregated["market_cap"] = aggregated["shares_outstanding"] * aggregated["close_price"]
    legacy.assert_unique_keys(aggregated, "market_div")
    return aggregated


def build_unlabeled_feature_frame(legacy: ModuleType, target_fiscal_year: int) -> pd.DataFrame:
    profile = legacy.load_profile()
    bs = legacy.load_statement("BS", legacy.BS_RENAME)
    income = legacy.load_statement("IS", legacy.IS_RENAME)
    cashflow = legacy.load_statement("CF", legacy.CF_RENAME)
    macro = legacy.load_macro()
    market_div = load_market_div_with_latest_year(legacy, latest_year=target_fiscal_year)

    panel = (
        profile.merge(bs, on=legacy.KEY_COLUMNS, how="inner")
        .merge(income, on=legacy.KEY_COLUMNS, how="inner")
        .merge(cashflow, on=legacy.KEY_COLUMNS, how="inner")
        .merge(market_div, on=legacy.KEY_COLUMNS, how="left")
    )
    panel = panel.sort_values(["market", "stock_code", "fiscal_year"]).reset_index(drop=True)
    panel["dividend_payer"] = panel["dividend_payer"].fillna(0).astype(int)

    panel["total_borrowings"] = (
        panel["short_term_borrowings"] + panel["long_term_borrowings"] + panel["bonds_payable"]
    )
    panel["market_to_book"] = legacy.safe_ratio(
        panel["market_cap"],
        panel["equity_total"] * 1000.0,
    )
    panel["current_ratio"] = legacy.safe_ratio(
        panel["current_assets"], panel["current_liabilities"]
    )
    panel["cash_ratio"] = legacy.safe_ratio(
        panel["cash_and_equivalents"],
        panel["current_liabilities"],
    )
    panel["equity_ratio"] = legacy.safe_ratio(panel["equity_total"], panel["assets_total"])
    panel["debt_ratio"] = legacy.safe_ratio(panel["liabilities_total"], panel["equity_total"])
    panel["total_borrowings_ratio"] = legacy.safe_ratio(
        panel["total_borrowings"],
        panel["assets_total"],
    )
    panel["capital_impairment_ratio"] = legacy.safe_ratio(
        panel["capital_stock"] - panel["equity_total"],
        panel["capital_stock"],
    )
    panel["net_margin"] = legacy.safe_ratio(panel["net_income"], panel["revenue"])
    panel["interest_coverage_ratio"] = legacy.capped_ratio(
        panel["operating_income"],
        panel["interest_expense"],
    )
    panel["pretax_roa"] = legacy.safe_ratio(panel["pretax_income"], panel["assets_total"])
    panel["operating_roa"] = legacy.safe_ratio(panel["operating_income"], panel["assets_total"])
    panel["pretax_roe"] = legacy.safe_ratio(panel["pretax_income"], panel["equity_total"])
    panel["ocf_to_total_liabilities"] = legacy.safe_ratio(panel["ocf"], panel["liabilities_total"])
    panel["ocf_to_total_borrowings"] = legacy.safe_ratio(panel["ocf"], panel["total_borrowings"])
    panel["ocf_to_sales"] = legacy.safe_ratio(panel["ocf"], panel["revenue"])
    panel["cashflow_coverage_ratio"] = legacy.capped_ratio(
        panel["ocf"],
        panel["interest_expense"],
    )
    panel["accruals_ratio"] = legacy.safe_ratio(
        panel["net_income"] - panel["ocf"],
        panel["assets_total"],
    )
    panel["intangible_assets_ratio"] = legacy.safe_ratio(
        panel["intangible_assets"],
        panel["assets_total"],
    )
    panel["total_debt_turnover"] = legacy.safe_ratio(panel["revenue"], panel["liabilities_total"])
    panel["short_term_borrowings_share"] = legacy.safe_ratio(
        panel["short_term_borrowings"],
        panel["total_borrowings"],
    )

    group = panel.groupby(["market", "stock_code"], sort=False)
    prev_assets_total = group["assets_total"].shift(1)
    prev_net_margin = group["net_margin"].shift(1)
    prev_ocf = group["ocf"].shift(1)
    prev_operating_income = group["operating_income"].shift(1)

    panel["total_assets_growth"] = legacy.growth_ratio(
        panel["assets_total"],
        prev_assets_total,
        abs_base=False,
    )
    panel["net_margin_diff"] = panel["net_margin"] - prev_net_margin
    panel["is_2y_consecutive_ocf_deficit"] = (
        ((panel["ocf"] < 0) & (prev_ocf < 0)).fillna(False).astype(int)
    )
    panel["icr_under_1"] = (panel["interest_coverage_ratio"] < 1).astype(int)
    panel["is_2y_consecutive_operating_loss"] = (
        ((panel["operating_income"] < 0) & (prev_operating_income < 0)).fillna(False).astype(int)
    )

    unlabeled = panel.merge(macro, on="fiscal_year", how="inner")
    unlabeled["corp_name"] = unlabeled["corp_name_profile"]
    unlabeled = unlabeled.drop(columns=["corp_name_profile", "delisted_date"], errors="ignore")
    unlabeled["eval_year"] = unlabeled["fiscal_year"] + 1
    unlabeled = unlabeled.loc[unlabeled["fiscal_year"] == target_fiscal_year].copy()
    unlabeled = unlabeled.loc[
        unlabeled["listed_year"].isna() | (unlabeled["fiscal_year"] >= unlabeled["listed_year"])
    ].copy()
    unlabeled = unlabeled.sort_values(["market", "stock_code"]).reset_index(drop=True)
    return unlabeled


def to_feature_43_inference(unlabeled: pd.DataFrame) -> pd.DataFrame:
    master_columns = pd.read_csv(
        FEATURE_MASTER_PATH, nrows=1, encoding="utf-8-sig"
    ).columns.tolist()
    feature_spec = json.loads(FEATURE_SPEC_PATH.read_text(encoding="utf-8"))
    model_features = [str(name) for name in feature_spec["model_features"]]

    output = unlabeled.copy()
    for feature in model_features:
        if feature.startswith("market_"):
            category = feature.removeprefix("market_")
            output[feature] = (output["market"] == category).astype(int)
        elif feature.startswith("firm_size_group_"):
            category = feature.removeprefix("firm_size_group_")
            output[feature] = (output["firm_size_group"] == category).astype(int)
        elif feature.startswith("industry_macro_category_"):
            category = feature.removeprefix("industry_macro_category_")
            output[feature] = (output["industry_macro_category"] == category).astype(int)

    ordered_columns = [column for column in master_columns if column != "is_speculative"]
    for column in ordered_columns:
        if column not in output.columns:
            output[column] = pd.NA
    output = output.loc[:, ordered_columns].copy()
    return output


def main() -> None:
    args = parse_args()
    legacy = load_legacy_builder()
    unlabeled = build_unlabeled_feature_frame(legacy, target_fiscal_year=args.target_fiscal_year)
    inference = to_feature_43_inference(unlabeled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inference.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[Saved] {args.output} ({len(inference):,} rows)")
    print(
        inference[["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
