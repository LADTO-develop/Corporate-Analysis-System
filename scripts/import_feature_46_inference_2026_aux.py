from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT.parent / "01_Raw_Data"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "raw" / "ts2000" / "feature_46_inference_2026_aux.csv"

KEY_COLUMNS = ["market", "stock_code", "fiscal_year"]
PROFILE_FIRM_SIZE_MAP = {
    "대기업": "large",
    "중견기업": "mid_sized",
    "중소기업": "small_and_medium",
    "기타": "other",
}
PROFILE_FIRM_SIZE_CODE_MAP = {
    "10": "large",
    "20": "small_and_medium",
    "30": "mid_sized",
    "90": "other",
}
MARKETS = ("KOSPI", "KOSDAQ")

PROFILE_COLUMNS = {
    "corp_name": "회사명",
    "stock_code": "거래소코드",
    "fiscal_year": "회계년도",
    "firm_size_code": "기업규모코드",
    "firm_size_name": "기업규모명",
}
BS_COLUMNS = {
    "stock_code": "거래소코드",
    "fiscal_year": "회계년도",
    "equity_total_source": "자본(*)(IFRS연결)(천원)",
}
MARKET_COLUMNS = {
    "stock_code": "거래소코드",
    "fiscal_year": "회계년도",
    "shares_outstanding_source": "발행주식의 총수 (현재 발행한 주식수 - 현재 감소한 주식수)(주)",
    "close_price_source": "종가(원)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import the minimal 2025 profile/market fields needed to repair the "
            "CAS-internal 2026 inference feature table."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-fiscal-year", type=int, default=2025)
    return parser.parse_args()


def normalize_stock_code(series: pd.Series) -> pd.Series:
    text = (
        series.astype("string")
        .fillna("")
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
    )
    return text.where(text == "", text.str.zfill(6))


def parse_year(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def parse_month(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"/(\d{1,2})", expand=False)
    month = pd.to_numeric(extracted, errors="coerce").astype("Int64")
    return month.fillna(12)


def parse_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .fillna("")
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_profile(source_dir: Path, target_fiscal_year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for market in MARKETS:
        path = source_dir / f"{market}_Profile.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"거래소코드": "string"})
        frame["market"] = market
        frame["stock_code"] = normalize_stock_code(frame[PROFILE_COLUMNS["stock_code"]])
        frame["fiscal_month"] = parse_month(frame[PROFILE_COLUMNS["fiscal_year"]])
        frame["fiscal_year"] = parse_year(frame[PROFILE_COLUMNS["fiscal_year"]])
        frame["corp_name_source"] = (
            frame[PROFILE_COLUMNS["corp_name"]].astype("string").fillna("").str.strip()
        )
        frame["firm_size_name_source"] = (
            frame[PROFILE_COLUMNS["firm_size_name"]].astype("string").fillna("").str.strip()
        )
        frame["firm_size_code_source"] = (
            frame[PROFILE_COLUMNS["firm_size_code"]]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.replace(r"\.0+$", "", regex=True)
        )
        group_from_code = frame["firm_size_code_source"].map(PROFILE_FIRM_SIZE_CODE_MAP)
        group_from_name = frame["firm_size_name_source"].map(PROFILE_FIRM_SIZE_MAP)
        frame["firm_size_group_raw"] = group_from_code.fillna(group_from_name).fillna("other")
        frames.append(
            frame[
                [
                    "market",
                    "stock_code",
                    "fiscal_year",
                    "fiscal_month",
                    "corp_name_source",
                    "firm_size_code_source",
                    "firm_size_name_source",
                    "firm_size_group_raw",
                ]
            ]
        )

    profile = pd.concat(frames, ignore_index=True)
    latest_profile = (
        profile.sort_values([*KEY_COLUMNS, "fiscal_month"])
        .drop_duplicates(KEY_COLUMNS, keep="last")
        .drop(columns=["fiscal_month"])
    )
    target_profile = latest_profile.loc[latest_profile["fiscal_year"].eq(target_fiscal_year)].copy()

    non_other_history = latest_profile.loc[
        latest_profile["firm_size_group_raw"].isin(["large", "mid_sized", "small_and_medium"])
    ].copy()
    latest_non_other = (
        non_other_history.sort_values(["market", "stock_code", "fiscal_year"])
        .groupby(["market", "stock_code"], as_index=False)
        .tail(1)[["market", "stock_code", "firm_size_group_raw", "fiscal_year"]]
        .rename(
            columns={
                "firm_size_group_raw": "firm_size_group_source",
                "fiscal_year": "firm_size_source_year",
            }
        )
    )

    target_profile = target_profile.merge(
        latest_non_other,
        on=["market", "stock_code"],
        how="left",
    )
    target_profile["firm_size_group_source"] = target_profile["firm_size_group_source"].fillna(
        target_profile["firm_size_group_raw"]
    )
    return target_profile.drop(columns=["firm_size_group_raw"])


def load_balance_sheet(source_dir: Path, target_fiscal_year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for market in MARKETS:
        path = source_dir / f"{market}_BS.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"거래소코드": "string"})
        frame["market"] = market
        frame["stock_code"] = normalize_stock_code(frame[BS_COLUMNS["stock_code"]])
        frame["fiscal_month"] = parse_month(frame[BS_COLUMNS["fiscal_year"]])
        frame["fiscal_year"] = parse_year(frame[BS_COLUMNS["fiscal_year"]])
        frame["equity_total_source"] = parse_numeric(frame[BS_COLUMNS["equity_total_source"]])
        frames.append(
            frame[
                [
                    "market",
                    "stock_code",
                    "fiscal_year",
                    "fiscal_month",
                    "equity_total_source",
                ]
            ]
        )

    balance_sheet = pd.concat(frames, ignore_index=True)
    return (
        balance_sheet.loc[balance_sheet["fiscal_year"].eq(target_fiscal_year)]
        .sort_values([*KEY_COLUMNS, "fiscal_month"])
        .drop_duplicates(KEY_COLUMNS, keep="last")
        .drop(columns=["fiscal_month"])
    )


def load_market(source_dir: Path, target_fiscal_year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for market in MARKETS:
        path = source_dir / f"{market}_market_and_div.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"거래소코드": "string"})
        frame["market"] = market
        frame["stock_code"] = normalize_stock_code(frame[MARKET_COLUMNS["stock_code"]])
        frame["fiscal_year"] = parse_year(frame[MARKET_COLUMNS["fiscal_year"]])
        frame["shares_outstanding_source"] = parse_numeric(
            frame[MARKET_COLUMNS["shares_outstanding_source"]]
        )
        frame["close_price_source"] = parse_numeric(frame[MARKET_COLUMNS["close_price_source"]])
        frames.append(
            frame[
                [
                    *KEY_COLUMNS,
                    "shares_outstanding_source",
                    "close_price_source",
                ]
            ]
        )

    market_frame = pd.concat(frames, ignore_index=True)
    target_market = market_frame.loc[market_frame["fiscal_year"].eq(target_fiscal_year)]
    return (
        target_market.groupby(KEY_COLUMNS, as_index=False)
        .agg(
            shares_outstanding_source=("shares_outstanding_source", "max"),
            close_price_source=("close_price_source", "max"),
        )
        .sort_values(KEY_COLUMNS)
    )


def build_auxiliary_frame(source_dir: Path, target_fiscal_year: int) -> pd.DataFrame:
    profile = load_profile(source_dir, target_fiscal_year)
    balance_sheet = load_balance_sheet(source_dir, target_fiscal_year)
    market = load_market(source_dir, target_fiscal_year)

    auxiliary = (
        profile.merge(balance_sheet, on=KEY_COLUMNS, how="left")
        .merge(market, on=KEY_COLUMNS, how="left")
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    auxiliary["market_cap_source"] = (
        auxiliary["shares_outstanding_source"] * auxiliary["close_price_source"]
    )
    valid_denominator = auxiliary["equity_total_source"].notna() & auxiliary[
        "equity_total_source"
    ].ne(0)
    auxiliary["market_to_book_source"] = np.nan
    auxiliary.loc[valid_denominator, "market_to_book_source"] = auxiliary.loc[
        valid_denominator, "market_cap_source"
    ] / (auxiliary.loc[valid_denominator, "equity_total_source"] * 1000.0)
    duplicates = int(auxiliary.duplicated(KEY_COLUMNS).sum())
    if duplicates:
        raise ValueError(f"Auxiliary inference source has duplicate key rows: {duplicates}")
    return auxiliary[
        [
            *KEY_COLUMNS,
            "corp_name_source",
            "firm_size_code_source",
            "firm_size_name_source",
            "firm_size_group_source",
            "firm_size_source_year",
            "equity_total_source",
            "shares_outstanding_source",
            "close_price_source",
            "market_cap_source",
            "market_to_book_source",
        ]
    ]


def main() -> None:
    args = parse_args()
    auxiliary = build_auxiliary_frame(args.source_dir, args.target_fiscal_year)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    auxiliary.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[Saved] {args.output} ({len(auxiliary):,} rows)")
    print(
        auxiliary[
            [
                "market",
                "stock_code",
                "corp_name_source",
                "fiscal_year",
                "firm_size_group_source",
                "market_to_book_source",
            ]
        ]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
