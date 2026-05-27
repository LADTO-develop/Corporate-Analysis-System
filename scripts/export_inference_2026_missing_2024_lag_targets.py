from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from apply_opendart_financial_supplements import normalize_stock_code

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INFERENCE_PATH = (
    ROOT / "data" / "input" / "credit_46_features" / "feature_46_inference_2026.csv"
)
DEFAULT_HISTORY_PATH = ROOT / "data" / "raw" / "ts2000" / "TS2000_Credit_Model_Dataset_Model_V1.csv"
DEFAULT_OUTPUT_PATH = (
    ROOT / "data" / "raw" / "opendart" / "inference_2026_missing_2024_lag_targets.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export 2026 inference companies whose 2024 fiscal-year row is missing "
            "from Model_V1. The output is used as the OpenDART 2024 lag collection target."
        )
    )
    parser.add_argument("--inference", type=Path, default=DEFAULT_INFERENCE_PATH)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--lag-fiscal-year", type=int, default=2024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference = pd.read_csv(args.inference, encoding="utf-8-sig", dtype={"stock_code": "string"})
    history = pd.read_csv(
        args.history, encoding="utf-8-sig", low_memory=False, dtype={"stock_code": "string"}
    )

    history_year = history.loc[
        pd.to_numeric(history["fiscal_year"], errors="coerce").eq(args.lag_fiscal_year),
        ["market", "stock_code"],
    ].copy()
    existing_keys = set(
        zip(
            history_year["market"],
            history_year["stock_code"].map(normalize_stock_code),
            strict=False,
        )
    )

    targets = inference.copy()
    targets["_stock_code_key"] = targets["stock_code"].map(normalize_stock_code)
    missing_lag_mask = [
        (market, stock_code) not in existing_keys
        for market, stock_code in zip(targets["market"], targets["_stock_code_key"], strict=False)
    ]
    targets = targets.loc[missing_lag_mask].drop(columns=["_stock_code_key"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        "[Saved] "
        f"{args.output} ({len(targets):,}/{len(inference):,} rows missing "
        f"{args.lag_fiscal_year} lag in Model_V1)"
    )


if __name__ == "__main__":
    main()
