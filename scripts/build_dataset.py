"""Build the supervised training dataset from raw TS2000 + ratings files.

Usage:
    python -m scripts.build_dataset
    bfd-build-dataset                           # installed entry point
    bfd-build-dataset --markets KOSPI KOSDAQ --start-year 2015 --end-year 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bfd.data.alignment import filter_to_annual_consolidated
from bfd.data.loaders.ratings import RatingsLoader
from bfd.data.loaders.ts2000 import TS2000Loader
from bfd.data.splitters import map_financials_to_next_year_rating
from bfd.ratings.targets import add_binary_target, aggregate_ratings_per_firm_year
from bfd.utils.io import write_parquet
from bfd.utils.logging import configure_logging, get_logger
from bfd.utils.seeds import set_seeds
from bfd.validation.leakage import run_all_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the BFD supervised dataset.")
    parser.add_argument("--ts2000-config", default="configs/data/ts2000.yaml")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument(
        "--markets",
        nargs="+",
        default=["KOSPI", "KOSDAQ"],
        help="Markets to include (space-separated).",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed",
        help="Root under which per-market parquet files are written.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    set_seeds()
    logger = get_logger(__name__)
    args = parse_args()

    # --- 1. Financials ---------------------------------------------------
    ts2000 = TS2000Loader(args.ts2000_config)
    wide = ts2000.load_range(
        start_year=args.start_year,
        end_year=args.end_year,
        markets=args.markets,
    )
    wide = filter_to_annual_consolidated(wide)
    logger.info("financials_loaded", n_rows=len(wide))

    # --- 2. Ratings ------------------------------------------------------
    ratings = RatingsLoader().load_all()
    ratings_per_firm_year = aggregate_ratings_per_firm_year(ratings)
    logger.info("ratings_aggregated", n_rows=len(ratings_per_firm_year))

    # --- 3. t → t+1 mapping + binary target -----------------------------
    supervised = map_financials_to_next_year_rating(
        financials=wide,
        ratings=ratings_per_firm_year,
    )
    supervised = add_binary_target(supervised)
    run_all_checks(supervised)
    logger.info("supervised_built", n_rows=len(supervised), pos_rate=float(supervised["target"].mean()))

    # --- 4. Write per-market parquet ------------------------------------
    root = Path(args.output_root)
    for market in args.markets:
        subset = supervised.loc[supervised["market"] == market]
        out_path = root / market.lower() / "supervised.parquet"
        write_parquet(subset, out_path)
        logger.info("market_subset_written", market=market, n_rows=len(subset), path=str(out_path))


if __name__ == "__main__":
    main()
