"""Train a market-specific ensemble using walk-forward CV.

Usage:
    bfd-train --market kospi
    bfd-train --market kosdaq --version v0.2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bfd.features.registry import REGISTRY
from bfd.models.ensemble import StackingEnsemble
from bfd.models.registry import ModelRegistry
from bfd.utils.io import ensure_dir, read_parquet, read_yaml, write_parquet
from bfd.utils.logging import configure_logging, get_logger
from bfd.utils.seeds import set_seeds
from bfd.validation.backtesting import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a market-specific ensemble.")
    parser.add_argument("--market", choices=["kospi", "kosdaq"], required=True)
    parser.add_argument("--version", default="v0.1.0")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override path to the supervised parquet (default: data/processed/{market}/supervised.parquet).",
    )
    parser.add_argument("--output-root", default="data/processed")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    set_seeds()
    logger = get_logger(__name__)
    args = parse_args()

    market_upper = args.market.upper()
    market_cfg = read_yaml(Path(f"configs/market/{args.market}.yaml"))
    cv_cfg = market_cfg["cv"]
    subset_name = market_cfg["feature_subset"]

    data_path = Path(
        args.data_path or f"{args.output_root}/{args.market}/supervised.parquet"
    )
    logger.info("load_dataset", path=str(data_path))
    dataset: pd.DataFrame = read_parquet(data_path)

    # Compute features — here we assume the parquet already has raw columns; a
    # real implementation would precompute features during build_dataset.py.
    feature_cols = [
        name
        for name in REGISTRY.names_for_subset(subset_name)
        if name in dataset.columns
    ]
    if not feature_cols:
        raise RuntimeError(
            f"No registered features of subset={subset_name!r} were found as columns in {data_path}. "
            "Run the feature-computation step before training."
        )
    logger.info("feature_cols_resolved", n=len(feature_cols))

    # --- Walk-forward backtest ------------------------------------------
    result = run_backtest(
        dataset=dataset,
        feature_cols=feature_cols,
        train_window=int(cv_cfg["train_years"]),
        val_window=int(cv_cfg.get("val_years", 1)),
        step=int(cv_cfg.get("step_years", 1)),
        min_train_year=cv_cfg.get("min_train_year"),
    )
    logger.info("backtest_done", overall=result.overall_metrics)

    # --- Save OOF predictions -------------------------------------------
    oof_path = Path(args.output_root) / args.market / "oof" / f"{args.version}.parquet"
    write_parquet(result.oof_predictions, oof_path)
    logger.info("oof_written", path=str(oof_path))

    # --- Fit final ensemble on the FULL dataset and register ------------
    ensemble = StackingEnsemble()
    X = dataset[feature_cols]
    y = dataset["target"]
    ensemble.fit_bases(X, y)
    if ensemble._strategy == "stacking":  # noqa: SLF001
        oof = result.oof_predictions
        meta_cols = [c for c in oof.columns if c.startswith("p_") and c != "p_ensemble"]
        ensemble.fit_meta_on_oof(oof[meta_cols], oof["target"])

    artifact_dir = Path(args.output_root) / args.market / "models" / args.version
    ensure_dir(artifact_dir)
    ensemble.save(artifact_dir)

    # --- Register artifact ----------------------------------------------
    ModelRegistry(root=args.output_root).register(
        market=market_upper,
        version=args.version,
        artifact_dir=str(artifact_dir),
        metrics=result.overall_metrics or {},
        feature_subset=subset_name,
        notes=f"Walk-forward with {len(result.per_fold_metrics)} folds.",
    )
    logger.info("artifact_registered", market=market_upper, version=args.version)


if __name__ == "__main__":
    main()
