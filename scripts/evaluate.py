"""Evaluate a trained ensemble on a held-out parquet.

Usage:
    bfd-evaluate --market kospi --holdout-path data/processed/kospi/holdout.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bfd.models.ensemble import StackingEnsemble
from bfd.models.registry import ModelRegistry
from bfd.utils.io import ensure_dir, read_parquet
from bfd.utils.logging import configure_logging, get_logger
from bfd.validation.drift import drift_report
from bfd.validation.metrics import compute_all_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained BFD ensemble.")
    parser.add_argument("--market", choices=["kospi", "kosdaq"], required=True)
    parser.add_argument("--version", default=None, help="Artifact version (default: latest).")
    parser.add_argument("--holdout-path", required=True)
    parser.add_argument("--reference-path", default=None, help="Training parquet for drift baseline.")
    parser.add_argument("--output-root", default="data/processed/eval")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    logger = get_logger(__name__)
    args = parse_args()
    market = args.market.upper()

    registry = ModelRegistry()
    record = registry.get(market, args.version) if args.version else registry.latest(market)
    if record is None:
        raise RuntimeError(f"No artifact found for market={market}.")
    logger.info("artifact_found", version=record.version, path=record.path)

    ensemble = StackingEnsemble.load(record.path)
    holdout: pd.DataFrame = read_parquet(args.holdout_path)

    feature_cols = [c for c in ensemble._bases[0].model._feature_names or [] if c in holdout.columns]  # noqa: SLF001
    if not feature_cols:
        raise RuntimeError("No overlap between artifact's feature set and holdout columns.")

    X = holdout[feature_cols]
    y = holdout["target"].values
    p_ensemble = ensemble.predict_proba(X)[:, 1]

    metrics = compute_all_metrics(y_true=y, p_score=p_ensemble)
    logger.info("eval_metrics", **metrics)

    out_dir = ensure_dir(Path(args.output_root) / args.market / record.version)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    preds_df = holdout[["corp_code", "fiscal_year", "target"]].copy()
    preds_df["p_ensemble"] = p_ensemble
    preds_df.to_parquet(out_dir / "predictions.parquet", index=False)

    # Optional drift report
    if args.reference_path:
        reference = read_parquet(args.reference_path)
        drift = drift_report(reference[feature_cols], holdout[feature_cols])
        drift.to_csv(out_dir / "drift_report.csv", index=False)
        logger.info("drift_written", path=str(out_dir / "drift_report.csv"))


if __name__ == "__main__":
    main()
