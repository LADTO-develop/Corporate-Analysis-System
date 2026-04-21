"""Walk-forward backtesting driver.

Runs the full train/predict cycle across every fold yielded by
``data.splitters.walk_forward_folds`` and returns OOF predictions plus
per-fold metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bfd.data.splitters import WalkForwardFold, slice_by_years, walk_forward_folds
from bfd.models.ensemble import StackingEnsemble
from bfd.utils.logging import get_logger
from bfd.validation.leakage import run_all_checks
from bfd.validation.metrics import compute_all_metrics

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Container for the outputs of a single backtest run."""

    oof_predictions: pd.DataFrame
    per_fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    overall_metrics: dict[str, Any] | None = None


def run_backtest(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    *,
    train_window: int,
    val_window: int = 1,
    step: int = 1,
    min_train_year: int | None = None,
    year_col: str = "fiscal_year",
    target_col: str = "target",
    key_cols: tuple[str, ...] = ("corp_code", "fiscal_year"),
) -> BacktestResult:
    """Run walk-forward CV and collect OOF predictions + per-fold metrics."""
    run_all_checks(dataset)

    years = sorted(dataset[year_col].unique().tolist())
    folds = list(
        walk_forward_folds(
            years,
            train_window=train_window,
            val_window=val_window,
            step=step,
            min_train_year=min_train_year,
        )
    )
    if not folds:
        raise ValueError("No folds produced — check train_window vs year range.")

    all_oof: list[pd.DataFrame] = []
    per_fold: list[dict[str, Any]] = []

    for fold in folds:
        fold_oof, fold_metrics = _run_fold(
            dataset=dataset,
            fold=fold,
            feature_cols=feature_cols,
            year_col=year_col,
            target_col=target_col,
            key_cols=key_cols,
        )
        all_oof.append(fold_oof)
        per_fold.append(fold_metrics)
        logger.info("fold_done", fold=fold.fold_index, **fold_metrics)

    oof = pd.concat(all_oof, ignore_index=True)

    overall: dict[str, Any] | None = None
    if len(oof) > 0:
        overall = compute_all_metrics(
            y_true=oof[target_col].values,
            p_score=oof["p_ensemble"].values,
        )

    return BacktestResult(oof_predictions=oof, per_fold_metrics=per_fold, overall_metrics=overall)


def _run_fold(
    *,
    dataset: pd.DataFrame,
    fold: WalkForwardFold,
    feature_cols: list[str],
    year_col: str,
    target_col: str,
    key_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train on fold.train_years, predict on fold.val_years, return OOF + metrics."""
    train_df = slice_by_years(dataset, fold.train_years, year_col=year_col)
    val_df = slice_by_years(dataset, fold.val_years, year_col=year_col)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    # Fit all bases on this fold's train, predict OOF on its val.
    ensemble = StackingEnsemble()
    ensemble.fit_bases(X_train, y_train, eval_set=(X_val, y_val))
    base_probs = ensemble.predict_proba_bases(X_val)

    p1_cols = {f"p_{name}": probs[:, 1] for name, probs in base_probs.items()}

    # For the first fold we don't yet have OOF-based meta-training data, so
    # use weighted-average fallback. Downstream meta-training aggregates OOF
    # across folds (see `train_market_model.py`).
    stacked_p1 = np.mean(np.column_stack(list(p1_cols.values())), axis=1)

    fold_oof = val_df[[*key_cols, year_col, target_col]].copy()
    for col, values in p1_cols.items():
        fold_oof[col] = values
    fold_oof["p_ensemble"] = stacked_p1
    fold_oof["fold_index"] = fold.fold_index

    metrics = compute_all_metrics(y_true=y_val.values, p_score=stacked_p1)
    metrics.update(
        {
            "fold_index": fold.fold_index,
            "train_years": list(fold.train_years),
            "val_years": list(fold.val_years),
        }
    )
    return fold_oof, metrics
