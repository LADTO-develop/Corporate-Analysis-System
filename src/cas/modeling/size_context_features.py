"""Feature transforms that reduce absolute-scale sensitivity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def signed_log1p(values: pd.Series) -> pd.Series:
    """Return a sign-preserving log1p transform for amount-like values."""
    numeric = pd.to_numeric(values, errors="coerce")
    return np.sign(numeric) * np.log1p(np.abs(numeric))


def add_signed_log_features(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    prefix: str = "log",
) -> tuple[pd.DataFrame, list[str]]:
    """Add signed log features for existing numeric columns."""
    output = frame.copy()
    added_columns: list[str] = []
    for column in columns:
        if column not in output.columns:
            continue
        feature = f"{prefix}_{column}"
        output[feature] = signed_log1p(output[column])
        added_columns.append(feature)
    return output, added_columns


def add_group_percentile_features(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_columns: list[str],
    suffix: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Add within-group percentile ranks for numeric columns."""
    missing_groups = [column for column in group_columns if column not in frame.columns]
    if missing_groups:
        raise KeyError(f"Missing group columns for percentile features: {missing_groups}")

    output = frame.copy()
    added_columns: list[str] = []
    for column in value_columns:
        if column not in output.columns:
            continue
        feature = f"{column}_{suffix}_pct"
        values = pd.to_numeric(output[column], errors="coerce")
        ranking_frame = output.loc[:, group_columns].copy()
        ranking_frame["_value"] = values
        output[feature] = ranking_frame.groupby(group_columns, dropna=False)["_value"].rank(
            pct=True,
            method="average",
        )
        added_columns.append(feature)
    return output, added_columns


def add_group_zscore_features(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_columns: list[str],
    suffix: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Add within-group z-scores, using zero as neutral for single-row groups."""
    missing_groups = [column for column in group_columns if column not in frame.columns]
    if missing_groups:
        raise KeyError(f"Missing group columns for z-score features: {missing_groups}")

    output = frame.copy()
    added_columns: list[str] = []
    grouped = output.groupby(group_columns, dropna=False)
    for column in value_columns:
        if column not in output.columns:
            continue
        feature = f"{column}_{suffix}_zscore"
        values = pd.to_numeric(output[column], errors="coerce")
        mean = grouped[column].transform(
            lambda series: pd.to_numeric(series, errors="coerce").mean()
        )
        std = grouped[column].transform(lambda series: pd.to_numeric(series, errors="coerce").std())
        zscore = (values - mean) / std.replace(0.0, np.nan)
        zscore = zscore.mask(values.notna() & zscore.isna(), 0.0)
        output[feature] = zscore
        added_columns.append(feature)
    return output, added_columns


def add_binary_group_context_features(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    value_columns: list[str],
    suffix: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Add peer-rate and peer-deviation features for binary context flags."""
    missing_groups = [column for column in group_columns if column not in frame.columns]
    if missing_groups:
        raise KeyError(f"Missing group columns for binary context features: {missing_groups}")

    output = frame.copy()
    added_columns: list[str] = []
    grouped = output.groupby(group_columns, dropna=False)
    for column in value_columns:
        if column not in output.columns:
            continue
        values = pd.to_numeric(output[column], errors="coerce")
        rate_feature = f"{column}_{suffix}_rate"
        deviation_feature = f"{column}_{suffix}_deviation"
        peer_rate = grouped[column].transform(
            lambda series: pd.to_numeric(series, errors="coerce").mean()
        )
        output[rate_feature] = peer_rate
        output[deviation_feature] = values - peer_rate
        added_columns.extend([rate_feature, deviation_feature])
    return output, added_columns
