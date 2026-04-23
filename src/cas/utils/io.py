"""File I/O helpers — parquet/yaml/json with consistent defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Dump a dict to YAML with UTF-8 + Korean-safe settings."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def read_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path, *, indent: int = 2) -> None:
    """Dump a Python object to JSON, UTF-8 + non-ASCII preserved."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=str)


def read_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a parquet file via pyarrow."""
    return pd.read_parquet(path, engine="pyarrow", **kwargs)


def write_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    compression: str = "snappy",
    **kwargs: Any,
) -> None:
    """Write a DataFrame to parquet via pyarrow."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression=compression, index=False, **kwargs)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
