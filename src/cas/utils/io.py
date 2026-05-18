"""File I/O helpers — parquet/yaml/json with consistent defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict. (절대 경로 보정 완료)"""
    target_path = Path(path)

    # 만약 입력된 경로가 상대 경로라면, 프로젝트 최상위(git_36_v2)를 기준으로 절대 경로를 강제 생성합니다.
    if not target_path.is_absolute():
        # io.py 파일의 위치(src/cas/utils)를 역추적하여 부모의 부모의 부모인 최상위 폴더를 찾습니다.
        project_root = Path(__file__).resolve().parents[3]
        target_path = project_root / path

    with open(target_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Dump a dict to YAML with UTF-8 + Korean-safe settings."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def read_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dict. (절대 경로 보정 완료)"""
    target_path = Path(path)

    # 입력된 경로가 상대 경로라면, 프로젝트 최상위(git_36_v2)를 기준으로 절대 경로 강제 생성
    if not target_path.is_absolute():
        project_root = Path(__file__).resolve().parents[3]
        target_path = project_root / path

    with open(target_path, encoding="utf-8") as f:
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
