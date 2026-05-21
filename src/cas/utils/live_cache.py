"""Small JSON cache helpers for live Stage 2 API calls."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "outputs" / "cache" / "stage2"
_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_VOLATILE_KEYS = {
    "cache_hit",
    "cache_key",
    "cache_path",
    "fetched_at",
    "generated_at",
    "run_id",
}


def live_cache_enabled(
    env_var: str = "CAS_STAGE2_CACHE_ENABLED",
    *,
    default: bool = True,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the live cache should be used for an API boundary."""
    source = os.environ if env is None else env
    value = source.get(env_var)
    if value is not None:
        return value.strip().lower() in _TRUE_VALUES
    shared_value = source.get("CAS_STAGE2_CACHE_ENABLED")
    if shared_value is not None:
        return shared_value.strip().lower() in _TRUE_VALUES
    if _running_pytest(source):
        return False
    return default


def live_cache_refresh(env: Mapping[str, str] | None = None) -> bool:
    """Return whether cache reads should be bypassed for the current run."""
    source = os.environ if env is None else env
    value = source.get("CAS_STAGE2_CACHE_REFRESH", "")
    return value.strip().lower() in _TRUE_VALUES


def live_cache_dir(env: Mapping[str, str] | None = None) -> Path:
    """Return the live cache directory, resolving relative paths from repo root."""
    source = os.environ if env is None else env
    configured = source.get("CAS_STAGE2_CACHE_DIR")
    if not configured:
        return _DEFAULT_CACHE_DIR
    path = Path(configured).expanduser()
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def stable_cache_key(payload: Mapping[str, object]) -> str:
    """Return a stable SHA-256 key for JSON-like payloads."""
    normalized = strip_cache_metadata(payload)
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json_cache(
    namespace: str,
    key: str,
    *,
    env_var: str = "CAS_STAGE2_CACHE_ENABLED",
    default: bool = True,
    env: Mapping[str, str] | None = None,
) -> dict[str, object] | None:
    """Read a cached JSON payload if caching is enabled and the file exists."""
    if live_cache_refresh(env) or not live_cache_enabled(env_var, default=default, env=env):
        return None
    path = _cache_path(namespace=namespace, key=key, env=env)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def write_json_cache(
    namespace: str,
    key: str,
    payload: Mapping[str, object],
    *,
    env_var: str = "CAS_STAGE2_CACHE_ENABLED",
    default: bool = True,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Write a JSON cache payload atomically when caching is enabled."""
    if not live_cache_enabled(env_var, default=default, env=env):
        return None
    path = _cache_path(namespace=namespace, key=key, env=env)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError:
        return None
    return path


def strip_cache_metadata(value: object) -> object:
    """Remove volatile cache/runtime fields before hashing request payloads."""
    if isinstance(value, Mapping):
        return {
            str(key): strip_cache_metadata(nested_value)
            for key, nested_value in value.items()
            if str(key) not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [strip_cache_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [strip_cache_metadata(item) for item in value]
    return value


def _cache_path(
    *,
    namespace: str,
    key: str,
    env: Mapping[str, str] | None,
) -> Path:
    safe_namespace = namespace.replace("/", "_").replace("..", "_")
    return live_cache_dir(env) / safe_namespace / f"{key}.json"


def _running_pytest(env: Mapping[str, str]) -> bool:
    if env.get("PYTEST_CURRENT_TEST"):
        return True
    return any("pytest" in arg for arg in sys.argv)
