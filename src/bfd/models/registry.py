"""Tracks trained model artifacts by market + version.

Artifacts land in ``data/processed/{market}/models/{version}/``. The registry
provides a thin index so the agent pipeline can load the latest artifact
without hard-coding a path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from bfd.utils.io import ensure_dir
from bfd.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ArtifactRecord:
    """A saved ensemble artifact and its provenance."""

    market: str
    version: str
    created_at: str
    path: str
    metrics: dict[str, float]
    feature_subset: str
    notes: str = ""


class ModelRegistry:
    """Filesystem-backed registry keyed on (market, version)."""

    def __init__(self, root: str | Path = "data/processed") -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------
    def _index_path(self, market: str) -> Path:
        return self.root / market.lower() / "models" / "index.json"

    def _read_index(self, market: str) -> list[dict[str, object]]:
        path = self._index_path(market)
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_index(self, market: str, records: list[dict[str, object]]) -> None:
        path = self._index_path(market)
        ensure_dir(path.parent)
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def register(
        self,
        *,
        market: str,
        version: str,
        artifact_dir: str | Path,
        metrics: dict[str, float],
        feature_subset: str,
        notes: str = "",
    ) -> ArtifactRecord:
        """Add a new artifact record to the index."""
        record = ArtifactRecord(
            market=market,
            version=version,
            created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            path=str(artifact_dir),
            metrics=metrics,
            feature_subset=feature_subset,
            notes=notes,
        )
        records = self._read_index(market)
        records.append(asdict(record))
        self._write_index(market, records)
        logger.info("artifact_registered", **asdict(record))
        return record

    def list(self, market: str) -> list[ArtifactRecord]:
        return [ArtifactRecord(**r) for r in self._read_index(market)]  # type: ignore[arg-type]

    def latest(self, market: str) -> ArtifactRecord | None:
        records = self.list(market)
        if not records:
            return None
        return sorted(records, key=lambda r: r.created_at)[-1]

    def get(self, market: str, version: str) -> ArtifactRecord:
        for r in self.list(market):
            if r.version == version:
                return r
        raise KeyError(f"No artifact {version} for market {market}")
