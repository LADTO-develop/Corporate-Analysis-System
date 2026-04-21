"""Loader for the local news/disclosure corpus.

Expected layout under ``data/raw/news/``::

    data/raw/news/
      ├── 005930/                 # corp_code
      │   ├── 2024-03-15_실적발표.txt
      │   └── 2024-04-02_공시.txt
      └── 000660/
          └── ...

Each file is a plain-text news article or disclosure. Filenames encode
``{YYYY-MM-DD}_{title}.txt`` for downstream chunking and dating.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from bfd.utils.io import read_yaml
from bfd.utils.logging import get_logger

logger = get_logger(__name__)

_FILENAME_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<title>.+)\.txt$")


class NewsCorpusLoader:
    """Walks ``data/raw/news/`` and returns a tabular representation."""

    def __init__(self, config_path: str | Path = "configs/data/news.yaml") -> None:
        self.config = read_yaml(config_path)
        self.raw_path = Path(self.config["corpus"]["raw_path"])

    def load_firm_corpus(self, corp_code: str) -> pd.DataFrame:
        """Return a DataFrame of news articles for a single firm."""
        firm_dir = self.raw_path / corp_code
        if not firm_dir.exists():
            logger.warning("news_firm_dir_missing", corp_code=corp_code, path=str(firm_dir))
            return pd.DataFrame(columns=["corp_code", "article_date", "title", "text"])

        rows: list[dict[str, object]] = []
        for txt_file in sorted(firm_dir.glob("*.txt")):
            parsed = _parse_filename(txt_file.name)
            if parsed is None:
                logger.debug("news_filename_skip", path=str(txt_file))
                continue
            article_date, title = parsed
            rows.append(
                {
                    "corp_code": corp_code,
                    "article_date": article_date,
                    "title": title,
                    "text": txt_file.read_text(encoding="utf-8"),
                    "source_path": str(txt_file),
                }
            )
        return pd.DataFrame(rows)

    def load_all(self) -> pd.DataFrame:
        """Return one DataFrame covering every firm directory found."""
        if not self.raw_path.exists():
            return pd.DataFrame(columns=["corp_code", "article_date", "title", "text"])

        frames: list[pd.DataFrame] = []
        for firm_dir in self.raw_path.iterdir():
            if firm_dir.is_dir():
                frames.append(self.load_firm_corpus(firm_dir.name))
        if not frames:
            return pd.DataFrame(columns=["corp_code", "article_date", "title", "text"])
        return pd.concat(frames, ignore_index=True)


def _parse_filename(name: str) -> tuple[date, str] | None:
    """Extract ``(date, title)`` from a filename of form ``YYYY-MM-DD_title.txt``."""
    m = _FILENAME_RE.match(name)
    if m is None:
        return None
    parts = m.group("date").split("-")
    return date(int(parts[0]), int(parts[1]), int(parts[2])), m.group("title")
