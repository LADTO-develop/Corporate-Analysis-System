"""Run the LangGraph agent pipeline for a single firm-year.

Usage:
    bfd-agent --corp-code 005930 --fiscal-year 2024
    bfd-agent --corp-code 000660 --fiscal-year 2024 --market KOSPI --thread-id run-1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bfd.agents.graph import run_once
from bfd.utils.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BFD agent for one firm-year.")
    parser.add_argument("--corp-code", required=True, help="KRX 6-digit 종목코드")
    parser.add_argument("--fiscal-year", type=int, required=True)
    parser.add_argument(
        "--market",
        choices=["KOSPI", "KOSDAQ"],
        default=None,
        help="If omitted, inferred from the TS2000 row.",
    )
    parser.add_argument("--thread-id", default=None)
    parser.add_argument(
        "--graph-config",
        default="configs/agent/graph.yaml",
    )
    return parser.parse_args()


def _infer_market(corp_code: str, fiscal_year: int) -> str:
    """Cheap market inference via a TS2000 probe. Falls back to KOSPI."""
    from bfd.data.loaders.ts2000 import TS2000Loader

    loader = TS2000Loader()
    try:
        wide = loader.load_wide(year=fiscal_year)
        row = wide.loc[wide["corp_code"] == corp_code]
        if not row.empty and "market" in row.columns:
            return str(row.iloc[0]["market"])
    except FileNotFoundError:
        pass
    return "KOSPI"


def main() -> None:
    configure_logging()
    logger = get_logger(__name__)
    args = parse_args()

    market = args.market or _infer_market(args.corp_code, args.fiscal_year)
    logger.info(
        "agent_run_start",
        corp_code=args.corp_code,
        market=market,
        fiscal_year=args.fiscal_year,
    )

    final_state = run_once(
        corp_code=args.corp_code,
        market=market,
        fiscal_year=args.fiscal_year,
        config_path=args.graph_config,
        thread_id=args.thread_id,
    )

    # Report has already been written by report_node; echo its path.
    artifacts = final_state.get("artifacts", {}) if isinstance(final_state, dict) else {}
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))

    md_path = artifacts.get("report_md")
    if md_path and Path(md_path).exists():
        print("\n" + "=" * 80)
        print(Path(md_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
