"""CLI entrypoint for the Corporate Analysis System."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from cas.agents.graph import run_once
from cas.utils.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single analysis run."""
    parser = argparse.ArgumentParser(
        description="Run the corporate investment suitability pipeline for one company."
    )
    parser.add_argument("--company-id", required=True, help="Profile file name under data/input/companies/")
    parser.add_argument("--analysis-year", type=int, default=None)
    parser.add_argument("--market", default=None, help="Optional override for the market label.")
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--graph-config", default="configs/agent/graph.yaml")
    return parser.parse_args()


def main() -> None:
    """Execute the LangGraph pipeline and print output artifact paths."""
    load_dotenv()
    configure_logging()
    logger = get_logger(__name__)
    args = parse_args()

    logger.info(
        "agent_run_start",
        company_id=args.company_id,
        market=args.market,
        analysis_year=args.analysis_year,
    )

    final_state = run_once(
        company_id=args.company_id,
        market=args.market,
        analysis_year=args.analysis_year,
        config_path=args.graph_config,
        thread_id=args.thread_id,
    )

    artifacts = final_state.get("artifacts", {}) if isinstance(final_state, dict) else {}
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))

    md_path = artifacts.get("report_md")
    if md_path and Path(md_path).exists():
        print("\n" + "=" * 80)
        print(Path(md_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
