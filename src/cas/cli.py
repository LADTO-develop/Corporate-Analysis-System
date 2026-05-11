"""CLI entrypoint for the Corporate Analysis System."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

from cas.agents.graph import run_once
from cas.utils.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single analysis run."""
    parser = argparse.ArgumentParser(
        description="Run the corporate investment suitability pipeline for one company."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--company-id",
        help="Profile file name under data/input/companies/",
    )
    input_group.add_argument(
        "--company-selection-json",
        help="Raw company_selection JSON payload from a web listing.",
    )
    input_group.add_argument(
        "--company-selection-file",
        type=Path,
        help="Path to a company_selection JSON payload file.",
    )
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
    company_selection = _load_company_selection(args)

    final_state = run_once(
        company_id=args.company_id,
        company_selection=company_selection,
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


def _load_company_selection(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.company_selection_json:
        return _parse_selection_payload(args.company_selection_json)
    if args.company_selection_file:
        return _parse_selection_payload(args.company_selection_file.read_text(encoding="utf-8"))
    return None


def _parse_selection_payload(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("company_selection payload must be a JSON object")
    return cast(dict[str, Any], payload)


if __name__ == "__main__":
    main()
