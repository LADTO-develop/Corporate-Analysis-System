from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from cas.macro.context_builder import build_macro_market_context, write_macro_market_context
from cas.macro.market_agent import evaluate_macro_market_context

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = ROOT / "configs" / "agent" / "ecos_indicator_registry.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "interim" / "macro_market"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch latest ECOS macro data and run MacroMarketAgent."
    )
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env",
        help="Dotenv file containing ECOS_API_KEY. Defaults to project .env.",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print the context and agent output without writing artifact files.",
    )
    parser.add_argument(
        "--as-of-date",
        type=_parse_date,
        default=None,
        help="Optional YYYY-MM-DD collection date. Defaults to today's date.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    api_key = os.getenv("ECOS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ECOS_API_KEY is required. Add it to .env or the process environment.")

    context = build_macro_market_context(
        registry_path=args.registry_path,
        api_key=api_key,
        as_of_date=args.as_of_date,
    )
    agent_output = evaluate_macro_market_context(context)
    if args.stdout_only:
        _print_json(
            {
                "macro_market_context": context.model_dump(mode="json"),
                "macro_market_agent_output": agent_output.model_dump(mode="json"),
            }
        )
        return

    context_paths = write_macro_market_context(context, output_dir=args.output_dir)
    agent_output_path = args.output_dir / "macro_market_agent_output.json"
    _write_json(agent_output_path, agent_output.model_dump(mode="json"))

    artifacts = {
        **{key: str(value) for key, value in context_paths.items()},
        "macro_market_agent_output": str(agent_output_path),
    }
    _print_json(artifacts)


def _parse_date(raw_value: str) -> date:
    return date.fromisoformat(raw_value)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
