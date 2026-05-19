"""Compatibility wrapper for the package CLI."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Run the package CLI from a source checkout."""
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from cas.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
