"""Convenience launcher for the TS2000 Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Run the TS2000 dashboard with Streamlit if available."""
    try:
        from streamlit.web import cli as stcli
    except ImportError as error:
        message = (
            "Streamlit is not installed in the current environment. "
            "Install it first, then rerun this script.\n"
            "Recommended command: python -m pip install streamlit"
        )
        raise SystemExit(message) from error

    app_path = (
        Path(__file__).resolve().parents[1] / "src" / "cas" / "dashboard" / "ts2000_app.py"
    )
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
