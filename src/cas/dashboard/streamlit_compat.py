"""Small Streamlit compatibility helpers for dashboard rendering."""

from __future__ import annotations

import streamlit as st


def _streamlit_supports_width_kwarg() -> bool:
    """Return whether Streamlit expects the newer width keyword."""
    version_text = str(getattr(st, "__version__", "0.0"))
    version_core = version_text.split("+", 1)[0].split("-", 1)[0]
    parts = version_core.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except (IndexError, ValueError):
        return False
    return (major, minor) >= (1, 57)


def _stretch_width_kwargs() -> dict[str, object]:
    """Build a full-width kwargs bundle for both old and new Streamlit APIs."""
    if _streamlit_supports_width_kwarg():
        return {"width": "stretch"}
    return {"use_container_width": True}


def stretch_altair_chart(chart: object) -> None:
    """Render an Altair chart at full container width."""
    vars(st)["altair_chart"](chart, **_stretch_width_kwargs())


def stretch_dataframe(data: object, **kwargs: object) -> None:
    """Render a dataframe at full container width."""
    vars(st)["dataframe"](data, **kwargs, **_stretch_width_kwargs())


def stretch_download_button(label: str | None = None, **kwargs: object) -> bool:
    """Render a download button at full container width."""
    if label is not None and "label" not in kwargs:
        kwargs["label"] = label
    return bool(vars(st)["download_button"](**kwargs, **_stretch_width_kwargs()))
