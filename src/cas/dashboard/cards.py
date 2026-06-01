"""Reusable card and badge render helpers for the credit dashboard."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

COLOR_CARD_BG = "var(--cas-panel)"
COLOR_CARD_BORDER = "var(--cas-border)"
COLOR_CARD_LABEL = "var(--cas-muted)"
COLOR_CARD_VALUE = "var(--cas-text)"
CARD_SHADOW = "var(--cas-shadow)"


def style_direction_badge(value: object) -> str:
    """Style feature interpretation cells with CAS theme variables."""
    text = str(value or "").strip()

    if "높을수록 대체로 긍정" in text or "긍정" in text:
        return (
            "background-color: var(--cas-success-soft); "
            "color: var(--cas-text); "
            "border: 1px solid var(--cas-success-border); "
            "font-weight: 700;"
        )

    if "낮을수록 대체로 긍정" in text:
        return (
            "background-color: var(--cas-warning-soft); "
            "color: var(--cas-text); "
            "border: 1px solid var(--cas-warning-border); "
            "font-weight: 700;"
        )

    if "위험" in text or "부정" in text:
        return (
            "background-color: var(--cas-risk-soft); "
            "color: var(--cas-text); "
            "border: 1px solid var(--cas-risk-border); "
            "font-weight: 700;"
        )

    return (
        "background-color: var(--cas-neutral-soft); "
        "color: var(--cas-text); "
        "border: 1px solid var(--cas-neutral-border); "
        "font-weight: 700;"
    )


def render_direction_badge_html(value: object) -> str:
    """Render an interpretation direction badge as HTML."""
    text = str(value)
    style = style_direction_badge(text)
    return f"<span style='{style}'>{escape(text)}</span>"


def render_risk_band_badge(risk_band: object) -> str:
    """Render a colored HTML badge for the risk band."""
    label = str(risk_band) if pd.notna(risk_band) else "-"
    style_map = {
        "안정": {
            "bg": "var(--cas-success-soft)",
            "fg": "var(--cas-success)",
            "border": "var(--cas-success-border)",
        },
        "관찰": {
            "bg": "var(--cas-warning-soft)",
            "fg": "var(--cas-warning)",
            "border": "var(--cas-warning-border)",
        },
        "고위험": {
            "bg": "var(--cas-risk-soft)",
            "fg": "var(--cas-risk)",
            "border": "var(--cas-risk-border)",
        },
    }
    style = style_map.get(
        label,
        {
            "bg": "var(--cas-neutral-soft)",
            "fg": "var(--cas-text)",
            "border": "var(--cas-neutral-border)",
        },
    )
    return (
        f"<div style='display:inline-block;padding:0.45rem 0.8rem;border-radius:999px;"
        f"background:{style['bg']};color:{style['fg']};border:1px solid {style['border']};"
        "font-weight:700;font-size:0.95rem;'>"
        f"{label}</div>"
    )


def render_decision_badge(label: object, *, muted: bool = False) -> str:
    """Render a colored badge for model and committee decisions."""
    text = str(label) if pd.notna(label) else "-"
    style_map = {
        "적격": {
            "bg": "var(--cas-success-soft)",
            "fg": "var(--cas-success)",
            "border": "var(--cas-success-border)",
        },
        "투자적격": {
            "bg": "var(--cas-success-soft)",
            "fg": "var(--cas-success)",
            "border": "var(--cas-success-border)",
        },
        "관찰": {
            "bg": "var(--cas-neutral-soft)",
            "fg": "var(--cas-text)",
            "border": "var(--cas-neutral-border)",
        },
        "경계 관찰": {
            "bg": "var(--cas-warning-soft)",
            "fg": "var(--cas-warning)",
            "border": "var(--cas-warning-border)",
        },
        "확인 필요": {
            "bg": "var(--cas-warning-soft)",
            "fg": "var(--cas-warning)",
            "border": "var(--cas-warning-border)",
        },
        "위험 주의": {
            "bg": "var(--cas-risk-soft)",
            "fg": "var(--cas-risk)",
            "border": "var(--cas-risk-border)",
        },
        "과민경고 완화": {
            "bg": "var(--cas-success-soft)",
            "fg": "var(--cas-success)",
            "border": "var(--cas-success-border)",
        },
        "경계등급 확인": {
            "bg": "var(--cas-warning-soft)",
            "fg": "var(--cas-warning)",
            "border": "var(--cas-warning-border)",
        },
        "근거 추가 확인": {
            "bg": "var(--cas-warning-soft)",
            "fg": "var(--cas-warning)",
            "border": "var(--cas-warning-border)",
        },
        "위험 신호 확인": {
            "bg": "var(--cas-risk-soft)",
            "fg": "var(--cas-risk)",
            "border": "var(--cas-risk-border)",
        },
        "보류": {
            "bg": "var(--cas-warning-soft)",
            "fg": "var(--cas-warning)",
            "border": "var(--cas-warning-border)",
        },
        "위험 보류": {
            "bg": "var(--cas-risk-soft)",
            "fg": "var(--cas-risk)",
            "border": "var(--cas-risk-border)",
        },
        "경계등급 보류": {
            "bg": "var(--cas-risk-soft)",
            "fg": "var(--cas-risk)",
            "border": "var(--cas-risk-border)",
        },
        "과민경고 완화 보류": {
            "bg": "var(--cas-success-soft)",
            "fg": "var(--cas-success)",
            "border": "var(--cas-success-border)",
        },
        "확인필요 보류": {
            "bg": "var(--cas-warning-soft)",
            "fg": "var(--cas-warning)",
            "border": "var(--cas-warning-border)",
        },
        "위험신호 있음": {
            "bg": "var(--cas-risk-soft)",
            "fg": "var(--cas-risk)",
            "border": "var(--cas-risk-border)",
        },
        "위험신호 아님": {
            "bg": "var(--cas-success-soft)",
            "fg": "var(--cas-success)",
            "border": "var(--cas-success-border)",
        },
        "부적격": {
            "bg": "var(--cas-risk-soft)",
            "fg": "var(--cas-risk)",
            "border": "var(--cas-risk-border)",
        },
        "투기등급": {
            "bg": "var(--cas-risk-soft)",
            "fg": "var(--cas-risk)",
            "border": "var(--cas-risk-border)",
        },
        "차이 있음": {
            "bg": "var(--cas-warning-soft)",
            "fg": "var(--cas-warning)",
            "border": "var(--cas-warning-border)",
        },
        "일치": {
            "bg": "var(--cas-success-soft)",
            "fg": "var(--cas-success)",
            "border": "var(--cas-success-border)",
        },
        "발동": {
            "bg": "var(--cas-risk-soft)",
            "fg": "var(--cas-risk)",
            "border": "var(--cas-risk-border)",
        },
        "후보 검토": {
            "bg": "var(--cas-warning-soft)",
            "fg": "var(--cas-warning)",
            "border": "var(--cas-warning-border)",
        },
        "미발동": {
            "bg": "var(--cas-neutral-soft)",
            "fg": "var(--cas-text)",
            "border": "var(--cas-neutral-border)",
        },
    }
    if muted:
        style = {
            "bg": "var(--cas-neutral-soft)",
            "fg": "var(--cas-text)",
            "border": "var(--cas-neutral-border)",
        }
    else:
        style = style_map.get(
            text,
            {
                "bg": "var(--cas-neutral-soft)",
                "fg": "var(--cas-text)",
                "border": "var(--cas-neutral-border)",
            },
        )
    return (
        f"<div style='display:inline-block;padding:0.45rem 0.8rem;border-radius:999px;"
        f"background:{style['bg']};color:{style['fg']};border:1px solid {style['border']};"
        "font-weight:700;font-size:0.95rem;'>"
        f"{escape(text)}</div>"
    )


def render_bold_value_block(
    container: st.delta_generator.DeltaGenerator, label: str, value: object
) -> None:
    """Render a bold label and value inside a consistent overview card."""
    container.markdown(
        (
            "<div style='min-height:104px;padding:0.9rem 1rem;"
            "border-radius:var(--cas-card-radius);background:var(--cas-card-bg);"
            "border:1px solid var(--cas-border-soft);"
            "display:flex;flex-direction:column;justify-content:space-between;"
            "margin-bottom:0.5rem;box-shadow:var(--cas-card-shadow);'>"
            f"<div style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};'>"
            f"{escape(label)}"
            "</div>"
            f"<div style='font-size:1.2rem;line-height:1.45;font-weight:700;color:{COLOR_CARD_VALUE};"
            "word-break:keep-all;'>"
            f"{escape(str(value))}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_badge_value_block(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    badge_html: str,
) -> None:
    """Render a bold label and badge inside the same overview card layout."""
    container.markdown(
        (
            "<div style='min-height:104px;padding:0.9rem 1rem;"
            "border-radius:var(--cas-card-radius);background:var(--cas-card-bg);"
            "border:1px solid var(--cas-border-soft);"
            "display:flex;flex-direction:column;justify-content:space-between;"
            "margin-bottom:0.5rem;box-shadow:var(--cas-card-shadow);'>"
            f"<div style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};'>"
            f"{escape(label)}"
            "</div>"
            f"<div style='line-height:1.45;'>{badge_html}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_value_detail_block(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    value: object,
    description: str | None = None,
    badge_html: str | None = None,
) -> None:
    """Render a value card with an optional short description."""
    badge_section = ""
    if badge_html:
        badge_section = f"<div style='margin-top:0.45rem;'>{badge_html}</div>"
    description_html = ""
    if description:
        description_html = (
            f"<div style='font-size:0.88rem;line-height:1.45;color:{COLOR_CARD_LABEL};"
            "margin-top:0.45rem;word-break:keep-all;'>"
            f"{escape(description)}"
            "</div>"
        )

    container.markdown(
        (
            f"<div style='min-height:136px;padding:0.9rem 1rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            "display:flex;flex-direction:column;justify-content:space-between;"
            f"margin-bottom:0.5rem;box-shadow:{CARD_SHADOW};'>"
            f"<div style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};'>"
            f"{escape(label)}"
            "</div>"
            f"<div style='font-size:1.18rem;line-height:1.45;font-weight:700;color:{COLOR_CARD_VALUE};"
            "word-break:keep-all;'>"
            f"{escape(str(value))}"
            "</div>"
            f"{badge_section}"
            f"{description_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_text_card(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    body: str,
) -> None:
    """Render an explanatory gray card with bold label."""
    container.markdown(
        (
            "<div style='min-height:120px;padding:0.95rem 1rem;"
            "border-radius:var(--cas-card-radius);background:var(--cas-card-bg);"
            "border:1px solid var(--cas-border-soft);"
            "display:flex;flex-direction:column;justify-content:space-between;"
            "margin-bottom:0.5rem;box-shadow:var(--cas-card-shadow);'>"
            f"<div style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};'>"
            f"{escape(label)}"
            "</div>"
            f"<div style='font-size:0.97rem;line-height:1.6;color:{COLOR_CARD_VALUE};"
            "word-break:keep-all;'>"
            f"{escape(body)}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_bullet_card(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    items: list[str],
    accent_color: str,
    empty_message: str,
) -> None:
    """Render a summary card with short bullet items."""
    if items:
        bullet_html = "".join(
            (f"<li style='margin:0 0 0.35rem 0;'>{escape(item)}</li>") for item in items
        )
        body_html = (
            "<ul style='margin:0.15rem 0 0 1rem;padding:0;"
            f"color:{COLOR_CARD_VALUE};font-size:0.94rem;line-height:1.6;'>"
            f"{bullet_html}"
            "</ul>"
        )
    else:
        body_html = (
            f"<div style='font-size:0.95rem;line-height:1.6;color:{COLOR_CARD_LABEL};'>"
            f"{escape(empty_message)}"
            "</div>"
        )

    container.markdown(
        (
            f"<div style='min-height:168px;padding:0.95rem 1rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            f"border-left:6px solid {accent_color};"
            "display:flex;flex-direction:column;justify-content:flex-start;"
            f"margin-bottom:0.5rem;box-shadow:{CARD_SHADOW};'>"
            f"<div style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};margin-bottom:0.35rem;'>"
            f"{escape(label)}"
            "</div>"
            f"{body_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_summary_banner(
    label: str,
    body: str,
    accent_color: str,
) -> None:
    """Render a wide summary banner for quick interpretation."""
    st.markdown(
        (
            "<div style='padding:0.95rem 1.05rem;border-radius:var(--cas-card-radius);"
            "background:var(--cas-card-bg);border:1px solid var(--cas-border-soft);"
            f"box-shadow:var(--cas-card-shadow), inset 0 3px 0 {accent_color};"
            "margin:0.25rem 0 0.9rem 0;'>"
            f"<div style='font-size:0.93rem;font-weight:700;color:{COLOR_CARD_LABEL};margin-bottom:0.3rem;'>"
            f"{escape(label)}"
            "</div>"
            f"<div style='font-size:1rem;line-height:1.65;color:{COLOR_CARD_VALUE};word-break:keep-all;'>"
            f"{escape(body)}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_list_card(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    items: list[str],
    accent_color: str,
) -> None:
    """Render a structured list card for short bullet summaries."""
    list_html = "".join(
        (f"<li style='margin-bottom:0.38rem;'>{escape(item)}</li>")
        for item in items
        if str(item).strip()
    )
    if not list_html:
        list_html = "<li>요약할 항목이 없습니다.</li>"

    container.markdown(
        (
            "<div style='min-height:188px;padding:0.95rem 1rem;"
            "border-radius:var(--cas-card-radius);background:var(--cas-card-bg);"
            "border:1px solid var(--cas-border-soft);"
            f"box-shadow:var(--cas-card-shadow), inset 0 3px 0 {accent_color};"
            "margin-bottom:0.5rem;'>"
            f"<div style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};margin-bottom:0.55rem;'>"
            f"{escape(label)}"
            "</div>"
            f"<ul style='margin:0;padding-left:1.15rem;font-size:0.97rem;line-height:1.65;color:{COLOR_CARD_VALUE};'>"
            f"{list_html}"
            "</ul>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_badge_hint_card(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    badge_items: list[tuple[str, str]],
    accent_color: str,
    empty_message: str,
) -> None:
    """Render a compact card containing inline interpretation badges."""
    if badge_items:
        badge_html = "".join(
            (
                "<div style='display:flex;align-items:center;gap:0.45rem;flex-wrap:wrap;"
                "margin:0 0 0.45rem 0;'>"
                f"<span style='font-size:0.92rem;font-weight:700;color:{COLOR_CARD_VALUE};'>{escape(name)}</span>"
                f"{render_direction_badge_html(direction)}"
                "</div>"
            )
            for name, direction in badge_items
        )
    else:
        badge_html = (
            f"<div style='font-size:0.93rem;line-height:1.55;color:{COLOR_CARD_LABEL};'>"
            f"{escape(empty_message)}"
            "</div>"
        )

    container.markdown(
        (
            f"<div style='min-height:112px;padding:0.85rem 0.95rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            f"border-top:4px solid {accent_color};box-shadow:{CARD_SHADOW};"
            "margin:-0.15rem 0 0.6rem 0;'>"
            f"<div style='font-size:0.92rem;font-weight:700;color:{COLOR_CARD_LABEL};margin-bottom:0.45rem;'>"
            f"{escape(label)}"
            "</div>"
            f"{badge_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_legend_card(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    description: str,
    color: str,
) -> None:
    """Render a compact colored legend card for comparison views."""
    container.markdown(
        (
            f"<div style='min-height:96px;padding:0.85rem 1rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            "display:flex;flex-direction:column;justify-content:space-between;"
            "margin-bottom:0.35rem;'>"
            "<div style='display:flex;align-items:center;gap:0.5rem;'>"
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:999px;background:{color};'></span>"
            f"<span style='font-size:0.95rem;font-weight:700;color:{COLOR_CARD_LABEL};'>{escape(label)}</span>"
            "</div>"
            f"<div style='font-size:0.95rem;line-height:1.45;color:{COLOR_CARD_VALUE};'>{escape(description)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_accent_summary_card(
    container: st.delta_generator.DeltaGenerator,
    label: str,
    value: object,
    note: str,
    color: str,
) -> None:
    """Render a compact summary card with a colored accent for quick scanning."""
    container.markdown(
        (
            f"<div style='min-height:120px;padding:0.95rem 1rem;border-radius:8px;"
            f"background:{COLOR_CARD_BG};border:1px solid {COLOR_CARD_BORDER};"
            f"border-top:4px solid {color};box-shadow:{CARD_SHADOW};"
            "display:flex;flex-direction:column;justify-content:space-between;"
            "margin-bottom:0.5rem;'>"
            f"<div style='font-size:0.93rem;font-weight:700;color:{COLOR_CARD_LABEL};'>"
            f"{escape(label)}"
            "</div>"
            f"<div style='font-size:1.08rem;line-height:1.45;font-weight:700;color:{COLOR_CARD_VALUE};"
            "word-break:keep-all;'>"
            f"{escape(str(value))}"
            "</div>"
            f"<div style='font-size:0.88rem;line-height:1.45;color:{COLOR_CARD_LABEL};word-break:keep-all;'>"
            f"{escape(note)}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
