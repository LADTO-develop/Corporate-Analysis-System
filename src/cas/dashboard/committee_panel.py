"""Committee review panel rendering helpers for the credit dashboard."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from html import escape

import pandas as pd
import streamlit as st

from cas.dashboard.committee_copy import (
    COMMITTEE_DECISION_STAGE_GUIDE,
    committee_decision_type_info,
)


@dataclass(frozen=True, slots=True)
class CommitteePanelRenderers:
    """Small set of host-dashboard callbacks used by committee panel cards."""

    render_decision_badge: Callable[[object], str]
    format_percent: Callable[[object], str]


COMMITTEE_SIGNAL_METRIC_GUIDE = [
    {
        "label": "2차 검토대상",
        "value": "Recall 100.0%",
        "tone": "warning",
        "body": (
            "보류와 부적격을 모두 포함한 넓은 그물입니다. 사용자가 놓치면 안 되는 위험 기업을 "
            "위원회 검토망에 올렸는지는 이 지표로 보는 게 맞습니다."
        ),
    },
    {
        "label": "강한 위험신호",
        "value": "Precision 88.9%",
        "tone": "risk",
        "body": (
            "화면에서 빨간 경고처럼 읽히는 신호입니다. 실제 위험 가능성이 높은 경우만 "
            "강하게 표시하도록 더 엄격하게 잡았습니다."
        ),
    },
    {
        "label": "위험신호 Recall",
        "value": "Recall 53.3%",
        "tone": "neutral",
        "body": (
            "낮아 보이지만 의도된 구조입니다. 나머지 위험 후보를 안전하다고 넘기는 것이 아니라 "
            "관찰 단계로 남겨 사용자가 추가로 확인할 수 있게 보여줍니다."
        ),
    },
]

COMMITTEE_HOLD_SUBTYPE_GUIDE = [
    {
        "label": "위험 보류",
        "signal": "위험신호 있음",
        "tone": "risk",
        "title": "위험 보류",
        "body": "외부근거와 재무 스트레스가 위험 쪽으로 맞물려, 적격으로 넘기기 어려운 보류입니다.",
        "action": "소송, 자금조달, 거래정지, 현금흐름 악화처럼 손실로 이어질 수 있는 근거를 먼저 확인합니다.",
    },
    {
        "label": "확인필요 보류",
        "signal": "위험신호 아님",
        "tone": "warning",
        "title": "확인필요 보류",
        "body": "보류는 유지하지만 빨간 위험 경고까지는 아닙니다. 근거의 직접성이나 최신성이 더 필요합니다.",
        "action": "단일 medium 공시나 키워드성 뉴스가 실제 부실 신호인지, 기준일 이전 근거인지 확인합니다.",
    },
    {
        "label": "과민경고 완화 보류",
        "signal": "위험신호 아님",
        "tone": "mitigate",
        "title": "과민경고 완화",
        "body": "1차 모델 경고를 바로 부적격으로 확정하지 않고, 방어 재무나 약한 외부근거를 반영해 낮춘 상태입니다.",
        "action": "유동성, 자본, 영업현금흐름이 방어적인지와 치명 공시 부재가 완화 근거로 충분한지 봅니다.",
    },
    {
        "label": "경계등급 보류",
        "signal": "위험신호 아님",
        "tone": "neutral",
        "title": "경계등급 보류",
        "body": "등급이나 확률이 기준선 근처라 판단을 세게 내리기보다 관찰로 남긴 보류입니다.",
        "action": "BBB-/BB+ 경계, 확률 기준선 근접, 최근 등급 방향을 함께 확인합니다.",
    },
]


def render_committee_signal_guide(
    *,
    decision_type_label: str,
    risk_signal: bool,
    renderers: CommitteePanelRenderers,
) -> None:
    """Show whether the current committee decision is a risk signal or review-only hold."""
    current_info = committee_decision_type_info(
        decision_type_label,
        risk_signal=risk_signal,
    )
    current_signal = "위험신호 있음" if risk_signal else current_info["signal"]
    current_detail = current_info.get("detail", "")
    current_html = (
        "<div class='committee-signal-card "
        f"{escape(current_info['tone'])} active'>"
        "<div class='committee-signal-eyebrow'>현재 기업의 판단유형</div>"
        f"<div class='committee-signal-title'>{escape(current_info['title'])}</div>"
        f"{renderers.render_decision_badge(current_signal)}"
        f"<div class='committee-signal-body' style='margin-top:0.55rem;'>"
        f"{escape(current_info['body'])}"
        "</div>"
        + (
            f"<div class='committee-signal-detail'>{escape(current_detail)}</div>"
            if current_detail
            else ""
        )
        + f"<div class='committee-signal-action'>{escape(current_info['action'])}</div>"
        + "</div>"
    )

    guide_cards = []
    current_stage_title = str(current_info.get("title") or "")
    for info in COMMITTEE_DECISION_STAGE_GUIDE:
        title = str(info["title"])
        active_class = " active" if title == current_stage_title else ""
        detail = info.get("detail", "")
        current_badge = (
            "<span class='committee-signal-current-badge'>현재 단계</span>" if active_class else ""
        )
        guide_cards.append(
            "<div class='committee-signal-card "
            f"{escape(info['tone'])}{active_class}'>"
            f"{current_badge}"
            "<div class='committee-signal-eyebrow'>위원회 판단 단계</div>"
            f"<div class='committee-signal-title'>{escape(title)}</div>"
            f"{renderers.render_decision_badge(info['signal'])}"
            f"<div class='committee-signal-body' style='margin-top:0.55rem;'>"
            f"{escape(info['body'])}"
            "</div>"
            + (f"<div class='committee-signal-detail'>{escape(detail)}</div>" if detail else "")
            + f"<div class='committee-signal-action'>{escape(info['action'])}</div>"
            + "</div>"
        )

    st.markdown(
        (
            f"<div class='committee-signal-guide' style='grid-template-columns:minmax(0, 1fr);'>{current_html}</div>"
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "판단유형은 단순히 적격/부적격만 나누기보다, 왜 보류로 보는지와 어떤 근거를 "
        "추가로 확인해야 하는지를 알려주는 해석 라벨입니다."
    )
    with st.expander("판단유형 전체 설명 보기", expanded=False):
        st.caption(
            "위원회 판단은 사용자가 빠르게 읽을 수 있도록 4단계로 단순화했습니다. "
            "세부 이유는 별도 판단 이유로 남기고, 아래 순서대로 위험 신호가 강해진다고 보면 됩니다."
        )
        st.markdown(
            (f"<div class='committee-signal-guide'>{''.join(guide_cards)}</div>"),
            unsafe_allow_html=True,
        )


def render_committee_hold_subtype_guide(
    *,
    decision_type_label: str,
    risk_signal: bool,
    renderers: CommitteePanelRenderers,
) -> None:
    """Show how Stage 2 hold subtypes differ from each other."""
    st.markdown("#### 보류 유형 구분")
    st.caption(
        "Stage 2는 보류를 한 덩어리로 보지 않고, 위험을 올린 보류와 경고를 완화한 보류를 "
        "분리해서 보여줍니다."
    )
    cards = []
    normalized_decision = str(decision_type_label or "").strip()
    for info in COMMITTEE_HOLD_SUBTYPE_GUIDE:
        label = str(info["label"])
        active = label == normalized_decision
        signal = "위험신호 있음" if active and risk_signal else str(info["signal"])
        current_badge = (
            "<span class='committee-signal-current-badge'>현재 유형</span>" if active else ""
        )
        cards.append(
            "<div class='committee-signal-card "
            f"{escape(str(info['tone']))}{' active' if active else ''}'>"
            f"{current_badge}"
            "<div class='committee-signal-eyebrow'>보류 세분화</div>"
            f"<div class='committee-signal-title'>{escape(str(info['title']))}</div>"
            f"{renderers.render_decision_badge(signal)}"
            f"<div class='committee-signal-body' style='margin-top:0.55rem;'>"
            f"{escape(str(info['body']))}"
            "</div>"
            f"<div class='committee-signal-action'>{escape(str(info['action']))}</div>"
            "</div>"
        )

    st.markdown(
        (
            "<div class='committee-signal-guide' "
            "style='grid-template-columns:repeat(auto-fit,minmax(220px,1fr));'>"
            f"{''.join(cards)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "발표 화면에서는 이 줄만 봐도 에이전트가 위험을 키운 건지, 보류만 유지한 건지, "
        "아니면 모델 경고를 완화한 건지 바로 구분할 수 있습니다."
    )


def render_committee_metric_guide() -> None:
    """Explain why review-target recall and strong risk-signal recall differ."""
    cards = []
    for item in COMMITTEE_SIGNAL_METRIC_GUIDE:
        cards.append(
            "<div class='committee-metric-card "
            f"{escape(item['tone'])}'>"
            f"<div class='committee-metric-label'>{escape(item['label'])}</div>"
            f"<div class='committee-metric-value'>{escape(item['value'])}</div>"
            f"<div class='committee-metric-card-body'>{escape(item['body'])}</div>"
            "</div>"
        )

    st.markdown(
        (
            "<div class='committee-metric-guide'>"
            "<div class='committee-metric-guide-title'>"
            "위원회 성능은 두 단계로 읽는 게 자연스럽습니다"
            "</div>"
            "<div class='committee-metric-guide-body'>"
            "사용자 입장에서는 위험 기업을 검토망에 올렸는지가 중요하므로 "
            "<b>보류+부적격 기준의 검토대상 Recall</b>을 먼저 봅니다. "
            "반면 <b>위험신호</b>는 빨간 경고에 가까운 강한 표현이라, "
            "오경보를 줄이기 위해 더 엄격하게 표시합니다."
            "</div>"
            f"<div class='committee-metric-grid'>{''.join(cards)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_committee_decision_trace(
    decision_trace: list[dict[str, object]],
    *,
    expanded: bool = False,
) -> None:
    """Render the deterministic Stage 2 gate trace in a user-friendly timeline."""
    clean_items = [
        item
        for item in decision_trace
        if isinstance(item, dict) and str(item.get("summary") or "").strip()
    ]
    if not clean_items:
        return

    trace_cards = []
    for index, item in enumerate(clean_items, start=1):
        severity = str(item.get("severity") or "info").strip().lower()
        tone = severity if severity in {"info", "watch", "risk", "mitigation"} else "info"
        triggered = bool(item.get("triggered", False))
        status_label = "작동" if triggered else "확인"
        label = str(item.get("label") or item.get("gate") or f"{index}단계")
        summary = _normalize_committee_text(item.get("summary"))
        trace_cards.append(
            "<div class='committee-trace-card "
            f"{escape(tone)} {'triggered' if triggered else 'muted'}'>"
            "<div class='committee-trace-head'>"
            f"<span class='committee-trace-step'>{index}</span>"
            f"<span class='committee-trace-title'>{escape(label)}</span>"
            f"<span class='committee-trace-status'>{escape(status_label)}</span>"
            "</div>"
            f"<div class='committee-trace-body'>{escape(summary)}</div>"
            "</div>"
        )

    with st.expander("판단 과정 보기", expanded=expanded):
        st.caption(
            "2차 위원회가 결론을 내릴 때 확인한 기준들을 순서대로 보여줍니다. "
            "`작동`은 해당 기준이 실제 판단에 영향을 준 신호이고, `확인`은 점검했지만 "
            "강하게 켜지지는 않은 기준입니다."
        )
        st.markdown(
            (
                "<style>"
                ".committee-trace-flow{display:grid;gap:0.65rem;margin-top:0.7rem;}"
                ".committee-trace-card{padding:0.9rem 1rem;border-radius:16px;"
                "background:var(--cas-card-bg);border:1px solid var(--cas-border-soft);"
                "box-shadow:var(--cas-card-shadow);}"
                ".committee-trace-card.muted{opacity:0.78;}"
                ".committee-trace-card.risk.triggered{border-color:var(--cas-risk-border);"
                "box-shadow:var(--cas-card-shadow),inset 4px 0 0 var(--cas-risk);}"
                ".committee-trace-card.watch.triggered{border-color:var(--cas-warning-border);"
                "box-shadow:var(--cas-card-shadow),inset 4px 0 0 var(--cas-warning);}"
                ".committee-trace-card.mitigation.triggered{border-color:var(--cas-success-border);"
                "box-shadow:var(--cas-card-shadow),inset 4px 0 0 var(--cas-success);}"
                ".committee-trace-card.info.triggered{border-color:var(--cas-neutral-border);"
                "box-shadow:var(--cas-card-shadow),inset 4px 0 0 var(--cas-neutral);}"
                ".committee-trace-head{display:flex;align-items:center;gap:0.55rem;flex-wrap:wrap;"
                "margin-bottom:0.45rem;}"
                ".committee-trace-step{display:inline-flex;align-items:center;justify-content:center;"
                "width:1.7rem;height:1.7rem;border-radius:999px;background:var(--cas-neutral-soft);"
                "color:var(--cas-text);font-size:0.82rem;font-weight:800;}"
                ".committee-trace-title{font-weight:800;color:var(--cas-text);}"
                ".committee-trace-status{margin-left:auto;padding:0.18rem 0.55rem;border-radius:999px;"
                "font-size:0.78rem;font-weight:800;background:var(--cas-neutral-soft);"
                "border:1px solid var(--cas-neutral-border);color:var(--cas-text);}"
                ".committee-trace-card.risk.triggered .committee-trace-status{"
                "background:var(--cas-risk-soft);border-color:var(--cas-risk-border);"
                "color:var(--cas-risk);}"
                ".committee-trace-card.watch.triggered .committee-trace-status{"
                "background:var(--cas-warning-soft);border-color:var(--cas-warning-border);"
                "color:var(--cas-warning);}"
                ".committee-trace-card.mitigation.triggered .committee-trace-status{"
                "background:var(--cas-success-soft);border-color:var(--cas-success-border);"
                "color:var(--cas-success);}"
                ".committee-trace-body{font-size:0.94rem;line-height:1.62;color:var(--cas-muted);"
                "word-break:keep-all;}"
                "</style>"
                f"<div class='committee-trace-flow'>{''.join(trace_cards)}</div>"
            ),
            unsafe_allow_html=True,
        )


def _normalize_committee_text(text: object) -> str:
    """Normalize committee text while preserving its full meaning."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _split_committee_highlight_fragments(text: object) -> list[str]:
    """Split dense numbered committee text into readable highlight fragments."""
    normalized = _normalize_committee_text(text)
    if not normalized:
        return []

    numbered_matches = re.findall(r"\(\d+\)\s*(.*?)(?=\s*\(\d+\)\s*|$)", normalized)
    if numbered_matches:
        return [match.strip(" ,;") for match in numbered_matches if match.strip()]

    sentence_parts = re.split(r"(?<=[.!?。])\s+", normalized)
    return [part.strip(" ,;") for part in sentence_parts if part.strip()]


def _committee_highlight_items(
    items: list[str],
    fallback: str,
    *,
    max_items: int = 3,
) -> list[str]:
    """Build short highlight items without truncating text."""
    highlights: list[str] = []
    for item in items:
        fragments = _split_committee_highlight_fragments(item)
        if len(fragments) > 1:
            highlights.extend(fragments)
        elif fragments:
            highlights.append(fragments[0])
        if len(highlights) >= max_items:
            break
    return highlights[:max_items] or [fallback]


def _committee_highlight_body_html(items: list[str]) -> str:
    """Render highlight items without ellipses."""
    clean_items = [item for item in (_normalize_committee_text(item) for item in items) if item]
    if len(clean_items) == 1:
        return escape(clean_items[0])
    list_html = "".join(f"<li>{escape(item)}</li>" for item in clean_items)
    return f"<ul>{list_html}</ul>"


def committee_review_tone_class(
    *,
    committee_label: str,
    decision_type_label: str,
    risk_signal: bool,
) -> str:
    """Return visual tone for the committee hero."""
    if committee_label == "부적격" or decision_type_label == "위험 보류" or risk_signal:
        return "risk"
    if committee_label == "적격" and decision_type_label not in {"확인필요 보류", "경계등급 보류"}:
        return "stable"
    return "watch"


def render_committee_review_hero(
    *,
    committee_label: str,
    committee_decision_type_label: str,
    committee_risk_signal_label: str,
    model_display_label: str,
    decision_gap_label: str,
    veto_label: str,
    final_confidence: object,
    summary_text: str,
    risk_signal: bool,
    renderers: CommitteePanelRenderers,
) -> None:
    """Render a polished committee-first result summary."""
    tone = committee_review_tone_class(
        committee_label=committee_label,
        decision_type_label=committee_decision_type_label,
        risk_signal=risk_signal,
    )
    normalized_summary = _normalize_committee_text(summary_text)
    fact_rows = [
        ("1차 모델", renderers.render_decision_badge(model_display_label)),
        ("판단 차이", renderers.render_decision_badge(decision_gap_label)),
        ("강제 경고", renderers.render_decision_badge(veto_label)),
        ("위원회 신뢰도", escape(renderers.format_percent(final_confidence))),
    ]
    fact_rows_html = "".join(
        (
            "<div class='committee-review-fact-row'>"
            f"<span class='committee-review-fact-label'>{escape(label)}</span>"
            f"<span class='committee-review-fact-value'>{value}</span>"
            "</div>"
        )
        for label, value in fact_rows
    )
    current_stage_key = re.sub(r"\s+", "", committee_label or "")
    stage_steps = [
        ("적격", "확인된 위험 낮음"),
        ("관찰", "추가 확인 필요"),
        ("위험주의", "먼저 볼 위험 있음"),
        ("부적격", "신용위험 높음"),
    ]
    stage_steps_html_parts = []
    for label, caption in stage_steps:
        active_class = "active" if re.sub(r"\s+", "", label) == current_stage_key else ""
        stage_steps_html_parts.append(
            "<div class='committee-stage-step "
            f"{active_class}'>"
            f"<span>{escape(label)}</span>"
            f"<small>{escape(caption)}</small>"
            "</div>"
        )
    stage_steps_html = "".join(stage_steps_html_parts)
    st.markdown(
        (
            f"<div class='committee-review-hero {escape(tone)}'>"
            "<div class='committee-review-layout'>"
            "<div>"
            "<div class='committee-review-eyebrow'>AI 위원회 결론</div>"
            "<div class='committee-review-title-row'>"
            f"<div class='committee-review-title'>{escape(committee_label)}</div>"
            f"{renderers.render_decision_badge(committee_decision_type_label)}"
            "</div>"
            f"<div class='committee-review-summary'>{escape(normalized_summary)}</div>"
            "<div class='committee-review-chip-row'>"
            f"<span class='committee-review-chip'>{escape(committee_risk_signal_label)}</span>"
            f"<span class='committee-review-chip'>신뢰도 {escape(renderers.format_percent(final_confidence))}</span>"
            "</div>"
            "</div>"
            "<div class='committee-review-facts'>"
            "<div class='committee-review-facts-title'>판단 근거 요약</div>"
            f"{fact_rows_html}"
            "</div>"
            "</div>"
            "</div>"
            "<div class='committee-stage-scale'>"
            "<div class='committee-stage-scale-label'>판단 단계</div>"
            f"<div class='committee-stage-track'>{stage_steps_html}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_committee_loading_state(
    container: st.delta_generator.DeltaGenerator,
    selected_row: pd.Series,
) -> None:
    """Show a friendly transient state while committee review is being prepared."""
    company_name = str(selected_row.get("corp_name") or "선택 기업")
    container.markdown(
        (
            "<div class='committee-loading-card'>"
            "<div class='committee-loading-orb'></div>"
            "<div>"
            "<div class='committee-loading-title'>"
            f"{escape(company_name)}의 위원회 검토를 준비하고 있어요"
            "</div>"
            "<div class='committee-loading-body'>"
            "재무 신호, 뉴스·웹·공시 근거, 모델 판단 차이를 함께 확인하는 중입니다. "
            "잠시만 기다리면 위원회 의견과 주요 근거가 정리됩니다."
            "</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_committee_section_divider(label: str) -> None:
    """Render a quiet divider that separates committee sections without adding more cards."""
    st.markdown(
        (
            "<div class='committee-section-divider'>"
            f"<span class='committee-section-kicker'>{escape(label)}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_committee_key_highlights(
    *,
    committee_label: str,
    committee_decision_type_label: str,
    committee_risk_signal_label: str,
    model_display_label: str,
    decision_gap_label: str,
    veto_label: str,
    final_confidence: object,
    summary_text: str,
    conflict_text: str,
    final_memo: str,
    risk_items: list[str],
    mitigation_items: list[str],
    renderers: CommitteePanelRenderers,
    max_highlight_items: int = 2,
) -> None:
    """Render the committee result as a quick executive summary."""
    top_risk_items = _committee_highlight_items(
        risk_items,
        "위원회가 별도로 강조한 위험 요인은 없습니다.",
        max_items=max_highlight_items,
    )
    top_mitigation_items = _committee_highlight_items(
        mitigation_items,
        "위원회가 별도로 강조한 완화 요인은 없습니다.",
        max_items=max_highlight_items,
    )
    checkpoint_text = final_memo or conflict_text or summary_text
    cards = [
        (
            "risk",
            "가장 먼저 볼 위험",
            _committee_highlight_body_html(top_risk_items),
        ),
        (
            "mitigate",
            "완화해서 본 근거",
            _committee_highlight_body_html(top_mitigation_items),
        ),
        (
            "warning",
            "사용자 체크 포인트",
            _committee_highlight_body_html(
                _committee_highlight_items(
                    [checkpoint_text],
                    checkpoint_text,
                    max_items=max_highlight_items,
                )
            ),
        ),
    ]
    card_html = "".join(
        (
            f"<div class='committee-highlight-card {escape(css_class)}'>"
            f"<div class='committee-highlight-title'>{escape(title)}</div>"
            f"<div class='committee-highlight-body'>{body}</div>"
            "</div>"
        )
        for css_class, title, body in cards
    )
    st.markdown(
        (
            "<div class='committee-highlights-heading'>핵심 확인 포인트</div>"
            f"<div class='committee-highlight-grid'>{card_html}</div>"
        ),
        unsafe_allow_html=True,
    )


def _committee_detail_section_html(title: str, body: str) -> str:
    if not body.strip():
        return ""
    return (
        "<div class='committee-detail-section'>"
        f"<div class='committee-detail-heading'>{escape(title)}</div>"
        f"<div class='committee-detail-text'>{escape(body)}</div>"
        "</div>"
    )


def _committee_detail_list_html(title: str, items: list[str], empty_message: str) -> str:
    clean_items = [str(item).strip() for item in items if str(item).strip()]
    if clean_items:
        list_html = "".join(f"<li>{escape(item)}</li>" for item in clean_items)
    else:
        list_html = f"<li>{escape(empty_message)}</li>"
    return (
        "<div class='committee-detail-section'>"
        f"<div class='committee-detail-heading'>{escape(title)}</div>"
        f"<ul>{list_html}</ul>"
        "</div>"
    )


def render_committee_full_review(
    *,
    summary_text: str,
    conflict_text: str,
    final_memo: str,
    risk_items: list[str],
    mitigation_items: list[str],
) -> None:
    """Render the complete committee rationale below the highlighted summary."""
    sections_html = "".join(
        [
            _committee_detail_section_html("판단 차이 해석", summary_text),
            _committee_detail_section_html("왜 이렇게 판단했나", conflict_text),
            _committee_detail_list_html(
                "주의해서 볼 점",
                risk_items,
                "위원회가 별도로 강조한 위험 요인은 없습니다.",
            ),
            _committee_detail_list_html(
                "긍정적으로 본 점",
                mitigation_items,
                "위원회가 별도로 강조한 완화 요인은 없습니다.",
            ),
            _committee_detail_section_html("최종 검토 의견", final_memo),
        ]
    )
    with st.expander("상세 판단 전문", expanded=False):
        st.markdown(
            (
                "<div class='committee-detail-flow'>"
                "<div class='committee-detail-title'>전체 검토 내용</div>"
                f"{sections_html}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
