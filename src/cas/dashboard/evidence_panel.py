"""External evidence panels for the Streamlit credit dashboard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import streamlit as st

EXTERNAL_EVIDENCE_STATUS_LABELS = {
    "not_requested": "아직 수집하지 않음",
    "ready": "수집 완료",
    "no_results": "검색 결과 없음",
    "missing_credentials": "API 키 미설정",
    "partial_error": "일부 수집 오류",
    "disabled": "수집 비활성화",
    "error": "수집 오류",
}

EXTERNAL_EVIDENCE_SOURCE_LABELS = {
    "opendart": "OpenDART 공시",
    "naver_news": "네이버 뉴스",
    "tavily": "웹 검색",
}

EXTERNAL_EVIDENCE_QUALITY_LABELS = {
    "high": "높음",
    "medium": "보통",
    "low": "낮음",
    "unknown": "미확인",
}

EXTERNAL_EVIDENCE_RELIABILITY_LABELS = {
    "high": "높음",
    "medium": "보통",
    "low": "낮음",
    "low_relevance": "관련성 낮음",
    "unknown": "미확인",
}


@dataclass(frozen=True, slots=True)
class EvidencePanelColors:
    """Color palette supplied by the parent dashboard."""

    risk: str
    mitigate: str
    neutral: str


@dataclass(frozen=True, slots=True)
class EvidencePanelRenderers:
    """Shared card renderers supplied by the parent dashboard."""

    render_badge_value_block: Callable[[st.delta_generator.DeltaGenerator, str, str], None]
    render_bold_value_block: Callable[[st.delta_generator.DeltaGenerator, str, object], None]
    render_decision_badge: Callable[[object], str]
    render_list_card: Callable[[st.delta_generator.DeltaGenerator, str, list[str], str], None]
    render_summary_banner: Callable[[str, str, str], None]
    render_text_card: Callable[[st.delta_generator.DeltaGenerator, str, str], None]


def external_veto_candidate_count(snapshot: dict[str, object] | None) -> int:
    """Count external evidence items that are veto candidates but not final veto yet."""
    if not snapshot:
        return 0
    configured_count = _optional_int(snapshot.get("veto_candidate_count"))
    if configured_count is not None:
        return configured_count
    items = snapshot.get("items", [])
    if not isinstance(items, list):
        return 0
    return sum(1 for item in items if isinstance(item, dict) and item.get("veto_candidate") is True)


def dashboard_veto_status_label(
    committee_view: dict[str, object],
    evidence_snapshot: dict[str, object] | None,
) -> str:
    """Return a user-facing three-state veto label."""
    if bool(committee_view.get("veto_triggered", False)):
        return "발동"
    if external_veto_candidate_count(evidence_snapshot) > 0:
        return "후보 검토"
    return "미발동"


def render_external_evidence_items(
    evidence_snapshot: dict[str, object] | None,
    *,
    expanded: bool,
    include_summary: bool,
    renderers: EvidencePanelRenderers,
) -> None:
    """Render detailed external evidence in a user-friendly table."""
    evidence_frame = _external_evidence_items_frame(evidence_snapshot)
    if evidence_frame.empty:
        st.info("자동 수집된 외부 뉴스/웹/공시 근거가 없습니다.")
        return

    total_count = len(evidence_frame)
    direct_count = sum(evidence_frame["관련성"].isin(["직접 관련", "종목코드 일치"]))
    verified_count = sum(evidence_frame["검증 품질"].isin(["높음", "보통"]))
    veto_candidate_count = sum(evidence_frame["강제 경고"] == "후보")
    summary_cols = st.columns(4)
    renderers.render_bold_value_block(summary_cols[0], "외부 근거", f"{total_count}건")
    renderers.render_bold_value_block(summary_cols[1], "직접 관련", f"{direct_count}건")
    renderers.render_bold_value_block(summary_cols[2], "검증 통과", f"{verified_count}건")
    renderers.render_bold_value_block(
        summary_cols[3],
        "강제 경고 후보",
        f"{veto_candidate_count}건",
    )

    with st.expander("외부 근거 상세 목록 보기", expanded=expanded):
        st.caption(
            "관련성 낮음 또는 키워드만 감지된 항목은 2차 위원회 판단을 바로 바꾸지 않고, "
            "검증 품질과 문맥을 함께 확인합니다."
        )
        display_columns = [
            "순번",
            "출처",
            "제목/공시명",
            "관련성",
            "검증 품질",
            "신뢰도",
            "위험 키워드",
            "강제 경고",
            "일자",
            "링크",
        ]
        if include_summary:
            display_columns.insert(-1, "요약")
        st.dataframe(
            evidence_frame.loc[:, display_columns],
            hide_index=True,
            width="stretch",
            column_config={
                "제목/공시명": st.column_config.TextColumn("제목/공시명", width="large"),
                "요약": st.column_config.TextColumn("요약", width="large"),
                "링크": st.column_config.LinkColumn("링크", display_text="열기"),
            },
        )


def render_external_evidence_judgment(
    evidence_snapshot: dict[str, object] | None,
    committee_view: dict[str, object],
    *,
    veto_label: str,
    colors: EvidencePanelColors,
    renderers: EvidencePanelRenderers,
) -> None:
    """Render the external-evidence interpretation as the main committee-tab focus."""
    st.subheader("외부 근거 판단")
    st.caption(
        "뉴스, 웹 검색, OpenDART 공시를 그냥 나열하지 않고 기업 직접 관련성, 근거 품질, "
        "위험 키워드의 문맥을 나누어 판단합니다."
    )

    summary_text, summary_color = _external_evidence_judgment_text(
        evidence_snapshot,
        veto_label=veto_label,
        colors=colors,
    )
    renderers.render_summary_banner("외부 근거를 이렇게 해석했어요", summary_text, summary_color)

    evidence_frame = _external_evidence_items_frame(evidence_snapshot)
    direct_count = (
        sum(evidence_frame["관련성"].isin(["직접 관련", "종목코드 일치"]))
        if not evidence_frame.empty
        else 0
    )
    verified_count = (
        sum(evidence_frame["검증 품질"].isin(["높음", "보통"])) if not evidence_frame.empty else 0
    )
    total_count = len(evidence_frame)
    caution_count = external_veto_candidate_count(evidence_snapshot)
    final_label = str(committee_view.get("final_committee_label") or "보류")

    status_cols = st.columns(4)
    renderers.render_badge_value_block(
        status_cols[0],
        "2차 위원회 판단",
        renderers.render_decision_badge(final_label),
    )
    renderers.render_bold_value_block(
        status_cols[1],
        "직접 관련 근거",
        f"{direct_count}/{total_count}건",
    )
    renderers.render_bold_value_block(status_cols[2], "검증 통과 근거", f"{verified_count}건")
    renderers.render_badge_value_block(
        status_cols[3],
        f"강제 경고 상태 ({caution_count}건)",
        renderers.render_decision_badge(veto_label),
    )

    evidence_cols = st.columns(3)
    renderers.render_list_card(
        evidence_cols[0],
        "판단에 쓴 핵심 근거",
        _external_evidence_item_summaries(evidence_snapshot, bucket="verified"),
        colors.mitigate,
    )
    renderers.render_list_card(
        evidence_cols[1],
        "추가로 살펴볼 신호",
        _external_evidence_item_summaries(evidence_snapshot, bucket="caution"),
        colors.risk,
    )
    renderers.render_text_card(
        evidence_cols[2],
        "수집 경로와 반영 방식",
        _external_evidence_collection_note(evidence_snapshot, veto_label=veto_label),
    )


def _external_evidence_judgment_text(
    snapshot: dict[str, object] | None,
    *,
    veto_label: str,
    colors: EvidencePanelColors,
) -> tuple[str, str]:
    """Return a friendly evidence-level judgment summary and accent color."""
    if not snapshot:
        return (
            "아직 외부 근거가 수집되지 않아 2차 판단은 모델과 재무지표 중심으로 해석됩니다.",
            colors.neutral,
        )
    status = str(snapshot.get("status") or "unknown")
    raw_items = snapshot.get("items", [])
    item_count = len(raw_items) if isinstance(raw_items, list) else 0
    direct_count = _optional_int(snapshot.get("direct_match_count")) or 0
    verified_count = _optional_int(snapshot.get("verified_item_count")) or 0
    critical_terms = _external_evidence_terms_text(snapshot.get("critical_terms"))

    if veto_label == "발동":
        terms_text = "" if critical_terms == "없음" else f" 감지 키워드는 {critical_terms}입니다."
        return (
            "외부 근거에서 최종 판단을 보수적으로 바꿀 만큼 강한 위험 신호가 확인됐습니다."
            f"{terms_text}",
            colors.risk,
        )
    if veto_label == "후보 검토":
        return (
            "위험 키워드가 일부 감지됐지만, 다중 출처나 고신뢰 문맥 조건이 충분하지 않아 "
            "즉시 경고로 확정하지 않고 후보 상태로 분리했습니다.",
            "#c0841a",
        )
    if status == "no_results" or item_count == 0:
        return (
            "현재 수집 범위에서는 기업과 직접 연결되는 외부 뉴스·웹·공시 근거가 많지 않습니다. "
            "따라서 2차 판단은 정량 결과와 재무지표 해석을 중심으로 유지됩니다.",
            colors.neutral,
        )
    return (
        f"총 {item_count}개 외부 근거 중 직접 관련 {direct_count}개, 검증 통과 {verified_count}개를 "
        "확인했습니다. 현재 수집된 근거만으로는 최종 판단을 뒤집을 강한 외부 위험 신호는 보이지 않습니다.",
        colors.mitigate,
    )


def _external_evidence_item_summaries(
    snapshot: dict[str, object] | None,
    *,
    bucket: str,
    limit: int = 3,
) -> list[str]:
    """Return short evidence-item summaries for judgment cards."""
    if not snapshot:
        return []
    raw_items = snapshot.get("items", [])
    if not isinstance(raw_items, list):
        return []
    items = [item for item in raw_items if isinstance(item, dict)]
    summaries: list[str] = []
    for item in items:
        match_label = _external_evidence_match_label(item)
        quality_label = _external_evidence_quality_label(item.get("evidence_quality"))
        caution_label = _external_evidence_veto_label(item)
        if bucket == "verified":
            if match_label not in {"직접 관련", "종목코드 일치"}:
                continue
            if quality_label not in {"높음", "보통"}:
                continue
        elif bucket == "caution":
            has_terms = _external_evidence_terms_text(item.get("critical_terms")) != "없음"
            if caution_label == "아님" and not has_terms and match_label != "관련성 낮음":
                continue
        else:
            continue
        source = _external_evidence_source_label(item.get("source"))
        title = _short_text(str(item.get("title") or "제목 없음"), limit=52)
        terms = _external_evidence_terms_text(item.get("critical_terms"))
        if bucket == "caution" and terms != "없음":
            summaries.append(f"{source}: {title} - 위험 키워드 {terms}, 상태 {caution_label}")
        else:
            summaries.append(f"{source}: {title} - {match_label}, 검증 {quality_label}")
        if len(summaries) >= limit:
            break
    if not summaries and bucket == "verified":
        return ["직접 관련성이 높고 검증을 통과한 외부 근거가 아직 없습니다."]
    if not summaries and bucket == "caution":
        return ["현재 수집된 근거에서 추가로 확인할 외부 위험 신호는 없습니다."]
    return summaries


def _external_evidence_collection_note(
    snapshot: dict[str, object] | None,
    *,
    veto_label: str,
) -> str:
    """Explain how collected evidence affects the committee decision."""
    provider_text = _external_evidence_provider_status_text(
        snapshot.get("providers") if snapshot else None
    )
    if not provider_text:
        provider_text = "뉴스, 웹 검색, OpenDART 공시를 기준으로 확인합니다."
    if veto_label == "발동":
        action_text = "강한 외부 위험 신호가 확인되어 위원회 판단을 보수적으로 조정합니다."
    elif veto_label == "후보 검토":
        action_text = "키워드는 감지됐지만 확정 근거가 부족해 후보로만 표시하고 판단을 과도하게 바꾸지 않습니다."
    else:
        action_text = "수집 근거는 판단 메모에 반영하되, 강한 경고 신호가 없으면 모델 판단을 무리하게 뒤집지 않습니다."
    return f"{provider_text} {action_text}"


def _external_evidence_items_frame(snapshot: dict[str, object] | None) -> pd.DataFrame:
    """Build a user-friendly table of collected external evidence items."""
    if not snapshot:
        return pd.DataFrame()
    raw_items = snapshot.get("items", [])
    if not isinstance(raw_items, list):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "순번": index,
                "출처": _external_evidence_source_label(item.get("source")),
                "제목/공시명": _short_text(str(item.get("title") or "제목 없음"), limit=110),
                "관련성": _external_evidence_match_label(item),
                "검증 품질": _external_evidence_quality_label(item.get("evidence_quality")),
                "신뢰도": _external_evidence_reliability_label(item.get("reliability")),
                "위험 키워드": _external_evidence_terms_text(item.get("critical_terms")),
                "강제 경고": _external_evidence_veto_label(item),
                "일자": _external_evidence_date_text(item.get("published_at")),
                "요약": _short_text(str(item.get("summary") or ""), limit=220),
                "링크": str(item.get("url") or ""),
            }
        )
    return pd.DataFrame(rows)


def _external_evidence_provider_status_text(providers: object) -> str:
    """Summarize provider-level evidence collection status."""
    if not isinstance(providers, dict) or not providers:
        return ""
    parts: list[str] = []
    provider_labels = {
        "naver_news": "네이버뉴스",
        "tavily": "웹검색",
        "opendart": "OpenDART",
    }
    for provider_name, raw_provider in providers.items():
        if not isinstance(raw_provider, dict):
            continue
        status = str(raw_provider.get("status") or "unknown")
        status_label = EXTERNAL_EVIDENCE_STATUS_LABELS.get(status, status)
        parts.append(
            f"{provider_labels.get(str(provider_name), str(provider_name))}: {status_label}"
        )
    if not parts:
        return ""
    return "제공자 상태는 " + ", ".join(parts) + "입니다."


def _external_evidence_source_label(value: object) -> str:
    """Return a Korean label for an external evidence source."""
    text = str(value or "unknown")
    return EXTERNAL_EVIDENCE_SOURCE_LABELS.get(text, text)


def _external_evidence_quality_label(value: object) -> str:
    """Return a Korean label for evidence verification quality."""
    text = str(value or "unknown").lower()
    return EXTERNAL_EVIDENCE_QUALITY_LABELS.get(text, text)


def _external_evidence_reliability_label(value: object) -> str:
    """Return a Korean label for source reliability."""
    text = str(value or "unknown").lower()
    return EXTERNAL_EVIDENCE_RELIABILITY_LABELS.get(text, text)


def _external_evidence_match_label(item: dict[str, object]) -> str:
    """Return a user-friendly company relevance label."""
    if item.get("company_match") is True:
        match_type = str(item.get("company_match_type") or "")
        if match_type == "stock_code":
            return "종목코드 일치"
        return "직접 관련"
    if item.get("company_match") is False:
        return "관련성 낮음"
    return "미확인"


def _external_evidence_terms_text(value: object) -> str:
    """Format critical evidence terms for display."""
    if isinstance(value, list | tuple):
        terms = [str(item) for item in value if str(item).strip()]
        return ", ".join(terms) if terms else "없음"
    text = str(value or "").strip()
    return text if text else "없음"


def _external_evidence_veto_label(item: dict[str, object]) -> str:
    """Return a precise label for external veto evidence status."""
    if item.get("veto_candidate") is True:
        return "후보"
    if _external_evidence_terms_text(item.get("critical_terms")) != "없음":
        if item.get("critical_context_confirmed") is False:
            return "키워드만 감지"
        return "검토 필요"
    return "아님"


def _external_evidence_date_text(value: object) -> str:
    """Format provider dates into a compact display value."""
    text = str(value or "").strip()
    if not text:
        return "-"
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8 and text.isdigit():
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return text[:16]


def _optional_int(value: object) -> int | None:
    """Return an integer when the value is numeric."""
    if pd.isna(value):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _short_text(text: str, *, limit: int) -> str:
    """Trim long evidence snippets for dataframe display."""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."
