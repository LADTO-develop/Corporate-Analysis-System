"""User-facing copy helpers for Stage 2 committee decisions."""

from __future__ import annotations

COMMITTEE_DECISION_TYPE_GUIDE = {
    "위험 보류": {
        "signal": "위험신호 있음",
        "tone": "risk",
        "title": "위험 주의",
        "body": "지금은 그냥 넘기기보다 한 번 더 들여다봐야 하는 기업입니다.",
        "detail": (
            "모델 확률, 보조 변수셋, 외부 근거 중 하나 이상이 위험 쪽으로 기울어 "
            "위원회가 안전하게 보류했습니다."
        ),
        "action": "먼저 손실 지속, 이자보상, 차입 부담, 직접 관련 공시·뉴스를 확인하세요.",
    },
    "경계등급 보류": {
        "signal": "위험신호 아님",
        "tone": "neutral",
        "title": "관찰",
        "body": "좋다/나쁘다를 딱 잘라 말하기 어려운 경계선 위의 기업입니다.",
        "detail": (
            "이전 공개등급이 BBB-/BB+ 근처이거나 모델 확률이 기준선 가까이에 있어 "
            "작은 정보 차이로 판단이 바뀔 수 있습니다."
        ),
        "action": "최근 등급 방향, 등급전망, 현금흐름 회복 여부, 외부근거의 방향성을 함께 보세요.",
    },
    "과민경고 완화 보류": {
        "signal": "위험신호 아님",
        "tone": "mitigate",
        "title": "관찰",
        "body": "모델은 경고했지만, 위원회가 보기에는 바로 부적격으로 단정하긴 이릅니다.",
        "detail": (
            "유동성, 자본비율, 영업현금흐름 같은 방어력이 있거나 외부근거가 치명적이지 "
            "않아 과민경고 가능성을 열어둔 상태입니다."
        ),
        "action": "모델을 자극한 SHAP 요인과 실제 재무 방어력이 서로 충돌하는지 확인하세요.",
    },
    "확인필요 보류": {
        "signal": "위험신호 아님",
        "tone": "neutral",
        "title": "관찰",
        "body": "현재 정보만으로는 결론을 세게 내기보다 근거를 더 모아야 하는 상태입니다.",
        "detail": (
            "모델과 외부근거가 뚜렷하게 같은 방향을 가리키지 않거나, 수집된 근거의 "
            "직접 관련성·최신성이 아직 충분하지 않습니다."
        ),
        "action": "누락된 공시, 최신 뉴스, 재무제표 주석, 동종업계 비교를 보완하세요.",
    },
}

COMMITTEE_DECISION_STAGE_GUIDE = [
    {
        "title": "적격",
        "signal": "위험신호 아님",
        "tone": "mitigate",
        "body": "현재 확인된 정보에서는 큰 위험 신호가 두드러지지 않는 단계입니다.",
        "detail": "모델 판단과 외부근거가 대체로 안정적인 방향으로 맞아떨어질 때 표시합니다.",
        "action": "정기 모니터링 관점에서 최신 공시와 등급 변화를 확인합니다.",
    },
    {
        "title": "관찰",
        "signal": "위험신호 아님",
        "tone": "neutral",
        "body": "당장 위험으로 단정하긴 어렵지만 흐름을 계속 지켜보면 좋은 단계입니다.",
        "detail": (
            "경계등급, 근거 부족, 모델 과민 가능성처럼 추가 확인이 필요한 경우를 "
            "이 단계로 묶어 보여줍니다."
        ),
        "action": "판단 이유를 함께 보고, 최신 공시·뉴스와 재무 방어력을 확인합니다.",
    },
    {
        "title": "위험 주의",
        "signal": "위험신호 있음",
        "tone": "risk",
        "body": "그냥 넘기기보다 먼저 확인해야 할 위험 신호가 있는 단계입니다.",
        "detail": "모델 확률, 보조 변수셋, 외부근거 중 하나 이상이 위험 쪽으로 기울 때 표시합니다.",
        "action": "손실 지속, 이자보상, 차입 부담, 직접 관련 공시·뉴스를 우선 확인합니다.",
    },
    {
        "title": "부적격",
        "signal": "위험신호 있음",
        "tone": "risk",
        "body": "정량·정성 근거를 종합할 때 신용위험이 높다고 보는 단계입니다.",
        "detail": "강한 재무 위험이나 신뢰도 높은 외부 위험 근거가 확인될 때 표시합니다.",
        "action": "투자 판단 전 핵심 위험 요인과 최신 공시를 반드시 재확인합니다.",
    },
]


def committee_user_stage_label(
    *,
    committee_label: str,
    decision_type_label: str,
    risk_signal: bool,
) -> str:
    """Map internal committee labels to a single user-facing decision stage."""
    if committee_label == "부적격":
        return "부적격"
    if committee_label == "적격":
        return "적격"
    if decision_type_label in {"과민경고 완화 보류", "경계등급 보류"}:
        return "관찰"
    if decision_type_label == "위험 보류" or risk_signal:
        return "위험 주의"
    if decision_type_label == "확인필요 보류" or committee_label == "보류":
        return "관찰"
    return committee_label or "관찰"


def committee_user_reason_label(decision_type_label: str, *, risk_signal: bool) -> str:
    """Return a short reason tag that avoids duplicating the final decision stage."""
    if decision_type_label == "과민경고 완화 보류":
        return "과민경고 완화"
    if decision_type_label == "경계등급 보류":
        return "경계등급 확인"
    if decision_type_label == "위험 보류" or risk_signal:
        return "위험 신호 확인"
    if decision_type_label == "확인필요 보류":
        return "근거 추가 확인"
    return decision_type_label or "근거 추가 확인"


def committee_decision_type_info(
    decision_type_label: str,
    *,
    risk_signal: bool,
) -> dict[str, str]:
    """Return user-facing copy for a committee decision subtype."""
    info = COMMITTEE_DECISION_TYPE_GUIDE.get(decision_type_label)
    if info is not None:
        return dict(info)
    if decision_type_label == "부적격" or risk_signal:
        return {
            "signal": "위험신호 있음",
            "tone": "risk",
            "title": decision_type_label or "위험 판단",
            "body": "위원회가 실제 위험 경고로 볼 만한 신호가 있다고 정리한 상태입니다.",
            "detail": "모델 판단만이 아니라 재무·외부근거를 함께 보아 위험 쪽으로 해석했습니다.",
            "action": "핵심 위험 요인과 외부 근거를 우선 확인합니다.",
        }
    if decision_type_label == "적격":
        return {
            "signal": "위험신호 아님",
            "tone": "mitigate",
            "title": "적격",
            "body": "현재 2차 위원회가 추가 위험신호를 강하게 보지 않은 상태입니다.",
            "detail": "모델 판단과 확인된 근거가 대체로 무리 없이 맞아떨어진 경우입니다.",
            "action": "다만 최신 공시나 뉴스가 바뀌면 다시 확인합니다.",
        }
    return {
        "signal": "위험신호 아님",
        "tone": "neutral",
        "title": "관찰",
        "body": "위험 여부를 단정하기보다 추가 확인이 필요한 상태입니다.",
        "detail": "아직 판단을 강하게 밀어줄 근거가 충분하지 않거나 근거끼리 방향이 엇갈립니다.",
        "action": "근거의 최신성, 직접 관련성, 재무 완충력을 함께 봅니다.",
    }
