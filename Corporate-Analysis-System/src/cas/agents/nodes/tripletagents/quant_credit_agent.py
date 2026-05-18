import json

from agno.agent import Agent
from agno.models.anthropic import Claude
from pydantic import BaseModel, Field

from cas.agents.state import AgentOutput, AgentState


# ==========================================
# 1. Pydantic 스키마 정의 (LLM의 사고 구조 강제)
# ==========================================
class QuantCreditOutput(BaseModel):
    """내부 재무 분석 에이전트의 출력 스키마입니다."""

    quantitative_interpretation: str = Field(
        description="XGBoost 예측 확률, SHAP 주요 변수, 산업/시장 평균 비교를 바탕으로 모델의 원판단 논리 요약"
    )
    fundamental_defense_capacity: str = Field(
        description="부채상환능력, 유동성, 현금흐름 지표를 바탕으로 실제 펀더멘털과 단기 방어력 진단"
    )
    key_risk_and_mitigation: str = Field(
        description="모델이 놓쳤거나 완화할 수 있는 재무 팩터 및 학술적 근거 기반의 취약 지점 요약"
    )
    internal_risk_level: str = Field(
        description="순수 재무 관점에서의 위험도 (안전 / 주의 / 위험 중 택 1)"
    )


# ==========================================
# 2. 에이전트 객체 선언 (Agno 기능 이식)
# ==========================================
quant_agent = Agent(
    name="QuantCredit_Agent",
    model=Claude(id="claude-3-5-sonnet-latest"),  # 공식 최신 모델 적용
    response_model=QuantCreditOutput,  # 💡 Agno가 알아서 Pydantic 객체로 변환
    instructions=[
        "당신은 3인 신용평가 위원회의 '내부 펀더멘털 총괄 책임자(QuantCreditAgent)'입니다.",
        "당신의 역할은 두 가지입니다: 1) XGBoost의 정량적 판단과 산업/시장 비교를 통한 위치 파악, 2) 부채상환능력, 유동성, 현금흐름을 통한 기업의 실제 단기 방어력 검토.",
        "분석 시 다음의 학술적/실무적 신용평가 기준을 반드시 적용하여 가중치를 판단하세요:",
        "- [Altman Z-Score 원칙]: 부채비율이 높더라도 영업이익(T44100) 및 이자보상배율이 우수하다면 방어력이 있는 것으로 평가하라.",
        "- [Beaver 현금흐름 원칙]: 당기순이익이 적자라도 영업활동현금흐름(T54000)이 양수이고 개선 중이면 긍정적 회생 신호로 가중치를 부여하라.",
        "- [S&P 유동성 룰]: 모델이 '적격'을 주었더라도 유동비율(T61000)이 100% 미만이면 즉각적인 유동성 위기 징후로 보고 강하게 경고하라.",
        "절대 외부 뉴스나 소송 등은 언급하지 마시고, 오직 제공된 '재무 데이터와 모델 결과'에만 집중하세요.",
    ],
)


# ==========================================
# 3. 외부 호출 래퍼(Wrapper) 함수
# ==========================================
def build_quant_agent_output(state: AgentState, xgb: dict) -> AgentOutput:
    """Quant 에이전트의 결과를 생성하는 함수입니다."""
    company_name = str(state.get("company_name") or "unknown")
    source_row = dict(state.get("source_feature_row") or {})
    peer_rows = state.get("peer_comparison_rows") or []

    prediction_label = str(xgb.get("prediction_label", "unknown"))
    probability = float(xgb.get("probability_speculative", 0.0) or 0.0)
    top_drivers = xgb.get("top_drivers", [])
    query = f"""
    기업명: {company_name}
    1단계 모델 라벨: {prediction_label} (투기등급 위험확률: {probability:.1%})
    [모델의 주요 판단 근거 (SHAP Top Drivers)]
    {json.dumps(top_drivers, ensure_ascii=False, indent=2)}
    [핵심 재무 지표 원본 (Source Row 추출)]
    - 부채비율(debt_ratio): {source_row.get("debt_ratio", "N/A")}
    - 유동비율(current_ratio): {source_row.get("current_ratio", "N/A")}
    - 이자보상배율(interest_coverage_ratio): {source_row.get("interest_coverage_ratio", "N/A")}
    - 2년 연속 영업현금 적자 여부: {source_row.get("is_2y_consecutive_ocf_deficit", "N/A")}
    [산업 및 시장 비교 백분위 데이터 (Peer Comparison)]
    {json.dumps(peer_rows, ensure_ascii=False, indent=2)}
    위 데이터를 종합하고 지시된 학술적 원칙(Altman, Beaver, S&P)을 적용하여 보고서를 작성하세요.
    """
    try:
        response = quant_agent.run(query)

        # 💡 Pydantic 파이썬 객체로 즉시 접근
        result_data = response.content

        summary = (
            f"QuantCreditAgent는 XGBoost의 {prediction_label}({probability:.1%}) 판단과 산업 백분위를 결합하고, "
            f"유동성/현금흐름 지표를 학술적 룰 기반으로 검증했습니다. 내부 재무 위험도는 '{result_data.internal_risk_level}'입니다."
        )

        findings = [
            f"정량 해석 및 비교: {result_data.quantitative_interpretation}",
            f"펀더멘털 방어력: {result_data.fundamental_defense_capacity}",
            f"위험 및 완화 요인: {result_data.key_risk_and_mitigation}",
        ]

    except Exception as e:
        summary = f"QuantCreditAgent 실행 중 에러 발생: {e!s}"
        findings = ["LLM 분석 수행 불가. 원본 model_view 참조 요망."]

    # committee_node.py가 기다리고 있는 AgentOutput 포맷으로 최종 반환
    return AgentOutput(
        role="quant_credit",
        summary=summary,
        findings=findings,
        confidence=0.85 if xgb else 0.35,
    )
