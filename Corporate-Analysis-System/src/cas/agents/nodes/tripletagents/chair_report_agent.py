import json
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.anthropic import Claude
from cas.agents.state import AgentState, AgentOutput

# ==========================================
# 1. Pydantic 스키마 정의 (원본 유지)
# ==========================================
class ChairReportOutput(BaseModel):
    final_committee_label: str = Field(
        description="최종 위원회 라벨 (적격 / 보류 / 부적격 중 하나로 제시)"
    )
    veto_triggered: bool = Field(
        description="리서치 결과 치명적 리스크로 인해 비토권(강제 강등)이 발동되었는지 여부 (True/False)"
    )
    conflict_resolution: str = Field(
        description="정량 해석과 외부 근거가 충돌할 경우, 어떤 근거에 가중치를 두어 판결했는지 논리 구성 (2~3문장)"
    )
    executive_summary: str = Field(
        description="최종 투자 심의 메모 (마크다운 포맷의 종합 보고서)"
    )

# ==========================================
# 2. 에이전트 객체 선언 (✨ Agno 기능 이식)
# ==========================================
chair_agent = Agent(
    name="ChairReport_Agent",
    model=Claude(id="claude-3-5-sonnet-latest"), # 공식 최신 모델 적용
    response_model=ChairReportOutput,            # 💡 Agno가 완벽한 객체로 파싱!
    instructions=[
        "당신은 3인 신용평가 위원회의 '최종 의사결정권자(ChairReportAgent)'입니다.",
        "내부 재무 분석가(QuantCredit)의 정량 해석과 외부 리스크 분석가(EvidenceAudit)의 증거를 종합하여 최종 위원회 의견을 작성하세요.",
        "절대 규칙 [Veto Power]: 외부 리스크 분석가가 'has_critical_risk'를 띄웠다면, 재무 점수가 아무리 좋아도 최종 라벨을 무조건 '부적격'으로 강등(Veto)시킵니다.",
        "두 에이전트의 의견이 충돌할 경우, 어떤 의견에 가중치를 두어 최종 결정을 내렸는지 'conflict_resolution'에 명확히 기재하세요."
        # (불필요한 JSON 문자열 출력 지시문 삭제 완료)
    ]
)

# ==========================================
# 3. 외부 호출 래퍼(Wrapper) 함수
# ==========================================
def build_chair_agent_output(state: AgentState, xgb: dict = None) -> AgentOutput:
    """
    committee_node.py에서 호출하는 함수. 
    앞선 두 에이전트의 결과를 state에서 꺼내어 최종 판결을 내립니다.
    """
    # 💡 committee_node에서 넘겨준 두 에이전트의 결과물을 꺼내옵니다.
    agent_outputs = state.get("agent_outputs", [])
    
    quant_findings_text = "결과 없음"
    evidence_findings_text = "결과 없음"
    confidence = 0.5
    
    if len(agent_outputs) >= 2:
        quant_findings_text = "\n".join(agent_outputs[0].findings)
        evidence_findings_text = "\n".join(agent_outputs[1].findings)
        # 앞선 두 에이전트의 평균 신뢰도를 계산 (최소 0.5 이상 보장)
        confidence = max(0.5, (agent_outputs[0].confidence + agent_outputs[1].confidence) / 2)

    prediction_label = str(xgb.get("prediction_label", "unknown")) if xgb else "unknown"
    
    query = f"""
    [1단계 기계 학습 원본 판단]
    모델 라벨: {prediction_label}
    
    [내부 재무 분석가 (QuantCredit) 소견]
    {quant_findings_text}
    
    [외부 리스크 분석가 (EvidenceAudit) 소견]
    {evidence_findings_text}
    
    위 3가지 의견을 종합하여, 비토권 발동 여부를 심사하고 최종 투자 심의 보고서를 작성하세요.
    """
    
    try:
        response = chair_agent.run(query)
        
        # 💡 지저분한 문자열 파싱 로직 삭제, Pydantic 객체로 바로 접근
        result_data = response.content 
        
        veto_status = "발동됨 (중대 리스크 발견)" if result_data.veto_triggered else "미발동"
        
        # executive_summary는 전체 보고서이므로 요약(summary) 필드에 담아 상위로 넘깁니다.
        summary = result_data.executive_summary
        
        findings = [
            f"최종 위원회 라벨: {result_data.final_committee_label}",
            f"비토권(Veto) 발동 여부: {veto_status}",
            f"의견 충돌 조율 논리: {result_data.conflict_resolution}",
            "종합 심사 메모: (최종 위원회 심사 결과 참조)" 
        ]
        
    except Exception as e:
        summary = f"ChairReportAgent 실행 중 에러 발생: {str(e)}"
        findings = ["최종 의견을 종합하지 못했습니다."]

    return AgentOutput(
        role="chair_report",
        summary=summary,
        findings=findings,
        confidence=confidence,
    )