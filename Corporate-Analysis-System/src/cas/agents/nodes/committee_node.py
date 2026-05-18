"""Run the Stage 2 three-agent Agno LLM review scaffold."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agents.nodes.tripletagents.chair_report_agent import build_chair_agent_output
from agents.nodes.tripletagents.evidence_audit_agent import build_evidence_agent_output

# 💡 우리가 만든 진짜 Agno 에이전트 모듈 임포트
# (선생님이 지정하신 agents/nodes/tripletagents/ 경로 기준)
from agents.nodes.tripletagents.quant_credit_agent import build_quant_agent_output

from cas.agents.state import AgentState, AuditEntry, CommitteeReview


def run(state: AgentState) -> dict[str, Any]:
    """Agno 기반 3인 위원회를 순차적으로 실행하여 최종 결과를 도출합니다."""
    # 1. 1단계 모델 예측 결과 및 초기 상태 확보
    xgb_result = dict(state.get("xgboost_result") or {})
    recommendation = state.get("final_recommendation", "review")

    print("\n======================================================================")
    print("🏛️ [Committee Node] Agno 3인 체제 위원회 심사를 시작합니다...")
    print("======================================================================\n")

    # ==========================================
    # 2. 에이전트 순차 실행 (Sequential Execution)
    # ==========================================

    # [Step 1] 재무/정량 분석가 실행
    quant_agent_output = build_quant_agent_output(state, xgb_result)
    print(f"✅ QuantCreditAgent 심사 완료 (위험도: {quant_agent_output.confidence:.2f})")

    # [Step 2] 외부/리스크 분석가 실행
    evidence_agent_output = build_evidence_agent_output(state, xgb_result)
    print(f"✅ EvidenceAuditAgent 심사 완료 (위험도: {evidence_agent_output.confidence:.2f})")

    # [Step 3] 위원장(의장) 실행 (앞선 두 에이전트의 결과를 종합하여 최종 판결)
    # state 버스에 임시로 두 에이전트의 결과를 싣고 의장에게 넘깁니다.
    state["agent_outputs"] = [quant_agent_output, evidence_agent_output]
    chair_agent_output = build_chair_agent_output(state, xgb_result)
    print("✅ ChairReportAgent 최종 심의 완료\n")

    # ==========================================
    # 3. 결과물 패키징 및 State 반환
    # ==========================================
    agents = [quant_agent_output, evidence_agent_output, chair_agent_output]

    # 시스템 스키마(Schema) 호환성을 위한 리뷰 객체 변환
    reviews = [
        CommitteeReview(
            perspective=agent.role,
            recommendation=recommendation,
            confidence=agent.confidence,
            rationale=agent.summary,
        )
        for agent in agents
    ]

    # 프론트엔드/대시보드 표시용 요약본 (가장 중요)
    agent_summary = {
        "final_recommendation": recommendation,
        "final_confidence": chair_agent_output.confidence,
        "synthesis": chair_agent_output.summary,
        "agents": {
            agent.role: {
                "summary": agent.summary,
                "findings": agent.findings,
                "confidence": agent.confidence,
            }
            for agent in agents
        },
    }

    # 감사 로그(Audit Trail) 기록
    audit = AuditEntry(
        node="agno_committee_node",
        timestamp=_now(),
        summary="Agno 기반 3인 체제 Stage 2 심사 완료",
        metrics={
            "n_agents": 3.0,
            "final_confidence": chair_agent_output.confidence,
        },
    )

    # AgentState 버스에 업데이트될 최종 데이터 반환
    return {
        "agent_outputs": agents,
        "committee_reviews": reviews,
        "agent_summary": agent_summary,
        "audit": [audit],
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
