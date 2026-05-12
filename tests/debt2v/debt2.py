import json
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    from agno.agent import Agent
    from agno.models.anthropic import Claude
    # from test_cases import dummy_case_1, dummy_case_2, dummy_case_3 # 모듈화 시 주석 해제
except ImportError as e:
    logging.error(f"모듈 Import 실패: {e}")
    raise SystemExit(1) from e

# ======== 테스트 케이스 1: 겉보기엔 멀쩡하지만 속은 곪은 기업 (유동성 부족) ========
dummy_case_1 = {
    "company_info": {"회사명": "(주)빛좋은테크", "거래소코드": "100001"},
    "model_view": "적격",
    "financial_metrics": {
        "T21000(부채비율)": "140% (업계 평균 수준)",
        "T61000(유동비율)": "55% (단기 현금 고갈 상태)",
        "T54000(영업활동현금흐름)": "10억 (흑자 유지 중)",
        "T56000(재무활동현금흐름)": "50억 (단기 차입 증가)"
    }
}

# ======== 테스트 케이스 2: 과거엔 망가졌으나 맹렬히 회복 중인 기업 (회생 가능성) ========
dummy_case_2 = {
    "company_info": {"회사명": "오뚝이산업(주)", "거래소코드": "200002"},
    "model_view": "부적격",
    "financial_metrics": {
        "T21000(부채비율)": "450% (과거 누적 적자로 인한 자본잠식 위험)",
        "T61000(유동비율)": "120% (단기 유동성 회복)",
        "T54000(영업활동현금흐름)": "+350억 (올해 대규모 흑자 전환 및 현금 유입)",
        "T56000(재무활동현금흐름)": "-200억 (과거 고금리 차입금 조기 상환 중)"
    }
}

# ======== 테스트 케이스 3: 전형적인 폰지(돌려막기) 구조 기업 (부채의 질 악화) ========
dummy_case_3 = {
    "company_info": {"회사명": "(주)신기루건설", "거래소코드": "300003"},
    "model_view": "보류",
    "financial_metrics": {
        "T21000(부채비율)": "300%",
        "T61000(유동비율)": "150% (현금 보유량은 많아 보임)",
        "T54000(영업활동현금흐름)": "-400억 (본업에서 심각한 현금 유출 지속)",
        "T56000(재무활동현금흐름)": "+450억 (대규모 신규 회사채 발행으로 연명)"
    }
}

# ======== 테스트용 리스트로 묶어서 내보내기 (main.py에서 import 하기 위함) ========
test_cases_list = [dummy_case_1, dummy_case_2, dummy_case_3]

# ======== 부채상환능력, 유동성, 현금흐름 집중 에이전트 ========
debt_liquidity_agent = Agent(
    name="DebtLiquidity_Agent",
    model=Claude(id="claude-opus-4-7"),
    instructions=[
        "당신은 기업의 부채상환능력, 유동성, 현금흐름, 단기 방어력을 전문적으로 검토하는 재무 애널리스트입니다.",
        "1단계 정량 모델의 결과(model_view)를 재무 지표로 정밀 검증하는 데 집중해야 합니다.",
        "당신의 임무는 XGBoost 정량 모델의 판단(적격/부적격)을 덮어쓰는 것이 아닙니다. 해당 판단을 지지하거나 경고할 수 있는 '부채 및 유동성 관점의 정성적 근거'를 마련하는 것입니다.",
        "입력된 재무 지표(부채, 유동성, 현금흐름)를 바탕으로, 기업이 외부 충격 시 실제로 버틸 수 있는 재무적 체력이 있는지 냉철하게 평가하세요.",
        "가중치를 부여하는 가장 주요 기준은 다음과 같습니다.",
        "1. [비판적 검토]: model_view가 '적격'이라도 T61000(유동비율)이 100% 미만이면 가중 처벌하여 리스크를 부각하십시오.",
        "2. [회생 가능성 검토]: model_view가 '부적격'이라도 T54000(영업현금흐름)이 양수(+)이면서 개선 중이면 가산점을 주어 의견을 완화하십시오.",
        "3. [부채의 질 평가]: T21000(부채비율)이 높더라도 재무활동현금흐름(T56000)이 상환 위주(-)라면 긍정적 신호로 해석하십시오.",
        "4. [반전가능성]: 반대의 경우로도 기능할 수 있습니다.",
        "그 중에서도 핵심 지표인 T21000(부채비율), T61000(유동비율), T54000(영업활동현금흐름), T56000(재무활동현금흐름)의 수치와 상호작용을 깊이 있게 분석하십시오.",
        "반드시 아래의 정확한 JSON 형식으로만 응답을 출력하세요. 마크다운 기호(```json)나 다른 설명 텍스트를 절대 포함하지 말고 순수 JSON 문자열만 출력해야 합니다.",
        "또한, 이모지는 추가하지 않습니다.",
        "{",
        '  "model_view_feedback": "1단계 결과에 대한 검증 의견 (동의/반박 및 근거)",',
        '  "weighted_score_rationale": "가중치 적용 핵심 사유 (2문장)",',
        '  "debt_repayment_capacity": "부채상환능력 상세 평가",',
        '  "domain_risk_level": "최종 위험도 (안전/주의/위험)"',
        "}"
    ]
)


# ======== 실행 로직 (유연성, 안정성, 기업명 연동 추가) ========
def run_weighted_analysis_test(test_data: dict):
    """
    테스트 케이스(딕셔너리)를 통째로 받아 기업명과 상태를 추출하고 분석합니다.
    """
    # 💡 기업명과 1단계 결과값을 데이터 구조에서 안전하게 추출합니다.
    company_name = test_data.get("company_info", {}).get("회사명", "알수없는 기업")
    model_status = test_data.get("model_view", "상태 미상")

    print("="*70)
    print(f"🔍 [{company_name}] 가중 분석 시작 (1단계 모델 판정: {model_status})")
    print("="*70)

    # 프롬프트 동적 생성
    query = f"1단계 결과인 '{model_status}'를 중심으로 아래 데이터를 가중 분석하세요: {json.dumps(test_data, ensure_ascii=False)}"

    # 에이전트 실행
    response = debt_liquidity_agent.run(query)

    # 💡 JSON 파싱 및 서버 에러 방지 (프로덕션 환경 필수)
    try:
        result = json.loads(response.content.strip())
        print(" 성공적으로 파싱된 JSON 결과입니다:\n")
        print(f" [1단계 피드백]: {result.get('model_view_feedback', 'N/A')}")
        print(f" [가중치 사유]: {result.get('weighted_score_rationale', 'N/A')}")
        print(f" [상환능력평가]: {result.get('debt_repayment_capacity', 'N/A')}")
        print(f" [최종위험도]: {result.get('domain_risk_level', 'N/A')}")
        print("="*70)
    except json.JSONDecodeError as e:
        logging.error(f"에이전트가 JSON 형식을 위반했습니다. 파싱 실패: {e}")
        logging.error(f"원시 응답:\n{response.content}")

# 실행부 (원하는 더미 데이터를 넣기만 하면 됩니다)
if __name__ == "__main__":
    # Part 2의 더미 데이터가 이 파일 안에 있다고 가정합니다.
    # run_weighted_analysis_test(dummy_case_1)
    # run_weighted_analysis_test(dummy_case_2)
    pass # 실제 구동 시 위에 있는 주석을 풀고 실행하세요.
