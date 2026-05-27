# 3에이전트 기반 신용위험 검토 구조 설계

## 1. 개편 배경

기존 Stage 2는 5개 에이전트 구조로 설계되어 있었다.
그러나 기존 세부 에이전트 중 일부는 모두 모델 결과와 재무/시장 위험을
해석한다는 점에서 역할 경계가 겹쳐 보일 수 있었다.

이번 개편은 에이전트 수를 줄이는 것이 목적이 아니라,
**추론이 필요한 역할만 남겨 각 에이전트의 책임을 선명하게 만드는 것**을 목표로 한다.

CAS의 Stage 2는 다음 3개 역할로 정리한다.

1. `QuantCreditAgent`: 정량 모델 판단의 근거를 해석한다.
2. `EvidenceAuditAgent`: 외부 근거와 숨은 꼬리 위험을 검증한다.
3. `ChairReportAgent`: 두 관점을 종합해 최종 위원회 의견을 작성한다.

현재 구현 상태는 설계 초안이 아니라, deterministic runner와 선택형 Agno runner가
같은 `Stage2AgentRunner` 계약으로 동작하는 파일럿 검증 단계다. `committee_view`
strict schema, 보류 세부 유형, `committee_risk_signal`, decision trace, live API
runbook, 성능 리포트 산출물까지 연결되어 있다.

파일럿 성능은 전체 기업 모집단 정확도가 아니라, FN/FP/경계등급/TP/TN을 의도적으로
섞은 hard sample에서 Stage 2가 1차 모델 판단을 얼마나 보완하는지 보는 지표다.
30건 stress sample 기준 1차 모델 F1은 `0.4243`, 2차 위험신호 F1은 `0.6666`,
2차 검토대상(`보류+부적격`) Recall은 `1.0000`이다.

## 2. 기본 원칙

Stage 2는 Stage 1 모델 결과를 덮어쓰지 않는다.

### model_view

- XGBoost 기반 원본 판단
- 예측확률(`y_proba`)
- 모델 라벨: `투자적격` / `부적격`
- 위험 밴드
- SHAP 기반 주요 변수

`model_view`는 모델의 원판단으로 간주한다.
에이전트나 LLM은 이 값을 직접 수정하지 않고, 해석과 보완 근거를 추가한다.

### committee_view

- 3개 에이전트가 정량 해석과 외부 근거를 검토한 뒤 제시하는 종합 의견
- 위원회 라벨: `적격` / `보류` / `부적격`
- 모델 판단 유지 여부, 경계 의견, 추가 점검 필요 여부를 설명형 의견으로 제시

최종 화면에서는 **모델이 본 이진 판단(model_view)**과
**위원회가 검토한 3단계 종합 의견(committee_view)**을 함께 보여준다.

## 3. Stage 2 에이전트 구성

### 1. QuantCreditAgent

역할:

- XGBoost 결과와 SHAP 기반 주요 변수 해석
- 핵심 재무변수와 산업/시장 비교 결과를 바탕으로 모델 판단의 근거 설명
- 모델 판단이 재무적으로 타당한지, 산업/시장 비교와 모순되는 부분은 없는지 점검

입력:

- 예측확률
- 모델 라벨
- 위험 밴드
- SHAP 상위 변수
- 핵심 재무지표
- 시장/산업 비교 결과

출력:

- 정량 해석 요약
- 핵심 위험 요인
- 완화 요인
- 모델 판단의 설명 가능성 점검

### 2. EvidenceAuditAgent

역할:

- 뉴스, DART 공시, 재무제표 주석, 거시환경, 산업 특성을 결합해 숨은 꼬리 위험 탐색
- 재무제표에 바로 드러나지 않는 위험 신호를 검토
- 부채상환능력, 유동성, 현금흐름, 시장 민감도를 외부 근거와 함께 교차 검증
- 근거의 출처 신뢰도, 최신성, 중복 여부, 루머 가능성을 점검

입력:

- 기업 관련 뉴스
- DART 공시
- 재무제표 주석
- 금리, 환율, 회사채 스프레드 등 거시 참고 지표
- 산업 특성
- 부채, 유동성, 현금흐름 관련 핵심 지표

출력:

- 핵심 외부 사건 요약
- 근거 신뢰도 정리
- 숨은 꼬리 위험 또는 완화 근거
- 부채/유동성 관점의 교차 검증 의견
- 거시/산업 환경 민감도 의견

### 3. ChairReportAgent

역할:

- QuantCreditAgent와 EvidenceAuditAgent의 의견을 종합
- 모델 원판단과 정성 검토 결과를 구분해 종합 심사 메모 작성
- 최종 위원회 라벨을 `적격` / `보류` / `부적격` 중 하나로 제시

입력:

- QuantCreditAgent 결과
- EvidenceAuditAgent 결과
- Stage 1 `model_view`
- 검증된 evidence bundle

출력:

- 최종 위원회 라벨 (`적격` / `보류` / `부적격`)
- 주요 위험 요인
- 완화 요인
- 종합 심사 메모

## 4. 회의 진행 구조

3에이전트 검토는 다음 3단계로 진행한다.

### 1라운드: 정량 판단 해석

`QuantCreditAgent`가 Stage 1 모델 결과를 해석한다.

- 모델이 왜 `투자적격` 또는 `부적격`으로 판단했는지 설명
- SHAP 상위 변수와 실제 재무지표를 연결
- 산업/시장 중앙값과 비교해 모델 판단이 직관적인지 점검

### 2라운드: 외부 근거 및 숨은 위험 검증

`EvidenceAuditAgent`가 모델과 재무제표만으로는 포착하기 어려운 정보를 검토한다.

- 뉴스/공시/주석에서 중요한 사건이 있는지 확인
- 금리, 환율, 회사채 스프레드 등 거시환경 변화에 취약한지 검토
- 유동성, 부채상환능력, 현금흐름이 모델 판단과 충돌하는지 확인
- 근거의 신뢰도와 최신성을 구분

예시:

- 정량상 `투자적격`이지만 공시상 유동성 위험이 큰 경우
- 정량상 `부적격`이지만 영업현금흐름과 차입금 상환이 빠르게 개선되는 경우
- 뉴스상 우려는 있으나 공시 근거가 약한 경우
- 거시환경 악화 시 차환 부담이 급격히 커질 가능성이 있는 경우

### 3라운드: 최종 종합

`ChairReportAgent`가 두 에이전트의 결과를 종합한다.

- `model_view`는 원본 판단으로 보존
- `committee_view`에는 정량 해석과 외부 검증 의견을 분리해 기록
- 최종 위원회 라벨은 `적격` / `보류` / `부적격` 중 하나로 제시

라벨 기준:

- `적격`: 모델 판단과 정성 근거를 종합할 때 투자적격으로 보는 경우
- `보류`: 당장 부적격으로 보기는 어렵지만 추가 점검 또는 보수적 해석이 필요한 경우
- `부적격`: 정량·정성 근거를 종합할 때 투자 위험이 높다고 보는 경우

## 5. 입력 번들 설계

에이전트에는 원시 CSV 전체를 넘기지 않는다.
정량 모델과 대시보드에서 이미 정리한 설명용 번들을 입력으로 사용한다.

권장 입력 필드:

- 기업명
- 종목코드
- 시장
- 산업
- 규모
- 회계연도
- 예측확률
- 모델 라벨
- 위험 밴드
- SHAP 기반 주요 변수
- 핵심 재무지표
- 시장/산업 비교 결과
- 뉴스 요약
- 공시 요약
- 재무제표 주석 요약
- 금리, 환율, 회사채 스프레드

### 외부 근거 수집 키

외부 API 키는 코드나 문서에 직접 적지 않고 `.env`에만 저장한다.
기본값은 외부 호출을 끈 상태이며, 로컬 데모에서만 명시적으로 켠다.

```bash
CAS_ENABLE_EXTERNAL_EVIDENCE=1
CAS_STAGE2_RUNNER=deterministic
CAS_STAGE2_AGNO_MODE=single
CAS_STAGE2_MODEL_PROVIDER=openai
CAS_STAGE2_MODEL=gpt-4.1-mini
OPENDART_API_KEY=...
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
TAVILY_API_KEY=...
# Keep real API keys out of commits and PR bodies.
OPENAI_API_KEY=
```

현재 CAS는 `news_cache` 노드에서 Naver 뉴스, Tavily 검색, OpenDART 공시를
선택적으로 수집할 수 있는 구조만 제공한다.
CI와 일반 로컬 실행의 재현성을 위해 `CAS_ENABLE_EXTERNAL_EVIDENCE=0`이면
외부 API를 호출하지 않는다.

OpenDART는 정확한 기업 공시 검색을 위해 `corp_code`가 필요하다.
`company_selection`에 `corp_code`가 있으면 그대로 사용하고, 없으면
OpenDART `corpCode.xml` 캐시에서 `stock_code -> corp_code`를 자동 보강한다.
공시는 분석 기준일(`as_of_date`, 없으면 `eval_year` 말일) 이전 자료만 조회해
미래 공시가 과거 판단에 섞이지 않도록 한다. 조회 대상은 주요사항보고,
외부감사관련, 거래소공시, 정기공시로 나누고, 횡령·배임·감사의견·상장폐지 등
위험 공시는 `provider_relevance=risk`로 우선 검토한다.

OpenDART 공시는 제목 키워드만으로 즉시 치명 리스크로 보지 않고,
`disclosure_severity`, `disclosure_event_class`, `disclosure_materiality`로 한 번 더
분류한다. 일정금액 미만 또는 자율공시 소송, 자율공시 단일 계약해지,
SPAC 합병 예비심사 등 절차성 거래정지는 `caution/procedural_or_one_off` 맥락으로
낮춰 EvidenceAuditAgent가 참고 근거로만 다루게 한다. 반대로 횡령·배임,
감사의견 거절, 상장폐지, 관리종목, 영업정지, 자본잠식처럼 실질 부실 사건은
`adverse` 또는 `veto`로 유지한다.

계약해지, 영업정지, 자금조달, 채무보증, 소송은 제목만으로는 실질성을 알기 어렵기
때문에 OpenDART 상세 공시 보강을 추가했다. `단일판매ㆍ공급계약해지` 후보는
`document.xml` 원문에서 매출 대비 계약해지 비율을 파싱하고, `영업정지` 후보는
`bsnSp.json`의 `sl_vs` 또는 `bsnsp_amt/rsl`로 매출 대비 영업정지 비율을 계산한다.
자금조달은 발행금액/자기자본과 희석률, 채무보증은 보증금액/자기자본, 소송은
청구금액/자기자본 또는 매출액을 파싱한다. 이 값은 `materiality_ratio`,
`materiality_basis`, `materiality_source`, 필요 시 `dilution_ratio`로
EvidenceAuditAgent에 전달된다. 기업 규모 대비 3% 미만은 절차성/일회성, 3~10%는
관찰 수준, 10% 이상은 실질 부정 공시로 유지해 정상기업 과잉 보류를 줄이는 재료로
사용한다.
다만 자금조달·채무보증의 10% 이상 materiality는 단독으로 정상기업을 `risk_hold`로
올리는 근거가 아니다. `veto_candidate`, `critical_context_confirmed`, 자본잠식·부도·
상장폐지 같은 hard distress 문맥, 또는 현금흐름/이자보상/손익/레버리지 중 2축 이상의
재무 스트레스가 함께 있을 때 숨은 꼬리위험 또는 RiskRecallQA의 실질 외부 위험으로 본다.
이 guardrail은 일회성 또는 계열사 지원성 채무보증을 실제 부실 전이와 구분하기 위한
정상기업 과잉 보류 방어 장치다.
반복 채무보증처럼 일부 재무약점과 함께 있어 보류 자체는 유지하되 치명 문맥과 현금흐름
악화가 없는 경우에는 hidden-tail-risk를 `risk_hold`가 아니라 `review_hold`로 표시한다.
`bsnSp.json`에 상세 비율이 없으면 `document.xml` 원문 fallback으로 `최근매출액 대비`,
`영업정지금액`, `최근매출액` 표 값을 다시 파싱한다. 이 fallback은 종속회사 영업정지가
모회사 신용위험으로 바로 전이되는지 판단하는 보조 근거다.

Agno/LLM 추론은 CI와 일반 재현 실행에서는 꺼 둔다.
`CAS_STAGE2_RUNNER=deterministic`이면 규칙 기반 scaffold가 실행되고,
`CAS_STAGE2_RUNNER=agno`로 바꾸면 `Stage2AgentRunner` 인터페이스를 통해 Agno
structured output 기반 실행으로 교체된다. 기본 live 모드는
`CAS_STAGE2_AGNO_MODE=single`이며 provider/model은 `CAS_STAGE2_MODEL_PROVIDER`,
`CAS_STAGE2_MODEL` 또는 batch CLI의 `--stage2-model-provider`, `--stage2-model`로
선택한다. 기본 provider/model catalog는 `configs/dashboard/llm_models.yaml`에서
대시보드와 Stage 2가 함께 읽는다. `single` 모드는 한 provider를 쓰되 QuantCredit/EvidenceAudit/ChairReport
세 역할 agent를 분리 실행한다. live 지연시간을 측정할 때는 LLM 응답 캐시를 끄기 위해
`CAS_STAGE2_LLM_CACHE_ENABLED=0` 또는 batch CLI의 `--no-stage2-llm-cache`를 사용한다.
여러 LLM 관점을 비교할 때만 `multi_llm_committee` 모드를 사용한다. 현재 기본 역할 배정은
`gemini_quant_claude_evidence_openai_chair`이며, QuantCredit는 Gemini 2.5 Flash,
EvidenceAudit는 Claude Sonnet, ChairReport는 OpenAI gpt-4.1-mini를 사용한다.

이때 모델 판단은 계속 `model_view`에 보존하고, Agno 결과는 `committee_view`를
설명·보완하는 용도로만 사용한다. 실제 기업-회계연도와 외부근거 질의를 API로
전송하는 live batch는 로컬 opt-in 실행으로 분리하며, Codex/CI 환경에서는 정책과
재현성 때문에 deterministic 경로 또는 기존 캐시 산출물을 사용한다.

## 6. 최종 출력 구조

### model_view

- 예측확률(`y_proba`)
- 모델 라벨 (`투자적격` / `부적격`)
- 위험 밴드
- 주요 변수 기여도
- 시장/산업 비교 요약

### committee_view

`committee_view`는 최종 위원회 결과를 설명하기 위한 구현된 출력 계약이다.

```json
{
  "final_committee_label": "보류",
  "committee_decision_type": "risk_hold",
  "committee_decision_type_label": "위험 보류",
  "committee_risk_signal": true,
  "veto_triggered": false,
  "hidden_tail_risk_flag": true,
  "hidden_tail_risk_reason": "모델은 투자적격으로 봤지만 직접 관련 외부 위험 근거가 확인되어 FN 가능성을 보수적으로 점검",
  "conflict_resolution": "모델 원판단과 외부 검증 근거가 충돌하는 지점을 어떻게 조율했는지 설명",
  "key_risk_factors": [
    "최종 판단에서 중요하게 본 위험 요인"
  ],
  "mitigating_factors": [
    "위험을 낮춰주는 완화 요인"
  ],
  "evidence_summary": [
    {
      "source": "model_view",
      "summary": "근거 요약",
      "reliability": "high"
    }
  ],
  "final_review_memo": "사람이 읽는 최종 심사 메모"
}
```

필드 의미:

- `final_committee_label`: 최종 위원회 라벨 (`적격` / `보류` / `부적격`)
- `committee_decision_type`: `보류`를 세분화한 내부 판단 유형 (`eligible`, `risk_hold`, `mitigation_hold`, `review_hold`, `reject`)
- `committee_decision_type_label`: 사용자에게 보여줄 세부 판단명 (`적격`, `위험 보류`, `과민경고 완화 보류`, `확인필요 보류`, `부적격`)
- `committee_risk_signal`: Precision/Recall 계산 시 실제 위험 신호로 볼지 여부. `위험 보류`와 `부적격`은 `true`, `과민경고 완화 보류`와 `확인필요 보류`는 기본적으로 `false`
- `veto_triggered`: 횡령, 배임, 상장폐지, 감사의견 거절 같은 치명적 외부 리스크가 있어 강제 경고가 필요한지 여부
- `hidden_tail_risk_flag`: 모델이 `투자적격`으로 본 기업에서 직접 관련 외부 위험 근거가 확인되어 false negative 가능성을 보완해야 하는지 여부
- `hidden_tail_risk_reason`: 숨은 꼬리위험 보완 플래그가 켜진 이유
- `conflict_resolution`: 모델 판단과 외부 근거가 충돌할 때 어떤 근거에 더 무게를 두었는지 설명
- `key_risk_factors`: 최종 판단에서 중요하게 본 위험 요인
- `mitigating_factors`: 위험을 낮춰주는 완화 요인
- `evidence_summary`: 사용한 근거의 출처, 요약, 신뢰도
- `final_review_memo`: 최종 종합 심사 메모

즉, 사용자는 모델이 본 이진 판단과
에이전트 위원회가 검토한 3단계 종합 의견을 함께 확인할 수 있다.

## 7. 코드 구조

현재 구현은 CI에서 안정적으로 검증 가능한 결정론적 scaffold와 선택형 Agno runner를
분리한다.

- `src/cas/agents/stage2_specs.py`: 3개 에이전트의 역할, 필수 입력, 출력 필드, Agno instruction 계약
- `src/cas/agents/stage2_bundle.py`: LangGraph `AgentState`를 Stage 2 전용 입력 번들로 정규화
- `src/cas/agents/stage2_outputs.py`: Agent별 Pydantic 출력 schema를 정의하고 공통 `AgentOutput`으로 변환
- `src/cas/agents/stage2_runner.py`: deterministic runner와 Agno runner가 공유할 실행 adapter 인터페이스
- `src/cas/agents/nodes/tripletagents/review_qa_agent.py`: 특정 조건에서만 실행되는 Agno ReviewQAAgent. 최종 라벨과 메모 일관성, `risk_hold` 세부유형, 외부근거 기준일, 정상기업 과잉 보류 가능성을 검수한다.
- `src/cas/agents/nodes/tripletagents/risk_recall_qa_agent.py`: 특정 조건에서만 실행되는 Agno RiskRecallQAAgent. 최종 `적격` 판단이 기준선/재무취약/외부근거 맥락에서 위험을 놓친 것은 아닌지 재검수한다.
- `src/cas/agents/committee_schema.py`: `committee_view` Pydantic strict schema 정의
- `src/cas/agents/signals/debt_liquidity_signals.py`: 부채상환능력, 유동성, 현금흐름 신호 계산
- `src/cas/agents/signals/macro_signals.py`: 거시·시장 맥락 신호 계산
- `src/cas/agents/signals/external_evidence_signals.py`: 뉴스·공시·검색 근거 신호 계산
- `src/cas/agents/signals/agent_disagreement_signals.py`: QuantCredit, EvidenceAudit, ChairReport/committee_view 사이의 판단 충돌 점수 계산
- `src/cas/agents/nodes/committee_node.py`: Stage 2 실행 순서 제어, 각 에이전트 결과 생성, audit 기록
- `src/cas/agents/committee_view.py`: `final_committee_label`, `veto_triggered`, `conflict_resolution` 등 최종 출력 JSON 조립
- `src/cas/veto_rules.py`: `configs/agent/committee.yaml`의 veto rule을 읽어 강제 경고 기준을 공유
- `src/cas/evidence/collectors.py`: Naver, Tavily, OpenDART 기반 외부 근거 수집 기능

이렇게 나누면 `committee_view` 출력 형식과 대시보드 계약은 유지하면서, 각 agent
내부 구현만 deterministic 또는 LLM 기반으로 선택할 수 있다. 특히
`Stage2InputBundle.to_prompt_payload()`는 Agno agent에 넘길 입력 payload의 기준
형태이고, `CommitteeViewPayload`는 LLM이 만든 최종 의견도 반드시 같은 출력 형식으로
검증하는 기준이다.

기본 본심 경로는 계속 `QuantCreditAgent → EvidenceAuditAgent → ChairReportAgent` 3개
역할이다. 세 agent 결과와 최종 `committee_view` 사이의 충돌은
`agent_disagreement_score`와 `agent_disagreement_reasons`로 기록한다. 예를 들어 정량
모델은 위험인데 EvidenceAudit은 watch/context 수준으로 보거나, 최종 `risk_hold`인데
치명 외부근거가 제한적인 경우 high disagreement가 될 수 있다.

Agno live 실행에서 선택형 `ReviewQAAgent`는 전수 호출하지 않는다. `agent_disagreement_level=high`
이더라도 치명 외부근거가 제한적이라는 이유만으로 호출하지 않고, 실제로 낮출 수 있는 경로가 있는지
한 번 더 확인한다. `risk_hold`는 1차 모델이 투자적격인데 위원회가 위험 보류로 올린 overhold 후보,
또는 라벨-메모 충돌 후보일 때 우선 검수한다. 1차 모델이 이미 부적격이고 위원회가 `risk_hold`로
완화한 케이스는 보정 가능성이 낮으면 ReviewQA를 건너뛴다. `medium`은
`chair_risk_without_critical_evidence`, `chair_reject_without_critical_evidence`,
`committee_label_memo_conflict`처럼 실제 라벨/근거 충돌을 설명하는 reason이 있을 때만 실행한다.
`low` disagreement 케이스는 ReviewQA를 건너뛰어 속도 비용을 줄인다. ReviewQA는
`agent_summary.agents.review_qa`와 runtime diagnostics에 advisory 결과를 남긴다. 최종
라벨은 직접 바꾸지 않는다. 다만 `risk_hold`가 치명 근거 없이 과도하다고 판단되고
veto 또는 hidden-tail-risk가 없을 때만 `committee_decision_type`을 `boundary_hold`로
낮추는 보수적 subtype 보정을 적용할 수 있다. ReviewQA live 응답의 표현이 조금 흔들려도
같은 결론을 재현할 수 있도록, 외부 공시가 모두 `caution/watch_context` 수준이고
중대성 비율 10% 이상·veto·hidden-tail-risk가 없는 경우에는 `risk_hold_without_critical_evidence`
downgrade 권고를 안정적으로 `boundary_hold` 보정에 연결한다.

RiskRecallQAAgent는 반대 방향의 안전망이다. 최종 위원회 라벨이 `적격`이고,
투기등급 확률이 기준선 근처라는 이유만으로는 실행하지 않는다. 기준선 근처와
유동성·현금흐름·이자보상·차입부담 취약 2축 이상이 함께 있거나, 재무 취약 3축
이상이거나, 실질 외부 위험 근거가 있을 때만 실행된다. 직접 관련 watch 공시나
BBB-/BB+ 경계 맥락은 단독 trigger가 아니라 이 핵심 조건에 붙는 보조 맥락으로만 쓴다.
이 에이전트는 정상기업 과잉 보류를 다시 늘리지 않도록 기본 권고를
`keep_committee_view`로 두고, 재무/외부근거가 정말 불안한 경우에만 `boundary_hold`
또는 제한적 `risk_hold` 상향을 권고한다. 특히
`eligible_with_substantive_evidence`는 routine 감사보고서나 단순 공시가 아니라,
`substantive_adverse`, veto/critical context, 또는 횡령·배임·상장폐지·감사의견 거절
같은 명시적 치명 제목에만 켜지도록 좁힌다. 자금조달·채무보증은 중대성 비율이
10% 이상이어도 재무 스트레스나 hard distress 문맥이 함께 있을 때만 실질 외부 위험으로 본다.
후속 적용부에서도 같은 원칙을 한 번 더 확인한다. RiskRecallQA가 상향을 권고하더라도
저품질 뉴스 스니펫이나 검색요약에 치명 키워드가 우연히 포함된 것만으로는 `적격`을
`보류`로 올리지 않는다. `risk_hold` 상향은 검증된 외부 중요근거 또는 매우 강한
재무취약성이 있어야 적용하고, `boundary_hold` 상향도 기준선 근처와 복수 재무취약성,
BBB-/BB+ 경계+재무취약성, 또는 검증된 외부근거가 있을 때만 적용한다.

횡령, 배임, 상장폐지, fraud 같은 강제 경고 키워드는 코드가 아니라
`configs/agent/committee.yaml`의 `veto_rules`에서 관리한다.
현재 기본 실행은 `DeterministicStage2AgentRunner`를 사용하며, live 실험에서는 같은
인터페이스의 `AgnoStage2AgentRunner`를 선택한다.

## 8. 구현 및 검증 상태

### 완료

- Stage 1 결과를 `QuantCreditAgent` 입력 번들로 정리
- `EvidenceAuditAgent`가 사용할 뉴스/공시/거시/산업/부채 신호 필드 정의
- `committee_view` strict schema와 decision subtype 정의
- deterministic runner와 선택형 Agno runner 연결
- Naver, Tavily, OpenDART 기반 외부근거 수집 노드 연결
- rolling/historical hard sample 성능 리포트와 속도 로그 생성
- `보류`를 `위험 보류`, `경계등급 보류`, `과민경고 완화 보류`, `확인필요 보류`로 분리
- 30건 stress sample 기준 1차 모델 대비 2차 위험신호 F1 개선 확인
- 정상기업 과잉 보류 guardrail 구현 및 로컬 회귀 검증
- 조건부 Agno ReviewQAAgent 추가. 전체 기업에 4번째 LLM 호출을 붙이지 않고, 보류/근거/메모 충돌 위험이 있는 케이스만 사후 검수한다.
- ReviewQA subtype advisory 안정화. `caution/watch_context` 외부근거만 있는 TN overhold 후보는 위험 보류가 아니라 경계등급 보류로 일관되게 낮추되, 자금조달·채무보증은 중대성 10% 이상이어도 재무 스트레스나 hard distress 문맥이 없으면 단독 `risk_hold` 근거로 쓰지 않는다.
- 조건부 Agno RiskRecallQAAgent 추가. 최종 적격 케이스 중 기준선/재무취약/외부근거 조건이 있는 경우만 적격 판단의 위험 누락 가능성을 사후 검수한다.
- RiskRecallQA escalation guardrail 추가. low-quality 뉴스 스니펫 단독으로는 적격을 보류로 올리지 않고, 구조화 외부근거 또는 복수 재무취약성이 확인될 때만 advisory를 적용한다.
- EvidenceAudit criticality hard gate 추가. LLM의 `has_critical_risk` 응답만으로는 치명 외부근거를 확정하지 않고, 구조화 근거 판정의 `critical_veto_review`, `hard_distress_detected`, 또는 `critical_evidence_count > 0`일 때만 `critical`로 올린다.
- Agent disagreement score 추가. QuantCredit, EvidenceAudit, ChairReport/committee_view의 방향이 충돌하는 경우를 점수화해 ReviewQA 호출 근거와 대시보드 설명 신호로 사용한다.

### 남은 개선 후보

- 정상기업 과잉 보류 guardrail의 live Agno/Claude 표본 재검증 확대
- 대시보드에서 decision subtype과 decision trace를 더 직관적으로 표시
- live Agno 설명 품질 비교 표본 확대
- 운영 환경에서 live API 호출 권한과 데이터 전송 승인 절차를 프로젝트 문서에 더 명확히 분리

## 9. 요약 메시지

Stage 2는 여러 역할을 단순히 많이 두는 방식이 아니라,
실제로 추론이 필요한 세 가지 판단 축으로 재구성한다.

`QuantCreditAgent`는 정량 모델의 판단 근거를 해석하고,
`EvidenceAuditAgent`는 외부 정보와 시장환경을 검토해 숨은 위험을 탐색하며,
`ChairReportAgent`는 두 관점을 종합해 최종 위원회 의견을 제시한다.
