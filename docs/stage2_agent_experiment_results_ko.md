# Stage 2 Agent Experiment Results

작성일: 2026-05-24
최신 업데이트: 2026-05-26

## 목적

이 문서는 Stage 2 Agno/OpenAI 에이전트 고도화 실험 결과를 PR 리뷰와 발표 자료에서
바로 확인할 수 있도록 한 곳에 모은 요약본이다.

주의할 점은 아래 수치가 전체 상장기업 모집단 정확도가 아니라는 점이다. 실험 샘플은
FN, FP, BBB-/BB+ 경계, TP, TN 과잉 보류 후보처럼 Stage 2 검토가 필요한 hard sample을
의도적으로 섞어 구성했다. 따라서 해석의 초점은 전체 모델 정확도보다 1차 XGBoost
판단의 오류를 2차 committee가 얼마나 보완했는지에 둔다.

## 누적 성능 증거

| 실험 묶음 | 건수 | 핵심 결과 | 해석 |
| --- | ---: | --- | --- |
| 초기 rolling pilot + Agno/Claude round 2 | 15 | 엄격 기준 12/15 = 80.0%, review-safe 14/15 = 93.3% | FN 보완과 FP 완화가 작동하기 시작한 기준선 |
| 30건 stress sample | 30 | 1차 모델 F1 0.4243 -> 2차 위험신호 F1 0.6666, 검토대상 Recall 1.0000 | 위험기업을 검토망에 올리는 조기경보 역할 확인 |
| OpenAI single 3-agent no-cache live | 8 | 엄격 기준 7/8 = 87.5%, review-safe 8/8 = 100.0%, cache hit 0 | 캐시 재평가가 아닌 실제 OpenAI Agno 3-agent 실행 증거 |
| TN ReviewQA/RiskRecallQA 20건 계열 | 20 | review-safe 20/20 = 100.0%, RiskRecallQA 호출 11건 -> 2건으로 축소 | 정상기업 과잉 QA 호출을 줄이면서 최종 분포 유지 |
| Mixed hard 40 combined | 40 | 엄격 기준 39/40 = 97.5%, review-safe 40/40 = 100.0%, run failure 0 | FN 8/8 보완, FP 12/12 완화, TP 12/12 위험 유지, TN 7/8 적격 유지 |
| Compact prompt smoke live | 8 | 엄격 기준 8/8 = 100.0%, review-safe 8/8 = 100.0%, cache hit 0 | role별 compact payload 적용 후 OpenAI Agno 3-agent 성능 유지 및 평균 14.90초 확인 |
| Explainability smoke live | 8 | 엄격 기준 7/8 = 87.5%, review-safe 8/8 = 100.0%, cache hit 0, EvidenceAudit 구조화 필드 8/8 | 실제 OpenAI Agno 3-agent에서 구조화 근거 판정과 `risk_hold` 이유 태그가 결과 CSV에 남는지 확인 |
| Agent disagreement smoke live | 10 | 엄격 기준 9/10 = 90.0%, review-safe 10/10 = 100.0%, cache hit 0, memo conflict 0 | Quant/Evidence/Chair 판단 충돌 점수화, high disagreement 2/10 모두 ReviewQA 실행 |
| Disagreement-gated ReviewQA live | 20 | 엄격 기준 18/20 = 90.0%, review-safe 20/20 = 100.0%, ReviewQA 5/20 | 내부 의견 차이가 낮은 케이스는 QA를 건너뛰고 필요한 경계 케이스에 호출 집중 |
| Disagreement-gated ReviewQA v2 live | 20 | 엄격 기준 19/20 = 95.0%, review-safe 20/20 = 100.0%, ReviewQA 3/20 | high disagreement 단독 호출을 제거해 QA 호출을 더 줄이고 성공률도 회복 |
| Disagreement-gated ReviewQA v2 full live | 40 | 엄격 기준 36/40 = 90.0%, review-safe 40/40 = 100.0%, ReviewQA 5/40 | QA 호출 절감은 유지됐지만 TN 4건 보류로 strict는 mixed hard 최고치보다 낮음 |
| EvidenceAudit criticality gate TN smoke | 10 | 엄격 기준 7/10 = 70.0%, review-safe 10/10 = 100.0%, cache hit 0 | TN overhold 후보 10건 중 7건 적격 유지, 3건은 경계 보류, 부적격 과잉 경고 0건 |

상세 누적 로그는 `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_agent_performance_evidence.md`에 남겨 두었다.
이 문서는 그중 PR에서 확인해야 할 핵심 숫자와 최신 materiality guardrail 검증만 요약한다.

## 최신 Materiality Guardrail 실험

최신 실험은 OpenDART 상세 공시에서 자금조달, 채무보증, 소송, 계약해지, 영업정지의
금액 중요도를 파싱한 뒤, "기업 규모 대비 큰 공시"와 "실제 부실 전이 위험"을 분리하는
방향으로 진행했다. 실행 조건은 OpenAI Agno single provider 3-agent, live external evidence,
`--no-stage2-llm-cache`, hard FP/TN 10건 샘플이다.

| 실행 | 건수 | 엄격 기준 | Review-safe | Wall time | Stage 2 평균 | Cache hit | 변화 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| materiality v7 baseline | 10 | 80.0% | 100.0% | 167.4초 | 14.97초 | 0 | 상세 중요도 컬럼 도입 전 기준 |
| materiality guardrail | 10 | 80.0% | 100.0% | 81.4초 | 15.83초 | 0 | 다스코, 제이엠티 `risk_hold` -> `mitigation_hold` |
| review-hold calibration | 10 | 80.0% | 100.0% | 91.7초 | 17.86초 | 0 | 일지테크 `risk_hold` -> `review_hold`, 하나투어는 `risk_hold` 유지 |

엄격 기준은 바뀌지 않았다. strict TN scoring은 실제 투자적격 기업이 `보류`로 남으면
실패로 보기 때문이다. 하지만 review-safe 기준은 계속 100.0%였고, 최신 개선의 핵심은
최종 라벨 자체보다 보류의 세부 위험 강도를 더 정확히 낮춘 데 있다.

## 케이스별 해석

| 기업 | 이전 판단 | 개선 후 판단 | 주요 materiality 근거 | 해석 |
| --- | --- | --- | --- | --- |
| 다스코 | `보류/risk_hold` | `보류/mitigation_hold` | 희석률 21.23%, 계약해지 watch 맥락 | 자금조달 규모는 크지만 hard distress나 재무 스트레스 결합이 약해 FP 완화 보류로 조정 |
| 제이엠티 | `보류/risk_hold` | `보류/mitigation_hold` | 채무보증금액/자기자본 12.80% | 단일 규모성 채무보증을 치명 위험으로 확정하지 않고 과민경고 완화 보류로 조정 |
| 일지테크 | `보류/risk_hold` | `보류/review_hold` | 채무보증금액/자기자본 14.90%, 반복 채무보증 | 보류는 유지하되 치명 문맥이나 현금흐름 악화가 제한적이라 위험신호는 `False`로 낮춤 |
| 하나투어 | `보류/risk_hold` | `보류/risk_hold` 유지 | 희석률 20.00%, 영업정지/자금조달 공시, 재무 스트레스 | 종속회사 영업정지와 반복 손실/OCF 적자가 결합되어 보수적 위험 보류가 타당 |

이 결과는 materiality guardrail이 단순히 보류를 모두 낮추는 장치가 아니라는 점을 보여준다.
자금조달·채무보증 비율만 큰 정상기업은 위험 보류에서 낮추되, 영업정지와 재무 스트레스가
같이 있는 케이스는 `risk_hold`를 유지한다.

## 구현 반영

- 자금조달·채무보증의 10% 이상 materiality는 단독으로 `risk_hold` 근거가 되지 않도록 조정했다.
- `veto_candidate`, `critical_context_confirmed`, hard distress 문맥, 또는 현금흐름/이자보상/손익/레버리지 중 2축 이상의 재무 스트레스가 함께 있을 때만 실질 외부 위험으로 본다.
- hidden-tail-risk를 `risk`와 `watch` 두 단계로 나눠, 보류 자체는 유지하되 사용자 화면의 위험신호를 낮출 수 있게 했다.
- batch 결과 CSV에 `materiality_event_count`, `materiality_substantive_count`, `materiality_watch_count`, `materiality_max_ratio`, `materiality_top_basis`, `materiality_event_classes`를 추가했다.
- `committee_view`, `EvidenceAudit`, `ReviewQA`, `RiskRecallQA`가 같은 materiality helper를 사용하도록 공통화해, 자금조달·채무보증 비율만으로 실질 위험을 과대평가하지 않게 했다.
- 확대 검증에서 발견된 BBB- 경계 FP 과잉 부적격 문제를 줄이기 위해, 치명 외부근거 또는 극단 재무위험이 없으면 고확률 모델 경고와 재무 watch 신호가 함께 있어도 `부적격` 확정 대신 `risk_hold`로 남기도록 reject confirmation gate를 보정했다.
- Agno live 호출 속도와 판단 안정성을 위해 role별 compact prompt payload를 추가했다. Quant/Evidence/Chair/QA 에이전트는 전체 원본 row 대신 필요한 재무 컬럼, 상위 driver, 압축된 공시 항목, materiality 요약만 받는다.
- compact prompt에는 `materiality_summary`를 공통 주입한다. 여기에는 실질 외부위험 여부, 자금조달/채무보증 개수, high-risk financing 개수, 최대 중요도 비율과 근거, event/materiality class가 들어간다.
- EvidenceAudit 출력에 `critical_evidence_count`, `watch_context_count`, `hard_distress_detected`, `recommended_evidence_treatment`를 추가해 Chair/QA가 prose를 다시 해석하지 않고 구조화된 근거 판정을 우선 참고하게 했다.
- 남은 strict miss인 휴맥스형 TN `risk_hold`를 무리하게 적격으로 낮추기보다, `financial_stress_hold`, `external_materiality_hold`, `combined_watch_hold` 같은 위험 보류 이유 태그와 요약을 남기도록 했다. 대시보드와 배치 CSV에서 "왜 보류로 남겼는지"를 설명할 수 있다.
- OpenAI structured output 호환성을 위해 Agno EvidenceAudit 응답 스키마에서는 자유형 `dict[str, Any]` 필드를 제거하고, `materiality_summary`는 프롬프트 컨텍스트와 deterministic evidence-treatment helper에서 계산해 최종 `EvidenceAuditOutput`에 주입한다.
- batch 결과 CSV에 `evidence_audit_structured_found`, `evidence_audit_critical_evidence_count`, `evidence_audit_watch_context_count`, `evidence_audit_hard_distress_detected`, `evidence_audit_recommended_evidence_treatment`, `evidence_audit_top_materiality_basis`를 추가했다.
- `agent_disagreement_score`, `agent_disagreement_level`, `agent_disagreement_reasons`, `agent_disagreement_summary`를 `committee_view`와 batch CSV에 추가했다. 정량 모델, EvidenceAudit, ChairReport가 서로 다른 방향을 볼 때 ReviewQA를 선택적으로 켜기 위한 진단 신호다.
- 최종 라벨과 메모 문구가 실제로 충돌할 때만 `committee_label_memo_conflict`로 잡도록 하되, "최종 적격으로 확정하지 않고 보류"처럼 부정형 문구는 오탐으로 보지 않게 보정했다.
- RiskRecallQA escalation guardrail을 추가했다. RiskRecallQA가 적격을 보류로 올리라고 권고해도, 저품질 뉴스 스니펫 단독 근거는 적용하지 않는다. `risk_hold` 상향은 검증된 외부 중요근거 또는 매우 강한 재무취약성이 필요하고, `boundary_hold` 상향도 기준선 근처+복수 재무취약성, BBB-/BB+ 경계+재무취약성, 또는 검증된 외부근거가 있어야 적용된다.
- EvidenceAudit criticality hard gate를 추가했다. Agno LLM이 `has_critical_risk=true`라고 응답해도, deterministic `structured_evidence_decision`에서 `critical_veto_review`, `hard_distress_detected`, 또는 `critical_evidence_count > 0`이 확인되지 않으면 `evidence_strength=critical`로 올리지 않는다. 이 변경은 저품질 뉴스/공시 요약을 치명 외부근거로 과대해석하는 경로를 줄이기 위한 안전장치다.

## EvidenceAudit Explainability Smoke

#52 병합 전, 새 구조화 필드와 `risk_hold` 이유 태그가 실제 OpenAI Agno live 경로에서도
채워지는지 8건 smoke test로 확인했다. 실행 조건은 OpenAI Agno single provider 3-agent,
live external evidence, `--no-stage2-llm-cache`, workers=1이다. 동일 설정의 1건 schema
smoke를 먼저 통과시킨 뒤 8건 전체를 실행했다.

| 건수 | 엄격 기준 | Review-safe | Cache hit | Evidence ready | EvidenceAudit 구조화 필드 | `risk_hold` 이유 태그 | Stage 2 평균 | Stage 2 최대 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 7/8 = 87.5% | 8/8 = 100.0% | 0/8 | 8/8 | 8/8 | 3/3 | 20.5624초 | 29.2241초 |

구조화 근거 판정은 8건 모두에서 추출됐다. `recommended_evidence_treatment` 분포는
`watch_context` 6건, `substantive_review` 2건이었다. `risk_hold`로 남은 라닉스,
대창솔루션, 휴맥스 3건은 모두 `combined_watch_hold`, `financial_stress_hold`,
`external_materiality_hold`, `secondary_radar_hold` 같은 reason tag가 함께 기록됐다.

엄격 기준 실패 1건은 휴맥스 TN이 `보류/risk_hold`로 남은 케이스다. 다만 review-safe
기준에서는 정상기업을 `부적격`으로 악화시키지 않았고, 휴맥스의 보류 이유도
희석률 20.19%, 재무 스트레스, 보조 레이더 맥락으로 설명 가능하게 남았다.

원시 산출물 폴더는 PR에 남기지 않고, 위 성능 수치와 해석만 이 문서에 보존한다.

## Agent Disagreement Score Smoke

ReviewQA를 전수 호출하지 않고 "AI 위원회 내부 판단이 엇갈리는 경우"에 더 정확히 켜기 위해
QuantCredit, EvidenceAudit, ChairReport/committee_view 사이의 충돌을 점수화했다. 주요 충돌
사유는 정량 모델은 위험인데 외부근거는 watch/context 수준인 경우, 최종 `risk_hold`인데
EvidenceAudit 치명 근거가 제한적인 경우, 또는 최종 라벨과 메모 문구가 충돌할 가능성이 있는
경우다.

OpenAI Agno single provider 3-agent, live external evidence, `--no-stage2-llm-cache`,
workers=1 조건으로 mixed hard 10건을 재실행했다. 이 실행은 memo conflict 오탐 보정 후의
최종 확인용 smoke test다.

| 건수 | 엄격 기준 | Review-safe | Evidence ready | Cache hit | Disagreement high | ReviewQA 호출 | Memo conflict | Stage 2 평균 | Stage 2 최대 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 9/10 = 90.0% | 10/10 = 100.0% | 10/10 | 0/10 | 2/10 | 3/10 | 0/10 | 24.3740초 | 54.5707초 |

`high` disagreement 2건은 모두 `(주)제닉`, `솔트웨어(주)` BBB-/BB+ 경계 FP였다. 두 케이스는
`quant_risk_evidence_watch_context / chair_risk_without_critical_evidence`로 기록됐고,
둘 다 ReviewQA가 실행됐다. 반대로 명확한 TP 위험 유지 케이스와 대부분의 TN/FP 완화 케이스는
low 또는 medium으로 남아 ReviewQA를 추가 호출하지 않았다.

엄격 기준 실패 1건은 `(주)엔에프씨` TN이 RiskRecallQA 적용 후 `보류/boundary_hold`로 올라간
케이스다. review-safe 기준에서는 정상기업을 `부적격`으로 악화시키지 않았기 때문에 성공으로 본다.
memo conflict 오탐은 0건으로, 이전에 발견된 "최종 적격으로 확정하지 않고 보류" 부정형 문구도
충돌로 잘못 잡지 않게 정리됐다.

원시 산출물 폴더는 PR에 남기지 않고, 위 성능 수치와 해석만 이 문서에 보존한다.

## ReviewQA Trigger 축소 실험

ReviewQA는 최종 위원회 판단 뒤에 붙는 advisory QA라서 안전장치 효과는 있지만 속도 비용이 크다.
최신 mixed hard 40 결과에서 ReviewQA는 30/40건 호출됐고, 실제 advisory 적용은 6건뿐이었다.
따라서 generic trigger인 `investment_model_hold`, `ambiguous_external_evidence`는 제거하고,
`risk_hold`/`reject` 중 치명 외부근거가 약하면서 watch-context 또는 재무 방어축이 있는 케이스에만 호출하도록 좁혔다.

동일 10건 샘플을 OpenAI Agno 3-agent, live external evidence, `--no-stage2-llm-cache` 조건으로 재검증했다.

| 실행 | 건수 | 엄격 기준 | Review-safe | ReviewQA 호출 | ReviewQA 적용 | ReviewQA 시간 합 | Stage 2 평균 | Stage 2 최대 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 broad trigger 동일 10건 | 10 | 100.0% | 100.0% | 8/10 | 3/10 | 85.3895초 | 25.6557초 | 61.6745초 |
| narrow trigger 재검증 | 10 | 100.0% | 100.0% | 3/10 | 3/10 | 18.1764초 | 22.5787초 | 47.9335초 |

결과적으로 ReviewQA 호출은 8건에서 3건으로 줄었고, ReviewQA 소요 시간 합은 약 78.7% 감소했다.
성능은 strict/review-safe 모두 100.0%로 유지됐다. FP 완화 보류(`mitigation_hold`)와 명확한 TP `reject` 케이스는
불필요한 ReviewQA를 건너뛰었고, 솔디펜스·제닉·솔트웨어처럼 실제 subtype advisory가 필요한 케이스만 남았다.

20건 확대 검증에서도 같은 경향이 유지됐다.

| 실행 | 건수 | 엄격 기준 | Review-safe | ReviewQA 호출 | ReviewQA 적용 | ReviewQA 시간 합 | Stage 2 평균 | Stage 2 최대 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 broad trigger 동일 20건 | 20 | 95.0% | 100.0% | 16/20 | 5/20 | 168.8237초 | 26.2304초 | 61.6745초 |
| narrow trigger 확대 검증 | 20 | 95.0% | 100.0% | 5/20 | 5/20 | 25.5863초 | 17.2261초 | 53.9036초 |

20건 기준 ReviewQA 호출은 68.75% 줄었고, ReviewQA 시간 합은 약 84.84% 감소했다.
적용 건수 5건은 그대로 유지되어, 불필요한 QA 호출을 줄이면서 실제 보정 효과는 보존한 것으로 해석한다.
엄격 기준 1건 실패는 하나투어 TN이 `보류/risk_hold`로 남은 케이스이며, review-safe 기준에서는 정상적으로 통과했다.

이후 대시보드 노출용 `agent_disagreement_level/reason`을 ReviewQA trigger에도 직접 연결한
v1 정책을 검증했다. `high`는 우선 QA를 켜고, `medium`은 `chair_risk_without_critical_evidence`,
`chair_reject_without_critical_evidence`, `committee_label_memo_conflict` reason이 있을 때만
실행하며, `low`는 스킵한다. 같은 mixed hard 20건을 OpenAI Agno live no-cache로 재검증한 결과는
strict 18/20 = 90.0%, review-safe 20/20 = 100.0%, cache hit 0/20이었다. ReviewQA는 5/20건,
advisory는 1/20건만 적용됐고, FP 완화 보류·TP 위험 유지·TN guardrail 케이스는 QA를 호출하지 않았다.
strict 실패 2건은 `(주)엔에프씨` TN `boundary_hold`와 `(주)하나투어` TN `risk_hold`였지만,
둘 다 부적격으로 악화하지 않아 review-safe 기준은 통과했다. Stage 2 평균은 22.9209초, 최대는
46.1780초였다.

이 live 결과에서 ReviewQA가 켜졌지만 적용되지 않은 4건을 줄이기 위해 후속 v2 구현에서는
`high` disagreement도 단독 호출 조건으로 쓰지 않는다. `risk_hold`는 1차 모델이 투자적격인데
위원회가 위험 보류로 올린 overhold 후보, 또는 라벨-메모 충돌 후보에 집중한다. 1차 모델이 이미
부적격이고 위원회가 보류로 완화한 케이스는 보정 가능성이 낮으면 QA를 건너뛰도록 해,
내부 의견 차이는 대시보드 설명 신호로 남기되 4번째 LLM 호출은 더 절제한다.

v2를 같은 mixed hard 20건으로 OpenAI Agno live no-cache 재검증한 결과, strict 19/20 = 95.0%,
review-safe 20/20 = 100.0%였다. ReviewQA 호출은 5/20건에서 3/20건으로 줄었고,
advisory 적용은 1/20건에서 2/20건으로 늘었다. 실행된 QA는 FN risk_hold 3건에만 집중됐고,
BBB-/BB+ 경계 FP, FP 완화, TP 위험 유지, TN guardrail 케이스는 모두 QA를 건너뛰었다.
LLM cache hit는 0/20, 외부근거 ready는 20/20, 실행 실패 행은 0건이었다. Stage 2 평균은
18.7126초, 최대는 31.4994초로 v1의 평균 22.9209초, 최대 46.1780초보다 줄었다.
strict 실패는 하나투어 TN `risk_hold` 1건뿐이며, 종속회사 영업정지/자금조달 중요도와 재무
스트레스가 결합된 보수적 보류라 review-safe 기준은 통과했다.

같은 v2 정책을 mixed hard 40건 전체로 확대한 live no-cache 재검증에서는 strict 36/40 = 90.0%,
review-safe 40/40 = 100.0%였다. ReviewQA는 5/40건만 실행됐고 advisory는 2건 적용됐다.
호출 대상은 FN risk_hold에만 집중됐으며, BBB-/BB+ 경계 FP 8건, FP 완화 8건, TP 12건,
TN 8건은 ReviewQA를 호출하지 않았다. LLM cache hit는 0/40, 외부근거 ready는 40/40,
실행 실패 행은 0건이었다. Stage 2 평균은 17.5488초, 최대는 34.0596초, wall time은
729.5342초였다.

다만 40건 확대에서는 TN 8건 중 4건이 `보류`로 남아 strict는 mixed hard 최고치인 39/40보다
낮았다. 실패 TN은 `(주)엔에프씨` `boundary_hold`, `(주)하나투어` `risk_hold`, `청광건설(주)`
`boundary_hold`, `(주)일지테크` `review_hold`였다. NFC와 청광건설은 RiskRecallQA/EvidenceAudit이
routine 공시 목록에서 치명 맥락을 과하게 읽어 적격을 보류로 올린 케이스이고, 일지테크는
채무보증금액/자기자본 14.90%를 확인필요 보류로 남긴 케이스다. 따라서 다음 개선은 ReviewQA가
아니라 RiskRecallQA escalation에 실제 evidence profile의 veto/substantive 근거를 요구하는
guardrail이다.

후속 구현으로 해당 RiskRecallQA escalation guardrail을 추가했다. 이제 저품질 뉴스 스니펫이나
검색요약에 `횡령`, `배임` 같은 치명 키워드가 우연히 들어간 것만으로는 적격을 보류로 올리지 않는다.
`risk_hold` 상향은 검증된 외부 중요근거 또는 매우 강한 재무취약성이 있어야 적용하고,
`boundary_hold` 상향도 기준선 근처+복수 재무취약성, BBB-/BB+ 경계+재무취약성, 또는 검증된
외부근거가 있어야 적용한다. 이 변경은 live API 재검증 전 구현 단계이며, 단위 테스트에서는
저품질 뉴스 단독 `boundary_hold`/`risk_hold` 상향이 차단되고, 기존 복수 재무취약성 및
OpenDART substantive evidence 상향 경로는 유지되는 것을 확인했다.

## Materiality 28건 확대 검증

FP/TN hard sample 28건을 OpenAI Agno 3-agent, live external evidence, `--no-stage2-llm-cache`
조건으로 확대 검증했다. 모든 행에서 외부근거 수집은 `ready`, LLM cache hit는 0건이었다.

| 건수 | 엄격 기준 | Review-safe | FP 완화 | TN 적격 유지 | ReviewQA 호출 | RiskRecallQA 호출 | Wall time | Stage 2 평균 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 28 | 24/28 = 85.7% | 25/28 = 89.3% | 15/18 | 9/10 | 2/28 | 3/28 | 475.0초 | 15.98초 |
| reject guardrail v2 | 27/28 = 96.4% | 28/28 = 100.0% | 18/18 | 9/10 | 3/28 | 3/28 | 430.0초 | 15.27초 |

확대 검증의 핵심 발견은 TN guardrail은 비교적 잘 작동했지만, BBB- 경계 FP 중 3건이 여전히
`부적격`으로 남았다는 점이다.

| 케이스 | 기존 최종 | materiality 근거 | 해석 |
| --- | --- | --- | --- |
| (주)에스디생명공학 | `부적격` | 희석률 37.03%, 소송/자금조달 | 치명 외부근거는 아니지만 고확률 모델 경고와 재무 watch가 결합되어 과잉 reject |
| (주)라닉스 | `부적격` | 희석률 13.61% | 자금조달 materiality와 재무취약이 결합됐으나 확정형 부실 근거는 제한적 |
| 대한광통신(주) | `부적격` | 희석률 4.81% | 비율 자체도 낮고 치명 외부근거가 약하지만 고확률/재무 watch로 reject 유지 |

이에 따라 reject confirmation gate를 보정했다. `veto_candidate`, `critical_context_confirmed`,
hard distress 문맥, 극단 재무위험 중 하나가 확인되는 경우에만 `부적격` 확정을 유지하고,
그 외의 고확률·재무취약·비치명 외부근거 조합은 `risk_hold`로 낮춰 검토 대상으로 남긴다.
이 변경은 실제 위험기업을 `적격`으로 낮추는 것이 아니라 보류로 유지하는 방식이라 recall 방어를
유지하면서 FP reject를 줄이는 목적이다.

패치 후 동일 28건을 다시 live 재검증한 결과, 위 3개 BBB- 경계 FP는 모두 `부적격`에서
`보류/risk_hold`로 내려갔다. 이에 따라 FP 18건은 전부 완화됐고, review-safe 기준은 100.0%로
회복됐다. 엄격 기준 실패 1건은 `(주)휴맥스` TN이 `보류/risk_hold`로 남은 케이스이며,
review-safe 기준에서는 정상 통과했다.

## Agno Compact Prompt Context

Agno 3-agent 속도 개선을 위해 full `Stage2InputBundle`을 그대로 보내지 않고 role별 compact
payload를 사용하도록 바꿨다. 내부 pipeline과 deterministic committee logic은 원본 입력을
그대로 쓰지만, LLM 프롬프트에는 다음처럼 축약된 입력만 들어간다.

| 에이전트 | 주요 입력 |
| --- | --- |
| QuantCredit | Stage 1 라벨/확률/기준선, top drivers, 핵심 재무지표, peer 요약, credit policy 요약 |
| EvidenceAudit | 핵심 재무지표, 압축된 외부근거 항목, provider 상태, materiality 요약 |
| ChairReport | Quant/Evidence 결과, Stage 1 요약, prior rating, materiality 요약, credit policy summary |
| ReviewQA/RiskRecallQA | 최종 committee view, 압축 외부근거, 핵심 재무지표, materiality 요약 |

로컬 샘플 JSON 길이 비교에서는 full payload 13,163자 대비 compact payload 2,061자로 약 84.3%
감소했다. 이후 `committee_review_error_risk_10_samples.csv` 중 8건을 OpenAI Agno 3-agent,
live external evidence, `--no-stage2-llm-cache` 조건으로 다시 실행해 live smoke를 확인했다.

| 실행 | 건수 | 엄격 기준 | Review-safe | Cache hit | Evidence ready | Stage 2 평균 | Stage 2 최대 | QA 호출 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compact prompt smoke | 8 | 8/8 = 100.0% | 8/8 = 100.0% | 0/8 | 8/8 | 14.9034초 | 19.1055초 | ReviewQA 3/8, RiskRecallQA 0/8 |

같은 8건 샘플의 이전 실행은 없으므로 속도 비교는 참고값으로만 본다. 다만 기존
OpenAI single 3-agent no-cache live 8건의 Stage 2 평균은 16.4786초였고, 이번 smoke는
14.9034초로 측정되어 compact prompt 적용 후에도 성능 저하 없이 속도 부담이 낮아지는
방향임을 확인했다.

## 보존 위치와 raw 파일 정리

최신 materiality live 실험의 원시 output 디렉터리와 샘플 CSV는 PR에 남기지 않는다. 재현에
필요한 실행 조건과 성능 수치만 아래 문서에 흡수해 보존한다.

| 보존 항목 | 경로 |
| --- | --- |
| PR/발표용 최신 요약 | `docs/stage2_agent_experiment_results_ko.md` |
| 누적 성능 증거 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_agent_performance_evidence.md` |
| Stage 2 고도화 한눈 요약 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_agent_improvement_summary.md` |
| 전체 실험 로그 CSV | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_agent_performance_experiment_log.csv` |
| 속도 실험 로그 CSV | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_agent_speed_experiment_log.csv` |

삭제한 raw output:

- `committee_review_materiality_v7_fp_tn_10_agno_openai_live_no_cache/`
- `committee_review_materiality_guardrail_fp_tn_10_agno_openai_live_no_cache/`
- `committee_review_materiality_review_hold_calibration_fp_tn_10_agno_openai_live_no_cache/`
- `committee_review_materiality_v7_fp_tn_10_samples.csv`
- `committee_review_explainability_schema_smoke_1_agno_openai_live_no_cache/`
- `committee_review_explainability_smoke_8_agno_openai_live_no_cache/`
- `committee_review_explainability_smoke_8_deterministic_preflight/`
- `committee_review_disagreement_smoke_10_agno_openai_live_no_cache/`
- `committee_review_disagreement_memo_fix_10_agno_openai_live_no_cache/`
- `committee_review_disagreement_memo_fix_final_10_agno_openai_live_no_cache/`
- `committee_review_disagreement_trigger_gated_20_agno_openai_live_no_cache/`
- `committee_review_disagreement_trigger_gated_v2_20_agno_openai_live_no_cache/`
- `committee_review_disagreement_trigger_gated_v2_40_agno_openai_live_no_cache/`

## Disagreement 기반 ReviewQA 트리거 구현

대시보드에 `agent_disagreement_level`, score, reason을 노출한 뒤, ReviewQA 호출 정책도
동일한 신호를 기준으로 좁혔다. `high` disagreement도 단독 호출 조건으로 쓰지 않고,
치명 외부근거가 제한적이며 실제 보정 가능성이 있는 `risk_hold`/`reject`에 집중한다.
`risk_hold`는 1차 모델이 투자적격인데 위원회가 위험 보류로 올린 overhold 후보나
라벨-메모 충돌 후보를 우선 검수하고, 1차 모델이 이미 부적격인데 위원회가 보류로 완화한
케이스는 보정 여지가 약하면 건너뛴다. `medium`은
`chair_risk_without_critical_evidence`, `chair_reject_without_critical_evidence`,
`committee_label_memo_conflict`처럼 실제 라벨/근거 충돌을 설명하는 reason이 있을 때만 켠다.
`low` disagreement는 ReviewQA를 건너뛰어 속도 비용을 줄인다.

단위 테스트에서는 low disagreement risk_hold가 ReviewQA를 건너뛰고, medium이라도 관련 없는
confidence gap만 있는 경우에는 스킵하며, high disagreement라도 1차 모델이 이미 부적격이고
별도 보정 경로가 없으면 호출하지 않는 것을 확인했다. 반대로 1차 모델 투자적격 overhold 후보이면서
watch-context 외부근거가 있는 high 케이스는 ReviewQA를 계속 호출한다.

## 검증

로컬 검증 기준:

- `pytest tests/unit -q`
- `ruff check src scripts tests/unit`
- `git diff --check`

최신 실험 산출물 기준:

- 10/10 rows external evidence `ready`
- 10/10 rows `stage2_backend_name=agno`
- 10/10 rows `stage2_llm_cache_hit=False`
- 실행 실패 0건
- high disagreement 2/10건은 대시보드 설명 신호로만 남기고 ReviewQA를 건너뜀
- ReviewQA 2/10건, RiskRecallQA 0/10건
- strict success 100.0%, review-safe success 100.0%

## PR #53 역할 분리 흡수 후 live smoke

`main`에 머지된 PR #53의 Quant/Evidence 역할 분리 의도와 credit signal policy 보강은
선별 반영하되, OpenAI 기본 provider, materiality guardrail, structured evidence treatment,
disagreement/RiskRecallQA 흐름은 유지했다. 이후 mixed hard 샘플 10건을 OpenAI Agno
single 3-agent, live external evidence, `--no-stage2-llm-cache` 조건으로 재검증했다.

| 실행 | 건수 | 엄격 기준 | Review-safe | Cache hit | Evidence ready | Stage 2 평균 | Stage 2 최대 | QA 호출 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after PR #53 role split smoke | 10 | 10/10 = 100.0% | 10/10 = 100.0% | 0/10 | 10/10 | 16.4039초 | 22.2847초 | ReviewQA 2/10, RiskRecallQA 0/10 |

FN 3건은 모두 `risk_hold`로 끌어올렸고, FP 6건은 모두 `mitigation_hold` 또는
review-safe 보류로 완화했으며, TP 1건은 위험 판단을 유지했다. 최종 `부적격`은 없었고
strict miss도 0건이었다. 따라서 PR #53의 역할 분리 아이디어를 현재 guardrail 위에
흡수해도 10건 smoke에서는 과잉 부적격이나 RiskRecallQA 과호출이 발생하지 않았다.

## EvidenceAudit Criticality Gate TN Smoke

EvidenceAudit criticality hard gate 적용 후 TN overhold 후보 10건을 OpenAI Agno single 3-agent,
live external evidence, `--no-stage2-llm-cache` 조건으로 재검증했다.

| 실행 | 건수 | 엄격 기준 | Review-safe | Cache hit | Evidence ready | Stage 2 평균 | Stage 2 최대 | QA 호출 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| criticality gate TN smoke | 10 | 7/10 = 70.0% | 10/10 = 100.0% | 0/10 | 10/10 | 24.3308초 | 42.3019초 | ReviewQA 1/10, RiskRecallQA 4/10 |

최종 라벨은 `적격` 7건, `보류/boundary_hold` 3건이었다. `부적격`은 0건이라 review-safe
기준은 유지됐다. 다만 EvidenceAudit의 deterministic 구조화 판정 자체가
`critical_veto_review` 4건, `substantive_review` 1건, `watch_context` 5건으로 나뉘었다.
따라서 hard gate는 LLM 단독 critical flag 승격을 막는 안전장치로 작동하지만, 다음 개선은
구조화 evidence-treatment 단계에서 routine 감사보고서/검색요약/과거 치명 키워드를
`critical_veto_review`로 잡는 품질을 더 좁히는 쪽이 맞다.

## 다음 확인 후보

다음 단계는 structured evidence-treatment의 critical 판정을 더 좁히는 것이다. 특히
routine 감사보고서, 저품질 검색요약, 회사 직접 관련성이 약한 과거 치명 키워드는
`critical_veto_review`가 아니라 `watch_context` 또는 `substantive_review`로 낮춰야 한다.
