# Stage 2 Agent Experiment Results

작성일: 2026-05-24

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

## 검증

로컬 검증 기준:

- `pytest tests/unit -q`
- `ruff check src scripts tests/unit`
- `git diff --check`

최신 실험 산출물 기준:

- 10/10 rows external evidence `ready`
- 10/10 rows `stage2_backend_name=agno`
- 10/10 rows `stage2_llm_cache_hit=False`
- strict success 80.0%, review-safe success 100.0%

## 다음 확인 후보

다음 단계는 같은 materiality guardrail을 20~30건 FP/TN mixed sample로 확대해 일반화 여부를
확인하는 것이다. 특히 일지테크처럼 보류는 유지하되 위험신호만 낮춘 케이스가 실제 운영에서
사용자에게 더 정확한 설명으로 보이는지 대시보드 표시까지 같이 점검하면 좋다.
