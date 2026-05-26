# TN Overhold Expanded 8 OpenAI Agno Live Analysis

TN 과잉 보류 30건 확대 샘플 중 대표 8건을 OpenAI single provider 3-agent Agno 경로로 live 재검증했다. 실행은 no-cache 조건이며 외부 뉴스/공시 수집을 켰다.

## Summary

- Rows: 8
- Runner: OpenAI Agno 3-agent, single provider
- LLM cache hits: 0
- External evidence status: ready 8/8
- Strict TN success: 2/8 = 25.0%
- Review-safe success: 8/8 = 100.0%
- Final labels: eligible 2, hold 6
- Wall time: 87.9967 sec
- Mean case time: 20.4052 sec
- Mean Stage 2 LLM time: 18.3034 sec
- Throughput: 5.4548 cases/min

## Deterministic vs Agno Live

| corp_name | fiscal_year | deterministic_label | agno_live_label | agno_type | interpretation |
| --- | ---: | --- | --- | --- | --- |
| (주)엔에프씨 | 2020 | 적격 | 적격 | eligible | 재무 방어축과 외부근거가 충돌하지 않아 적격 유지 |
| (주)휴니드테크놀러지스 | 2020 | 적격 | 적격 | eligible | 외부근거 ready 상태에서도 적격 유지 |
| (주)머큐리 | 2020 | 적격 | 보류 | risk_hold | 전환사채 공시 1건이 보수적 보류를 유발; 단일 medium financing 공시 민감도 후보 |
| (주)레몬 | 2020 | 적격 | 보류 | risk_hold | 유상증자/전환사채 공시가 반복 수집되어 deterministic guardrail을 막음 |
| 현대무벡스(주) | 2020 | 보류 | 보류 | risk_hold | SPAC 합병 관련 거래정지 공시와 ICR 약점으로 보류 유지 |
| (주)한울반도체 | 2020 | 보류 | 보류 | boundary_hold | OCF/수익성 약점이 있어 경계등급 보류 유지 |
| (주)화승알앤에이 | 2021 | 보류 | 보류 | boundary_hold | 약한 자본/이자보상과 채무보증 공시로 보류 유지 |
| (주)하나투어 | 2022 | 보류 | 보류 | risk_hold | 영업정지 공시와 OCF/ICR/수익성 약점으로 보류 유지 |

## Key Findings

- Agno live 경로 자체는 정상 작동했다. 8건 모두 `stage2_backend_name=agno`, LLM cache hit 0, 역할별 실행시간 기록이 존재했다.
- Review-safe 기준은 8/8 = 100.0%로 유지됐다. 실제 투자적격 TN을 부적격으로 확정하지는 않았다.
- Strict TN 기준은 2/8 = 25.0%로 낮다. 다만 이 8건은 적격 후보만 모은 샘플이 아니라, 보류 유지가 맞는 TN과 과잉 보류 후보를 섞어 만든 stress subset이므로 전체 정확도처럼 해석하면 안 된다.
- deterministic 대비 바뀐 2건은 모두 자금조달성 DART 공시가 있는 기업이다. 특히 `(주)머큐리`는 전환사채 공시 1건만으로 보류가 되어, 단일 medium financing 공시가 과하게 강한 blocker로 작동하는지 재점검이 필요하다.
- `(주)레몬`은 유상증자/전환사채 공시가 반복 수집되어 보류가 된 점은 설명 가능하지만, chair memo가 "투자적격 판단 유지"라고 쓰면서 최종 라벨은 `보류`인 설명 충돌이 있다.

## Next Improvement

다음 guardrail 개선은 보류를 무작정 줄이는 것이 아니라, 외부근거의 강도를 세분화하는 방향이 안전하다.

1. 단일 medium 자금조달 공시는 `risk_hold`가 아니라 `review_hold` 또는 guardrail 허용 대상으로 낮춘다.
2. 반복 자금조달, 고위험 disclosure severity, 또는 재무 차단 신호와 결합된 자금조달만 `risk_hold` blocker로 둔다.
3. chair memo가 최종 committee label과 충돌하지 않도록, agent memo는 "보강 의견"으로만 붙이고 최종 라벨 문구가 우선하도록 정리한다.

## Follow-up Guardrail

Agno live 결과를 반영해 단일 medium 자금조달 공시와 반복·고위험 자금조달 공시를 분리했다. 단일 medium DART 전환사채/유상증자 공시는 TN overhold guardrail을 막지 않고, 반복 자금조달 2건 이상 또는 high-risk/adverse 자금조달 근거만 보류 보강 근거로 둔다. 또한 최종 라벨이 `보류`인데 chair memo가 "투자적격 판단 유지"처럼 읽히는 경우에는 해당 보강 메모를 붙이지 않도록 정리했다.

같은 8건과 같은 외부근거 캐시로 deterministic committee replay를 수행한 결과는 다음과 같다.

| 실행 | 엄격 기준 | Review-safe | 적격 | 보류 | Wall time | 비고 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Agno live no-cache before follow-up | 2/8 = 25.0% | 8/8 = 100.0% | 2 | 6 | 87.9967 sec | 단일 전환사채 공시가 있는 머큐리도 risk_hold |
| Cached-evidence replay after follow-up | 3/8 = 37.5% | 8/8 = 100.0% | 3 | 5 | 1.7129 sec | 머큐리만 적격으로 개선, 반복 자금조달 레몬은 보류 유지 |
| Agno live no-cache after follow-up | 3/8 = 37.5% | 8/8 = 100.0% | 3 | 5 | 95.4372 sec | cache hit 0, 머큐리 적격 개선 live 확인 |

최종 live 재실행에서도 cache hit는 0건이었다. `(주)머큐리`는 단일 medium 전환사채 공시만으로는 보류 보강 근거가 충분하지 않아 `적격`으로 내려갔고, `(주)레몬`은 반복 유상증자/전환사채 공시가 있어 `보류`로 남았다. 따라서 이 guardrail은 "자금조달 공시를 무시"하는 방식이 아니라, 단일 medium 공시와 반복·고위험 자금조달 공시를 분리하는 방식으로 작동한다.

## SPAC Procedural Halt Follow-up

남은 `risk_hold` 중 현대무벡스는 SPAC 합병 예비심사 때문에 거래정지가 발생했고, 같은 평가 기준일 이전에 `거래정지해제(상장예비심사결과 통지(승인))`가 확인됐다. 이를 상장폐지·관리종목·감사의견 이슈와 같은 실질 부실 이벤트가 아니라 절차성 거래정지로 분리했다. 다만 ICR 1 미만은 남아 있으므로 적격으로 낮추지는 않고, OCF/총부채·cashflow coverage·자본비율·차입 부담이 방어적인 단일 ICR 약점 케이스로 보아 `risk_hold` 대신 `boundary_hold`로 표시한다.

| 실행 | 엄격 기준 | Review-safe | 적격 | 보류 | Risk signal TN holds | 비고 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Agno live after financing guardrail | 3/8 = 37.5% | 8/8 = 100.0% | 3 | 5 | 3 | 현대무벡스는 SPAC 거래정지 때문에 risk_hold |
| Cached-evidence replay after SPAC guardrail | 3/8 = 37.5% | 8/8 = 100.0% | 3 | 5 | 2 | 현대무벡스 risk_hold→boundary_hold, hidden tail risk 해제 |

이 개선은 strict TN 성공률을 올리는 변화가 아니라, 정상기업을 과도하게 `위험 보류`로 표시하는 문제를 줄이는 subtype 품질 개선이다.

## Agno Chair Memo Consistency Follow-up

Agno live 결과에서 최종 위원회 라벨은 `보류`인데 chair memo가 "모델 라벨을 유지하되" 또는 "최종 라벨은 투자적격 유지하되"처럼 읽히는 표현이 남을 수 있음을 확인했다. 따라서 최종 라벨이 `적격`이 아닌 경우에는 `투자적격 판단을 유지`, `모델 라벨을 유지`, `모델 라벨을 존중`, `최종 라벨은 투자적격` 계열 문장을 보강 메모로 붙이지 않도록 필터를 확장했다.

이 변경은 최종 라벨이나 strict/review-safe 성능을 바꾸는 guardrail이 아니라, Agno 3-agent 출력이 사용자에게 전달될 때 최종 판단과 설명 문구가 충돌하지 않도록 하는 설명 품질 개선이다. 회귀 테스트에는 `(주)레몬` 반복 자금조달 보류 케이스를 사용했고, 실제 Agno live에서 관찰된 표현 변형을 함께 추가했다.

전체 8건을 다시 OpenAI Agno 3-agent no-cache live로 실행해 설명 충돌 제거를 확인했다.

| 실행 | 건수 | 엄격 기준 | Review-safe | 적격 | 보류 | Risk signal TN holds | Memo conflicts | Wall time | Stage 2 평균 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Agno live no-cache after memo filter | 8 | 3/8 = 37.5% | 8/8 = 100.0% | 3 | 5 | 2 | 0 | 67.3331 sec | 15.0851 sec |

재검증 결과 `stage2_backend_name=agno` 8/8, `stage2_llm_cache_hit=False` 8/8, `stage2_parallel_independent_agents=True` 8/8, 외부근거 `ready` 8/8이었다. 최종 라벨 분포는 SPAC 절차성 거래정지 보정 이후와 동일하게 `적격` 3건, `boundary_hold` 3건, `risk_hold` 2건이며, 최종 라벨이 `보류`인 케이스에서 투자적격 유지처럼 읽히는 chair memo 충돌은 0건이었다.

## Conditional ReviewQA Follow-up

Agno 본심 3-agent 뒤에 조건부 ReviewQAAgent를 추가해 같은 8건을 live no-cache로 재검증했다. 첫 실행에서는 애매한 외부공시만 있어도 QA가 켜져 `적격` 3건까지 포함한 8/8건이 QA를 탔고, wall time은 112.5576초였다. 이후 trigger를 조정해 최종 라벨이 `적격`인 단순 애매공시 케이스는 QA를 건너뛰도록 바꿨다.

| 실행 | 건수 | 엄격 기준 | Review-safe | QA triggered | QA action | Wall time | 평균 case time |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| ReviewQA before trigger tuning | 8 | 3/8 = 37.5% | 8/8 = 100.0% | 8/8 | keep 6, downgrade risk_hold→boundary_hold 2 | 112.5576 sec | 27.4583 sec |
| ReviewQA after trigger tuning | 8 | 3/8 = 37.5% | 8/8 = 100.0% | 5/8 | keep 4, downgrade risk_hold→boundary_hold 1 | 83.7142 sec | 19.5163 sec |
| ReviewQA subtype advisory applied | 8 | 3/8 = 37.5% | 8/8 = 100.0% | 5/8 | applied 1, keep 4 | 241.5522 sec | 57.6304 sec |

Trigger tuning 이후에도 `stage2_backend_name=agno` 8/8, `stage2_llm_cache_hit=False` 8/8, 외부근거 `ready` 8/8, memo conflict 0건을 유지했다. 최종 라벨 분포도 `적격` 3건, `boundary_hold` 3건, `risk_hold` 2건으로 유지됐다. ReviewQA는 `(주)레몬`의 반복 자금조달성 공시 기반 `risk_hold`에 대해 `downgrade_risk_hold_to_boundary_hold`를 권고했고, 하나투어의 영업정지/재무 약점 결합 `risk_hold`는 유지 의견을 냈다. 따라서 ReviewQA는 최종 라벨을 직접 덮어쓰지 않는 advisory layer로 두되, 향후 `risk_hold` subtype 자동 재분류 후보를 찾는 근거로 사용할 수 있다.

ReviewQA subtype advisory 적용 후 live no-cache 재검증에서는 최종 라벨은 그대로 `적격` 3건, `보류` 5건을 유지하면서, `(주)레몬` 1건만 `risk_hold`에서 `boundary_hold`로 낮아졌다. 따라서 `committee_risk_signal=True`인 TN 위험신호 보류는 2건에서 1건으로 줄었고, `stage2_review_qa_advisory_applied=True`는 1/8건이었다. 하나투어는 ReviewQA가 `keep_committee_view`를 권고해 `risk_hold`로 유지됐다. 이번 run의 wall time은 241.5522초로 길었지만, QA 평균은 6.0396초였고 지연은 주로 QuantCredit/ChairReport OpenAI 응답 시간 변동에서 발생했다.

## ReviewQA 20-Case Expansion

이전 8건만으로는 표본이 작아, 같은 TN 과잉 보류 stress 목적의 20건 샘플로 ReviewQA advisory 적용을 확대 검증했다. 샘플은 이전 live 8건, deterministic 보류 TN 추가 4건, 기준선 근처 TN 추가 8건으로 구성했다.

| 실행 | 건수 | 엄격 기준 | Review-safe | 적격 | 보류 | QA triggered | Advisory applied | Wall time | Stage 2 평균 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ReviewQA advisory applied 20 live no-cache | 20 | 10/20 = 50.0% | 20/20 = 100.0% | 10 | 10 | 10/20 | 2/20 | 190.1121 sec | 17.9042 sec |

확대 실행에서도 `stage2_backend_name=agno` 20/20, `stage2_llm_cache_hit=False` 20/20, 외부근거 `ready` 20/20, 실행 오류 0건이었다. ReviewQA는 최종 `적격` 10건을 건너뛰고 `보류` 10건에만 켜졌다. 권고는 keep 7건, `risk_hold → boundary_hold` 3건이었고, 실제 자동 적용은 `(주)레몬`, `신원종합개발(주)` 2건이었다. `다스코(주)`는 ReviewQA가 downgrade를 권고했지만 `hidden_tail_risk_flag=True`라 자동 적용이 차단되어, 안전장치가 의도대로 작동했다.

이 결과는 ReviewQA를 전체 기업에 항상 붙이는 구조가 아니라, 보류/위험근거 충돌 후보에만 붙이는 구조가 운영상 더 적합하다는 근거다. 세부 분석은 `tn_overhold_expanded_20_reviewqa_live_analysis.md`에 별도 저장했다.
