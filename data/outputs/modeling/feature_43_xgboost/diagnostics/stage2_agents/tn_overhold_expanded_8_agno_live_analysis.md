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
