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
