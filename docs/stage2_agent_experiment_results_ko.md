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

## 산출물 위치

| 산출물 | 경로 |
| --- | --- |
| 최신 baseline report | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_materiality_v7_fp_tn_10_agno_openai_live_no_cache/committee_review_batch_report.md` |
| materiality guardrail comparison | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_materiality_guardrail_fp_tn_10_agno_openai_live_no_cache/materiality_guardrail_before_after_comparison.md` |
| review-hold calibration comparison | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_materiality_review_hold_calibration_fp_tn_10_agno_openai_live_no_cache/review_hold_calibration_before_after_comparison.md` |
| 누적 성능 증거 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_agent_performance_evidence.md` |

원시 output 디렉터리는 재현성과 용량 관리를 위해 PR 커밋에는 포함하지 않고, 경로와 핵심
수치만 문서에 남긴다.

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
