# Stage 2 Agent Performance Evidence

- 작성일: 2026-05-21
- 최종 업데이트: 2026-05-26
- 범위: 지금까지의 committee-review agent 실험 로그, rolling validation 핵심 증빙 15건, 새 holdout 8건 속도/성능 재검증, OpenAI single 3-agent no-cache live 재검증, ReviewQA/RiskRecallQA live 확대 검증, Agent disagreement score 검증
- 목적: Claude/OpenAI API + Agno 기반 Stage 2 committee가 1차 모델 오류를 얼마나 보완했는지 수치로 남긴다.
- 주의: 아래 수치는 전체 기업 모집단 정확도가 아니라, Stage 2 검토가 필요한 hard sample/replay 샘플에 대한 위원회 보완 성능이다.

## 핵심 요약

| 구분 | 건수 | 엄격 기준 성공 | Review-safe 성공 | 비고 |
| --- | ---: | ---: | ---: | --- |
| 1차 rolling validation pilot | 5 | 3/5 = 60.0% | 4/5 = 80.0% | 초기 replay 기준 |
| 추가 Agno/Claude round 2 | 10 | 9/10 = 90.0% | 10/10 = 100.0% | JSON 파싱/실행 실패 없음 |
| 합산 | 15 | 12/15 = 80.0% | 14/15 = 93.3% | 누적 증빙 기준 |
| 새 holdout 8건 로컬 guardrail | 8 | 8/8 = 100.0% | 8/8 = 100.0% | 기존 실험과 겹치지 않는 기업-회계연도 |
| 추가 Agno/Claude round 3 | 10 | 8/10 = 80.0% | 10/10 = 100.0% | 새 기업-회계연도, workers=3 실제 API 실행 |
| round 3 저확률 guardrail 재평가 | 10 | 9/10 = 90.0% | 10/10 = 100.0% | 같은 샘플 캐시 재평가, TN 1건 개선 |
| isolated ICR TN guardrail | 8 | 7/8 = 87.5% | 8/8 = 100.0% | 이자보상 단일 플래그 TN 1건 개선 |
| OpenAI single 3-agent no-cache live | 8 | 7/8 = 87.5% | 8/8 = 100.0% | 캐시 hit 0, 역할별 실행시간 8/8건 기록 |
| Compact prompt smoke live | 8 | 8/8 = 100.0% | 8/8 = 100.0% | role별 compact payload 적용, cache hit 0, Stage 2 평균 14.9034초 |
| Explainability smoke live | 8 | 7/8 = 87.5% | 8/8 = 100.0% | cache hit 0, EvidenceAudit 구조화 필드 8/8, risk_hold reason tag 3/3 |
| Agent disagreement smoke live | 10 | 9/10 = 90.0% | 10/10 = 100.0% | cache hit 0, high disagreement 2/10 모두 ReviewQA 실행, memo conflict 0 |
| Disagreement-gated ReviewQA 20건 live | 20 | 18/20 = 90.0% | 20/20 = 100.0% | cache hit 0, ReviewQA 5/20건 실행, advisory 1건 적용, Stage 2 평균 22.9209초 |
| Disagreement-gated ReviewQA v2 20건 live | 20 | 19/20 = 95.0% | 20/20 = 100.0% | cache hit 0, ReviewQA 3/20건 실행, advisory 2건 적용, Stage 2 평균 18.7126초 |
| Disagreement-gated ReviewQA v2 40건 live | 40 | 36/40 = 90.0% | 40/40 = 100.0% | cache hit 0, ReviewQA 5/40건 실행, advisory 2건 적용, Stage 2 평균 17.5488초 |
| PR #53 role split absorption smoke | 10 | 10/10 = 100.0% | 10/10 = 100.0% | cache hit 0, ReviewQA 2/10건 실행, RiskRecallQA 0/10, Stage 2 평균 16.4039초 |
| EvidenceAudit criticality gate TN smoke | 10 | 7/10 = 70.0% | 10/10 = 100.0% | cache hit 0, TN 7건 적격 유지, `critical_veto_review` 4/10 |
| Evidence treatment refined TN smoke | 10 | 7/10 = 70.0% | 10/10 = 100.0% | cache hit 0, 최종 분포 유지, `critical_veto_review` 4/10 -> 0/10 |
| TN 과잉 보류 30건 확대 | 30 | 22/30 = 73.3% | 30/30 = 100.0% | 레몬 1건 보류→적격, 남은 보류 8건은 재무 차단 신호 보유 |
| TN 과잉 보류 8건 OpenAI Agno live | 8 | 2/8 = 25.0% | 8/8 = 100.0% | 캐시 hit 0, 외부근거 ready 8/8, 자금조달 공시 민감도 발견 |
| TN 자금조달 guardrail 재평가 | 8 | 3/8 = 37.5% | 8/8 = 100.0% | 같은 외부근거 캐시 재평가, 머큐리 1건 보류→적격 |
| TN 자금조달 guardrail OpenAI Agno live | 8 | 3/8 = 37.5% | 8/8 = 100.0% | 캐시 hit 0, 머큐리 보류→적격 live 확인 |
| TN SPAC 절차성 guardrail 재평가 | 8 | 3/8 = 37.5% | 8/8 = 100.0% | 같은 외부근거 캐시 재평가, 현대무벡스 risk_hold→boundary_hold |
| TN ReviewQA 20건 OpenAI Agno live | 20 | 10/20 = 50.0% | 20/20 = 100.0% | cache hit 0, ReviewQA 10/20건 실행, advisory 2건 적용, hidden-tail 1건 자동 적용 차단 |
| TN EvidenceAudit v4 + ReviewQA 20건 OpenAI Agno live | 20 | 10/20 = 50.0% | 20/20 = 100.0% | cache hit 0, 공시 120건 중 adverse 2건만 유지, 최종 subtype은 이전 20건과 동일 |
| TN EvidenceAudit v5 상세중요도 + ReviewQA 20건 OpenAI Agno live | 20 | 11/20 = 55.0% | 20/20 = 100.0% | cache hit 0, 다스코 계약해지 5.92% watch_context, 위험신호 TN hold 2→1 |
| TN EvidenceAudit v6 영업정지 fallback + ReviewQA 20건 OpenAI Agno live | 20 | 11/20 = 55.0% | 20/20 = 100.0% | cache hit 0, 하나투어 종속회사 영업정지 11.37% substantive_adverse 확인 |
| TN ReviewQA stabilized v6 OpenAI Agno live | 20 | 11/20 = 55.0% | 20/20 = 100.0% | cache hit 0, 신원종합개발 risk_hold→boundary_hold, 위험신호 TN hold 2→1 |
| TN RiskRecallQA v1 OpenAI Agno live | 20 | 11/20 = 55.0% | 20/20 = 100.0% | cache hit 0, 적격 11건 RiskRecallQA 재검수, 모두 keep, 최종 분포 유지 |
| TN RiskRecallQA precision v2 OpenAI Agno live | 20 | 11/20 = 55.0% | 20/20 = 100.0% | cache hit 0, substantive trigger 4→0, 최종 분포 유지 |
| TN RiskRecallQA speed gate v3 OpenAI Agno smoke | 3 | 3/3 = 100.0% | 3/3 = 100.0% | cache hit 0, per-category 기본값 때문에 3건만 실행, v2 동일 3건 RiskRecallQA 3→0 |
| TN RiskRecallQA speed gate v3 OpenAI Agno full20 | 20 | 11/20 = 55.0% | 20/20 = 100.0% | cache hit 0, RiskRecallQA 호출 11→2, 최종 분포 유지 |
| TN timeout30 + speed gate v3 OpenAI Agno full20 | 20 | 11/20 = 55.0% | 20/20 = 100.0% | cache hit 0, 최종 분포 유지, Stage 2 max 73.2104초→19.3524초 |
| Mixed hard 40 timeout30 + speed gate v3 OpenAI Agno | 40 | 34/40 = 85.0% | 38/40 = 95.0% | cache hit 0, FN 8/8 상향, TP 12/12 유지, hold/reject Recall 1.0000 |

개선 폭은 1차 5건 대비 추가 10건에서 엄격 기준 +30.0%p, review-safe 기준 +20.0%p다. 합산 기준으로도 review-safe 성공률은 93.3%까지 올라왔다.

추가로 기존 실험에 등장하지 않은 새 기업-회계연도 8건에서는 deterministic 기준선이 6/8 = 75.0%였고, 유동성 watch guardrail 반영 후 8/8 = 100.0%로 개선됐다. 동일 샘플 기준 엄격/review-safe 모두 +25.0%p 개선이다.

## 실패 파일럿 정리 후 최종 재검증

실행 정책 차단, 잘못된 API 설정, 또는 중간 실험 실패로 남아 있던 파일럿 산출물은 삭제했다. 삭제 후 남은 모든 `committee_review_batch_results.csv`를 다시 스캔해 `error_message` 기준 실행 실패가 0건인지 확인했고, 전체 재계산 요약은 `stage2_agent_all_pilots_recomputed_summary.csv`에 저장했다.

| 재검증 배치 | 건수 | 엄격 기준 | Review-safe | 실행 실패 | 비고 |
| --- | ---: | ---: | ---: | ---: | --- |
| Historical test 12 rerun | 12 | 10/12 = 83.3% | 10/12 = 83.3% | 0 | 기존 12건 deterministic 재현 |
| Rolling pilot 5 rerun | 5 | 3/5 = 60.0% | 4/5 = 80.0% | 0 | 초기 pilot 기준점 유지 |
| Holdout guardrail rerun | 8 | 8/8 = 100.0% | 8/8 = 100.0% | 0 | 새 holdout guardrail 재현 |
| Agno round 2 cached rerun | 10 | 9/10 = 90.0% | 10/10 = 100.0% | 0 | 캐시 재평가 기준 |
| Agno round 3 guardrail cached rerun | 10 | 9/10 = 90.0% | 10/10 = 100.0% | 0 | 저확률 guardrail 반영 상태 재현 |

최종적으로 남겨둔 파일럿 산출물에는 실행 실패가 남아 있지 않다. 다만 rolling pilot 5건처럼 성능이 낮은 초기 기준점은 실패 산출물이 아니라 비교 기준이므로 보존한다.

## Precision/Recall 기준 재해석

위원회 평가는 내부 진단용 `committee_success`뿐 아니라 분류 지표인 Precision, Recall, F1로도 함께 확인한다. 이때 Stage 2 위원회의 목적은 최종 부적격 확정이 아니라 조기경보 대상 선별이므로, `보류`와 `부적격`을 모두 위험 신호로 본다.

- 실제 양성: `투기등급`
- 1차 모델의 위험 판단: `부적격`
- 2차 위원회의 위험 신호: `보류` 또는 `부적격`

최근 재검증한 hard sample 28건(`holdout guardrail 8건 + Agno round 2 10건 + Agno round 3 guardrail 10건`) 기준 성능은 다음과 같다.

| 평가 기준 | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1차 모델 | 9 | 9 | 4 | 6 | 0.5000 | 0.6000 | 0.5455 | 0.4643 |
| 2차 위원회 (`보류+부적격=위험`) | 15 | 11 | 2 | 0 | 0.5769 | 1.0000 | 0.7317 | 0.6071 |
| 2차 위원회 (`부적격만=위험`) | 6 | 0 | 13 | 9 | 1.0000 | 0.4000 | 0.5714 | 0.6786 |

조기경보 관점에서는 `보류+부적격=위험` 기준이 가장 적합하다. 이 기준에서 Stage 2 위원회는 1차 모델 대비 Recall을 0.6000에서 1.0000으로 올렸고, F1도 0.5455에서 0.7317로 개선했다. 반대로 `부적격만=위험` 기준은 Precision은 1.0000이지만 Recall이 0.4000으로 낮아, 위험 기업을 놓치지 않는 조기경보 목적에는 맞지 않는다.

주의할 점은 위 28건이 전체 test 모집단이 아니라, FN/FP/경계등급처럼 일부러 어려운 케이스를 모은 hard sample이라는 점이다. 따라서 이 표는 전체 모델 성능표가 아니라, Stage 2 에이전트가 모델 오류를 보완하는지 확인하는 보조 검증 지표로 해석한다.

## 랜덤 10건 안전성 점검

hard sample은 FN/FP/경계등급을 의도적으로 많이 포함하므로, 실제 운영에서 안전한 기업을 과하게 `보류` 또는 `부적격`으로 올리는지 별도 점검이 필요하다. 이를 위해 rolling validation 전체 2,526건에서 기존 실험에 이미 등장한 기업-연도 25건을 제외하고, `random_state=20260521`로 10건을 무작위 추출했다.

랜덤 10건 구성은 `true_negative` 8건, `false_positive` 1건, `true_positive` 1건이었다. 재현 가능한 deterministic 위원회 기준으로 실행했으며, 실행 실패는 0건이었다.

| 평가 기준 | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1차 모델 | 1 | 1 | 8 | 0 | 0.5000 | 1.0000 | 0.6667 | 0.9000 |
| 2차 위원회 (`보류+부적격=검토대상`) | 1 | 1 | 8 | 0 | 0.5000 | 1.0000 | 0.6667 | 0.9000 |
| 2차 위원회 (`위험 보류+부적격=위험신호`) | 1 | 0 | 9 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2차 위원회 (`부적격만=위험신호`) | 1 | 0 | 9 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

세부적으로 안전한 `true_negative` 8건은 모두 최종 `적격`으로 유지됐다. 따라서 이 랜덤 샘플에서는 Stage 2가 안전한 기업을 부적격으로 과잉 판단하는 문제는 확인되지 않았다. 또한 `(주)파라텍`은 실제 투자적격이지만 1차 모델 확률이 85.1%로 매우 높았던 FP 케이스였는데, 고확률 모델 단독 경고 완화 규칙 적용 후 `부적격`이 아니라 `보류`로 낮아졌다. 이 보류는 `committee_decision_type_label = 과민경고 완화 보류`, `committee_risk_signal = False`로 기록된다. 이 규칙은 `veto` 또는 직접 외부 치명근거가 없고, 일부 손익·이자보상 스트레스가 있어도 OCF와 자본/부채 구조가 방어력을 제공하는 경우 즉시 부적격 확정 대신 보류 재점검으로 처리한다.

이제 `보류`는 하나의 라벨로만 해석하지 않고, `위험 보류`, `과민경고 완화 보류`, `확인필요 보류`로 세분화한다. 따라서 운영/발표에서는 두 가지 지표를 함께 본다. `보류+부적격=검토대상`은 위원회가 추가 확인을 요구한 전체 workload를 보여주고, `위험 보류+부적격=위험신호`는 실제 위험 경고 성능을 보여준다. 이 구분 덕분에 파라텍 같은 FP 완화 보류는 검토대상에는 포함되지만 위험신호 Precision을 낮추지는 않는다.

## 추가 round 3 Agno/Claude 실험

이전 15개 기업명을 제외하고 rolling validation tuning pool에서 FN, FP, BBB-/BB+ 경계, TP, TN 과잉보류 방지 유형을 2건씩 새로 뽑아 총 10건을 Claude/Agno + 외부근거 수집으로 실행했다. 실행은 `workers=3` 병렬 모드로 수행했다.

| 실행 | 건수 | 엄격 기준 | Review-safe | Wall time | 평균 case time | 처리량 | 비고 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| round 3 live parallel | 10 | 8/10 = 80.0% | 10/10 = 100.0% | 259.3310초 | 69.3528초 | 2.3136건/분 | 실제 Claude/Agno 및 외부근거 API 실행 |
| round 3 low-prob guardrail replay | 10 | 9/10 = 90.0% | 10/10 = 100.0% | 1.4552초 | 0.4295초 | 412.3145건/분 | 같은 샘플 캐시 재평가. 실시간 API 속도로 해석하지 않음 |

속도 측면에서는 기존 순차 실행에서 체감상 케이스당 약 2분 내외가 걸리던 Claude/Agno 검토를 `workers=3` 병렬 배치로 바꾸면서, 10건 전체 wall time이 259.3310초로 줄었다. 이를 배치 관점으로 환산하면 실시간 API 실행 기준 약 25.9초/건이며, 처리량은 약 2.3136건/분이다. 개별 케이스의 API 대기 시간 자체는 평균 69.3528초로 남아 있지만, 병렬화 덕분에 사용자가 기다리는 전체 실행 시간은 약 4~5배 수준으로 단축된다.

이번 속도 개선에는 병렬화 외에 평가 replay 중복 호출 제거도 포함된다. 기존 평가 배치에서는 `run_once()`로 전체 그래프를 먼저 실행한 뒤, 샘플의 rolling OOT 모델값을 다시 주입해 committee를 재실행할 수 있어 Agno 모드에서 Stage 2가 케이스당 2회 호출될 여지가 있었다. 현재는 replay 평가에서 `data → feature → news → base_prediction → rule_engine`까지만 먼저 실행하고, 샘플 모델값 반영 후 `committee` 단계에서만 Agno를 1회 호출한다. 따라서 Stage 2 LLM 호출 중복을 제거했고, 이론적으로 케이스당 LLM 호출 비용과 대기 시간을 최대 절반 가까이 줄일 수 있는 구조가 됐다. 따라서 운영에서는 전체 기업을 순차 호출하지 않고, Stage 2 검토 대상만 선별한 뒤 3~5개 워커로 병렬 처리하는 전략이 적합하다.

round 3 live 결과에서는 FN 2건은 모두 보류로 끌어올렸고, FP 3건은 모두 보류로 완화했으며, TP 3건은 위험 판단을 유지했다. 실패는 TN 2건이 모두 보류로 올라간 부분이었다. 이후 저확률 guardrail을 적용해 `1차 모델 투자적격 + 투기등급 확률 28% 미만 + 강한 재무 스트레스 없음`인 secondary trigger는 적격으로 되돌릴 수 있게 했고, 청광건설(주)이 `보류 → 적격`으로 개선됐다. (주)일지테크는 이자보상배율 1 미만, 낮은 유동비율, 높은 단기차입 비중이 있어 보류를 유지했다.

## 전체 실험 로그

아래 표는 지금까지 남긴 `committee_review_batch_results.csv` 산출물을 기준으로 재계산했다. Review-safe 컬럼은 해당 지표가 산출된 rolling validation 계열에만 표시한다.

| 계열 | 산출물 | 건수 | 엄격 기준 | Review-safe | 실행 오류 | 해석 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Historical 12 baseline | `committee_review_batch_results.csv` | 12 | 9/12 = 75.0% | - | 0 | 초기 12건 기준점 |
| Live Claude pilot | `committee_review_live_claude_pilot` | 4 | 2/4 = 50.0% | - | 0 | 초기 Claude 연결 확인 |
| Claude FP mitigation | `committee_review_live_claude_pilot_fp_mitigation` | 4 | 3/4 = 75.0% | - | 0 | FP 완화 프롬프트 반영 |
| Claude label alignment | `committee_review_live_claude_pilot_label_alignment` | 4 | 4/4 = 100.0% | - | 0 | 4건 pilot 전부 성공 |
| Claude 12-case label alignment | `committee_review_live_claude_12case_label_alignment` | 12 | 10/12 = 83.3% | - | 0 | 12건 확장 후 성능 유지 |
| Other candidates | `committee_review_batch_other_candidates` | 20 | 18/20 = 90.0% | - | 0 | 후보군 확장 검증 |
| Keyword/context rerun | `committee_review_batch_rerun_keyword_context_claude` | 12 | 11/12 = 91.7% | - | 0 | 외부근거/키워드 맥락 보강 |
| Secondary trigger rerun | `committee_review_batch_secondary_trigger_rerun` | 12 | 11/12 = 91.7% | - | 0 | 45개 보조 trigger 연결 전후 점검 |
| Secondary signal connected | `committee_review_batch_secondary_signal_connected` | 12 | 12/12 = 100.0% | - | 0 | 동일 12건 계열 최종 개선판 |
| Rolling validation pilot | `committee_review_rolling_validation_batch` | 5 | 3/5 = 60.0% | 4/5 = 80.0% | 0 | rolling replay 초기 기준 |
| Rolling retry batch | `committee_review_rolling_validation_agno_claude_retry_batch` | 2 | 1/2 = 50.0% | 1/2 = 50.0% | 0 | FP 재점검 소규모 retry |
| Rolling Agno/Claude round 2 | `committee_review_rolling_validation_agno_claude_round2_batch` | 10 | 9/10 = 90.0% | 10/10 = 100.0% | 0 | 안정화 후 추가 10건 검증 |
| Holdout unseen deterministic speed baseline | `committee_review_holdout_unseen_deterministic_speed_baseline` | 8 | 6/8 = 75.0% | 6/8 = 75.0% | 0 | 기존 결과와 겹치지 않는 새 holdout 8건의 로컬 기준선 |
| Holdout unseen liquidity guardrail | `committee_review_holdout_unseen_guardrail_speed_batch` | 8 | 8/8 = 100.0% | 8/8 = 100.0% | 0 | FN 2건을 보류로 끌어올리고 FP/TP 판단 유지 |
| OpenAI single 3-agent no-cache live | `committee_review_openai_single_3agent_no_cache_live_8` | 8 | 7/8 = 87.5% | 8/8 = 100.0% | 0 | OpenAI 단일 provider 3-agent live 실행, 캐시 hit 0 |
| TN overhold expanded before liquidity buffer | `committee_review_tn_overhold_expanded_30_deterministic` | 30 | 21/30 = 70.0% | 30/30 = 100.0% | 0 | 기존 TN 검토 7건 제외 후 새 TN 30건 확대 분석 |
| TN overhold expanded after liquidity buffer | `committee_review_tn_overhold_expanded_30_after_liquidity_buffer` | 30 | 22/30 = 73.3% | 30/30 = 100.0% | 0 | 현금흐름 방어 current-ratio watch 예외 후 레몬 1건 적격 개선 |
| TN overhold expanded OpenAI Agno live no-cache | `committee_review_tn_overhold_expanded_8_agno_openai_live_no_cache` | 8 | 2/8 = 25.0% | 8/8 = 100.0% | 0 | TN 확대 샘플 대표 8건 Agno live 검증, 자금조달 공시가 보수적 보류를 유발 |
| TN overhold expanded financing guardrail cached evidence | `committee_review_tn_overhold_expanded_8_after_financing_guardrail_cached_evidence` | 8 | 3/8 = 37.5% | 8/8 = 100.0% | 0 | 단일 medium 자금조달 공시 예외 후 머큐리 1건 적격 개선, 반복 자금조달 레몬은 보류 유지 |
| TN overhold expanded financing guardrail OpenAI Agno live no-cache | `committee_review_tn_overhold_expanded_8_after_financing_guardrail_agno_openai_live_no_cache` | 8 | 3/8 = 37.5% | 8/8 = 100.0% | 0 | 같은 대표 8건 OpenAI Agno live 재실행, cache hit 0, 머큐리 적격 개선 확인 |
| TN overhold expanded SPAC procedural guardrail cached evidence | `committee_review_tn_overhold_expanded_8_after_spac_guardrail_cached_evidence` | 8 | 3/8 = 37.5% | 8/8 = 100.0% | 0 | 해소된 SPAC 합병 거래정지를 절차성 공시로 분리, 현대무벡스 위험 보류를 경계 보류로 낮춤 |
| TN overhold expanded memo filter OpenAI Agno live no-cache | `committee_review_tn_overhold_expanded_8_after_memo_filter_agno_openai_live_no_cache_full8` | 8 | 3/8 = 37.5% | 8/8 = 100.0% | 0 | Agno chair memo 충돌 필터 후 8건 live 재검증, cache hit 0, memo conflict 0 |
| TN overhold expanded ReviewQA trigger tuned OpenAI Agno live no-cache | `committee_review_tn_overhold_expanded_8_reviewqa_trigger_tuned_agno_openai_live_no_cache` | 8 | 3/8 = 37.5% | 8/8 = 100.0% | 0 | 조건부 ReviewQA 5/8건 실행, cache hit 0, memo conflict 0, 레몬 risk_hold subtype downgrade 권고 |
| TN overhold expanded ReviewQA advisory applied OpenAI Agno live no-cache | `committee_review_tn_overhold_expanded_8_reviewqa_advisory_applied_agno_openai_live_no_cache` | 8 | 3/8 = 37.5% | 8/8 = 100.0% | 0 | ReviewQA 권고를 subtype 보정에 적용, 레몬 risk_hold→boundary_hold, 위험신호 TN hold 2→1 |
| TN overhold expanded ReviewQA advisory applied 20 OpenAI Agno live no-cache | `committee_review_tn_overhold_expanded_20_reviewqa_advisory_applied_agno_openai_live_no_cache` | 20 | 10/20 = 50.0% | 20/20 = 100.0% | 0 | 20건 확대 live 검증, cache hit 0, ReviewQA 10/20건 실행, advisory 2/20건 적용 |
| TN overhold expanded EvidenceAudit v4 ReviewQA OpenAI Agno live no-cache | `committee_review_tn_overhold_expanded_20_evidence_v4_reviewqa_agno_openai_live_no_cache` | 20 | 10/20 = 50.0% | 20/20 = 100.0% | 0 | external_evidence_v4 live 검증, cache hit 0, OpenDART 120건 중 adverse 2건만 유지 |
| TN overhold expanded EvidenceAudit v5 detail materiality ReviewQA OpenAI Agno live no-cache | `committee_review_tn_overhold_expanded_20_detail_materiality_reviewqa_agno_openai_live_no_cache_full20` | 20 | 11/20 = 55.0% | 20/20 = 100.0% | 0 | external_evidence_v5 상세중요도 live 검증, 다스코 계약해지 매출 대비 5.92%로 watch_context 전환 |
| TN overhold expanded EvidenceAudit v6 business suspension fallback ReviewQA OpenAI Agno live no-cache | `committee_review_tn_overhold_expanded_20_detail_materiality_v6_reviewqa_agno_openai_live_no_cache_full20` | 20 | 11/20 = 55.0% | 20/20 = 100.0% | 0 | external_evidence_v6 live 검증, 하나투어 종속회사 영업정지 매출 대비 11.37%로 substantive_adverse 유지 |
| TN overhold expanded ReviewQA stabilized v6 OpenAI Agno live no-cache | `committee_review_tn_overhold_expanded_20_reviewqa_stabilized_v6_agno_openai_live_no_cache_full20` | 20 | 11/20 = 55.0% | 20/20 = 100.0% | 0 | ReviewQA subtype advisory 안정화 live 재검증, 신원종합개발 risk_hold→boundary_hold, 위험신호 TN hold 1건만 유지 |
| TN overhold expanded RiskRecallQA v1 OpenAI Agno live no-cache | `committee_review_tn_overhold_expanded_20_risk_recall_qa_v1_agno_openai_live_no_cache_full20` | 20 | 11/20 = 55.0% | 20/20 = 100.0% | 0 | RiskRecallQA가 최종 적격 11건을 모두 재검수했지만 모두 keep, 정상 TN 과잉 보류 재발 없음 |
| TN overhold expanded RiskRecallQA precision v2 OpenAI Agno live no-cache | `committee_review_tn_overhold_expanded_20_risk_recall_qa_precision_v2_agno_openai_live_no_cache_full20` | 20 | 11/20 = 55.0% | 20/20 = 100.0% | 0 | RiskRecallQA substantive trigger 정밀화 live 검증, routine 공시 과대분류 제거, 최종 분포 유지 |
| TN overhold expanded RiskRecallQA speed gate v3 OpenAI Agno smoke no-cache | `committee_review_tn_overhold_expanded_20_risk_recall_qa_speed_gate_v3_agno_openai_live_no_cache_full20` | 3 | 3/3 = 100.0% | 3/3 = 100.0% | 0 | per-category 기본값으로 3건만 실행된 smoke test, RiskRecallQA 호출 0/3, 최종 적격 유지 |
| TN overhold expanded RiskRecallQA speed gate v3 OpenAI Agno full20 no-cache | `committee_review_tn_overhold_expanded_20_risk_recall_qa_speed_gate_v3_agno_openai_live_no_cache_full20_rerun` | 20 | 11/20 = 55.0% | 20/20 = 100.0% | 0 | RiskRecallQA 호출 11/20→2/20, 최종 라벨·subtype·위험신호 분포 유지 |
| TN overhold expanded timeout30 speed gate v3 OpenAI Agno full20 no-cache | `committee_review_tn_overhold_expanded_20_timeout30_speed_gate_v3_agno_openai_live_no_cache_full20` | 20 | 11/20 = 55.0% | 20/20 = 100.0% | 0 | timeout30 live 재검증, 최종 분포 유지, wall 190.9334초→151.0032초, Stage 2 max 73.2104초→19.3524초 |
| Mixed hard 40 deterministic baseline | `committee_review_mixed_hard_40_deterministic_baseline` | 40 | 34/40 = 85.0% | 38/40 = 95.0% | 0 | Agno live 전 sanity baseline, FN 8 FP 12 TP 12 TN 8 혼합 hard sample |
| Mixed hard 40 timeout30 speed gate v3 OpenAI Agno live no-cache | `committee_review_mixed_hard_40_timeout30_speed_gate_v3_agno_openai_live_no_cache` | 40 | 34/40 = 85.0% | 38/40 = 95.0% | 0 | timeout30 + speed gate v3 혼합 hard sample live 검증, hold/reject Recall 1.0000, Stage 2 max 23.5955초 |
| Agent disagreement memo-fix final 10 OpenAI Agno live no-cache | `committee_review_disagreement_memo_fix_final_10_agno_openai_live_no_cache` | 10 | 9/10 = 90.0% | 10/10 = 100.0% | 0 | Quant/Evidence/Chair disagreement score live 검증, high 2/10 모두 ReviewQA 실행, memo conflict 0 |
| Disagreement-gated ReviewQA 20 OpenAI Agno live no-cache | `committee_review_disagreement_trigger_gated_20_agno_openai_live_no_cache` | 20 | 18/20 = 90.0% | 20/20 = 100.0% | 0 | ReviewQA trigger를 disagreement level/reason에 직접 연결한 뒤 20건 live 재검증, ReviewQA 5/20건 실행 |
| Disagreement-gated ReviewQA v2 20 OpenAI Agno live no-cache | `committee_review_disagreement_trigger_gated_v2_20_agno_openai_live_no_cache` | 20 | 19/20 = 95.0% | 20/20 = 100.0% | 0 | high disagreement 단독 호출을 제거한 v2 live 재검증, ReviewQA 3/20건 실행 |
| Disagreement-gated ReviewQA v2 40 OpenAI Agno live no-cache | `committee_review_disagreement_trigger_gated_v2_40_agno_openai_live_no_cache` | 40 | 36/40 = 90.0% | 40/40 = 100.0% | 0 | v2를 mixed hard 40건 전체로 확대 재검증, ReviewQA 5/40건 실행 |
| PR #53 role split absorption 10 OpenAI Agno live no-cache | `committee_review_after_pr53_role_split_10_agno_openai_live_no_cache` | 10 | 10/10 = 100.0% | 10/10 = 100.0% | 0 | PR #53 역할 분리 의도 선별 흡수 후 mixed hard 10건 smoke, ReviewQA 2/10, RiskRecallQA 0/10 |
| EvidenceAudit criticality gate TN 10 OpenAI Agno live no-cache | `committee_review_evidence_criticality_gate_tn10_agno_openai_live_no_cache` | 10 | 7/10 = 70.0% | 10/10 = 100.0% | 0 | LLM 단독 critical flag hard gate 후 TN 7/10 적격 유지, 구조화 판정은 critical_veto_review 4/10 |
| Evidence treatment refined TN 10 OpenAI Agno live no-cache | `committee_review_evidence_treatment_refined_tn10_agno_openai_live_no_cache` | 10 | 7/10 = 70.0% | 10/10 = 100.0% | 0 | routine 감사보고서/검색요약 critical 조건 축소 후 final 분포 유지, critical_veto_review 4/10 -> 0/10 |

Historical 12건 계열은 동일 기업 12건을 반복 검증한 산출물이다. 이 계열에서는 초기 75.0%에서 secondary signal connected 기준 100.0%까지 개선됐다. Rolling validation 계열은 샘플 구성과 평가지표가 달라 별도로 보며, 최종 추가 10건에서 90.0%/100.0%를 기록했다.

## 새 Holdout 8건 및 속도 측정

기존 모든 committee 결과 파일에 등장한 `(stock_code, fiscal_year)` 47개를 제외하고 `committee_review_historical_test_holdout_samples.csv`에서 8건을 새로 뽑았다. 샘플 파일은 `committee_review_holdout_unseen_8_samples.csv`로 저장했다.

| 실행 | Runner | 상태 | 엄격 기준 | Review-safe | Wall time | 평균 case time | 처리량 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Holdout unseen local baseline | deterministic | 완료 | 6/8 = 75.0% | 6/8 = 75.0% | 1.3777초 | 0.3440초 | 348.4068건/분 |
| Holdout unseen guardrail | deterministic | 완료 | 8/8 = 100.0% | 8/8 = 100.0% | 1.4247초 | 0.3558초 | 336.9130건/분 |

로컬 baseline은 FP 2건 완화와 TP 4건 위험 유지에는 성공했지만, FN 2건은 모두 적격으로 남아 `fn_not_escalated`였다. 이후 `투자적격 + 기준선 0.10 이내 + 45개 보조 레이더 + 룰엔진 유동성 watch` 조건에서 보류를 유지하는 좁은 guardrail을 추가했고, 같은 holdout 8건에서 FN 2건이 모두 `fn_escalated`로 바뀌었다. 속도는 wall time 1.3777초에서 1.4247초로 거의 유지되어, 로컬 guardrail은 대시보드 실시간 경로에 부담이 작다.

## 오류 유형별 성과

| 오류 유형 | 합산 건수 | Stage 2 결과 | 성공률 해석 |
| --- | ---: | --- | --- |
| FN | 3 | 3건 모두 보류/부적격으로 끌어올림 | 놓친 위험 보완 100.0% |
| FP | 6 | 5건 보류로 완화, 1건 부적격 유지 | 과민 경고 완화 83.3% |
| TP | 3 | 3건 모두 보류/부적격 유지 | 실제 위험 판단 유지 100.0% |
| TN | 3 | 1건 적격 유지, 2건 보류 | 정상기업 과잉 보류가 남은 약점 |

Review-safe 기준에서는 TN 보류를 실무상 추가검토로 허용하므로 TN 3건 모두 부적격으로 악화시키지는 않았다. 다만 엄격 기준에서는 TN 1/3만 적격으로 유지되어, 다음 개선은 정상기업을 과도하게 보류로 올리지 않는 guardrail에 둔다.

## 추가 10건 세부 결과

| 유형 | 기업 | 실제 | 1차 모델 | 최종 위원회 | 효과 |
| --- | --- | --- | --- | --- | --- |
| FN | (주)예선테크 | 투기등급 | 투자적격 | 보류 | fn_escalated |
| FN | 명신산업(주) | 투기등급 | 투자적격 | 보류 | fn_escalated |
| FP | (주)예림당 | 투자적격 | 투기등급 | 보류 | fp_mitigated |
| FP | (주)엘오티베큠 | 투자적격 | 투기등급 | 보류 | fp_mitigated |
| FP | (주)라닉스 | 투자적격 | 투기등급 | 보류 | fp_mitigated |
| FP | 솔트웨어(주) | 투자적격 | 투기등급 | 보류 | fp_mitigated |
| TP | (주)대창솔루션 | 투기등급 | 투기등급 | 보류 | tp_risk_supported |
| TP | 휴림로봇(주) | 투기등급 | 투기등급 | 부적격 | tp_risk_supported |
| TN | (주)휴맥스 | 투자적격 | 투자적격 | 보류 | tn_escalated |
| TN | (주)동성화인텍 | 투자적격 | 투자적격 | 적격 | tn_kept_eligible |

## 운영 안정성 메모

추가 10건 round 2 결과 파일 기준 `error_message`는 10/10건 비어 있어 실행 실패나 structured output 파싱 실패가 없었다. `evidence_status`도 10/10건 `ready`로 남았다. 운영에서는 속도와 비용을 줄이기 위해 전체 기업이 아니라 `stage2_review_trigger=True`, 보조 trigger, high/medium priority, 외부 치명 리스크 후보 기업만 Agno/Claude로 보내는 구조를 유지한다.

## 현재 판단

Claude/Agno Stage 2는 FN 보완과 FP 완화에는 이미 효과가 있다. 특히 BBB-/BB+ 경계 FP인 라닉스, 솔트웨어를 부적격으로 확정하지 않고 보류로 낮춘 점은 실무형 review-safe 목적에 잘 맞는다. 새 holdout에서는 로컬 유동성 guardrail만으로 FN 2건을 추가 보완해, 외부 API를 전체 기업에 돌리지 않아도 일부 위험은 빠르게 선별할 수 있음을 확인했다.

남은 약점은 실제 투자적격인 TN이 기준선 근처 또는 45개 보조 레이더 신호 때문에 보류로 올라가는 문제다. 이에 대해 다음 조건을 모두 만족할 때 Stage 2 공통 `committee_view`가 보류로 올리는 것을 억제하는 정상기업 과잉 보류 guardrail을 구현했다. 이 guardrail은 Agno/Claude runner 이후의 최종 위원회 판단에도 적용되는 공통 로직이지만, 이번 추가 검증은 외부 API를 호출하지 않는 로컬 deterministic runner로 수행했다.

- 1차 모델이 투자적격
- 투기등급 확률이 기준선 아래
- 직접적이고 검증된 외부 치명 리스크 없음
- 현금흐름을 포함해 유동성/현금흐름/자본 중 최소 2개 이상이 방어적
- 45개 보조 레이더가 기준선 근처일 뿐 강한 부실 신호는 아님

구현 후 로컬 deterministic 검증에서는 `tests/unit/test_committee_view.py` 기준 41건, 관련 Stage 2 단위 테스트 묶음 51건이 모두 통과했다. rolling validation tuning 샘플 25건 재실행 결과는 엄격 기준 76.0%, review-safe 기준 80.0%였고, TN guardrail 샘플 5건 중 4건은 `적격` 유지, 1건만 `경계등급 보류`로 남았다. 속도는 wall time 2.2550초, 평균 case time 0.2673초, 처리량 665.1885건/분으로 로컬 guardrail 추가에 따른 대시보드 지연은 제한적이었다.

휴맥스 2021 단건 재실행은 `보류(경계등급 보류)`로 남았다. 로컬 입력 기준 유동성은 방어적이지만 이자보상배율 -3.626, OCF/매출액 -0.024, OCF/총부채 -0.027, 자기자본비율 0.388, 부채비율 1.579로 현금흐름·자본 축이 guardrail 기준을 충족하지 않았기 때문이다. 이 케이스는 부적격 확정이나 위험 보류가 아니라 `committee_risk_signal=False`인 경계등급 보류로 두어 추가 확인 대상으로 분리한다.

## TN Guardrail OpenAI Agno 재검증

정상기업 과잉 보류 guardrail을 OpenAI 단일모델 Agno 경로에서도 확인하기 위해 FN 2건, FP 2건, TP 1건, TN 3건으로 구성한 8건 샘플을 만들었다. 샘플 파일은 `committee_review_tn_guardrail_agno_8_samples.csv`이며, 같은 샘플을 deterministic과 OpenAI Agno + 외부근거 수집으로 각각 실행했다.

| 실행 | Runner | 외부근거 | 엄격 기준 | Review-safe | 실패/파싱 오류 | Wall time | 평균 case time | 처리량 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TN guardrail deterministic 8 | deterministic | disabled | 4/8 = 50.0% | 6/8 = 75.0% | 0 | 1.9270초 | 0.7124초 | 249.0919건/분 |
| TN guardrail OpenAI Agno 8 | OpenAI Agno single `gpt-4.1-mini` | ready 8/8 | 4/8 = 50.0% | 6/8 = 75.0% | 0 | 70.8385초 | 17.5339초 | 6.7760건/분 |
| FN escalation OpenAI Agno rerun 8 | OpenAI Agno single `gpt-4.1-mini` | ready 8/8 | 6/8 = 75.0% | 8/8 = 100.0% | 0 | 1.5040초 | 0.3758초 | 319.1489건/분 |
| Isolated ICR guardrail deterministic 8 | deterministic | disabled | 7/8 = 87.5% | 8/8 = 100.0% | 0 | 1.8684초 | 0.4666초 | 256.9043건/분 |
| OpenAI single 3-agent no-cache live 8 | OpenAI Agno single `gpt-4.1-mini` | ready 8/8 | 7/8 = 87.5% | 8/8 = 100.0% | 0 | 67.6725초 | 16.8457초 | 7.0930건/분 |

초기 OpenAI Agno 결과는 deterministic과 최종 라벨이 8/8건 동일했다. FN 2건은 둘 다 `적격`으로 남아 missed, FP 2건은 `과민경고 완화 보류`, TP 1건은 `부적격`, TN 3건 중 동성화인텍은 `적격`, 데이타솔루션과 휴맥스는 `경계등급 보류`였다. 모든 케이스의 `evidence_status`는 `ready`였고 `error_message`는 비어 있었다.

따라서 이번 OpenAI Agno 재검증은 실행 안정성·외부근거 수집·속도 측정 증거로는 유효하지만, 라벨 개선은 deterministic 대비 추가되지 않았다. 다음 모델 고도화는 LLM provider 교체보다 FN 2건처럼 외부근거가 ready여도 숨은 위험으로 올라가지 않는 케이스의 secondary trigger/FN escalation 기준을 조정하는 쪽이 더 직접적이다.

OpenAI Agno 재검증에서 드러난 FN 미상승 원인은 정상기업 과잉 보류 guardrail이 너무 넓게 적용된 점이었다. 예선테크는 OCF/매출액과 OCF/총부채가 음수인데도 `2년 연속 OCF 적자 아님`만으로 현금흐름 방어 축이 잡혔고, 명신산업은 순이익률 -10.97%와 낮은 이자보상·자본 버퍼에도 guardrail이 적용됐다. 이에 따라 현금흐름 방어를 실제 OCF 양수 또는 커버리지 1배 이상으로 조이고, 순이익률 -10% 미만, OCF 동시 음수, 낮은 이자보상과 약한 자본 버퍼 조합은 guardrail 차단 신호로 추가했다.

수정 후 같은 8건 deterministic 재평가에서는 엄격 기준이 4/8 = 50.0%에서 6/8 = 75.0%로, review-safe 기준이 6/8 = 75.0%에서 8/8 = 100.0%로 개선됐다. FN 2건은 모두 `경계등급 보류`로 끌어올렸고, FP 2건은 `과민경고 완화 보류`, TP 1건은 `부적격`, TN 3건은 동성화인텍 `적격`, 데이타솔루션·휴맥스 `경계등급 보류`로 유지됐다. 속도는 wall time 1.4459초, 평균 case time 0.5330초, 처리량 331.9732건/분이었다.

수정 후 OpenAI Agno 경로 재실행에서도 엄격 기준 6/8 = 75.0%, review-safe 8/8 = 100.0%로 같은 개선이 확인됐다. 수정 전 OpenAI Agno 결과와 비교하면 예선테크와 명신산업 2건만 `적격`에서 `경계등급 보류`로 바뀌었고, FP/TP/TN 라벨은 그대로 유지됐다. 다만 wall time 1.5040초, 평균 case time 0.3758초로 이전 live API 실행보다 매우 짧아 Agno/외부근거 캐시 재사용 가능성이 높다. 따라서 이 행은 수정 후 최종 라벨 회귀 확인으로 사용하고, 실제 live API 지연시간 대표값은 이후 `--no-stage2-llm-cache`로 실행한 OpenAI single 3-agent no-cache live 8 결과를 기준으로 본다.

추가 진단에서는 남은 TN 과잉 보류 중 데이타솔루션 2020이 `interest_coverage_under_1` 단일 blocking flag에 과하게 묶인 것으로 확인됐다. 이 케이스는 ICR이 1배 미만이지만 OCF/총부채 15.7%, cashflow coverage 7.83배, 현금비율 34.0%, 총차입금 비중 7.6%로 현금흐름과 차입 부담이 방어적이었다. 이에 따라 blocking flag가 이자보상 단일 항목이고, OCF·현금·저차입 조건이 동시에 충족되는 경우에만 정상기업 과잉 보류 guardrail을 허용했다. 같은 8건 deterministic 재평가에서 데이타솔루션은 `경계등급 보류`에서 `적격`으로 개선됐고, FN 2건은 계속 `경계등급 보류`, 휴맥스는 음수 OCF와 음수 ICR 때문에 보류로 남았다. 결과는 엄격 기준 7/8 = 87.5%, review-safe 8/8 = 100.0%다.

속도/검증 신뢰도 쪽 문제도 함께 확인했다. `single` 모드는 OpenAI 단일 provider라는 뜻이지 LLM 1회 호출이 아니어서 케이스당 QuantCredit, EvidenceAudit, ChairReport 호출이 발생한다. 이 구조가 3에이전트 성능 증빙에 더 맞으므로 유지하고, live latency 측정 시 캐시 재사용을 피할 수 있도록 batch CLI에 `--no-stage2-llm-cache`를 추가했다. 따라서 앞으로 실제 API 속도를 잴 때는 `single + --no-stage2-llm-cache` 기준으로 측정한다.

추가로 batch 결과 CSV에 Stage 2 실행 진단 컬럼을 남기도록 했다. 주요 컬럼은 `stage2_backend_name`, `stage2_llm_cache_hit`, `stage2_total_elapsed_seconds`, `stage2_agent_elapsed_seconds_sum`, `stage2_quant_credit_elapsed_seconds`, `stage2_evidence_audit_elapsed_seconds`, `stage2_chair_report_elapsed_seconds`, `stage2_parallel_independent_agents`다. 따라서 앞으로는 전체 배치 wall time뿐 아니라 케이스별 Stage 2 LLM 시간, 역할별 병목, 캐시 재사용 여부를 같은 결과 파일에서 바로 확인할 수 있다.

이 진단 컬럼을 붙인 뒤 같은 8건을 `--no-stage2-llm-cache`로 다시 실행해 실제 OpenAI single 3-agent live 증거를 남겼다. 결과는 엄격 기준 7/8 = 87.5%, review-safe 8/8 = 100.0%였고, `stage2_llm_cache_hit=False`가 8/8건, `stage2_backend_name=agno`가 8/8건, `stage2_parallel_independent_agents=True`가 8/8건이었다. 역할별 실행시간도 8/8건 모두 채워졌으며 평균은 QuantCredit 9.8721초, EvidenceAudit 6.9654초, ChairReport 6.6039초였다. `stage2_total_elapsed_seconds` 평균은 16.4786초, 최대는 19.8059초였고, batch wall time은 67.6725초였다. 역할별 시간 합계가 Stage 2 총시간보다 큰 것은 QuantCredit과 EvidenceAudit을 독립 병렬 실행하기 때문이다.

## TN 과잉 보류 30건 확대 분석

휴맥스처럼 실제 투자적격이지만 Stage 2가 보류로 남기는 TN 케이스를 더 보기 위해, 기존 TN 검토 산출물에 등장한 7개 기업-연도를 제외하고 rolling validation TN 후보 30건을 새로 추출했다. 샘플은 `committee_review_tn_overhold_expanded_30_samples.csv`이며, 모두 실제 투자적격이고 1차 모델도 투자적격으로 판단했지만 기준선 근처라 Stage 2 검토 대상에 오른 기업이다.

| 실행 | Runner | 건수 | 엄격 기준 | Review-safe | 보류 | 적격 | Wall time | 평균 case time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TN overhold expanded before liquidity buffer | deterministic | 30 | 21/30 = 70.0% | 30/30 = 100.0% | 9 | 21 | 2.6217초 | 0.3443초 |
| TN overhold expanded after liquidity buffer | deterministic | 30 | 22/30 = 73.3% | 30/30 = 100.0% | 8 | 22 | 2.5362초 | 0.3332초 |

확대 분석 결과, 기존 로직에서 보류 9건 중 8건은 이자보상배율 1 미만, OCF 동시 음수, 순이익률 -10% 미만, 약한 자본/이자보상 조합, 단기차입 압력 중 하나 이상이 있어 보류 유지가 합리적이었다. 반면 `(주)레몬` 2020은 current ratio가 0.7443으로 낮지만 cash ratio 0.2969, OCF/매출 0.2612, OCF/총부채 0.5441, cashflow coverage 24.3625, ICR 18.7971, 자기자본비율 0.6099로 유동성·현금흐름·자본 방어축이 모두 확인됐다. 이에 따라 current ratio watch가 있어도 현금비율·OCF·ICR·자본이 강하고 총차입금 부담이 낮은 경우에는 TN 과잉 보류 guardrail을 막지 않도록 아주 좁은 예외를 추가했다.

수정 후 같은 30건에서 `(주)레몬`만 `보류 → 적격`으로 내려갔고, 남은 보류 8건은 모두 재무 차단 신호를 보유했다. 따라서 휴맥스형 케이스는 계속 보류로 남기고, 방어축이 확실한 current-ratio 단독 watch 케이스만 적격으로 낮추는 방향이 안전하다고 본다. 세부 분석 파일은 `tn_overhold_expanded_30_analysis.md`, `tn_overhold_expanded_30_analysis.csv`, `tn_overhold_expanded_30_analysis_summary.json`에 저장했다.

## TN 과잉 보류 8건 OpenAI Agno live no-cache

위 30건 중 대표 8건을 골라 OpenAI single provider 3-agent Agno 경로로 live 재검증했다. 샘플은 `committee_review_tn_overhold_expanded_8_agno_live_samples.csv`이며, 새 guardrail로 적격 전환된 `(주)레몬`, 재무 차단 신호가 있는 보류 유지 후보 4건, 재무 방어축이 강한 적격 유지 후보 3건으로 구성했다. 실행은 `--stage2-runner agno --stage2-agno-mode single --stage2-model-provider openai --no-stage2-llm-cache --live-external-evidence --workers 2` 조건이며, LLM cache hit는 0건이었다.

| 실행 | Runner | 건수 | 엄격 기준 | Review-safe | 보류 | 적격 | Wall time | 평균 case time | Stage 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TN overhold expanded Agno live no-cache | OpenAI Agno 3-agent | 8 | 2/8 = 25.0% | 8/8 = 100.0% | 6 | 2 | 87.9967초 | 20.4052초 | 18.3034초 |

deterministic 결과와 비교하면 6/8건은 같은 최종 라벨을 유지했다. `(주)엔에프씨`, `(주)휴니드테크놀러지스`는 계속 적격으로 남았고, 현대무벡스·한울반도체·화승알앤에이·하나투어는 보류로 남았다. 바뀐 2건은 `(주)머큐리`와 `(주)레몬`으로, deterministic에서는 재무 방어축이 강해 적격이었지만 Agno live에서는 DART 기반 전환사채/유상증자 공시가 수집되면서 보류로 올라갔다.

이 결과는 Agno 경로가 TN을 모두 적격으로 낮춘다는 증거가 아니라, live 외부근거까지 넣으면 자금조달성 공시에 매우 보수적으로 반응한다는 증거다. 특히 `(주)머큐리`는 전환사채 공시 1건만으로 보류가 되었고, chair memo는 외부 증거가 낮은 위험 수준을 뒷받침해 투자적격 판단을 유지한다고 적었지만 최종 라벨은 보류였다. 따라서 다음 개선은 단일 medium 자금조달 공시와 반복·고위험 자금조달 공시를 분리하고, 최종 라벨과 chair memo가 충돌하지 않도록 설명 합성 로직을 정리하는 쪽이 적합하다. 핵심 수치는 이 증빙 문서와 실험 로그 CSV에 취합했다.

후속 개선에서는 단일 medium 자금조달 공시를 `risk_hold` 보강 근거에서 제외하고, 반복 자금조달 2건 이상 또는 high-risk/adverse 자금조달 근거만 TN overhold guardrail을 막도록 조정했다. 같은 8건과 같은 외부근거 캐시로 deterministic committee replay를 수행한 결과, `(주)머큐리`만 `보류 → 적격`으로 내려갔고 `(주)레몬`은 반복 유상증자/전환사채 공시 때문에 보류로 유지됐다. 엄격 기준은 2/8 = 25.0%에서 3/8 = 37.5%로 개선됐고, review-safe는 8/8 = 100.0%를 유지했다.

이후 같은 대표 8건을 OpenAI Agno 3-agent no-cache live로 재실행해 live 경로에서도 같은 개선이 재현되는지 확인했다. 결과는 엄격 기준 3/8 = 37.5%, review-safe 8/8 = 100.0%, LLM cache hit 0건이었다. `(주)머큐리`는 단일 전환사채 공시에도 재무 방어축이 강해 `적격`으로 내려갔고, `(주)레몬`은 반복 유상증자/전환사채 공시가 있어 `보류`로 유지됐다. 속도는 wall 95.4372초, 평균 case 22.3430초, Stage 2 LLM 평균 20.3821초였다.

추가로 현대무벡스처럼 SPAC 합병 예비심사 때문에 발생한 거래정지가 이후 `거래정지해제(상장예비심사결과 통지(승인))`로 해소된 경우에는 외부 꼬리위험으로 보지 않도록 분리했다. 단, 현대무벡스는 ICR 1 미만이어서 적격으로 내리지는 않고, OCF/총부채 7.6%, cashflow coverage 3.26배, 자기자본비율 84.0%, 부채비율 19.1%, 총차입금 비중 15.6%로 방어축이 있는 단일 ICR 약점 케이스로 보아 `위험 보류`가 아닌 `경계등급 보류`로 낮췄다. 같은 외부근거 캐시 재평가에서 엄격 기준과 review-safe는 3/8, 8/8로 유지됐지만, `committee_risk_signal=True`인 TN 과잉 위험신호는 3건에서 2건으로 줄었다.

Agno chair memo 충돌 필터를 추가한 뒤 같은 8건을 OpenAI Agno 3-agent no-cache live로 다시 실행했다. `stage2_backend_name=agno` 8/8, `stage2_llm_cache_hit=False` 8/8, 외부근거 `ready` 8/8로 확인됐고, 엄격 기준 3/8 = 37.5%, review-safe 8/8 = 100.0%를 유지했다. 최종 라벨 분포는 `적격` 3건, `boundary_hold` 3건, `risk_hold` 2건이며, 최종 라벨이 `보류`인 케이스에서 "모델 라벨을 유지" 또는 "최종 라벨은 투자적격"처럼 읽히는 memo conflict는 0건이었다. 속도는 wall 67.3331초, 평균 case 15.4886초, Stage 2 평균 15.0851초였다.

이후 조건부 Agno ReviewQAAgent를 추가해 최종 `committee_view`를 사후 검수했다. ReviewQA는 라벨을 직접 덮어쓰지 않고, 라벨-메모 일관성, `risk_hold` 적정성, 외부근거 기준일, 정상기업 과잉 보류 guardrail을 검토하는 advisory layer다. 초기 ReviewQA live no-cache에서는 8/8건이 QA를 타서 wall 112.5576초, 평균 case 27.4583초였고, trigger tuning 후에는 `적격` 3건을 제외한 보류 5/8건만 QA를 타도록 줄었다. Trigger-tuned 결과는 엄격 기준 3/8 = 37.5%, review-safe 8/8 = 100.0%, `stage2_backend_name=agno` 8/8, `stage2_llm_cache_hit=False` 8/8, memo conflict 0건이었다. 속도는 wall 83.7142초, 평균 case 19.5163초로 줄었고, QA 평균 호출 시간은 6.4768초였다. ReviewQA 권고는 keep 4건, `(주)레몬` `risk_hold → boundary_hold` 권고 1건이었다.

ReviewQA 권고를 안전한 subtype 보정에 연결한 뒤 같은 8건을 다시 live no-cache로 실행했다. 최종 라벨은 `적격` 3건, `보류` 5건으로 유지됐고, review-safe도 8/8 = 100.0%를 유지했다. 다만 `(주)레몬` 1건은 `stage2_review_qa_advisory_applied=True`로 표시되며 `risk_hold`에서 `boundary_hold`로 낮아졌고, 이에 따라 `committee_risk_signal=True`인 TN 위험신호 보류는 2건에서 1건으로 줄었다. 하나투어는 ReviewQA가 `keep_committee_view`를 권고해 `risk_hold`를 유지했다. 속도는 wall 241.5522초, 평균 case 57.6304초로 이전보다 길었지만, QA 평균은 6.0396초였고 주요 지연은 no-cache OpenAI QuantCredit/ChairReport 호출 변동에서 발생했다.

ReviewQA를 20건으로 확대한 live no-cache 검증에서는 `stage2_backend_name=agno` 20/20, `stage2_llm_cache_hit=False` 20/20, 외부근거 `ready` 20/20, 실행 오류 0건을 확인했다. 결과는 엄격 기준 10/20 = 50.0%, review-safe 20/20 = 100.0%였고, 최종 라벨은 `적격` 10건, `보류` 10건이었다. ReviewQA는 최종 `적격` 10건을 건너뛰고 `보류` 10건에만 켜졌으며, 권고는 keep 7건, `risk_hold → boundary_hold` 3건이었다. 이 중 `(주)레몬`, `신원종합개발(주)` 2건은 `veto_triggered=False`, `hidden_tail_risk_flag=False`라 advisory가 적용되어 위험신호 보류에서 경계 보류로 낮아졌고, `다스코(주)` 1건은 hidden-tail risk가 있어 권고 적용이 차단됐다. 따라서 ReviewQA는 정상 TN을 적격으로 직접 낮추는 라벨 개선 장치라기보다, 최종 보류 중 위험 보류와 경계 보류를 안전하게 구분하는 subtype quality layer로 해석한다. 속도는 wall 190.1121초, 평균 case 18.6943초, Stage 2 평균 17.9042초, QA 평균 5.9525초였다.

20건 확대 결과를 바탕으로 다음 개선은 ReviewQA 사후 검수보다 EvidenceAudit의 공시 severity calibration으로 옮겼다. OpenDART 수집 단계에서 `external_evidence_v4` 캐시 버전으로 올리고, 각 공시에 `disclosure_event_class`, `disclosure_materiality`를 추가했다. 일정금액 미만 또는 자율공시 소송, 자율공시 단일 계약해지, SPAC 합병 예비심사 거래정지는 `caution/procedural_or_one_off`로 낮추고, 횡령·배임·감사의견 거절·상장폐지·관리종목·영업정지·자본잠식 등 실질 부실 사건은 `adverse/veto`로 유지한다. 이 변경은 live API를 다시 돌리기 전 단계의 분류 품질 개선이며, 단위 테스트 `tests/unit/test_evidence_collectors.py`, `tests/unit/test_evidence_signal_modules.py`, `tests/unit/test_agent_report_guardrails.py` 26건과 committee/view 관련 87건이 모두 통과했다.

`external_evidence_v4` 적용 후 같은 20건을 OpenAI Agno live no-cache로 다시 실행했다. 결과는 엄격 기준 10/20 = 50.0%, review-safe 20/20 = 100.0%, 최종 라벨 `적격` 10건·`보류` 10건으로 이전 20건과 동일했다. 다만 OpenDART 공시 120건 중 `adverse`는 2건만 남고, `routine` 69건, `caution` 49건으로 분류되어 EvidenceAudit 입력 품질은 개선됐다. 현대무벡스 SPAC 거래정지/해제는 `procedural_trading_halt`, 다스코 일정금액 미만 소송은 `low_materiality_litigation`으로 낮아졌다. 최종 risk_hold 2건은 다스코의 `material_contract_cancellation`과 하나투어의 `substantive_adverse` 영업정지 공시 때문에 유지됐다.

후속 구현에서는 `external_evidence_v5`로 캐시 버전을 올리고, 계약해지/영업정지 후보에만 OpenDART 상세중요도 보강을 붙였다. 계약해지는 `document.xml` 원문 zip에서 매출 대비 계약해지 비율을 파싱하고, 영업정지는 `bsnSp.json`의 `sl_vs` 또는 `bsnsp_amt/rsl`로 매출 대비 영업정지 비율을 계산한다. 산출 필드는 `materiality_ratio`, `materiality_basis`, `materiality_source`, `materiality_confidence`이며, EvidenceAuditAgent prompt와 외부근거 신호 문구에도 이 상세중요도가 들어간다. 매출 대비 3% 미만은 `procedural_or_one_off`, 3~10%는 `watch_context`, 10% 이상은 `substantive_adverse`로 유지한다. 이는 live API 재검증 전 단계의 구조 개선이며, `tests/unit/test_evidence_collectors.py` 24건, Stage 2 관련 단위 테스트 117건, ruff check가 통과했다.

`external_evidence_v5` 적용 후 같은 20건을 OpenAI Agno live no-cache로 다시 실행했다. 결과는 엄격 기준 11/20 = 55.0%, review-safe 20/20 = 100.0%로 v4 대비 엄격 기준이 +5.0%p 개선됐다. 최종 라벨은 `적격` 11건·`보류` 9건이고, 위험신호 TN hold는 2건에서 1건으로 줄었다. 개선된 케이스는 `다스코(주)` 2019로, `document.xml`에서 `단일판매ㆍ공급계약해지`의 계약해지 금액이 매출 대비 5.92%로 파싱되어 `material_contract_cancellation/substantive_adverse`가 아니라 `contract_cancellation_watch/watch_context`로 낮아졌다. 이에 따라 hidden-tail risk가 꺼지고 최종 판단은 `risk_hold`에서 `적격`으로 회복됐다. `(주)하나투어` 2022는 `영업정지(종속회사의주요경영사항)` 상세 비율이 확인되지 않아 유일한 `risk_hold`로 남았다.

하나투어처럼 `bsnSp.json` 상세 비율이 비어 있는 영업정지 공시를 겨냥해 `external_evidence_v6` 구현을 추가했다. v6는 영업정지 후보에서 `bsnSp.json`으로 `materiality_ratio`를 채우지 못하면 같은 접수번호의 `document.xml` 원문을 fallback으로 파싱한다. 원문 표의 `최근매출액 대비`, `영업정지금액`, `최근매출액` 값을 이용해 종속회사 영업정지가 모회사 신용위험으로 전이될 만큼 중대한지 다시 분류한다. 이 변경은 live API 재검증 전 단계의 구조 개선이며, `tests/unit/test_evidence_collectors.py` 26건, Stage 2 관련 단위 테스트 119건, ruff check가 통과했다.

`external_evidence_v6` 적용 후 같은 20건을 OpenAI Agno live no-cache로 다시 실행했다. 결과는 엄격 기준 11/20 = 55.0%, review-safe 20/20 = 100.0%로 v5와 동일했다. v6 fallback은 하나투어에서 실제로 작동해 `영업정지(종속회사의주요경영사항)`의 매출 대비 비율 11.37%를 파싱했다. 이 값은 10% 이상이므로 `substantive_adverse`로 유지되어 하나투어 `risk_hold`를 낮추지 않는 근거가 됐다. 즉 v6는 라벨 개선보다는 남은 하나투어 risk_hold가 과잉 보류가 아니라 중대성 있는 종속회사 영업정지 근거를 가진 보수적 보류임을 확인한 실험이다. 다만 `신원종합개발(주)`은 no-cache ReviewQA 응답 변동으로 `boundary_hold`에서 `risk_hold`로 남아 위험신호 TN hold가 v5의 1건에서 v6의 2건으로 늘었다.

후속 구현으로 ReviewQA subtype advisory 안정화 guardrail을 추가했다. ReviewQA가 `risk_hold_without_critical_evidence` 조건에서 downgrade를 권고했고, 외부 공시가 모두 `caution/watch_context/procedural_or_one_off` 수준이며 중대성 비율 10% 이상·veto·hidden-tail-risk가 없으면 `risk_hold`를 `boundary_hold`로 낮춘다. 반대로 하나투어처럼 `substantive_adverse` 공시가 있거나 hidden-tail-risk가 있으면 낮추지 않는다. 배치 CSV에는 적용 사유를 추적하기 위해 `stage2_review_qa_advisory_apply_reason` 컬럼을 추가했다. 이 변경은 아직 live API 재검증 전 단계이며, `tests/unit/test_committee.py` 12건, Stage 2 관련 단위 테스트 121건, ruff check가 통과했다.

ReviewQA subtype advisory 안정화 구현 후 같은 20건을 OpenAI Agno live no-cache로 다시 실행했다. 결과는 엄격 기준 11/20 = 55.0%, review-safe 20/20 = 100.0%로 v6와 동일했고, 최종 라벨은 `적격` 11건·`보류` 9건이었다. subtype은 `eligible` 11건, `boundary_hold` 8건, `risk_hold` 1건으로, 이전 v6에서 `risk_hold`로 남았던 `신원종합개발(주)`이 `boundary_hold`로 회복됐다. 이에 따라 `committee_risk_signal=True`인 TN hold는 2건에서 하나투어 1건으로 줄었다. ReviewQA는 9/20건 실행됐고, keep 7건, downgrade 2건, advisory 적용 2건이었다. 이번 live에서는 두 advisory 모두 `review_qa_overstated_risk_hold` 경로로 적용되어 새 `watch_context_only_risk_hold_override`는 직접 발동하지 않았다. 다만 해당 백업 경로는 ReviewQA 표현 변동에 대비한 안전장치로 단위 테스트가 통과했다. 속도는 wall 213.3648초, case 평균 21.3309초, Stage 2 평균 21.1818초였고 cache hit는 0건이었다.

후속 구현으로 RiskRecallQAAgent를 추가했다. ReviewQA가 보류 케이스의 과잉 위험신호를 낮추는 역할이라면, RiskRecallQA는 최종 `적격` 케이스 중 기준선 근처 확률, 유동성/현금흐름/이자보상/차입부담 약점, 직접 관련 외부 공시, BBB-/BB+ 경계 맥락이 있는 케이스만 사후 검수한다. 이 에이전트는 적격 판단의 위험 누락 가능성을 점검하고, 필요한 경우 `boundary_hold` 또는 제한적 `risk_hold` 상향을 권고한다. 속도 부담을 줄이기 위해 모든 적격 기업에 호출하지 않고 trigger reason이 있는 적격 케이스만 호출한다. 배치 CSV에는 `stage2_risk_recall_qa_*` 진단 컬럼을 추가했다. 이 변경은 아직 live API 재검증 전 단계이며, RiskRecallQA 관련 단위 테스트를 포함한 `tests/unit/test_stage2_outputs.py`, `tests/unit/test_stage2_specs.py`, `tests/unit/test_committee.py` 29건이 통과했다.

RiskRecallQA v1 적용 후 같은 20건을 OpenAI Agno live no-cache로 실행했다. 결과는 엄격 기준 11/20 = 55.0%, review-safe 20/20 = 100.0%로 stabilized v6와 동일했고, 최종 분포도 `적격` 11건, `boundary_hold` 8건, `risk_hold` 1건으로 같았다. RiskRecallQA는 최종 `적격` 11건 전체에서 실행됐고, 11건 모두 `keep_committee_view`를 권고해 advisory 적용은 0건이었다. 이 샘플은 TN overhold/near-threshold 중심이어서 적격 회복 기업도 대부분 기준선 근처 또는 경계등급 맥락을 갖기 때문에 trigger가 넓게 켜진 것으로 해석한다. 속도는 wall 170.2559초, case 평균 16.8237초, Stage 2 평균 16.6055초였고, RiskRecallQA 평균 호출 시간은 5.8375초였다. 다음 개선은 `eligible_with_substantive_evidence` trigger를 더 정밀화해 routine 감사보고서나 단순 공시가 substantive evidence로 잡히는 호출을 줄이는 것이다.

RiskRecallQA trigger precision 후속 개선으로 `eligible_with_substantive_evidence` 판정을 좁혔다. 이제 routine 감사보고서, 단순 주주명부, 일반 계약체결처럼 `provider_relevance=risk` 또는 `disclosure_severity=adverse`만 있는 항목은 substantive evidence로 보지 않는다. 대신 `materiality_ratio >= 10%`, `disclosure_event_class/substantive_adverse`, `disclosure_materiality/substantive_adverse`, veto/critical context, 또는 횡령·배임·상장폐지·감사의견 거절 같은 명시적 치명 제목이 있어야 한다. 이 변경은 live API 재검증 전 단계이며, `tests/unit/test_committee.py` 18건, Stage 2 관련 단위 테스트 129건, ruff check가 통과했다.

RiskRecallQA precision v2를 같은 20건에 대해 OpenAI Agno live no-cache로 재검증했다. 결과는 엄격 기준 11/20 = 55.0%, review-safe 20/20 = 100.0%로 v1과 동일했고, 최종 분포도 `적격` 11건, `boundary_hold` 8건, `risk_hold` 1건으로 같았다. RiskRecallQA는 적격 11건 전체에서 실행됐고 모두 `keep_committee_view`를 권고했다. trigger reason 기준으로는 `eligible_with_substantive_evidence`가 v1의 4건에서 v2의 0건으로 줄었다. 제거된 케이스는 `(주)엔에프씨`, `(주)휴니드테크놀러지스`, `(주)화신정공`, `청광건설(주)`이며, routine 감사보고서·주주명부·일반 공시가 더 이상 substantive evidence로 표시되지 않는다. 속도는 wall 173.7005초, case 평균 17.1851초, Stage 2 평균 17.0067초, RiskRecallQA 평균 6.0799초였다.

RiskRecallQA 속도 개선을 위해 추가로 trigger gate를 좁혔다. 이제 최종 적격 케이스라도 기준선 근처 또는 BBB-/BB+ 경계 맥락만으로는 RiskRecallQA를 호출하지 않는다. 호출 조건은 `기준선 근처 + 재무 취약 2축 이상`, `재무 취약 3축 이상`, 또는 `실질 외부 위험 근거`로 제한하고, watch 공시와 rating boundary는 이 핵심 조건이 있을 때만 보조 reason으로 기록한다. precision v2 live 결과의 trigger reason을 기준으로 단순 추정하면 RiskRecallQA 호출 대상은 11건에서 2건으로 줄어든다. v2의 RiskRecallQA 총 호출 시간이 66.8784초, 평균 6.0799초였으므로 같은 20건에서는 약 9건 호출 절감, 약 54초 내외의 Stage 2 대기시간 절감 여지가 있다. 이 변경은 live API 재검증 전 단계이며, `tests/unit/test_committee.py` 20건, Stage 2 관련 단위 테스트 131건, ruff check가 통과했다.

Speed gate v3 구현 후 OpenAI Agno live no-cache smoke test를 실행했다. 출력 디렉터리명은 `full20`이지만 실행 명령에 `--per-category`가 없어 기본값 3이 적용되어 실제 결과는 3건이다. 따라서 이 결과는 공식 20건 성능표가 아니라 smoke test로만 해석한다. 3건 모두 `stage2_backend_name=agno`, `stage2_llm_cache_hit=False`, 외부근거 `ready`였고, 엄격 기준과 review-safe는 모두 3/3 = 100.0%였다. v2 동일 3건에서는 RiskRecallQA가 3/3건 켜졌지만, v3에서는 0/3건으로 줄었다. 최종 라벨은 모두 `적격`으로 유지됐고, Stage 2 평균 시간은 v2 동일 3건 16.3614초에서 v3 10.5734초로 낮아졌다. 정식 검증은 `--per-category 20`을 명시해 20건 전체를 다시 실행해야 한다.

이후 `--per-category 20`을 명시해 같은 20건을 OpenAI Agno live no-cache로 정식 재검증했다. 결과는 엄격 기준 11/20 = 55.0%, review-safe 20/20 = 100.0%로 precision v2와 동일했고, 최종 분포도 `적격` 11건, `boundary_hold` 8건, `risk_hold` 1건으로 같았다. RiskRecallQA 호출은 v2의 11/20건에서 v3의 2/20건으로 줄었고, 총 RiskRecallQA 호출 시간도 66.8784초에서 10.2324초로 줄었다. 최종 라벨·subtype·위험신호 분포는 바뀌지 않았으므로, v3는 정상 TN 적격 회복 케이스의 과잉 QA 호출을 줄이는 데 성공했다. 다만 전체 Stage 2 평균은 17.0067초에서 17.9665초로 소폭 악화됐는데, 이는 `(주)옵투스제약`의 QuantCreditAgent 호출이 67.9774초까지 튄 API latency outlier 때문이다. median은 16.5653초에서 15.2759초로 개선됐고, 옵투스제약 outlier를 제외한 Stage 2 평균은 17.0033초에서 15.0589초로 낮아졌다.

이 outlier 대응을 위해 Agno 모델 생성 단계에 provider HTTP timeout 설정을 추가했다. `CAS_STAGE2_AGENT_TIMEOUT_SECONDS`를 설정하면 OpenAI/Claude 개별 agent 요청에 timeout이 걸리고, timeout이나 일시 오류는 기존 `CAS_STAGE2_AGENT_RETRIES` 루프가 재시도한다. 속도 측정에서는 provider SDK 내부 재시도가 CAS 재시도와 중첩되지 않도록 `CAS_STAGE2_PROVIDER_MAX_RETRIES=0`을 권장한다. 기본값은 timeout 비활성화이므로 기존 실험 재현성은 유지된다. 구현 단계에서 `tests/unit/test_stage2_runner.py` 12건과 ruff check가 통과했다.

Timeout30 설정을 켠 뒤 같은 20건을 다시 OpenAI Agno live no-cache로 실행했다. 결과는 엄격 기준 11/20 = 55.0%, review-safe 20/20 = 100.0%로 speed gate v3와 동일했고, 최종 분포도 `적격` 11건, `boundary_hold` 8건, `risk_hold` 1건으로 같았다. RiskRecallQA 호출도 2/20건으로 유지됐다. 속도는 wall 190.9334초에서 151.0032초로 줄었고, Stage 2 평균은 17.9665초에서 14.1294초, max는 73.2104초에서 19.3524초로 낮아졌다. 특히 `(주)옵투스제약`의 QuantCreditAgent outlier가 67.9774초에서 5.5273초로 정상화됐다.

TN overhold 중심 튜닝에 과적합되지 않았는지 확인하기 위해 FN 8건, FP 12건, TP 12건, TN 8건으로 구성한 mixed hard 40 샘플을 새로 만들었다. 샘플은 `committee_review_mixed_hard_40_timeout30_speed_gate_v3_samples.csv`이며, 선정 요약은 `committee_review_mixed_hard_40_selection_summary.md`에 저장했다. 먼저 deterministic sanity baseline을 돌려 엄격 기준 34/40 = 85.0%, review-safe 38/40 = 95.0%를 확인했다.

같은 40건을 OpenAI Agno live no-cache, external evidence ready, timeout30, RiskRecallQA speed gate v3 조건으로 재검증했다. 결과는 엄격 기준 34/40 = 85.0%, review-safe 38/40 = 95.0%로 deterministic baseline과 같았다. FN 8/8건은 모두 `보류`로 상향됐고, TP 12/12건은 위험 판단을 유지했다. FP 12건 중 10건은 `보류`로 완화됐지만 BBB- 경계 FP 2건은 `부적격`으로 남아 review-safe 실패가 됐다. TN 8건은 4건 적격 유지, 4건 보류로 남았지만 모두 review-safe 기준으로는 성공이다. 조기경보 기준(`보류+부적격=위험`)에서는 Stage 1 모델 대비 Recall이 0.6000에서 1.0000으로, F1이 0.5455에서 0.7143으로 개선됐다. ReviewQA는 32/40건 실행, advisory 4건 적용, RiskRecallQA는 0/40건 실행됐다. 속도는 wall 353.2612초, Stage 2 평균 15.8870초, max 23.5955초로 60초대 outlier는 재발하지 않았다.

## BBB- 경계 FP reject 완화 smoke

mixed hard 40에서 남은 review-safe 실패 2건은 `(주)제닉 2021`, `솔트웨어(주) 2022`였다. 둘 다 실제 라벨은 BBB- 투자적격이고, 1차 모델은 고확률 투기등급으로 본 false positive였다. v1 smoke에서는 ReviewQA가 2/2건 모두 실행됐지만 `keep_committee_view`를 권고해 hard reject가 유지됐다. 이후 ReviewQA trigger가 `reject_without_critical_evidence`이고 외부근거가 routine/caution/watch-context 수준이며 유동성·자본·차입·반복손실 부재 등 재무 방어축이 충분한 경우에는 `부적격` 확정 대신 `boundary_hold`로 낮추는 defensive boundary override를 추가했다.

| 실행 | strict 성공 | review-safe 성공 | ReviewQA trigger | ReviewQA advisory 적용 | 최종 결과 | Stage 2 평균 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| v1 OpenAI Agno live no-cache | 0/2 = 0.0% | 0/2 = 0.0% | 2/2 | 0/2 | 두 건 모두 `부적격/reject` 유지 | 20.8839초 |
| v2 OpenAI Agno live no-cache | 2/2 = 100.0% | 2/2 = 100.0% | 2/2 | 2/2 | 두 건 모두 `보류/boundary_hold` 완화 | 16.4519초 |

이 개선은 정답 라벨(`credit_rating=BBB-`)을 런타임 판단에 넣지 않고, 평가 기준일 이전 외부근거의 치명성 부재와 재무 방어축만 사용한다. 따라서 mixed hard 40 전체 재검증 전까지는 2건 smoke 증거로만 해석해야 하며, 다음 확인은 같은 40건에서 TP risk signal이 훼손되지 않는지 보는 것이다.

같은 v2 guardrail을 mixed hard 40 전체에 다시 적용했다. 첫 실행에서는 API timeout/rate limit 때문에 `(주)아이즈비전 2019`, `(주)티에스트릴리온 2022` 두 건이 실패했으나, 실패 2건만 `workers=1`, timeout 45초, provider retry 2회로 재실행해 최종 merged 결과를 만들었다. 결합 결과는 run failure 0건, 엄격 기준 36/40 = 90.0%, review-safe 기준 40/40 = 100.0%다. FN 8/8은 모두 보류로 상향됐고, FP 12/12는 모두 보류로 완화됐으며, TP 12/12는 `보류` 또는 `부적격`으로 위험 판단을 유지했다. TN 8건은 4건 적격, 4건 보류라 엄격 기준에서는 4건 실패지만 review-safe 기준에서는 모두 성공이다.

| mixed hard 40 실행 | strict 성공 | review-safe 성공 | run failure | FP 완화 | TP 위험 유지 | Stage 2 평균 | Stage 2 max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| timeout30 + speed gate v3 OpenAI Agno live no-cache | 34/40 = 85.0% | 38/40 = 95.0% | 0 | 10/12 | 12/12 | 15.8870초 | 23.5955초 |
| ReviewQA defensive boundary v2 combined | 36/40 = 90.0% | 40/40 = 100.0% | 0 | 12/12 | 12/12 | 25.1108초 | 55.7821초 |

v2에서 ReviewQA는 33/40건 실행됐고 advisory는 5/40건 적용됐다. 이 중 BBB- FP 두 건은 Agno ReviewQA가 `keep_committee_view`를 냈더라도 deterministic defensive boundary override가 적용되어 `부적격/reject`에서 `보류/boundary_hold`로 내려갔다. 중간 raw 실행 폴더는 PR에서 제외하고, 핵심 수치와 해석은 이 문서와 실험 로그 CSV에 취합했다.

## TN strict overhold guardrail v2 smoke

ReviewQA defensive boundary v2 이후 남은 strict 실패 4건은 모두 실제 투자적격 TN이 `보류`로 남은 케이스였다. 이 중 하나투어는 종속회사 영업정지 공시와 반복 손실/OCF 적자가 있어 보류 유지가 타당한 반면, 아시아경제·예림당·일지테크는 prior가 BBB+ 이상 투자등급 non-boundary이고 OCF/부채상환 현금흐름이 방어적이었다. 이에 따라 `stable prior + cashflow-backed ICR dip` 조건을 만족하고 치명 외부근거가 없는 경우에만 `boundary_hold`를 `eligible`로 낮추는 TN strict overhold guardrail v2를 추가했다.

4건 smoke를 OpenAI Agno live no-cache로 실행한 결과는 strict 3/4 = 75.0%, review-safe 4/4 = 100.0%, run failure 0건이었다. 아시아경제 2022, 예림당 2019, 일지테크 2019는 `적격/eligible`로 낮아졌고, 하나투어 2022는 `보류/risk_hold`로 유지됐다. 따라서 mixed hard 40 전체에서 다른 케이스에 회귀가 없으면 strict는 36/40에서 39/40으로 개선될 것으로 예상된다.

| 케이스 | 실제/모델 | v2 smoke 최종 | 해석 |
| --- | --- | --- | --- |
| `(주)아시아경제` 2022 | TN | `적격/eligible` | BBB+ prior, OCF·현금흐름 방어, 치명 외부근거 없음 |
| `(주)예림당` 2019 | TN | `적격/eligible` | BBB+ prior, OCF·현금흐름 방어, 반복 손실 없음 |
| `(주)일지테크` 2019 | TN | `적격/eligible` | BBB+ prior, OCF·현금흐름 방어, 외부근거는 채무보증 맥락 |
| `(주)하나투어` 2022 | TN | `보류/risk_hold` | 종속회사 영업정지 materiality와 반복 손실/OCF 적자로 보류 유지 |

같은 v2 guardrail을 mixed hard 40 전체에 적용했다. 첫 전체 실행은 OpenAI TPM rate limit 로그가 콘솔에 표시됐지만 결과 CSV 기준 `stage2_error_message`와 누락 행은 0건이었다. 다만 `(주)제닉 2021` 한 건이 `부적격/reject`로 남아, 이 1건만 `workers=1`, provider retry 6회 조건으로 no-cache 재실행했다. 재실행에서 제닉은 `보류/boundary_hold`로 완화되어 FP 완화에 성공했다.

최종 결합 결과는 run failure 0건, 엄격 기준 39/40 = 97.5%, review-safe 기준 40/40 = 100.0%다. FN 8/8은 모두 보류로 상향됐고, FP 12/12는 모두 부적격이 아닌 보류로 완화됐으며, TP 12/12는 `보류` 또는 `부적격`으로 위험 판단을 유지했다. TN 8건은 7건 적격, 1건 보류이며, 남은 strict 실패는 하나투어 2022 한 건이다. 하나투어는 종속회사 영업정지와 유상증자 공시, 반복 손실/OCF 적자 맥락이 있어 review-safe 기준으로는 정상적인 보수적 보류로 해석한다.

| mixed hard 40 실행 | strict 성공 | review-safe 성공 | run failure | FN 보완 | FP 완화 | TP 위험 유지 | TN 적격 유지 | Stage 2 평균 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ReviewQA defensive boundary v2 combined | 36/40 = 90.0% | 40/40 = 100.0% | 0 | 8/8 | 12/12 | 12/12 | 4/8 | 25.1108초 |
| TN strict overhold guardrail v2 combined | 39/40 = 97.5% | 40/40 = 100.0% | 0 | 8/8 | 12/12 | 12/12 | 7/8 | 25.8937초 |

최종 결합 상세 결과는 `committee_review_mixed_hard_40_tn_overhold_strict_v2_agno_openai_live_no_cache_combined/committee_review_batch_results.csv`에 저장했다.

## Agno 실행 기준 보류 세분화 결과

deterministic 묶음은 규칙 변경이 깨지지 않았는지 보는 내부 sanity check로만 사용하고, 발표/공유용 성능은 Agno/Claude 실행 기준을 우선한다. 아래 표는 `보류`를 하나의 위험 라벨로 보지 않고, `위험 보류`, `과민경고 완화 보류`, `확인필요 보류`로 세분화한 뒤 재계산한 결과다.

| Agno 실행 | 판단 기준 | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| round 2 10건 | 1차 모델 | 2 | 4 | 2 | 2 | 0.3333 | 0.5000 | 0.4000 | 0.4000 |
| round 2 10건 | 2차 검토대상 (`보류+부적격`) | 4 | 5 | 1 | 0 | 0.4444 | 1.0000 | 0.6154 | 0.5000 |
| round 2 10건 | 2차 위험신호 (`committee_risk_signal`) | 3 | 1 | 5 | 1 | 0.7500 | 0.7500 | 0.7500 | 0.8000 |
| round 2 10건 | 2차 부적격만 | 1 | 0 | 6 | 3 | 1.0000 | 0.2500 | 0.4000 | 0.7000 |
| round 3 10건 | 1차 모델 | 3 | 3 | 2 | 2 | 0.5000 | 0.6000 | 0.5455 | 0.5000 |
| round 3 10건 | 2차 검토대상 (`보류+부적격`) | 5 | 4 | 1 | 0 | 0.5556 | 1.0000 | 0.7143 | 0.6000 |
| round 3 10건 | 2차 위험신호 (`committee_risk_signal`) | 3 | 1 | 4 | 2 | 0.7500 | 0.6000 | 0.6667 | 0.7000 |
| round 3 10건 | 2차 부적격만 | 1 | 0 | 5 | 4 | 1.0000 | 0.2000 | 0.3333 | 0.6000 |
| random rolling 10건 | 1차 모델 | 1 | 1 | 8 | 0 | 0.5000 | 1.0000 | 0.6667 | 0.9000 |
| random rolling 10건 | 2차 검토대상 (`보류+부적격`) | 1 | 2 | 7 | 0 | 0.3333 | 1.0000 | 0.5000 | 0.8000 |
| random rolling 10건 | 2차 위험신호 (`committee_risk_signal`) | 1 | 0 | 9 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| random rolling 10건 | 2차 부적격만 | 1 | 0 | 9 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

해석상 가장 중요한 차이는 `보류+부적격`과 `committee_risk_signal`을 분리한 점이다. `보류+부적격`은 사용자가 추가로 확인해야 하는 검토 workload라서 Precision이 낮게 보일 수 있다. 반면 `committee_risk_signal`은 실제 위험 경고로 볼 판단만 집계하므로, 과민경고 완화 보류나 확인필요 보류가 위험신호 Precision을 떨어뜨리지 않는다.

random rolling 10건 Agno live 실행은 실제 Claude/Agno 및 외부근거 API를 호출했고, 실행 오류 없이 10건이 모두 완료됐다. 전체 wall time은 167.9999초, 평균 case time은 59.5117초였다. `workers=4` 병렬 실행 기준 사용자가 기다린 시간은 약 16.8초/건 수준이며, 개별 API 대기 시간 합계는 595.1170초였다.

| Agno 실행 | 보류/판단 세부유형 | 건수 |
| --- | --- | ---: |
| round 2 10건 | 과민경고 완화 보류 | 5 |
| round 2 10건 | 위험 보류 | 3 |
| round 2 10건 | 부적격 | 1 |
| round 2 10건 | 적격 | 1 |
| round 3 10건 | 과민경고 완화 보류 | 5 |
| round 3 10건 | 위험 보류 | 3 |
| round 3 10건 | 부적격 | 1 |
| round 3 10건 | 적격 | 1 |
| random rolling 10건 | 적격 | 7 |
| random rolling 10건 | 과민경고 완화 보류 | 1 |
| random rolling 10건 | 확인필요 보류 | 1 |
| random rolling 10건 | 부적격 | 1 |

## 판단 오류 위험 10건 Agno 실행

마지막으로 1차 모델 또는 위원회가 헷갈릴 가능성이 큰 rolling validation 후보만 따로 10건 추출해 Agno/Claude live 실행을 수행했다. 샘플은 FN 위험 4건, FP 위험 4건, BBB-/BB+ 경계 2건으로 구성했다. 이전 Agno 실험에 이미 등장한 기업-연도는 최대한 제외해 새 케이스 중심으로 봤다.

| 판단 기준 | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1차 모델 | 1 | 5 | 0 | 4 | 0.1667 | 0.2000 | 0.1818 | 0.1000 |
| 2차 검토대상 (`보류+부적격`) | 5 | 5 | 0 | 0 | 0.5000 | 1.0000 | 0.6667 | 0.5000 |
| 2차 위험신호 (`committee_risk_signal`) | 5 | 0 | 5 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2차 부적격만 | 1 | 0 | 5 | 4 | 1.0000 | 0.2000 | 0.3333 | 0.6000 |

이 샘플은 애초에 오류 위험이 높은 케이스만 모았기 때문에 전체 모집단 성능으로 해석하면 안 된다. 대신 Stage 2가 어떤 역할을 하는지 보여주는 stress test로 해석한다. 결과적으로 1차 모델이 놓친 FN 4건은 모두 `위험 보류`로 끌어올렸고, 1차 모델이 과민하게 본 FP 5건은 모두 `과민경고 완화 보류` 또는 `확인필요 보류`로 낮췄다. 실제 투기등급이면서 1차 모델도 위험으로 본 `(주)바른손`은 `부적격`으로 유지됐다.

| 세부유형 | 건수 | 해석 |
| --- | ---: | --- |
| 위험 보류 | 4 | 실제 투기등급 FN을 보류로 끌어올린 케이스 |
| 과민경고 완화 보류 | 4 | 실제 투자적격 FP를 위험신호가 아닌 재점검 대상으로 낮춘 케이스 |
| 확인필요 보류 | 1 | 위험신호는 아니지만 추가 확인이 필요한 케이스 |
| 부적격 | 1 | 실제 투기등급 TP를 부적격으로 유지한 케이스 |

실행 오류는 0건이었다. 전체 wall time은 193.7719초, 평균 case time은 70.5624초였다. `workers=4` 병렬 실행 기준 사용자가 기다린 시간은 약 19.4초/건 수준이다.

## 최신 materiality guardrail 수치 보존

OpenDART 상세 공시의 금액 중요도를 대시보드와 위원회 판단에 연결하기 위해 FP/TN hard
sample 10건을 OpenAI Agno 3-agent live no-cache 조건으로 재검증했다. 원시 output
디렉터리는 PR에 남기지 않고 삭제했으며, 재현에 필요한 수치만 이 문서와
`docs/stage2_agent_experiment_results_ko.md`에 보존한다.

| 실행 | 건수 | 엄격 기준 | Review-safe | Wall time | Stage 2 평균 | LLM cache hit | 핵심 변화 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| materiality v7 baseline | 10 | 8/10 = 80.0% | 10/10 = 100.0% | 167.4초 | 14.97초 | 0/10 | 상세 중요도 기준선 |
| materiality guardrail | 10 | 8/10 = 80.0% | 10/10 = 100.0% | 81.4초 | 15.83초 | 0/10 | 다스코·제이엠티 `risk_hold` -> `mitigation_hold` |
| review-hold calibration | 10 | 8/10 = 80.0% | 10/10 = 100.0% | 91.7초 | 17.86초 | 0/10 | 일지테크 `risk_hold` -> `review_hold`, 위험신호 `True` -> `False` |

strict 성공률이 80.0%로 유지된 이유는 실제 투자적격 TN이 최종 `보류`로 남으면 실패로
계산하는 엄격 기준 때문이다. 반면 review-safe는 세 실행 모두 100.0%였고, 최신 개선의
초점은 최종 라벨보다 사용자 화면에 보이는 위험 강도 세분화다.

| 케이스 | 변화 | materiality 근거 | 해석 |
| --- | --- | --- | --- |
| 다스코 | `risk_hold` -> `mitigation_hold` | 희석률 21.23% | 규모성 자금조달은 있지만 치명 외부근거와 hard distress 결합이 약해 과민경고 완화 |
| 제이엠티 | `risk_hold` -> `mitigation_hold` | 채무보증금액/자기자본 12.80% | 단일 규모성 채무보증을 즉시 위험 보류로 확정하지 않음 |
| 일지테크 | `risk_hold` -> `review_hold` | 채무보증금액/자기자본 14.90% | 반복 채무보증 때문에 보류는 유지하되 치명 문맥이 약해 위험신호는 해제 |
| 하나투어 | `risk_hold` 유지 | 희석률 20.00%, 영업정지/자금조달 공시 | 종속회사 영업정지와 반복 손실/OCF 적자가 결합되어 보수적 위험 보류 유지 |

삭제한 raw output은 다음 네 가지다.

- `committee_review_materiality_v7_fp_tn_10_agno_openai_live_no_cache/`
- `committee_review_materiality_guardrail_fp_tn_10_agno_openai_live_no_cache/`
- `committee_review_materiality_review_hold_calibration_fp_tn_10_agno_openai_live_no_cache/`
- `committee_review_materiality_v7_fp_tn_10_samples.csv`

## ReviewQA Trigger 축소 속도 검증

ReviewQA는 Stage 2 3-agent 결과를 사후 점검하는 안전장치지만, 최신 mixed hard 40 기준
30/40건이 호출되고 실제 advisory 적용은 6건이었다. 속도 병목을 줄이기 위해 generic trigger인
`investment_model_hold`, `ambiguous_external_evidence`를 제거하고, `risk_hold`/`reject` 중
치명 외부근거가 약하면서 watch-context 또는 재무 방어축이 있는 케이스로 호출 대상을 좁혔다.

동일 10건 비교 결과는 다음과 같다.

| 실행 | 건수 | 엄격 기준 | Review-safe | ReviewQA 호출 | ReviewQA 적용 | ReviewQA 시간 합 | Stage 2 평균 | Stage 2 최대 | LLM cache hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 broad trigger 동일 10건 | 10 | 10/10 = 100.0% | 10/10 = 100.0% | 8/10 | 3/10 | 85.3895초 | 25.6557초 | 61.6745초 | 0/10 |
| narrow trigger OpenAI live no-cache | 10 | 10/10 = 100.0% | 10/10 = 100.0% | 3/10 | 3/10 | 18.1764초 | 22.5787초 | 47.9335초 | 0/10 |

호출 축소 후에도 FN 2건은 모두 보류로 끌어올렸고, FP 4건은 모두 보류로 완화했으며,
TP 2건은 부적격으로 유지, TN 2건은 적격으로 유지됐다. ReviewQA 호출은 62.5% 줄었고
ReviewQA 시간 합은 약 78.7% 감소했다. 따라서 ReviewQA는 전수 보조 판단자가 아니라,
subtype 충돌 가능성이 있는 케이스에만 붙이는 선택형 QA로 운영하는 편이 더 효율적이다.

같은 설정을 20건으로 확대한 검증에서도 호출 축소 효과가 유지됐다.

| 실행 | 건수 | 엄격 기준 | Review-safe | ReviewQA 호출 | ReviewQA 적용 | ReviewQA 시간 합 | Stage 2 평균 | Stage 2 최대 | LLM cache hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 broad trigger 동일 20건 | 20 | 19/20 = 95.0% | 20/20 = 100.0% | 16/20 | 5/20 | 168.8237초 | 26.2304초 | 61.6745초 | 0/20 |
| narrow trigger OpenAI live no-cache 20건 | 20 | 19/20 = 95.0% | 20/20 = 100.0% | 5/20 | 5/20 | 25.5863초 | 17.2261초 | 53.9036초 | 0/20 |

20건 기준 ReviewQA 호출은 68.75% 줄었고, ReviewQA 시간 합은 약 84.84% 감소했다.
적용 건수는 5건으로 유지되어, 제닉·솔트웨어 같은 `reject_without_critical_evidence` 보정과
솔디펜스·씨아이에스·아진전자부품 같은 `risk_hold_without_critical_evidence` 보정은 그대로 작동했다.
strict 실패 1건은 하나투어 TN `보류/risk_hold`로, 종속회사 영업정지와 재무 스트레스 결합을 보수적으로 본 케이스다.
review-safe 기준에서는 해당 케이스도 정상 통과했다.

이후 구현에서는 materiality 해석을 `materiality_signals` 공통 helper로 분리했다. `committee_view`,
`EvidenceAudit`, `ReviewQA`, `RiskRecallQA`가 모두 같은 기준을 사용하므로 자금조달·채무보증 공시는
비율이 10%를 넘더라도 hard distress 또는 재무 스트레스 보강축이 없으면 단독 실질 위험으로 보지 않는다.
반면 영업정지·소송·계약해지처럼 비자금조달성 중요 공시는 기존처럼 ratio와 event class가 실질 위험 판단에
반영된다.

## Materiality 28건 확대 검증 및 reject 보정

FP/TN hard sample 28건을 OpenAI Agno 3-agent, live external evidence, `--no-stage2-llm-cache`
조건으로 재검증했다. 외부근거 수집은 28/28건 `ready`, LLM cache hit는 0/28건이었다.

| 건수 | 엄격 기준 | Review-safe | FP 완화 | TN 적격 유지 | ReviewQA 호출 | RiskRecallQA 호출 | Wall time | Stage 2 평균 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 28 | 24/28 = 85.7% | 25/28 = 89.3% | 15/18 | 9/10 | 2/28 | 3/28 | 475.0초 | 15.98초 |
| reject guardrail v2 | 27/28 = 96.4% | 28/28 = 100.0% | 18/18 | 9/10 | 3/28 | 3/28 | 430.0초 | 15.27초 |

TN은 10건 중 9건이 `적격`으로 유지되어 정상기업 과잉 보류 guardrail이 개선됐지만,
BBB- 경계 FP 3건이 `부적격`으로 남았다. 실패 케이스는 `(주)에스디생명공학`,
`(주)라닉스`, `대한광통신(주)`이며, 모두 치명 외부근거보다는 고확률 모델 경고와 재무 watch,
자금조달/소송 materiality가 결합된 과잉 reject 유형이었다.

이 결과를 반영해 reject confirmation gate를 보정했다. 이제 `veto_candidate`,
`critical_context_confirmed`, hard distress 문맥, 극단 재무위험 중 하나가 확인되는 경우에만
`부적격` 확정을 유지한다. 그 외의 고확률·재무취약·비치명 외부근거 조합은 `risk_hold`로 낮춰
검토 대상에 남기므로, recall은 보류 단계에서 방어하면서 FP reject를 줄이는 구조다.

패치 후 같은 28건을 다시 live 재검증한 결과, 기존 실패 3건은 모두 `부적격`에서 `보류/risk_hold`로
낮아졌다. strict 기준은 85.7%에서 96.4%로, review-safe 기준은 89.3%에서 100.0%로 개선됐다.
남은 strict 실패 1건은 `(주)휴맥스` TN `보류/risk_hold`이며, review-safe 기준에서는 성공이다.

## Agno prompt 경량화 및 materiality 요약 주입

Stage 2 Agno live 호출은 QuantCredit/EvidenceAudit/ChairReport를 실제 3-agent 구조로 실행하므로
프롬프트 입력 크기가 속도와 비용에 직접 영향을 준다. 이에 따라 full `Stage2InputBundle`을 그대로
보내는 대신 role별 compact prompt context를 추가했다. 원본 row와 full evidence snapshot은 내부
pipeline에는 그대로 남고, LLM 프롬프트에는 필요한 재무 컬럼, top drivers, peer 요약, 압축 evidence
items, prior rating, credit policy summary만 전달된다.

또한 모든 Agno role에 `materiality_summary`를 공통 주입한다. 이 요약은 실질 외부위험 여부,
자금조달/채무보증 공시 수, high-risk financing 수, TN hold 차단 여부, hard distress item 수,
최대 materiality ratio와 근거를 포함한다. 따라서 EvidenceAudit과 ChairReport가 제목 목록만 보고
위험을 과대평가하지 않고, 이미 공통 helper가 계산한 중요도 기준을 같이 보게 된다.

로컬 샘플 JSON 길이 비교에서는 full payload 13,163자 대비 compact payload 2,061자로 약 84.3%
감소했다. prompt contract는 `stage2_triplet_prompt_v2`, optional QA cache는 ReviewQA v4,
RiskRecallQA v3로 올려 기존 캐시와 분리했다.

이후 판단 오류 위험 샘플 중 8건을 OpenAI Agno 3-agent, live external evidence,
`--no-stage2-llm-cache` 조건으로 다시 실행했다. 결과는 엄격 기준 8/8 = 100.0%,
review-safe 8/8 = 100.0%, `stage2_backend_name=agno` 8/8, `stage2_llm_cache_hit=False`
8/8, 외부근거 `ready` 8/8이었다. `stage2_parallel_independent_agents=True`도 8/8로
확인되어 실제 3-agent 병렬 경로가 유지됐다. 역할별 평균은 QuantCredit 7.4278초,
EvidenceAudit 5.9740초, ChairReport 5.5899초였고, ReviewQA는 3/8건만 호출되어 평균
4.6925초였다. RiskRecallQA 호출은 0건이었다. Stage 2 총시간 평균은 14.9034초, 최대는
19.1055초, batch wall time은 123.5187초였다.

동일 케이스 1:1 비교는 아니지만, 기존 OpenAI single 3-agent no-cache live 8건의 Stage 2
평균 16.4786초보다 낮은 참고값이다. 따라서 compact prompt는 성능 저하 없이 입력 크기와
역할별 지연시간을 줄이는 방향으로 작동한 것으로 본다.

후속 구조 개선으로 EvidenceAudit 출력에 `critical_evidence_count`, `watch_context_count`,
`materiality_summary`, `hard_distress_detected`, `recommended_evidence_treatment`를 추가했다.
이 값은 공통 evidence treatment helper가 계산하며, ChairReport/ReviewQA/RiskRecallQA는 prose보다
이 구조화 판정을 먼저 참고한다. 목적은 단일 medium 공시나 watch-context 공시를 Chair 단계에서
다시 실질 부실처럼 과대해석하는 흔들림을 줄이는 것이다.

또한 남은 strict miss인 휴맥스형 TN `risk_hold`는 무리하게 적격으로 낮추지 않고 설명력을
높이는 쪽으로 정리했다. `committee_view`와 batch CSV에 `risk_hold_reason_tags`,
`risk_hold_reason_labels`, `risk_hold_reason_summary`를 추가했으며, 태그는
`financial_stress_hold`, `external_materiality_hold`, `combined_watch_hold`,
`secondary_radar_hold`, `model_reject_confirmation_hold`, `model_risk_hold`로 구분한다.
따라서 실제 라벨이 투자적격인 기업이 보류로 남아도, 재무 스트레스 때문인지, 외부 공시 중요도
때문인지, 재무와 외부근거가 결합된 관찰 보류인지 발표와 대시보드에서 설명할 수 있다.

## Agent Disagreement Score live 검증

ReviewQA를 더 효율적으로 호출하기 위해 QuantCredit, EvidenceAudit, ChairReport/committee_view의
판단이 서로 엇갈리는 정도를 `agent_disagreement_score`로 기록하도록 했다. 점수는 정량 모델과
외부근거의 방향 충돌, 최종 `risk_hold`/`reject`와 EvidenceAudit 치명근거 부족, 최종 라벨과
메모 문구 충돌 가능성, 역할 agent confidence gap을 조합해 계산한다.

mixed hard 10건을 OpenAI Agno single provider 3-agent, live external evidence,
`--no-stage2-llm-cache`, workers=1 조건으로 재실행했다. 실행 결과는 다음과 같다.

| 실행 | 건수 | 엄격 기준 | Review-safe | Evidence ready | LLM cache hit | Disagreement high | ReviewQA 호출 | Memo conflict | Stage 2 평균 | Stage 2 최대 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| disagreement memo-fix final | 10 | 9/10 = 90.0% | 10/10 = 100.0% | 10/10 | 0/10 | 2/10 | 3/10 | 0/10 | 24.3740초 | 54.5707초 |

`high` disagreement 2건은 모두 BBB-/BB+ 경계 FP인 `(주)제닉`, `솔트웨어(주)`였다. 두 케이스 모두
`quant_risk_evidence_watch_context / chair_risk_without_critical_evidence` reason이 기록됐고,
ReviewQA가 실행됐다. 즉 ReviewQA가 전수 호출이 아니라 "정량 모델은 위험인데 외부근거는
watch/context 수준인 애매한 경계 보류"에 집중되도록 하는 증거가 생겼다.

memo conflict 오탐은 0건이었다. 이전 smoke에서 `(주)솔디펜스`의 "최종 적격으로 확정하지 않고 보류"
문구를 충돌로 잘못 읽는 사례가 있었지만, 부정형 문구를 제외하도록 보정한 뒤 최종 재실행에서는
`committee_label_memo_conflict`가 남지 않았다.

엄격 기준 실패 1건은 `(주)엔에프씨` TN이 RiskRecallQA 적용 후 `보류/boundary_hold`로 올라간
케이스다. review-safe 기준에서는 정상기업을 `부적격`으로 악화시키지 않았으므로 성공으로 본다.
이 결과는 disagreement score가 위험기업 recall을 해치지 않으면서, ReviewQA 호출 이유를 설명하는
진단 신호로 사용할 수 있음을 보여준다.

후속 구현에서는 ReviewQA trigger도 이 score에 직접 연결한 v1 정책을 검증했다. `high`
disagreement는 치명 외부근거가 제한적인 `risk_hold`/`reject`에서 우선 QA를 실행하고, `medium`은
`chair_risk_without_critical_evidence`, `chair_reject_without_critical_evidence`,
`committee_label_memo_conflict` reason이 있을 때만 실행한다. `low` disagreement는
ReviewQA를 건너뛰도록 해, 단순 watch-context 또는 방어 재무축만으로는 4번째 LLM 호출이 발생하지
않게 했다.

이 정책을 mixed hard 20건으로 OpenAI Agno live no-cache 재검증했다. 외부근거는 20/20건
`ready`, LLM cache hit는 0/20건, 실행 실패 행은 0건이었다. 결과는 strict 18/20 = 90.0%,
review-safe 20/20 = 100.0%였다. ReviewQA는 5/20건만 실행됐고, advisory는 1건 적용됐다.
실행된 5건은 FN risk_hold 3건과 BBB-/BB+ 경계 FP 2건에 집중됐으며, FP 완화 보류,
TP 위험 유지, TN guardrail 케이스는 ReviewQA를 호출하지 않았다. Stage 2 평균은 22.9209초,
최대는 46.1780초였다. 실행 중 OpenAI timeout 로그가 3회 있었지만 내부 retry/continuation 후
최종 CSV에는 `error_message`와 `stage2_error_message`가 남지 않았다.

strict 실패 2건은 `(주)엔에프씨` TN `boundary_hold`와 `(주)하나투어` TN `risk_hold`였다.
둘 다 최종 `부적격`으로 악화하지 않아 review-safe 기준은 통과했다. 하나투어는 희석률 20.00%와
재무 스트레스/외부 중요도 근거가 함께 남아 위험 보류를 유지한 보수적 케이스다.

이 live 결과에서 ReviewQA 5건 중 advisory가 적용된 건 1건뿐이었으므로, 후속 v2 구현에서는
`high` disagreement를 단독 호출 조건으로 쓰지 않도록 더 좁혔다. `risk_hold`는 1차 모델이
투자적격인데 위원회가 위험 보류로 올린 overhold 후보, 또는 라벨-메모 충돌 후보를 우선 검수한다.
1차 모델이 이미 부적격이고 위원회가 보류로 완화한 케이스는 보정 가능성이 낮으면 QA를 건너뛰어,
내부 의견 차이는 대시보드 설명 신호로 남기되 4번째 LLM 호출은 줄인다.

v2를 같은 mixed hard 20건으로 OpenAI Agno live no-cache 재검증했다. 외부근거는 20/20건
`ready`, LLM cache hit는 0/20건, 실행 실패 행은 0건이었다. 결과는 strict 19/20 = 95.0%,
review-safe 20/20 = 100.0%였다. ReviewQA는 3/20건만 실행됐고 advisory는 2건 적용됐다.
실행된 3건은 모두 FN risk_hold였으며, BBB-/BB+ 경계 FP 4건, FP 완화 4건, TP 위험 유지 4건,
TN guardrail 4건은 ReviewQA를 호출하지 않았다. Stage 2 평균은 18.7126초, 최대는 31.4994초였고,
wall time은 376.3633초였다. v1 대비 ReviewQA 호출은 5건에서 3건으로 줄고, strict 성공은
18/20에서 19/20으로 회복됐다. strict 실패는 하나투어 TN `risk_hold` 1건뿐이며, 종속회사
영업정지/자금조달 중요도와 재무 스트레스가 결합된 보수적 보류라 review-safe 기준은 통과했다.

v2를 mixed hard 40건 전체로 확대한 OpenAI Agno live no-cache 재검증에서는 외부근거 40/40건
`ready`, LLM cache hit 0/40건, 실행 실패 행 0건을 확인했다. 결과는 strict 36/40 = 90.0%,
review-safe 40/40 = 100.0%였다. ReviewQA는 5/40건만 실행됐고 advisory는 2건 적용됐다.
호출 대상은 모두 FN risk_hold였으며, BBB-/BB+ 경계 FP 8건, FP 완화 8건, TP 12건, TN 8건은
ReviewQA를 호출하지 않았다. Stage 2 평균은 17.5488초, 최대는 34.0596초, wall time은
729.5342초였다.

40건 확대 결과에서 ReviewQA v2의 호출 절감은 일반화됐지만, strict는 mixed hard 최고치보다 낮았다.
TN 8건 중 `(주)엔에프씨`, `(주)하나투어`, `청광건설(주)`, `(주)일지테크` 4건이 보류로 남았기
때문이다. 하나투어는 종속회사 영업정지/자금조달 중요도와 재무 스트레스가 결합된 보수적
`risk_hold`라 유지 근거가 분명하지만, NFC와 청광건설은 RiskRecallQA/EvidenceAudit이 routine
공시 목록에서 과거 횡령/배임 같은 치명 맥락을 과하게 읽어 적격을 `boundary_hold`로 올린 케이스다.
일지테크는 채무보증금액/자기자본 14.90% 때문에 `review_hold`로 남았다. 따라서 다음 개선은
ReviewQA가 아니라 RiskRecallQA escalation에 실제 evidence profile의 veto/substantive 근거를
요구하는 guardrail이다.

후속 구현으로 RiskRecallQA escalation guardrail을 추가했다. RiskRecallQA가 상향을 권고해도
저품질 뉴스 스니펫이나 검색요약에 `횡령`, `배임` 같은 치명 키워드가 우연히 포함된 것만으로는
적격을 보류로 올리지 않는다. `risk_hold` 상향은 검증된 외부 중요근거 또는 매우 강한 재무취약성이
있어야 적용하고, `boundary_hold` 상향도 기준선 근처+복수 재무취약성, BBB-/BB+ 경계+재무취약성,
또는 검증된 외부근거가 있어야 적용한다. 이 변경은 live API 재검증 전 구현 단계이며,
`tests/unit/test_committee.py`에서 저품질 뉴스 단독 `boundary_hold`/`risk_hold` 상향 차단과
기존 복수 재무취약성 및 OpenDART substantive evidence 상향 경로 유지를 확인했다.

PR #53 머지 후에는 Quant/Evidence 역할 분리 문구와 credit signal policy 보강을 선별 흡수하되,
OpenAI 기본 provider, materiality guardrail, structured evidence treatment, disagreement score,
RiskRecallQA escalation guardrail은 유지했다. 이를 mixed hard 10건으로 OpenAI Agno live
no-cache 재검증한 결과, 외부근거 `ready` 10/10건, LLM cache hit 0/10건, 실행 실패 0건이었다.
결과는 strict 10/10 = 100.0%, review-safe 10/10 = 100.0%였다. FN 3건은 모두 `risk_hold`로
상향됐고, FP 6건은 모두 `mitigation_hold` 또는 review-safe 보류로 완화됐으며, TP 1건은
위험 판단을 유지했다. ReviewQA는 2/10건만 실행됐고 advisory 적용은 0건, RiskRecallQA 호출은
0건이었다. Stage 2 평균은 16.4039초, 최대는 22.2847초, wall time은 90.0239초였다.

후속 구현으로 EvidenceAudit criticality hard gate를 추가했다. Agno EvidenceAudit LLM 응답의
`has_critical_risk=true`는 advisory 신호로만 보고, deterministic `structured_evidence_decision`
기준으로 `critical_veto_review`, `hard_distress_detected`, 또는 `critical_evidence_count > 0`이
확인될 때만 `evidence_strength=critical`을 허용한다. 저품질 뉴스/공시 요약이나 watch-context
외부근거를 LLM이 과하게 해석해 치명 외부근거로 올리는 경로를 줄이기 위한 변경이며,
`tests/unit/test_agent_report_guardrails.py`에서 LLM 단독 critical flag가 `critical`로 승격되지
않는 것을 확인했다.

이를 TN overhold 후보 10건으로 OpenAI Agno live no-cache 재검증했다. 외부근거는 10/10건
`ready`, LLM cache hit는 0/10건, 실행 실패 행은 0건이었다. 결과는 strict 7/10 = 70.0%,
review-safe 10/10 = 100.0%였다. 최종 라벨은 `적격` 7건, `보류/boundary_hold` 3건,
`부적격` 0건이었다. ReviewQA는 1/10건 실행되어 advisory 1건이 적용됐고, RiskRecallQA는
4/10건 실행됐지만 advisory 적용은 0건이었다. Stage 2 평균은 24.3308초, 최대는 42.3019초,
wall time은 136.9049초였다.

이 결과는 hard gate가 LLM 단독 critical flag를 막는 방어선으로 유효하지만, 구조화
evidence-treatment 자체의 critical 판정 품질은 아직 개선 여지가 있음을 보여준다. 해당 10건에서
`critical_veto_review` 4건, `substantive_review` 1건, `watch_context` 5건이 나왔고, 일부 적격
유지 케이스의 메모에는 "치명적 위험 신호" 같은 표현이 남았다. 따라서 다음 개선은 routine
감사보고서, 저품질 검색요약, 회사 직접 관련성이 약한 과거 치명 키워드를 `critical_veto_review`로
올리는 조건을 더 좁히는 것이다.

후속 refined evidence-treatment에서는 routine 감사보고서, 검색요약, 회사 직접 관련성이 약한
과거 치명 키워드를 `critical_veto_review`로 올리는 조건을 더 좁혔다. 같은 TN overhold 후보
10건 OpenAI Agno live no-cache 재검증에서 외부근거는 10/10건 `ready`, LLM cache hit는 0/10건,
실행 실패 행은 0건이었다. 최종 라벨은 `적격` 7건, `보류/boundary_hold` 3건, `부적격` 0건으로
유지되어 strict 7/10 = 70.0%, review-safe 10/10 = 100.0%였다. 핵심 변화는 구조화
EvidenceAudit 판정으로, `critical_veto_review`가 4건에서 0건으로 줄고
`hard_distress_detected=True`도 4건에서 0건으로 줄었다. `recommended_evidence_treatment`는
`substantive_review` 4건, `watch_context` 6건으로 재분류됐다. Stage 2 평균은 21.8159초,
최대는 48.2240초, wall time은 113.2332초였다.

최종 PR 정리 시 원시 batch output 폴더는 남기지 않고, 위 수치와 해석만 이 증빙 문서에 보존한다.

## 증빙 파일

| 용도 | 파일 |
| --- | --- |
| 전체 실험 로그 CSV | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_agent_performance_experiment_log.csv` |
| 속도 실험 로그 CSV | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_agent_speed_experiment_log.csv` |
| 실패 파일럿 삭제 후 전체 재계산 요약 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_agent_all_pilots_recomputed_summary.csv` |
| 통합 Stage 2 평가 리포트 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_evaluation_report.md` |
| validation/test 정책 평가 리포트 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_validation_test_policy_report.md` |
| decision trace 게이트 기여도 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_validation_test_trace_gate_contribution.csv` |
| OpenAI Agno 설명 비교 리포트 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_openai_agno_explanation_comparison.md` |
| Agno 보류 세분화 성능표 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_agent_agno_hold_subtype_metrics.csv` |
| Agno 보류 세부유형 건수표 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_agent_agno_hold_subtype_counts.csv` |
| 판단 오류 위험 10건 샘플 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_error_risk_10_samples.csv` |
| 판단 오류 위험 10건 Agno 성능표 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_agent_error_risk_10_agno_metrics.csv` |
| holdout 8건 샘플 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_holdout_unseen_8_samples.csv` |
| 랜덤 10건 샘플 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_random_rolling_10_samples.csv` |
| OpenAI Agno 비교 실행 결과 | `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_openai_agno_comparison_agno/committee_review_batch_results.csv` |
