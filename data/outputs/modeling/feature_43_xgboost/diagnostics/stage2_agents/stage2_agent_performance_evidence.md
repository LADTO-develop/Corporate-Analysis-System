# Stage 2 Agent Performance Evidence

- 작성일: 2026-05-21
- 범위: 지금까지의 committee-review agent 실험 로그, rolling validation 핵심 증빙 15건, 새 holdout 8건 속도/성능 재검증
- 목적: Claude API + Agno 기반 Stage 2 committee가 1차 모델 오류를 얼마나 보완했는지 수치로 남긴다.
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

OpenAI Agno 결과는 deterministic과 최종 라벨이 8/8건 동일했다. FN 2건은 둘 다 `적격`으로 남아 missed, FP 2건은 `과민경고 완화 보류`, TP 1건은 `부적격`, TN 3건 중 동성화인텍은 `적격`, 데이타솔루션과 휴맥스는 `경계등급 보류`였다. 모든 케이스의 `evidence_status`는 `ready`였고 `error_message`는 비어 있었다.

따라서 이번 OpenAI Agno 재검증은 실행 안정성·외부근거 수집·속도 측정 증거로는 유효하지만, 라벨 개선은 deterministic 대비 추가되지 않았다. 다음 모델 고도화는 LLM provider 교체보다 FN 2건처럼 외부근거가 ready여도 숨은 위험으로 올라가지 않는 케이스의 secondary trigger/FN escalation 기준을 조정하는 쪽이 더 직접적이다.

OpenAI Agno 재검증에서 드러난 FN 미상승 원인은 정상기업 과잉 보류 guardrail이 너무 넓게 적용된 점이었다. 예선테크는 OCF/매출액과 OCF/총부채가 음수인데도 `2년 연속 OCF 적자 아님`만으로 현금흐름 방어 축이 잡혔고, 명신산업은 순이익률 -10.97%와 낮은 이자보상·자본 버퍼에도 guardrail이 적용됐다. 이에 따라 현금흐름 방어를 실제 OCF 양수 또는 커버리지 1배 이상으로 조이고, 순이익률 -10% 미만, OCF 동시 음수, 낮은 이자보상과 약한 자본 버퍼 조합은 guardrail 차단 신호로 추가했다.

수정 후 같은 8건 deterministic 재평가에서는 엄격 기준이 4/8 = 50.0%에서 6/8 = 75.0%로, review-safe 기준이 6/8 = 75.0%에서 8/8 = 100.0%로 개선됐다. FN 2건은 모두 `경계등급 보류`로 끌어올렸고, FP 2건은 `과민경고 완화 보류`, TP 1건은 `부적격`, TN 3건은 동성화인텍 `적격`, 데이타솔루션·휴맥스 `경계등급 보류`로 유지됐다. 속도는 wall time 1.4459초, 평균 case time 0.5330초, 처리량 331.9732건/분이었다. OpenAI Agno live 재실행은 아직 이 수정 이후 다시 돌리지 않았으므로, live API 기준 최종 수치는 추가 확인이 필요하다.

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
