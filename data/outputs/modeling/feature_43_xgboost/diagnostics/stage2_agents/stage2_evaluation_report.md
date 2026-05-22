# Stage 2 Evaluation Report

- 생성시각(UTC): `2026-05-22T04:34:00Z`
- 목적: Stage 2 에이전트 위원회의 보완 효과, 위험신호 성능, 실행 안정성, 속도 개선을 한 번에 점검한다.

## 해석 주의

- 이 리포트는 대시보드에 노출하는 사용자용 정확도 지표가 아니다.
- 아래 수치는 과거 validation/test 기업-연도 replay와 Agno 파일럿 샘플을 기준으로 한다.
- 2026년 추론 대상 기업 전체의 실제 정답률이나, 현재 선택 기업의 개별 정확도로 해석하면 안 된다.
- Agno 파일럿 표본은 FN/FP/경계등급 등 어려운 케이스를 의도적으로 포함하므로 전체 모집단 성능으로 해석하지 않는다.
- 발표에서는 “과거 오류 사례에서 Stage 2가 1차 모델 판단을 얼마나 보완했는지 보는 검증 자료”로 설명한다.

## 핵심 요약

- 파일럿 표본 내 위험신호 F1 최고값: `agno_random_rolling_10` F1 1.0000, Precision 1.0000, Recall 1.0000
- 파일럿 표본 내 검토대상 Recall 최고값: `agno_random_rolling_10` Recall 1.0000
- 최신 배치 기준 검토대상 Recall: 0.8333
- validation/test trace 기준 FN 보완 최다 게이트: `부적격 확정 게이트` 8건
- validation/test trace 기준 FP 완화 최다 게이트: `과민경고 완화 점검` 57건

## Stage 2 성능 요약

아래 표는 Agno/Claude 파일럿 및 오류위험 샘플의 성능표를 합쳐 run 단위로 재정리한 결과다.

| run                     | n  | stage1_f1 | review_recall | risk_precision | risk_recall | risk_f1 | risk_f1_delta_vs_stage1 | review_recall_delta_vs_stage1 |
| ----------------------- | -- | --------- | ------------- | -------------- | ----------- | ------- | ----------------------- | ----------------------------- |
| agno_random_rolling_10  | 10 | 0.6667    | 1.0000        | 1.0000         | 1.0000      | 1.0000  | 0.3333                  | 0.0000                        |
| agno_round2_10          | 10 | 0.4000    | 1.0000        | 0.7500         | 0.7500      | 0.7500  | 0.3500                  | 0.5000                        |
| agno_round3_10          | 10 | 0.5455    | 1.0000        | 0.7500         | 0.6000      | 0.6667  | 0.1212                  | 0.4000                        |
| error_risk_10_agno_live | 10 | 0.1818    | 1.0000        | 1.0000         | 1.0000      | 1.0000  | 0.8182                  | 0.8000                        |

## 통합 분류 성능표

| run                     | target               | n  | TP | FP | TN | FN | Precision | Recall | F1     | Accuracy |
| ----------------------- | -------------------- | -- | -- | -- | -- | -- | --------- | ------ | ------ | -------- |
| agno_round2_10          | 1차 모델                | 10 | 2  | 4  | 2  | 2  | 0.3333    | 0.5000 | 0.4000 | 0.4000   |
| agno_round2_10          | 2차 검토대상(보류+부적격)      | 10 | 4  | 5  | 1  | 0  | 0.4444    | 1.0000 | 0.6154 | 0.5000   |
| agno_round2_10          | 2차 위험신호(risk_signal) | 10 | 3  | 1  | 5  | 1  | 0.7500    | 0.7500 | 0.7500 | 0.8000   |
| agno_round2_10          | 2차 부적격만              | 10 | 1  | 0  | 6  | 3  | 1.0000    | 0.2500 | 0.4000 | 0.7000   |
| agno_round3_10          | 1차 모델                | 10 | 3  | 3  | 2  | 2  | 0.5000    | 0.6000 | 0.5455 | 0.5000   |
| agno_round3_10          | 2차 검토대상(보류+부적격)      | 10 | 5  | 4  | 1  | 0  | 0.5556    | 1.0000 | 0.7143 | 0.6000   |
| agno_round3_10          | 2차 위험신호(risk_signal) | 10 | 3  | 1  | 4  | 2  | 0.7500    | 0.6000 | 0.6667 | 0.7000   |
| agno_round3_10          | 2차 부적격만              | 10 | 1  | 0  | 5  | 4  | 1.0000    | 0.2000 | 0.3333 | 0.6000   |
| agno_random_rolling_10  | 1차 모델                | 10 | 1  | 1  | 8  | 0  | 0.5000    | 1.0000 | 0.6667 | 0.9000   |
| agno_random_rolling_10  | 2차 검토대상(보류+부적격)      | 10 | 1  | 2  | 7  | 0  | 0.3333    | 1.0000 | 0.5000 | 0.8000   |
| agno_random_rolling_10  | 2차 위험신호(risk_signal) | 10 | 1  | 0  | 9  | 0  | 1.0000    | 1.0000 | 1.0000 | 1.0000   |
| agno_random_rolling_10  | 2차 부적격만              | 10 | 1  | 0  | 9  | 0  | 1.0000    | 1.0000 | 1.0000 | 1.0000   |
| error_risk_10_agno_live | 1차 모델                | 10 | 1  | 5  | 0  | 4  | 0.1667    | 0.2000 | 0.1818 | 0.1000   |
| error_risk_10_agno_live | 2차 검토대상(보류+부적격)      | 10 | 5  | 5  | 0  | 0  | 0.5000    | 1.0000 | 0.6667 | 0.5000   |
| error_risk_10_agno_live | 2차 위험신호(risk_signal) | 10 | 5  | 0  | 5  | 0  | 1.0000    | 1.0000 | 1.0000 | 1.0000   |
| error_risk_10_agno_live | 2차 부적격만              | 10 | 1  | 0  | 5  | 4  | 1.0000    | 0.2000 | 0.3333 | 0.6000   |

## 최신 배치 결과 재계산

현재 `committee_review_batch_results.csv`가 남아 있으면 같은 기준으로 즉시 재계산한다.

| target                              | n  | TP | FP | TN | FN | Precision | Recall | F1     | Accuracy |
| ----------------------------------- | -- | -- | -- | -- | -- | --------- | ------ | ------ | -------- |
| 1차 모델                               | 12 | 3  | 6  | 0  | 3  | 0.3333    | 0.5000 | 0.4000 | 0.2500   |
| 2차 검토대상(보류+부적격)                     | 12 | 5  | 6  | 0  | 1  | 0.4545    | 0.8333 | 0.5882 | 0.4167   |
| 2차 위험신호(risk_signal 미제공; 보류+부적격 대체) | 12 | 5  | 6  | 0  | 1  | 0.4545    | 0.8333 | 0.5882 | 0.4167   |
| 2차 부적격만                             | 12 | 4  | 2  | 4  | 2  | 0.6667    | 0.6667 | 0.6667 | 0.6667   |

## 파일럿 성공률 로그

| experiment_group                              | rows | strict_success_rate | review_safe_success_rate | run_failures | note                                                                                                                                                       |
| --------------------------------------------- | ---- | ------------------- | ------------------------ | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| historical_12_baseline                        | 12   | 0.7500              |                          | 0            | Initial 12-case baseline                                                                                                                                   |
| live_claude_pilot                             | 4    | 0.5000              |                          | 0            | Initial Claude connectivity and quality pilot                                                                                                              |
| live_claude_fp_mitigation                     | 4    | 0.7500              |                          | 0            | FP mitigation prompt update                                                                                                                                |
| live_claude_label_alignment                   | 4    | 1.0000              |                          | 0            | 4-case label alignment success                                                                                                                             |
| live_claude_12case_label_alignment            | 12   | 0.8330              |                          | 0            | Expanded 12-case Claude label alignment                                                                                                                    |
| other_candidates                              | 20   | 0.9000              |                          | 0            | Expanded candidate set                                                                                                                                     |
| keyword_context_rerun                         | 12   | 0.9170              |                          | 0            | Keyword and evidence-context rerun                                                                                                                         |
| secondary_trigger_rerun                       | 12   | 0.9170              |                          | 0            | Secondary trigger rerun                                                                                                                                    |
| secondary_signal_connected                    | 12   | 1.0000              |                          | 0            | Same 12-case series after secondary signal connection                                                                                                      |
| rolling_validation_pilot                      | 5    | 0.6000              | 0.8000                   | 0            | Initial rolling validation replay                                                                                                                          |
| rolling_agno_claude_retry_batch               | 2    | 0.5000              | 0.5000                   | 0            | Small FP retry batch                                                                                                                                       |
| rolling_agno_claude_round2                    | 10   | 0.9000              | 1.0000                   | 0            | Stabilized 10-case Agno/Claude round 2                                                                                                                     |
| rolling_validation_combined_15                | 15   | 0.8000              | 0.9330                   | 0            | Combined rolling validation evidence set                                                                                                                   |
| holdout_unseen_deterministic_speed_baseline   | 8    | 0.7500              | 0.7500                   | 0            | Local deterministic baseline on unseen holdout 8 cases; FN guardrail remains the improvement target                                                        |
| holdout_unseen_liquidity_guardrail            | 8    | 1.0000              | 1.0000                   | 0            | Local deterministic liquidity-watch guardrail on unseen holdout 8 cases; improved baseline by 25.0 percentage points                                       |
| rolling_agno_claude_round3_live_parallel      | 10   | 0.8000              | 1.0000                   | 0            | New unseen rolling validation Agno/Claude round 3 with workers=3; TN over-hold remained the main weakness                                                  |
| rolling_agno_claude_round3_low_prob_guardrail | 10   | 0.9000              | 1.0000                   | 0            | Same round 3 samples after low-absolute-probability secondary-trigger guardrail; one TN moved from hold to eligible                                        |
| rerun_check_historical_test_12                | 12   | 0.8330              | 0.8330                   | 0            | Final rerun after failed pilot cleanup; deterministic replay of historical test 12                                                                         |
| rerun_check_rolling_pilot_5                   | 5    | 0.6000              | 0.8000                   | 0            | Final rerun after failed pilot cleanup; original rolling pilot baseline retained for comparison                                                            |
| rerun_check_holdout_guardrail                 | 8    | 1.0000              | 1.0000                   | 0            | Final rerun after failed pilot cleanup; unseen holdout guardrail remains 100 percent                                                                       |
| rerun_check_round2_agno                       | 10   | 0.9000              | 1.0000                   | 0            | Final cached Agno rerun after failed pilot cleanup; round 2 remains review-safe 100 percent                                                                |
| rerun_check_round3_after_guardrail            | 10   | 0.9000              | 1.0000                   | 0            | Final cached Agno rerun after failed pilot cleanup; round 3 guardrail remains review-safe 100 percent                                                      |
| rerun_check_random_rolling_10                 | 10   | 1.0000              | 1.0000                   | 0            | Random rolling validation sanity check after FP mitigation; 8 true-negative safe firms stayed eligible and 1 high-probability FP moved from reject to hold |

## 속도 로그

| experiment_group                                     | runner        | rows | batch_wall_time_seconds | case_elapsed_seconds_mean | throughput_cases_per_minute | note                                                                                                                                                                               |
| ---------------------------------------------------- | ------------- | ---- | ----------------------- | ------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| holdout_unseen_deterministic_speed_baseline          | deterministic | 8    | 1.3777                  | 0.3440                    | 348.4068                    | Local deterministic baseline on unseen holdout 8 cases                                                                                                                             |
| holdout_unseen_liquidity_guardrail                   | deterministic | 8    | 1.4247                  | 0.3558                    | 336.9130                    | Local deterministic liquidity-watch guardrail on unseen holdout 8 cases                                                                                                            |
| rolling_agno_claude_round3_live_parallel             | agno          | 10   | 259.3310                | 69.3528                   | 2.3136                      | Live Claude/Agno + external evidence round 3 with workers=3 and replay duplicate Stage 2 call removed effective wall-clock about 25.9 sec per case versus roughly 2 min sequential |
| rolling_agno_claude_round3_low_prob_guardrail_cached | agno          | 10   | 1.4552                  | 0.4295                    | 412.3145                    | Cache replay after low-probability guardrail; not live API latency                                                                                                                 |
| rerun_check_historical_test_12                       | deterministic | 12   | 1.8694                  | 0.4665                    | 385.1503                    | Final deterministic rerun after failed pilot cleanup                                                                                                                               |
| rerun_check_rolling_pilot_5                          | deterministic | 5    | 1.6207                  | 0.9572                    | 185.1052                    | Final deterministic rerun of original rolling pilot baseline                                                                                                                       |
| rerun_check_holdout_guardrail                        | deterministic | 8    | 1.1707                  | 0.4302                    | 410.0111                    | Final deterministic rerun of unseen holdout guardrail                                                                                                                              |
| rerun_check_round2_agno                              | agno          | 10   | 1.2696                  | 0.3738                    | 472.5898                    | Final cached Agno rerun of round 2 after failed pilot cleanup                                                                                                                      |
| rerun_check_round3_after_guardrail                   | agno          | 10   | 1.2387                  | 0.3644                    | 484.3788                    | Final cached Agno rerun of round 3 guardrail after failed pilot cleanup                                                                                                            |
| rerun_check_random_rolling_10                        | deterministic | 10   | 1.6494                  | 0.6451                    | 363.7686                    | Random rolling validation sanity check after FP mitigation with hold subtype fields and workers=4                                                                                  |

## 전체 파일럿 재검증 요약

| artifact_dir                                                                 | rows | strict_success_rate | review_safe_success_rate | run_failures | speed_wall_sec | throughput_cases_per_minute |
| ---------------------------------------------------------------------------- | ---- | ------------------- | ------------------------ | ------------ | -------------- | --------------------------- |
| committee_review_batch_secondary_signal_connected                            | 12   | 1.0000              |                          | 0            |                |                             |
| rerun_check_random_rolling_10                                                | 10   | 1.0000              | 1.0000                   | 0            | 1.6494         | 363.7686                    |
| committee_review_holdout_unseen_guardrail_speed_batch                        | 8    | 1.0000              | 1.0000                   | 0            | 1.4247         | 336.9130                    |
| rerun_check_holdout_guardrail                                                | 8    | 1.0000              | 1.0000                   | 0            | 1.1707         | 410.0111                    |
| committee_review_live_claude_pilot_label_alignment                           | 4    | 1.0000              |                          | 0            |                |                             |
| committee_review_batch_rerun_keyword_context_claude                          | 12   | 0.9167              |                          | 0            |                |                             |
| committee_review_batch_secondary_trigger_rerun                               | 12   | 0.9167              |                          | 0            |                |                             |
| committee_review_batch_other_candidates                                      | 20   | 0.9000              |                          | 0            |                |                             |
| committee_review_rolling_validation_agno_claude_round2_batch                 | 10   | 0.9000              | 1.0000                   | 0            |                |                             |
| committee_review_rolling_validation_agno_claude_round3_after_guardrail_batch | 10   | 0.9000              | 1.0000                   | 0            | 1.4552         | 412.3145                    |
| rerun_check_round2_agno                                                      | 10   | 0.9000              | 1.0000                   | 0            | 1.2696         | 472.5898                    |
| rerun_check_round3_after_guardrail                                           | 10   | 0.9000              | 1.0000                   | 0            | 1.2387         | 484.3788                    |
| committee_review_live_claude_12case_label_alignment                          | 12   | 0.8333              |                          | 0            |                |                             |
| rerun_check_historical_test_12                                               | 12   | 0.8333              | 0.8333                   | 0            | 1.8694         | 385.1503                    |
| committee_review_rolling_validation_agno_claude_round3_batch                 | 10   | 0.8000              | 1.0000                   | 0            | 259.3310       | 2.3136                      |
| .                                                                            | 12   | 0.7500              |                          | 0            |                |                             |
| committee_review_holdout_unseen_deterministic_speed_baseline                 | 8    | 0.7500              | 0.7500                   | 0            | 1.3777         | 348.4068                    |
| committee_review_live_claude_pilot_fp_mitigation                             | 4    | 0.7500              |                          | 0            |                |                             |
| committee_review_rolling_validation_batch                                    | 5    | 0.6000              | 0.8000                   | 0            |                |                             |
| rerun_check_rolling_pilot_5                                                  | 5    | 0.6000              | 0.8000                   | 0            | 1.6207         | 185.1052                    |
| committee_review_live_claude_pilot                                           | 4    | 0.5000              |                          | 0            |                |                             |
| committee_review_rolling_validation_agno_claude_retry_batch                  | 2    | 0.5000              | 0.5000                   | 0            |                |                             |

## Validation/Test 정책 성능

정책 선택은 validation 기준으로 보고, test는 확인용으로만 해석한다.

| split | policy                             | precision | recall | f1     | tp  | fp  | fn  | tn  | predicted_count | delta_fp_vs_stage1 | delta_fn_vs_stage1 | delta_recall_vs_stage1 | delta_precision_vs_stage1 | delta_f1_vs_stage1 |
| ----- | ---------------------------------- | --------- | ------ | ------ | --- | --- | --- | --- | --------------- | ------------------ | ------------------ | ---------------------- | ------------------------- | ------------------ |
| test  | current_committee_hold_or_reject   | 0.4879    | 0.8916 | 0.6307 | 181 | 190 | 22  | 531 | 371             | 116                | -8                 | 0.0394                 | -0.2125                   | -0.1382            |
| test  | current_committee_reject_only      | 0.9800    | 0.2414 | 0.3874 | 49  | 1   | 154 | 720 | 50              | -73                | 124                | -0.6108                | 0.2796                    | -0.3815            |
| test  | stage1_minus_overwarning_candidate | 0.7042    | 0.8325 | 0.7630 | 169 | 71  | 34  | 650 | 240             | -3                 | 4                  | -0.0197                | 0.0038                    | -0.0059            |
| test  | stage1_model                       | 0.7004    | 0.8522 | 0.7689 | 173 | 74  | 30  | 647 | 247             | 0                  | 0                  | 0.0000                 | 0.0000                    | 0.0000             |
| test  | stage1_or_45                       | 0.6797    | 0.8571 | 0.7582 | 174 | 82  | 29  | 639 | 256             | 8                  | -1                 | 0.0049                 | -0.0207                   | -0.0107            |
| test  | stage1_or_45_high_margin           | 0.6948    | 0.8522 | 0.7655 | 173 | 76  | 30  | 645 | 249             | 2                  | 0                  | 0.0000                 | -0.0056                   | -0.0034            |
| test  | stage1_or_45_no_it_low_threshold   | 0.6797    | 0.8571 | 0.7582 | 174 | 82  | 29  | 639 | 256             | 8                  | -1                 | 0.0049                 | -0.0207                   | -0.0107            |
| test  | stage1_or_45_or_it_low_threshold   | 0.6718    | 0.8571 | 0.7532 | 174 | 85  | 29  | 636 | 259             | 11                 | -1                 | 0.0049                 | -0.0286                   | -0.0156            |
| test  | stage1_or_it_low_threshold         | 0.6920    | 0.8522 | 0.7638 | 173 | 77  | 30  | 644 | 250             | 3                  | 0                  | 0.0000                 | -0.0084                   | -0.0051            |
| valid | current_committee_hold_or_reject   | 0.5880    | 0.8920 | 0.7088 | 157 | 110 | 19  | 390 | 267             | 54                 | -6                 | 0.0341                 | -0.1415                   | -0.0797            |
| valid | current_committee_reject_only      | 0.9583    | 0.2614 | 0.4107 | 46  | 2   | 130 | 498 | 48              | -54                | 105                | -0.5966                | 0.2289                    | -0.3778            |
| valid | stage1_minus_overwarning_candidate | 0.7363    | 0.8409 | 0.7851 | 148 | 53  | 28  | 447 | 201             | -3                 | 3                  | -0.0170                | 0.0068                    | -0.0034            |
| valid | stage1_model                       | 0.7295    | 0.8580 | 0.7885 | 151 | 56  | 25  | 444 | 207             | 0                  | 0                  | 0.0000                 | 0.0000                    | 0.0000             |
| valid | stage1_or_45                       | 0.7273    | 0.8636 | 0.7896 | 152 | 57  | 24  | 443 | 209             | 1                  | -1                 | 0.0057                 | -0.0022                   | 0.0011             |
| valid | stage1_or_45_high_margin           | 0.7308    | 0.8636 | 0.7917 | 152 | 56  | 24  | 444 | 208             | 0                  | -1                 | 0.0057                 | 0.0013                    | 0.0032             |
| valid | stage1_or_45_no_it_low_threshold   | 0.7308    | 0.8636 | 0.7917 | 152 | 56  | 24  | 444 | 208             | 0                  | -1                 | 0.0057                 | 0.0013                    | 0.0032             |
| valid | stage1_or_45_or_it_low_threshold   | 0.7264    | 0.8750 | 0.7938 | 154 | 58  | 22  | 442 | 212             | 2                  | -3                 | 0.0170                 | -0.0031                   | 0.0053             |
| valid | stage1_or_it_low_threshold         | 0.7251    | 0.8693 | 0.7907 | 153 | 58  | 23  | 442 | 211             | 2                  | -2                 | 0.0114                 | -0.0044                   | 0.0022             |

## Decision Trace 게이트 기여도

아래 표는 deterministic committee replay의 `decision_trace`를 이용해, 어떤 게이트가 1차 모델의 FN 끌어올림 또는 FP 완화에 함께 작동했는지 집계한 결과다.
한 기업에서 여러 게이트가 동시에 켜질 수 있으므로 게이트별 건수는 서로 배타적이지 않다.

| split | gate_label | triggered_count | fn_escalated_count | fn_escalation_share | fp_softened_count | fp_softening_share | dominant_effect |
| ----- | ---------- | --------------- | ------------------ | ------------------- | ----------------- | ------------------ | --------------- |
| valid | 부적격 확정 게이트 | 518             | 6                  | 0.2400              | 1                 | 0.0179             | fn_and_fp       |
| valid | 2차 보조 레이더  | 4               | 3                  | 0.1200              | 0                 | 0.0000             | fn_escalation   |
| valid | 경계등급 점검    | 12              | 1                  | 0.0400              | 4                 | 0.0714             | fn_and_fp       |
| valid | 과민경고 완화 점검 | 106             | 0                  | 0.0000              | 42                | 0.7500             | fp_softening    |
| valid | 강제 경고 게이트  | 0               | 0                  | 0.0000              | 0                 | 0.0000             | none            |
| valid | 숨은 꼬리위험 점검 | 0               | 0                  | 0.0000              | 0                 | 0.0000             | none            |
| test  | 부적격 확정 게이트 | 727             | 8                  | 0.2667              | 0                 | 0.0000             | fn_escalation   |
| test  | 경계등급 점검    | 17              | 1                  | 0.0333              | 9                 | 0.1216             | fn_and_fp       |
| test  | 2차 보조 레이더  | 11              | 1                  | 0.0333              | 0                 | 0.0000             | fn_escalation   |
| test  | 과민경고 완화 점검 | 131             | 0                  | 0.0000              | 57                | 0.7703             | fp_softening    |
| test  | 강제 경고 게이트  | 0               | 0                  | 0.0000              | 0                 | 0.0000             | none            |
| test  | 숨은 꼬리위험 점검 | 0               | 0                  | 0.0000              | 0                 | 0.0000             | none            |

## OpenAI Agno 설명 품질 비교

같은 샘플을 deterministic과 OpenAI Agno로 각각 실행한 뒤 저장된 결과가 있으면, 최종 라벨 변화와 설명 품질 점수를 비교한다.
현재 Codex 세션에서 실제 OpenAI 호출이 차단된 경우 이 표는 비어 있을 수 있다.

| corp_name | model_error_type | stage1_label | deterministic_label | agno_label | deterministic_quality_score | agno_quality_score | quality_delta |
| --------- | ---------------- | ------------ | ------------------- | ---------- | --------------------------- | ------------------ | ------------- |
| (주)이수앱지스  | false_negative   | 투자적격         | 보류                  | 보류         | 0.7750                      | 0.7750             | 0.0000        |
| (주)타이거일렉  | false_positive   | 투기등급         | 보류                  | 보류         | 0.7729                      | 1.0000             | 0.2271        |
| (주)엠젠솔루션  | true_positive    | 투기등급         | 보류                  | 보류         | 0.6236                      | 0.7981             | 0.1745        |
| (주)플라즈맵   | true_positive    | 투기등급         | 부적격                 | 부적격        | 0.9250                      | 1.0000             | 0.0750        |

## 해석 가이드

- `2차 검토대상(보류+부적격)`은 조기경보 관점의 넓은 그물이다. Recall이 높을수록 위험 기업을 검토망에 올리는 능력이 좋다.
- `2차 위험신호(risk_signal)`은 실제 빨간 경고에 가까운 신호다. Precision과 Recall을 함께 본다.
- `2차 부적격만`은 가장 엄격한 확정 판단이다. Precision은 높을 수 있지만 Recall이 낮아질 수 있다.
- `과민경고 완화 보류`, `확인필요 보류`, `경계등급 보류`는 위험 확정이 아니라 추가 확인 상태로 해석한다.

## 입력 파일 상태

| file                                                  | exists | rows | columns | modified_at_utc      |
| ----------------------------------------------------- | ------ | ---- | ------- | -------------------- |
| stage2_agent_agno_hold_subtype_metrics.csv            | True   | 12   | 13      | 2026-05-22T02:12:38Z |
| stage2_agent_error_risk_10_agno_metrics.csv           | True   | 4    | 11      | 2026-05-22T02:12:38Z |
| stage2_agent_performance_experiment_log.csv           | True   | 23   | 9       | 2026-05-22T02:12:38Z |
| stage2_agent_speed_experiment_log.csv                 | True   | 10   | 12      | 2026-05-22T02:12:38Z |
| stage2_agent_all_pilots_recomputed_summary.csv        | True   | 22   | 8       | 2026-05-22T02:12:38Z |
| committee_review_batch_results.csv                    | True   | 12   | 38      | 2026-05-21T00:42:41Z |
| stage2_validation_test_policy_metrics.csv             | True   | 18   | 15      | 2026-05-22T04:33:45Z |
| stage2_validation_test_trace_gate_contribution.csv    | True   | 12   | 15      | 2026-05-22T04:33:45Z |
| stage2_openai_agno_explanation_comparison_details.csv | True   | 4    | 19      | 2026-05-22T04:33:41Z |
