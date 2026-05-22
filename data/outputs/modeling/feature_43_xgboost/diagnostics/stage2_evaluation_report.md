# Stage 2 Evaluation Report

- 생성시각(UTC): `2026-05-22T02:17:50Z`
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
| test  | current_committee_hold_or_reject   | 0.4674    | 0.8818 | 0.6109 | 179 | 204 | 24  | 517 | 383             | 115                | -6                 | 0.0296                 | -0.1929                   | -0.1332            |
| test  | current_committee_reject_only      | 0.8276    | 0.4729 | 0.6019 | 96  | 20  | 107 | 701 | 116             | -69                | 77                 | -0.3793                | 0.1673                    | -0.1422            |
| test  | stage1_minus_overwarning_candidate | 0.6793    | 0.7931 | 0.7318 | 161 | 76  | 42  | 645 | 237             | -13                | 12                 | -0.0591                | 0.0190                    | -0.0123            |
| test  | stage1_model                       | 0.6603    | 0.8522 | 0.7441 | 173 | 89  | 30  | 632 | 262             | 0                  | 0                  | 0.0000                 | 0.0000                    | 0.0000             |
| test  | stage1_or_45                       | 0.6460    | 0.8719 | 0.7421 | 177 | 97  | 26  | 624 | 274             | 8                  | -4                 | 0.0197                 | -0.0143                   | -0.0019            |
| test  | stage1_or_45_high_margin           | 0.6578    | 0.8522 | 0.7425 | 173 | 90  | 30  | 631 | 263             | 1                  | 0                  | 0.0000                 | -0.0025                   | -0.0016            |
| test  | stage1_or_45_no_it_low_threshold   | 0.6471    | 0.8670 | 0.7411 | 176 | 96  | 27  | 625 | 272             | 7                  | -3                 | 0.0148                 | -0.0132                   | -0.0030            |
| test  | stage1_or_45_or_it_low_threshold   | 0.6312    | 0.8768 | 0.7340 | 178 | 104 | 25  | 617 | 282             | 15                 | -5                 | 0.0246                 | -0.0291                   | -0.0101            |
| test  | stage1_or_it_low_threshold         | 0.6434    | 0.8621 | 0.7368 | 175 | 97  | 28  | 624 | 272             | 8                  | -2                 | 0.0099                 | -0.0169                   | -0.0072            |
| valid | current_committee_hold_or_reject   | 0.5576    | 0.8807 | 0.6828 | 155 | 123 | 21  | 377 | 278             | 47                 | -4                 | 0.0227                 | -0.1076                   | -0.0666            |
| valid | current_committee_reject_only      | 0.8491    | 0.5114 | 0.6383 | 90  | 16  | 86  | 484 | 106             | -60                | 61                 | -0.3466                | 0.1839                    | -0.1111            |
| valid | stage1_minus_overwarning_candidate | 0.7000    | 0.8352 | 0.7617 | 147 | 63  | 29  | 437 | 210             | -13                | 4                  | -0.0227                | 0.0348                    | 0.0123             |
| valid | stage1_model                       | 0.6652    | 0.8580 | 0.7494 | 151 | 76  | 25  | 424 | 227             | 0                  | 0                  | 0.0000                 | 0.0000                    | 0.0000             |
| valid | stage1_or_45                       | 0.6540    | 0.8807 | 0.7506 | 155 | 82  | 21  | 418 | 237             | 6                  | -4                 | 0.0227                 | -0.0112                   | 0.0012             |
| valid | stage1_or_45_high_margin           | 0.6667    | 0.8636 | 0.7525 | 152 | 76  | 24  | 424 | 228             | 0                  | -1                 | 0.0057                 | 0.0015                    | 0.0031             |
| valid | stage1_or_45_no_it_low_threshold   | 0.6568    | 0.8807 | 0.7524 | 155 | 81  | 21  | 419 | 236             | 5                  | -4                 | 0.0227                 | -0.0084                   | 0.0030             |
| valid | stage1_or_45_or_it_low_threshold   | 0.6434    | 0.8920 | 0.7476 | 157 | 87  | 19  | 413 | 244             | 11                 | -6                 | 0.0341                 | -0.0218                   | -0.0018            |
| valid | stage1_or_it_low_threshold         | 0.6511    | 0.8693 | 0.7445 | 153 | 82  | 23  | 418 | 235             | 6                  | -2                 | 0.0114                 | -0.0141                   | -0.0049            |

## 해석 가이드

- `2차 검토대상(보류+부적격)`은 조기경보 관점의 넓은 그물이다. Recall이 높을수록 위험 기업을 검토망에 올리는 능력이 좋다.
- `2차 위험신호(risk_signal)`은 실제 빨간 경고에 가까운 신호다. Precision과 Recall을 함께 본다.
- `2차 부적격만`은 가장 엄격한 확정 판단이다. Precision은 높을 수 있지만 Recall이 낮아질 수 있다.
- `과민경고 완화 보류`, `확인필요 보류`, `경계등급 보류`는 위험 확정이 아니라 추가 확인 상태로 해석한다.

## 입력 파일 상태

| file                                           | exists | rows | columns | modified_at_utc      |
| ---------------------------------------------- | ------ | ---- | ------- | -------------------- |
| stage2_agent_agno_hold_subtype_metrics.csv     | True   | 12   | 13      | 2026-05-22T02:12:38Z |
| stage2_agent_error_risk_10_agno_metrics.csv    | True   | 4    | 11      | 2026-05-22T02:12:38Z |
| stage2_agent_performance_experiment_log.csv    | True   | 23   | 9       | 2026-05-22T02:12:38Z |
| stage2_agent_speed_experiment_log.csv          | True   | 10   | 12      | 2026-05-22T02:12:38Z |
| stage2_agent_all_pilots_recomputed_summary.csv | True   | 22   | 8       | 2026-05-22T02:12:38Z |
| committee_review_batch_results.csv             | True   | 12   | 38      | 2026-05-21T00:42:41Z |
| stage2_validation_test_policy_metrics.csv      | True   | 18   | 15      | 2026-05-21T00:42:41Z |
