# Stage 2 Rolling Validation Tuning Samples

rolling OOT 예측값을 기준으로 Stage 2 에이전트 튜닝 샘플을 구성했습니다.

## 원칙

- 각 rolling_eval_year는 그 이전 데이터만 사용한 모델로 예측합니다.
- 이 파일은 에이전트 규칙/프롬프트 개선용 validation pool입니다.
- test holdout과 2026 외부검증 라벨은 튜닝에 사용하지 않습니다.

## Fold Summary

| rolling_eval_year | policy_year | train_year_min | train_year_max | train_rows | policy_rows | eval_rows | threshold | policy_precision | policy_recall | policy_f1 | best_iteration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 2018 | 2014 | 2017 | 1490 | 511 | 574 | 0.225 | 0.5300546448087432 | 0.8584070796460177 | 0.6554054054054054 | 270 |
| 2020 | 2019 | 2014 | 2018 | 2001 | 574 | 603 | 0.325 | 0.6740331491712708 | 0.8714285714285714 | 0.7601246105919003 | 248 |
| 2021 | 2020 | 2014 | 2019 | 2575 | 603 | 673 | 0.31 | 0.6615384615384615 | 0.8543046357615894 | 0.7456647398843931 | 399 |
| 2022 | 2021 | 2014 | 2020 | 3178 | 673 | 676 | 0.25 | 0.5875 | 0.8545454545454545 | 0.6962962962962963 | 361 |

## Case Counts

| rolling_eval_year | model_error_type | rows |
| --- | --- | --- |
| 2019 | false_negative | 12 |
| 2019 | false_positive | 101 |
| 2019 | true_negative | 333 |
| 2019 | true_positive | 128 |
| 2020 | false_negative | 37 |
| 2020 | false_positive | 49 |
| 2020 | true_negative | 403 |
| 2020 | true_positive | 114 |
| 2021 | false_negative | 38 |
| 2021 | false_positive | 93 |
| 2021 | true_negative | 415 |
| 2021 | true_positive | 127 |
| 2022 | false_negative | 24 |
| 2022 | false_positive | 89 |
| 2022 | true_negative | 411 |
| 2022 | true_positive | 152 |

## Sample Counts

| committee_policy | sample_category | rows |
| --- | --- | --- |
| rolling_recall_first_mid_mfg_prob_0_10 | bbb_minus_bb_plus_boundary | 15 |
| rolling_recall_first_mid_mfg_prob_0_10 | fn_caught_by_stage2_review | 15 |
| rolling_recall_first_mid_mfg_prob_0_10 | fp_needing_committee_mitigation | 15 |
| rolling_recall_first_mid_mfg_prob_0_10 | true_negative_overescalation_guardrail | 15 |
| rolling_recall_first_mid_mfg_prob_0_10 | true_positive_risk_explanation | 14 |
| rolling_stage1_or_near_threshold_0_10 | bbb_minus_bb_plus_boundary | 15 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | 15 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | 15 |
| rolling_stage1_or_near_threshold_0_10 | true_negative_overescalation_guardrail | 15 |
| rolling_stage1_or_near_threshold_0_10 | true_positive_risk_explanation | 14 |

## Sample Preview

| committee_policy | sample_category | corp_name | fiscal_year | eval_year | actual_label_name | model_predicted_label_name | credit_rating | prob_speculative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | 참좋은여행(주) | 2020 | 2021 | 투기등급 | 투자적격 | BB+ | 0.3221653699874878 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)예선테크 | 2020 | 2021 | 투기등급 | 투자적격 | BB- | 0.314068466424942 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)누보 | 2020 | 2021 | 투기등급 | 투자적격 | BB- | 0.3112805485725403 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)휴럼 | 2020 | 2021 | 투기등급 | 투자적격 | BB+ | 0.3112805485725403 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | 명신산업(주) | 2020 | 2021 | 투기등급 | 투자적격 | BB | 0.31103089451789856 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)픽셀플러스 | 2020 | 2021 | 투기등급 | 투자적격 | BB | 0.30166950821876526 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)솔디펜스 | 2020 | 2021 | 투기등급 | 투자적격 | BB- | 0.30166950821876526 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | 씨아이에스(주) | 2020 | 2021 | 투기등급 | 투자적격 | BB | 0.30166950821876526 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | 아진전자부품(주) | 2020 | 2021 | 투기등급 | 투자적격 | BB | 0.3001667261123657 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)아즈텍더블유비이 | 2020 | 2021 | 투기등급 | 투자적격 | BB+ | 0.2987302541732788 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | 와이엠티(주) | 2020 | 2021 | 투기등급 | 투자적격 | BB | 0.2972560226917267 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)덱스터스튜디오 | 2021 | 2022 | 투기등급 | 투자적격 | BB+ | 0.2888428270816803 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)에스엠컬처앤콘텐츠 | 2021 | 2022 | 투기등급 | 투자적격 | BB+ | 0.2877834439277649 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | 핸즈코퍼레이션(주) | 2021 | 2022 | 투기등급 | 투자적격 | BB+ | 0.2842256426811218 |
| rolling_stage1_or_near_threshold_0_10 | fn_caught_by_stage2_review | 와이엠티(주) | 2021 | 2022 | 투기등급 | 투자적격 | BB | 0.2834196090698242 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)엘오티베큠 | 2019 | 2020 | 투자적격 | 투기등급 | A- | 0.22718480229377747 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)예림당 | 2020 | 2021 | 투자적격 | 투기등급 | BBB+ | 0.32790717482566833 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)퍼시스 | 2019 | 2020 | 투자적격 | 투기등급 | A+ | 0.22844743728637695 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | 동일제강(주) | 2019 | 2020 | 투자적격 | 투기등급 | BBB+ | 0.22905565798282623 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)에이프로 | 2020 | 2021 | 투자적격 | 투기등급 | BBB- | 0.3307737112045288 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)지란지교시큐리티 | 2020 | 2021 | 투자적격 | 투기등급 | BBB- | 0.331118643283844 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)대성미생물연구소 | 2021 | 2022 | 투자적격 | 투기등급 | A | 0.317757248878479 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)파세코 | 2021 | 2022 | 투자적격 | 투기등급 | A- | 0.317757248878479 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | 현대에이치티(주) | 2021 | 2022 | 투자적격 | 투기등급 | A | 0.317757248878479 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | 다스코(주) | 2022 | 2023 | 투자적격 | 투기등급 | BBB | 0.258312851190567 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)아이즈비전 | 2019 | 2020 | 투자적격 | 투기등급 | BBB | 0.23360560834407806 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | 제이엠티(주) | 2019 | 2020 | 투자적격 | 투기등급 | BBB- | 0.23422381281852722 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | 와이엠씨(주) | 2019 | 2020 | 투자적격 | 투기등급 | BBB+ | 0.23508048057556152 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)디지캡 | 2021 | 2022 | 투자적격 | 투기등급 | BBB | 0.321814626455307 |
| rolling_stage1_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)한국큐빅 | 2022 | 2023 | 투자적격 | 투기등급 | BBB+ | 0.2620837688446045 |

Total rolling score rows: 2526
Total tuning sample rows: 148
