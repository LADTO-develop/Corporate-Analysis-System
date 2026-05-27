# Stage 2 Rolling Validation Tuning Samples

rolling OOT 예측값을 기준으로 Stage 2 에이전트 튜닝 샘플을 구성했습니다.

## 원칙

- 각 rolling_eval_year는 그 이전 데이터만 사용한 모델로 예측합니다.
- Stage 2 trigger는 feature_46 공식 모델과 `full_review_trigger_73` 보조 트리거를 함께 사용합니다.
- 이 파일은 에이전트 규칙/프롬프트 개선용 validation pool입니다.
- test holdout과 2026 외부검증 라벨은 튜닝에 사용하지 않습니다.

## Fold Summary

| rolling_eval_year | policy_year | train_year_min | train_year_max | train_rows | policy_rows | eval_rows | stage1_threshold | stage2_trigger_candidate | stage2_trigger_feature_count | stage2_aux_threshold | stage2_aux_it_services_threshold | stage1_policy_precision | stage1_policy_recall | stage1_policy_f1 | stage2_aux_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 2018 | 2014 | 2017 | 1490 | 511 | 574 | 0.26 | full_review_trigger_73 | 73 | 0.26 | 0.05 |  |  |  | 0.833564604153781 |
| 2020 | 2019 | 2014 | 2018 | 2001 | 574 | 603 | 0.355 | full_review_trigger_73 | 73 | 0.37 | 0.325 |  |  |  | 0.8354160705551165 |
| 2021 | 2020 | 2014 | 2019 | 2575 | 603 | 673 | 0.305 | full_review_trigger_73 | 73 | 0.285 | 0.115 |  |  |  | 0.8117134176163583 |
| 2022 | 2021 | 2014 | 2020 | 3178 | 673 | 676 | 0.24 | full_review_trigger_73 | 73 | 0.215 | 0.17 |  |  |  | 0.831640025663161 |

## Case Counts

| rolling_eval_year | model_error_type | rows |
| --- | --- | --- |
| 2019 | false_negative | 16 |
| 2019 | false_positive | 59 |
| 2019 | true_negative | 375 |
| 2019 | true_positive | 124 |
| 2020 | false_negative | 31 |
| 2020 | false_positive | 42 |
| 2020 | true_negative | 410 |
| 2020 | true_positive | 120 |
| 2021 | false_negative | 32 |
| 2021 | false_positive | 73 |
| 2021 | true_negative | 435 |
| 2021 | true_positive | 133 |
| 2022 | false_negative | 20 |
| 2022 | false_positive | 66 |
| 2022 | true_negative | 434 |
| 2022 | true_positive | 156 |

## Sample Counts

| committee_policy | sample_category | rows |
| --- | --- | --- |
| feature46_full_review_trigger_73 | bbb_minus_bb_plus_boundary | 15 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | 15 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | 15 |
| feature46_full_review_trigger_73 | true_negative_overescalation_guardrail | 15 |
| feature46_full_review_trigger_73 | true_positive_risk_explanation | 15 |

## Sample Preview

| committee_policy | sample_category | corp_name | fiscal_year | eval_year | actual_label_name | model_predicted_label_name | credit_rating | prob_speculative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | (주)토박스코리아 | 2020 | 2021 | 투기등급 | 투자적격 | B+ | 0.34956336431540824 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | 대한방직(주) | 2021 | 2022 | 투기등급 | 투자적격 | BB | 0.3023972413809951 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | (주)아모텍 | 2021 | 2022 | 투기등급 | 투자적격 | BB+ | 0.29776710840988724 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | (주)인스코비 | 2021 | 2022 | 투기등급 | 투자적격 | B+ | 0.293712258599165 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | (주)신성이엔지 | 2020 | 2021 | 투기등급 | 투자적격 | B+ | 0.2896245801718076 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | 일성건설(주) | 2021 | 2022 | 투기등급 | 투자적격 | BB+ | 0.2877610293583873 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | (주)센코 | 2020 | 2021 | 투기등급 | 투자적격 | BB+ | 0.281290907442824 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | (주)쏠리드 | 2021 | 2022 | 투기등급 | 투자적격 | BB | 0.26459844225116697 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | 씨아이에스(주) | 2020 | 2021 | 투기등급 | 투자적격 | BB | 0.261941370412471 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | (주)정다운 | 2021 | 2022 | 투기등급 | 투자적격 | BB | 0.2595973211900678 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | (주)시너지이노베이션 | 2021 | 2022 | 투기등급 | 투자적격 | BB- | 0.2587064692304214 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | (주)미래아이앤지 | 2019 | 2020 | 투기등급 | 투자적격 | BB | 0.25013030517857854 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | (주)서산 | 2019 | 2020 | 투기등급 | 투자적격 | BB | 0.23231535805776804 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | (주)금호에이치티 | 2019 | 2020 | 투기등급 | 투자적격 | BB+ | 0.22671578656934563 |
| feature46_full_review_trigger_73 | fn_caught_by_stage2_review | (주)지앤비에스에코 | 2022 | 2023 | 투기등급 | 투자적격 | B | 0.22176722143306157 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | (주)힘스 | 2021 | 2022 | 투자적격 | 투기등급 | BBB+ | 0.30533971818456945 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | 에코플라스틱(주) | 2021 | 2022 | 투자적격 | 투기등급 | BBB | 0.3058486939162164 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | (주)혜인 | 2022 | 2023 | 투자적격 | 투기등급 | BBB | 0.24101277244866964 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | (주)신원 | 2021 | 2022 | 투자적격 | 투기등급 | BBB- | 0.3084814181010812 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | (주)휴니드테크놀러지스 | 2019 | 2020 | 투자적격 | 투기등급 | A | 0.2670231359362365 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | 에스지씨에너지(주) | 2020 | 2021 | 투자적격 | 투기등급 | A+ | 0.36292854502901867 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | (주)티웨이항공 | 2019 | 2020 | 투자적격 | 투기등급 | BBB | 0.2706391338563727 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | (주)동방 | 2019 | 2020 | 투자적격 | 투기등급 | BBB- | 0.2714466237814927 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | (주)디지캡 | 2021 | 2022 | 투자적격 | 투기등급 | BBB | 0.31758011592883034 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | (주)대성미생물연구소 | 2022 | 2023 | 투자적격 | 투기등급 | A | 0.25324996277972156 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | 디와이피(주) | 2019 | 2020 | 투자적격 | 투기등급 | BBB | 0.2740194122980457 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | 신원종합개발(주) | 2019 | 2020 | 투자적격 | 투기등급 | A | 0.2752681759618527 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | (주)이상네트웍스 | 2019 | 2020 | 투자적격 | 투기등급 | A | 0.27612484987490754 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | (주)디케이앤디 | 2022 | 2023 | 투자적격 | 투기등급 | BBB | 0.2582390193073228 |
| feature46_full_review_trigger_73 | fp_needing_committee_mitigation | 다스코(주) | 2019 | 2020 | 투자적격 | 투기등급 | A- | 0.27877667846898896 |

Total rolling score rows: 2526
Total tuning sample rows: 75
