# 46-Feature Macro Interaction Feature Pack Experiments

공식 `credit_46_features` 입력에 macro regime 변화량과 macro shock × 재무 취약도 interaction 후보를 추가해 walk-forward rolling OOT로 비교했습니다.

Rolling 평가연도는 `2019, 2020, 2021, 2022`이고, Final Test는 공식 test split인 2023~2024 구간입니다.
각 fold는 `과거 연도 학습 -> 직전 1년 Platt calibration/threshold 선택 -> 다음 1년 평가` 구조입니다.

## 1. 결론

- Baseline rolling F1/PR-AUC: `0.7589` / `0.8363`
- Rolling 기준 최상위 후보: `baseline_46_native` (rolling F1 `0.7589`, PR-AUC `0.8363`)
- Rolling F1 변화: `+0.0000`
- Final Test F1 변화: `+0.0000`
- Rolling OOT 기준에서도 현재 46-feature baseline이 가장 안정적입니다. macro interaction 후보는 공식 모델에 바로 반영하지 않는 편이 안전합니다.

## 2. 후보별 Rolling + Final Test 비교

| Variant | Added | Roll PR | Roll P | Roll R | Roll F1 | Roll dF1 | Roll FP | Roll FN | Final PR | Final P | Final R | Final F1 | Final dF1 | Final FP | Final FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_46_native | 0 | 0.8363 | 0.6917 | 0.8432 | 0.7589 | +0.0000 | 240 | 99 | 0.8321 | 0.6941 | 0.8719 | 0.7729 | +0.0000 | 78 | 26 |
| macro_regime_51_raw | 5 | 0.8379 | 0.6871 | 0.8489 | 0.7584 | -0.0005 | 245 | 96 | 0.8267 | 0.7029 | 0.8276 | 0.7602 | -0.0127 | 71 | 35 |
| macro_shock_components_51 | 5 | 0.8377 | 0.6872 | 0.8487 | 0.7582 | -0.0006 | 245 | 96 | 0.8231 | 0.7155 | 0.8177 | 0.7632 | -0.0097 | 66 | 37 |
| macro_vulnerability_interactions_56 | 10 | 0.8363 | 0.6938 | 0.8433 | 0.7582 | -0.0007 | 243 | 99 | 0.8279 | 0.6880 | 0.8473 | 0.7594 | -0.0135 | 78 | 31 |
| macro_full_pressure_70 | 24 | 0.8399 | 0.6811 | 0.8512 | 0.7550 | -0.0038 | 257 | 94 | 0.8252 | 0.7155 | 0.8177 | 0.7632 | -0.0097 | 66 | 37 |
| macro_shock_plus_interactions_61 | 15 | 0.8335 | 0.6749 | 0.8607 | 0.7542 | -0.0047 | 267 | 88 | 0.8292 | 0.7025 | 0.8374 | 0.7640 | -0.0089 | 72 | 33 |
| macro_regime_plus_interactions_61 | 15 | 0.8331 | 0.6728 | 0.8443 | 0.7465 | -0.0124 | 264 | 99 | 0.8313 | 0.7000 | 0.8621 | 0.7726 | -0.0003 | 75 | 28 |

## 3. Baseline 연도별 Rolling 성능

| Eval Year | Threshold | PR-AUC | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 0.2600 | 0.8339 | 0.6776 | 0.8857 | 0.7678 | 59 | 16 |
| 2020 | 0.3550 | 0.8540 | 0.7407 | 0.7947 | 0.7668 | 42 | 31 |
| 2021 | 0.3050 | 0.8217 | 0.6456 | 0.8061 | 0.7170 | 73 | 32 |
| 2022 | 0.2400 | 0.8355 | 0.7027 | 0.8864 | 0.7839 | 66 | 20 |

## 4. 최상위 후보 연도별 Rolling 성능

| Eval Year | Threshold | PR-AUC | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 0.2600 | 0.8339 | 0.6776 | 0.8857 | 0.7678 | 59 | 16 |
| 2020 | 0.3550 | 0.8540 | 0.7407 | 0.7947 | 0.7668 | 42 | 31 |
| 2021 | 0.3050 | 0.8217 | 0.6456 | 0.8061 | 0.7170 | 73 | 32 |
| 2022 | 0.2400 | 0.8355 | 0.7027 | 0.8864 | 0.7839 | 66 | 20 |

## 5. 참고용 Final Test 순위

| Variant | Threshold | PR-AUC | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_46_native | 0.3000 | 0.8321 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 |
| macro_regime_plus_interactions_61 | 0.3000 | 0.8313 | 0.7000 | 0.8621 | 0.7726 | 75 | 28 |
| macro_shock_plus_interactions_61 | 0.3000 | 0.8292 | 0.7025 | 0.8374 | 0.7640 | 72 | 33 |
| macro_full_pressure_70 | 0.3400 | 0.8252 | 0.7155 | 0.8177 | 0.7632 | 66 | 37 |
| macro_shock_components_51 | 0.3400 | 0.8231 | 0.7155 | 0.8177 | 0.7632 | 66 | 37 |
| macro_regime_51_raw | 0.3300 | 0.8267 | 0.7029 | 0.8276 | 0.7602 | 71 | 35 |
| macro_vulnerability_interactions_56 | 0.3050 | 0.8279 | 0.6880 | 0.8473 | 0.7594 | 78 | 31 |

## 6. Final Test 세그먼트

| Variant | Segment | Rows | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_46_native | it_services | 176 | 0.6538 | 0.8095 | 0.7234 | 9 | 4 |
| baseline_46_native | manufacturing | 598 | 0.7035 | 0.8642 | 0.7756 | 59 | 22 |
| baseline_46_native | KOSDAQ | 427 | 0.7015 | 0.8650 | 0.7747 | 60 | 22 |
| baseline_46_native | KOSPI | 497 | 0.6667 | 0.9000 | 0.7660 | 18 | 4 |
| baseline_46_native | all | 924 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 |
| macro_full_pressure_70 | it_services | 176 | 0.6667 | 0.6667 | 0.6667 | 7 | 7 |
| macro_full_pressure_70 | manufacturing | 598 | 0.7181 | 0.8333 | 0.7714 | 53 | 27 |
| macro_full_pressure_70 | KOSDAQ | 427 | 0.7198 | 0.8037 | 0.7594 | 51 | 32 |
| macro_full_pressure_70 | KOSPI | 497 | 0.7000 | 0.8750 | 0.7778 | 15 | 5 |
| macro_full_pressure_70 | all | 924 | 0.7155 | 0.8177 | 0.7632 | 66 | 37 |
| macro_regime_51_raw | it_services | 176 | 0.7273 | 0.7619 | 0.7442 | 6 | 5 |
| macro_regime_51_raw | manufacturing | 598 | 0.7000 | 0.8210 | 0.7557 | 57 | 29 |
| macro_regime_51_raw | KOSDAQ | 427 | 0.7074 | 0.8160 | 0.7578 | 55 | 30 |
| macro_regime_51_raw | KOSPI | 497 | 0.6863 | 0.8750 | 0.7692 | 16 | 5 |
| macro_regime_51_raw | all | 924 | 0.7029 | 0.8276 | 0.7602 | 71 | 35 |
| macro_regime_plus_interactions_61 | it_services | 176 | 0.6667 | 0.7619 | 0.7111 | 8 | 5 |
| macro_regime_plus_interactions_61 | manufacturing | 598 | 0.7056 | 0.8580 | 0.7744 | 58 | 23 |
| macro_regime_plus_interactions_61 | KOSDAQ | 427 | 0.7107 | 0.8589 | 0.7778 | 57 | 23 |
| macro_regime_plus_interactions_61 | KOSPI | 497 | 0.6604 | 0.8750 | 0.7527 | 18 | 5 |
| macro_regime_plus_interactions_61 | all | 924 | 0.7000 | 0.8621 | 0.7726 | 75 | 28 |
| macro_shock_components_51 | it_services | 176 | 0.7143 | 0.7143 | 0.7143 | 6 | 6 |
| macro_shock_components_51 | manufacturing | 598 | 0.7097 | 0.8148 | 0.7586 | 54 | 30 |
| macro_shock_components_51 | KOSDAQ | 427 | 0.7213 | 0.8098 | 0.7630 | 51 | 31 |
| macro_shock_components_51 | KOSPI | 497 | 0.6939 | 0.8500 | 0.7640 | 15 | 6 |
| macro_shock_components_51 | all | 924 | 0.7155 | 0.8177 | 0.7632 | 66 | 37 |
| macro_shock_plus_interactions_61 | it_services | 176 | 0.6400 | 0.7619 | 0.6957 | 9 | 5 |
| macro_shock_plus_interactions_61 | manufacturing | 598 | 0.7098 | 0.8457 | 0.7718 | 56 | 25 |
| macro_shock_plus_interactions_61 | KOSDAQ | 427 | 0.7120 | 0.8344 | 0.7684 | 55 | 27 |
| macro_shock_plus_interactions_61 | KOSPI | 497 | 0.6667 | 0.8500 | 0.7473 | 17 | 6 |
| macro_shock_plus_interactions_61 | all | 924 | 0.7025 | 0.8374 | 0.7640 | 72 | 33 |
| macro_vulnerability_interactions_56 | it_services | 176 | 0.5926 | 0.7619 | 0.6667 | 11 | 5 |
| macro_vulnerability_interactions_56 | manufacturing | 598 | 0.7005 | 0.8519 | 0.7688 | 59 | 24 |
| macro_vulnerability_interactions_56 | KOSDAQ | 427 | 0.6869 | 0.8344 | 0.7535 | 62 | 27 |
| macro_vulnerability_interactions_56 | KOSPI | 497 | 0.6923 | 0.9000 | 0.7826 | 16 | 4 |
| macro_vulnerability_interactions_56 | all | 924 | 0.6880 | 0.8473 | 0.7594 | 78 | 31 |

## 7. 후보 변수

- Raw macro regime: `market_spread_diff`, `spec_spread_diff`, `base_rate_diff`, `treasury_3y_diff`, `usd_krw_diff`
- Macro shock component: `macro_market_spread_widening`, `macro_spec_spread_widening`, `macro_base_rate_hike`, `macro_treasury_3y_hike`, `macro_usd_krw_up`
- Vulnerability proxy: `macro_short_term_borrowings_pressure`, `macro_total_borrowings_pressure`, `macro_interest_coverage_pressure`, `macro_cash_ratio_inverse`
- Macro × vulnerability interaction: `macro_market_spread_widening_x_short_term_borrowings_share`, `macro_market_spread_widening_x_total_borrowings_ratio`, `macro_market_spread_widening_x_interest_coverage_pressure`, `macro_market_spread_widening_x_cash_ratio_inverse`, `macro_spec_spread_widening_x_short_term_borrowings_share`, `macro_spec_spread_widening_x_total_borrowings_ratio`, `macro_base_rate_hike_x_short_term_borrowings_share`, `macro_base_rate_hike_x_interest_coverage_pressure`, `macro_treasury_3y_hike_x_total_borrowings_ratio`, `macro_usd_krw_up_x_cash_ratio_inverse`

## 8. 해석 주의

- 후보 선택은 rolling OOT 평균 기준으로 판단했습니다.
- Final Test는 공식 test split 사후 확인용입니다.
- Macro 변수는 연도별 공통 신호가 강하므로 특정 국면을 외우는지 주의해야 합니다.
- F1이 좋아도 FN이 늘면 조기경보 모델로는 보수적으로 해석합니다.
- 이 실험은 official artifact를 덮어쓰지 않는 feature-pack screening입니다.