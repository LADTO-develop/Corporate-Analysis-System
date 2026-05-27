# 46-Feature Calibration + Operating Threshold Experiments

공식 `feature_46_xgboost` raw score는 유지하고, probability calibration과 dashboard 운영 threshold mode를 분리해 비교한 실험입니다.

## 1. 결론

- Rolling calibration recommendation: `fold_platt` (No rolling calibration variant improved ECE while preserving Brier/logloss; keep fold Platt as the operating baseline.)
- Final Test calibration check: `segment_market_platt` (Improves ECE without worsening Brier/logloss on Final Test.)
- Operating mode default: `balanced`
- Dashboard에 `Recall 우선`, `균형`, `FP 축소 Global`, `FP 축소 KOSDAQ` 모드를 노출하면 모델 재학습 없이 review-load trade-off를 설명할 수 있습니다.

## 2. Rolling Validation 운영 모드

각 rolling fold는 `과거 연도 학습 -> 직전 1년 calibration/threshold 선택 -> 다음 1년 평가` 구조입니다.

| Mode | Folds | Precision Mean | Recall Mean | F1 Mean | Pooled Precision | Pooled Recall | Pooled F1 | FP Sum | FN Sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 4 | 0.6917 | 0.8432 | 0.7589 | 0.6895 | 0.8434 | 0.7587 | 240 | 99 |
| fp_reduction_global | 4 | 0.7410 | 0.7939 | 0.7654 | 0.7378 | 0.7927 | 0.7643 | 178 | 131 |
| fp_reduction_kosdaq | 4 | 0.7354 | 0.7823 | 0.7568 | 0.7336 | 0.7801 | 0.7561 | 179 | 139 |
| recall_first | 4 | 0.6213 | 0.8906 | 0.7296 | 0.6173 | 0.8908 | 0.7293 | 349 | 69 |

## 3. Rolling Validation 연도별 운영 모드

| Eval Year | Policy Year | Mode | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 2018 | balanced | 0.6776 | 0.8857 | 0.7678 | 59 | 16 |
| 2019 | 2018 | fp_reduction_global | 0.7301 | 0.8500 | 0.7855 | 44 | 21 |
| 2019 | 2018 | fp_reduction_kosdaq | 0.7041 | 0.8500 | 0.7702 | 50 | 21 |
| 2019 | 2018 | recall_first | 0.5696 | 0.9357 | 0.7081 | 99 | 9 |
| 2020 | 2019 | balanced | 0.7407 | 0.7947 | 0.7668 | 42 | 31 |
| 2020 | 2019 | fp_reduction_global | 0.8056 | 0.7682 | 0.7864 | 28 | 35 |
| 2020 | 2019 | fp_reduction_kosdaq | 0.7891 | 0.7682 | 0.7785 | 31 | 35 |
| 2020 | 2019 | recall_first | 0.6889 | 0.8212 | 0.7492 | 56 | 27 |
| 2021 | 2020 | balanced | 0.6456 | 0.8061 | 0.7170 | 73 | 32 |
| 2021 | 2020 | fp_reduction_global | 0.7011 | 0.7394 | 0.7198 | 52 | 43 |
| 2021 | 2020 | fp_reduction_kosdaq | 0.7169 | 0.7212 | 0.7190 | 47 | 46 |
| 2021 | 2020 | recall_first | 0.6100 | 0.8909 | 0.7241 | 94 | 18 |
| 2022 | 2021 | balanced | 0.7027 | 0.8864 | 0.7839 | 66 | 20 |
| 2022 | 2021 | fp_reduction_global | 0.7273 | 0.8182 | 0.7701 | 54 | 32 |
| 2022 | 2021 | fp_reduction_kosdaq | 0.7316 | 0.7898 | 0.7596 | 51 | 37 |
| 2022 | 2021 | recall_first | 0.6169 | 0.9148 | 0.7368 | 100 | 15 |

## 4. Rolling Validation Calibration 비교

| Variant | Fit | Folds | PR-AUC Mean | Brier Mean | Logloss Mean | ECE Mean | Bias Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold_segment_industry_platt | rolling_policy_year_industry | 4 | 0.8343 | 0.0870 | 0.2850 | 0.0319 | 0.0035 |
| fold_beta | rolling_policy_year | 4 | 0.8363 | 0.0863 | 0.2819 | 0.0321 | 0.0022 |
| fold_platt | rolling_policy_year | 4 | 0.8363 | 0.0862 | 0.2818 | 0.0321 | 0.0021 |
| fold_segment_market_platt | rolling_policy_year_market | 4 | 0.8304 | 0.0866 | 0.2834 | 0.0339 | 0.0023 |
| fold_isotonic | rolling_policy_year | 4 | 0.8080 | 0.0877 | 0.3158 | 0.0356 | 0.0034 |

## 5. Final Test Calibration 비교

| Variant | Fit | PR-AUC | Brier | Logloss | ECE | MCE | Mean P | Bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| segment_industry_platt | valid_2022_industry | 0.8301 | 0.0761 | 0.2485 | 0.0213 | 0.0921 | 0.2286 | 0.0089 |
| segment_market_platt | valid_2022_market | 0.8348 | 0.0756 | 0.2450 | 0.0227 | 0.1056 | 0.2255 | 0.0059 |
| refit_platt_valid | valid_2022 | 0.8321 | 0.0758 | 0.2470 | 0.0268 | 0.1095 | 0.2259 | 0.0062 |
| current_platt | saved_valid_platt | 0.8321 | 0.0758 | 0.2470 | 0.0268 | 0.1095 | 0.2259 | 0.0062 |
| rolling_oof_isotonic | rolling_oof_2018_2022 | 0.8157 | 0.0758 | 0.2572 | 0.0271 | 0.0902 | 0.2214 | 0.0017 |
| beta_valid | valid_2022 | 0.8321 | 0.0760 | 0.2476 | 0.0288 | 0.1062 | 0.2264 | 0.0067 |
| rolling_oof_platt | rolling_oof_2018_2022 | 0.8321 | 0.0758 | 0.2466 | 0.0296 | 0.1379 | 0.2230 | 0.0033 |
| rolling_oof_beta | rolling_oof_2018_2022 | 0.8321 | 0.0758 | 0.2466 | 0.0302 | 0.1176 | 0.2230 | 0.0033 |
| isotonic_valid | valid_2022 | 0.8065 | 0.0771 | 0.2619 | 0.0311 | 0.2058 | 0.2267 | 0.0070 |

## 6. Final Test Current Platt 운영 모드

| Mode | Threshold | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| balanced | global:0.300; valid recall floor 0.85 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 |
| fp_reduction_global | global:0.380; valid recall floor 0.80 | 0.7222 | 0.8325 | 0.7735 | 65 | 34 |
| fp_reduction_kosdaq | market=KOSDAQ:0.380; fallback:0.300 | 0.7203 | 0.8374 | 0.7745 | 66 | 33 |
| recall_first | global:0.225; valid recall floor 0.90 | 0.6411 | 0.9064 | 0.7510 | 103 | 19 |

## 7. Final Test Current Platt Calibration Bin

| Bin | Rows | Mean P | Actual | Gap |
| --- | --- | --- | --- | --- |
| (-0.001, 0.1] | 542 | 0.0218 | 0.0258 | -0.0040 |
| (0.1, 0.2] | 86 | 0.1406 | 0.0581 | 0.0825 |
| (0.2, 0.3] | 41 | 0.2483 | 0.1707 | 0.0776 |
| (0.3, 0.4] | 26 | 0.3541 | 0.3846 | -0.0305 |
| (0.4, 0.5] | 37 | 0.4465 | 0.5405 | -0.0940 |
| (0.5, 0.6] | 32 | 0.5433 | 0.5938 | -0.0504 |
| (0.6, 0.7] | 38 | 0.6513 | 0.6053 | 0.0460 |
| (0.7, 0.8] | 28 | 0.7524 | 0.6429 | 0.1095 |
| (0.8, 0.9] | 41 | 0.8575 | 0.8537 | 0.0038 |
| (0.9, 1.0] | 53 | 0.9548 | 0.9811 | -0.0263 |

## 8. Final Test Current Platt 주요 세그먼트별 운영 모드

| Mode | Segment | Rows | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | it_services | 176 | 0.6538 | 0.8095 | 0.7234 | 9 | 4 |
| balanced | manufacturing | 598 | 0.7035 | 0.8642 | 0.7756 | 59 | 22 |
| balanced | KOSDAQ | 427 | 0.7015 | 0.8650 | 0.7747 | 60 | 22 |
| balanced | KOSPI | 497 | 0.6667 | 0.9000 | 0.7660 | 18 | 4 |
| balanced | all | 924 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 |
| fp_reduction_kosdaq | it_services | 176 | 0.6818 | 0.7143 | 0.6977 | 7 | 6 |
| fp_reduction_kosdaq | manufacturing | 598 | 0.7273 | 0.8395 | 0.7794 | 51 | 26 |
| fp_reduction_kosdaq | KOSDAQ | 427 | 0.7363 | 0.8221 | 0.7768 | 48 | 29 |
| fp_reduction_kosdaq | KOSPI | 497 | 0.6667 | 0.9000 | 0.7660 | 18 | 4 |
| fp_reduction_kosdaq | all | 924 | 0.7203 | 0.8374 | 0.7745 | 66 | 33 |
| recall_first | it_services | 176 | 0.6207 | 0.8571 | 0.7200 | 11 | 3 |
| recall_first | manufacturing | 598 | 0.6518 | 0.9012 | 0.7565 | 78 | 16 |
| recall_first | KOSDAQ | 427 | 0.6549 | 0.9080 | 0.7609 | 78 | 15 |
| recall_first | KOSPI | 497 | 0.5902 | 0.9000 | 0.7129 | 25 | 4 |
| recall_first | all | 924 | 0.6411 | 0.9064 | 0.7510 | 103 | 19 |

## 9. 해석 주의

- 운영 모드 선택은 rolling validation을 우선 기준으로 보고, Final Test는 마지막 확인용입니다.
- Final Test의 segment calibration 후보 선택은 사후 확인용 결과입니다.
- Operating threshold mode는 확률 자체를 바꾸지 않고 리뷰 민감도를 바꾸는 UI 정책입니다.