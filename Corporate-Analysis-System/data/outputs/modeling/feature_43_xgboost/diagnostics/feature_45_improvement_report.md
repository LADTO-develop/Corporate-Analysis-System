# Feature 45 Improvement Experiments

43개 baseline에 `delta_accruals_ratio`, `is_3y_consecutive_operating_loss`를 추가한 45개 변수셋의 개선 여지를 하이퍼파라미터, threshold 정책, segment threshold 관점에서 확인했습니다.
하이퍼파라미터 탐색은 baseline 1개와 deterministic sample `48`개입니다.

## 1. 결론

- 43개 baseline test F1/Recall/Precision: `0.7347` / `0.8623` / `0.6400`
- 45개 변수셋 기본 test F1/Recall/Precision: `0.7318` / `0.8743` / `0.6293`
- 45개 validation 기준 선택 후보: `feature_45_tuned_006` (test F1 `0.6972`, test Recall `0.8204`)
- 45개 참고용 test F1 최상위 후보: `feature_45_default` (test F1 `0.7318`)
- 45개 기본 모델 threshold 정책 최상위: `valid_recall85_max_precision` (test F1 `0.7318`)
- 43개 또는 45개 중 하나라도 위험으로 보는 union trigger는 FN을 `19`개까지 줄이지만 FP는 `89`개로 늘어납니다.
- 가장 현실적인 Stage 2 위원회 검토 트리거는 `45개 변수셋 + IT서비스 threshold 완화`입니다. test Recall `0.8862`, F1 `0.7255`, 추가 검토 `9`개입니다.
- Recall을 더 높이는 정책은 가능하지만, 추가로 잡는 위험 기업보다 추가 검토되는 정상 기업 증가가 더 빠릅니다.
- 현재 탐색 범위에서는 45개를 운영 모델로 바로 교체하기보다, Recall 보완 후보 또는 Stage 2 검토 트리거로 쓰는 전략이 더 안전합니다.

## 2. 핵심 모델 비교

| Variant | Features | Threshold | Test PR | Test ROC | Test P | Test R | Test F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_43_native | 43 | 0.3150 | 0.7744 | 0.9110 | 0.6400 | 0.8623 | 0.7347 | 81 | 23 |
| feature_45_default | 45 | 0.3150 | 0.7750 | 0.9102 | 0.6293 | 0.8743 | 0.7318 | 86 | 21 |
| feature_45_tuned_006 | 45 | 0.3100 | 0.7733 | 0.9062 | 0.6062 | 0.8204 | 0.6972 | 89 | 30 |

## 3. 45개 하이퍼파라미터 Validation 상위

| Variant | Depth | Child | Lambda | SPW x | Threshold | Valid F1 | Test P | Test R | Test F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature_45_tuned_006 | 3 | 3.0000 | 0.5000 | 1.0000 | 0.3100 | 0.7569 | 0.6062 | 0.8204 | 0.6972 | 89 | 30 |
| feature_45_tuned_042 | 5 | 5.0000 | 1.0000 | 1.2000 | 0.3350 | 0.7550 | 0.6398 | 0.8084 | 0.7143 | 76 | 32 |
| feature_45_tuned_008 | 3 | 3.0000 | 6.0000 | 1.0000 | 0.3100 | 0.7538 | 0.6116 | 0.8204 | 0.7008 | 87 | 30 |
| feature_45_tuned_043 | 5 | 5.0000 | 3.0000 | 1.2000 | 0.3300 | 0.7519 | 0.6256 | 0.7904 | 0.6984 | 79 | 35 |
| feature_45_tuned_007 | 3 | 3.0000 | 1.0000 | 1.2000 | 0.3250 | 0.7519 | 0.6193 | 0.8084 | 0.7013 | 83 | 32 |
| feature_45_tuned_021 | 4 | 3.0000 | 6.0000 | 1.5000 | 0.3300 | 0.7494 | 0.6136 | 0.8084 | 0.6977 | 85 | 32 |
| feature_45_tuned_046 | 5 | 8.0000 | 3.0000 | 1.5000 | 0.3350 | 0.7481 | 0.6071 | 0.8144 | 0.6957 | 88 | 31 |
| feature_45_tuned_013 | 4 | 1.0000 | 0.5000 | 1.2000 | 0.3250 | 0.7475 | 0.5983 | 0.8383 | 0.6983 | 94 | 27 |
| feature_45_tuned_034 | 5 | 3.0000 | 0.5000 | 1.0000 | 0.3150 | 0.7469 | 0.6000 | 0.8443 | 0.7015 | 94 | 26 |
| feature_45_tuned_045 | 5 | 8.0000 | 0.5000 | 1.2000 | 0.3250 | 0.7463 | 0.6018 | 0.8144 | 0.6921 | 90 | 31 |

## 4. 45개 기본 모델 Threshold 정책

| Policy | Thresholds | Test P | Test R | Test F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| valid_recall85_max_precision | global:0.315 | 0.6293 | 0.8743 | 0.7318 | 86 | 21 |
| global_max_precision_recall_ge_0.85 | global:0.315 | 0.6293 | 0.8743 | 0.7318 | 86 | 21 |
| targeted_industry_macro_category_it_services_recall_ge_0.80 | industry_macro_category=it_services:0.360; fallback:0.315 | 0.6316 | 0.8623 | 0.7291 | 84 | 23 |
| targeted_market_KOSDAQ_recall_ge_0.80 | market=KOSDAQ:0.360; fallback:0.315 | 0.6462 | 0.8204 | 0.7230 | 75 | 30 |
| targeted_market_KOSDAQ_recall_ge_0.85 | market=KOSDAQ:0.335; fallback:0.315 | 0.6376 | 0.8323 | 0.7221 | 79 | 28 |
| global_max_precision_recall_ge_0.80 | global:0.335 | 0.6359 | 0.8263 | 0.7188 | 79 | 29 |
| global_best_valid_f1 | global:0.335 | 0.6359 | 0.8263 | 0.7188 | 79 | 29 |
| market_segment_best_valid_f1 | KOSDAQ:0.335; KOSPI:0.290; fallback:0.315 | 0.6318 | 0.8323 | 0.7183 | 81 | 28 |
| targeted_industry_macro_category_manufacturing_recall_ge_0.80 | industry_macro_category=manufacturing:0.335; fallback:0.315 | 0.6318 | 0.8323 | 0.7183 | 81 | 28 |
| industry_macro_category_segment_best_valid_f1 | it_services:0.335; manufacturing:0.335; wholesale_retail:0.505; fallback:0.315 | 0.6343 | 0.8204 | 0.7154 | 79 | 30 |
| global_max_precision_recall_ge_0.88 | global:0.245 | 0.5731 | 0.8922 | 0.6979 | 111 | 18 |

## 5. 45개를 Stage 2 보조 트리거로 쓸 때

| Policy | Precision | Recall | F1 | FP | FN | 45-only cases | 45-only risk | 45-only normal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 43_baseline | 0.6400 | 0.8623 | 0.7347 | 81 | 23 | 12 | 4 | 8 |
| 45_feature_set | 0.6293 | 0.8743 | 0.7318 | 86 | 21 | 12 | 4 | 8 |
| union_43_or_45_review_trigger | 0.6245 | 0.8862 | 0.7327 | 89 | 19 | 12 | 4 | 8 |
| intersection_43_and_45_strict | 0.6455 | 0.8503 | 0.7339 | 78 | 25 | 12 | 4 | 8 |

45개 변수셋만 추가로 위험하다고 본 기업은 12개였고, 이 중 실제 투기등급은 4개였습니다.
따라서 45개 모델은 최종 라벨을 직접 바꾸기보다, 43개 모델이 낮게 본 기업 중 일부를 에이전트 검토 대상으로 올리는 신호로 활용하는 편이 더 적합합니다.

## 6. Recall 우선 정책

아래 정책은 45개 변수셋의 threshold를 낮추어 더 넓게 잡는 방식입니다. 최종 부적격 라벨로 바로 쓰기보다는 위원회 검토 대상으로 올리는 review trigger 후보로 해석하는 편이 안전합니다.
현실적인 운영 후보는 `targeted_industry_macro_category_it_services_valid_recall_ge_0.90`입니다. IT서비스 기업에만 threshold `0.175`를 적용하고, 나머지는 기본 threshold `0.315`를 유지합니다.

| Policy | Thresholds | Test P | Test R | Test F1 | FP | FN | Added | Added Risk | Added Normal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_valid_recall_ge_0.95 | global:0.175 | 0.5256 | 0.9222 | 0.6696 | 139 | 13 | 61 | 8 | 53 |
| targeted_market_KOSDAQ_valid_recall_ge_0.95 | market=KOSDAQ:0.195; fallback:0.315 | 0.5523 | 0.9162 | 0.6892 | 124 | 14 | 45 | 7 | 38 |
| global_valid_recall_ge_0.92 | global:0.220 | 0.5527 | 0.9102 | 0.6878 | 123 | 15 | 43 | 6 | 37 |
| targeted_market_KOSDAQ_valid_recall_ge_0.92 | market=KOSDAQ:0.230; fallback:0.315 | 0.5741 | 0.9042 | 0.7023 | 112 | 16 | 31 | 5 | 26 |
| targeted_industry_macro_category_manufacturing_valid_recall_ge_0.92 | industry_macro_category=manufacturing:0.220; fallback:0.315 | 0.5725 | 0.8982 | 0.6993 | 112 | 17 | 30 | 4 | 26 |
| global_valid_recall_ge_0.90 | global:0.235 | 0.5639 | 0.8982 | 0.6928 | 116 | 17 | 34 | 4 | 30 |
| targeted_market_KOSDAQ_valid_recall_ge_0.90 | market=KOSDAQ:0.245; fallback:0.315 | 0.5843 | 0.8922 | 0.7062 | 106 | 18 | 23 | 3 | 20 |
| targeted_industry_macro_category_it_services_valid_recall_ge_0.92 | industry_macro_category=it_services:0.070; fallback:0.315 | 0.5843 | 0.8922 | 0.7062 | 106 | 18 | 23 | 3 | 20 |
| global_valid_recall_ge_0.88 | global:0.245 | 0.5731 | 0.8922 | 0.6979 | 111 | 18 | 28 | 3 | 25 |
| targeted_industry_macro_category_it_services_valid_recall_ge_0.90 | industry_macro_category=it_services:0.175; fallback:0.315 | 0.6141 | 0.8862 | 0.7255 | 93 | 19 | 9 | 2 | 7 |
| targeted_industry_macro_category_manufacturing_valid_recall_ge_0.90 | industry_macro_category=manufacturing:0.245; fallback:0.315 | 0.5857 | 0.8802 | 0.7033 | 104 | 20 | 19 | 1 | 18 |
| global_valid_recall_ge_0.85 | global:0.315 | 0.6293 | 0.8743 | 0.7318 | 86 | 21 | 0 | 0 | 0 |

## 7. 해석

- 하이퍼파라미터 튜닝은 validation 기준으로만 선택해야 하며, test 최상위 후보는 참고용입니다.
- 45개 변수셋은 기본적으로 FN을 줄이는 방향이지만 FP도 같이 늘어나는 경향이 있습니다.
- Recall만 우선하면 threshold `0.220~0.235` 구간에서 FN을 추가로 줄일 수 있지만, 정상 기업까지 위원회 검토로 많이 올라옵니다.
- segment threshold가 FP를 줄여도 Recall/F1이 같이 악화되면 운영 모델 교체 근거로는 약합니다.
- 45개 모델을 단독 최종 라벨로 쓰기보다 43개 모델 옆의 보조 경고 신호로 쓰면 Stage 2 에이전트 구조와 더 잘 맞습니다.