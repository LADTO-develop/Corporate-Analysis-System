# SHAP-Driven Feature Improvement Experiments

오류 사례 SHAP 분석에서 반복적으로 나타난 절대금액, 기업규모, 산업 내 위치, 전년 대비 악화 신호를 변수 후보로 만들어 비교한 실험입니다.
모든 실험은 XGBoost native missing, Platt scaling, validation 기준 `recall >= 0.85` 조건에서 precision 최대 threshold를 사용했습니다.

## 1. 결론

- Baseline F1: `0.7347`, Precision: `0.6400`, Recall: `0.8623`, FP/FN: `81/23`
- Best F1 variant: `baseline_43_native` (F1 `0.7347`, Precision `0.6400`, Recall `0.8623`, FP/FN `81/23`)
- Best vs baseline: F1 `+0.0000`, FP `+0`, FN `+0`
- 현재 기준에서는 새 변수 추가보다 기존 43개 변수셋이 가장 안정적입니다. 성능 개선은 변수 추가보다 오류 사례 기반 라벨/원천 변수 보강 쪽이 더 유망합니다.

## 2. 전체 성능 비교

| Variant | Features | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_43_native | 43 | 0.3150 | 0.7744 | 0.9110 | 0.6400 | 0.8623 | 0.7347 | 81 | 23 |
| drop_firm_size_dummies_native | 39 | 0.2900 | 0.7711 | 0.9066 | 0.6102 | 0.8623 | 0.7146 | 92 | 23 |
| scale_adjusted_amounts_add_native | 46 | 0.3050 | 0.7745 | 0.9101 | 0.6094 | 0.8503 | 0.7100 | 91 | 25 |
| amount_log_replace_native | 43 | 0.3100 | 0.7679 | 0.9073 | 0.6094 | 0.8503 | 0.7100 | 91 | 25 |
| log_amounts_drop_firm_size_native | 39 | 0.2950 | 0.7710 | 0.9062 | 0.5992 | 0.8683 | 0.7090 | 97 | 22 |
| amount_log_add_native | 46 | 0.2600 | 0.7738 | 0.9093 | 0.5896 | 0.8862 | 0.7081 | 103 | 19 |
| full_shap_context_add_native | 69 | 0.3250 | 0.7634 | 0.9040 | 0.6133 | 0.8263 | 0.7041 | 87 | 29 |
| lag_delta_key_ratios_add_native | 52 | 0.3200 | 0.7725 | 0.9064 | 0.5949 | 0.8443 | 0.6980 | 96 | 26 |
| ratio_industry_pct_add_native | 52 | 0.3200 | 0.7665 | 0.9057 | 0.5851 | 0.8443 | 0.6912 | 100 | 26 |
| industry_amount_pct_add_native | 46 | 0.3200 | 0.7623 | 0.9041 | 0.5865 | 0.8323 | 0.6881 | 98 | 28 |
| industry_amount_pct_replace_native | 43 | 0.2750 | 0.7641 | 0.9034 | 0.5675 | 0.8563 | 0.6826 | 109 | 24 |

## 3. KOSDAQ FP 관점

- Baseline KOSDAQ FP: `70`, FN: `19`
- KOSDAQ FP 최소 variant: `baseline_43_native` (FP `70`, FN `19`, F1 `0.7278`)

| Variant | Rows | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_43_native | 384 | 0.6296 | 0.8623 | 0.7278 | 70 | 19 |
| scale_adjusted_amounts_add_native | 384 | 0.6082 | 0.8551 | 0.7108 | 76 | 20 |
| drop_firm_size_dummies_native | 384 | 0.6070 | 0.8841 | 0.7198 | 79 | 16 |
| amount_log_replace_native | 384 | 0.5939 | 0.8478 | 0.6985 | 80 | 21 |
| full_shap_context_add_native | 384 | 0.5897 | 0.8333 | 0.6907 | 80 | 23 |
| log_amounts_drop_firm_size_native | 384 | 0.5902 | 0.8768 | 0.7055 | 84 | 17 |
| lag_delta_key_ratios_add_native | 384 | 0.5821 | 0.8478 | 0.6903 | 84 | 21 |
| amount_log_add_native | 384 | 0.5829 | 0.8913 | 0.7049 | 88 | 15 |

## 4. 해석

- 절대금액을 log나 산업 백분위로 바꾸는 실험은 FP를 줄일 수 있는지 확인하기 위한 실험입니다.
- 전년 대비 변화량은 SK이노베이션, KG모빌리티처럼 규모가 큰 기업의 위험을 너무 안정적으로 보는 FN 문제를 줄일 수 있는지 확인하기 위한 실험입니다.
- 개선 폭이 작거나 특정 구간만 좋아지는 경우에는 운영 모델을 즉시 교체하지 않고, 추가 feature 후보 또는 세그먼트별 보정 후보로만 관리합니다.