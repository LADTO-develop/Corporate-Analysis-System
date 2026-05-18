# Rolling OOT Validation Experiments

1년 validation에 대한 과신을 줄이기 위해 walk-forward rolling OOT 방식으로 비교했습니다.
각 fold는 `과거 연도 학습 -> 직전 1년 Platt calibration/threshold 선택 -> 다음 1년 평가` 구조입니다.
기존 단일 validation은 특정 경기/시장 국면에 우연히 잘 맞은 후보를 과대평가할 수 있기 때문에,
여러 평가연도에서 같은 후보가 반복적으로 안정적인지 확인하는 용도로 rolling validation을 사용했습니다.
최종 test 구간은 후보 선택에 쓰지 않고 마지막 확인용으로만 남깁니다.

## 1. 결론

- Baseline rolling mean F1: `0.7022` (mean PR-AUC `0.7955`)
- Rolling 평균 최상위 후보: `val_best_interest_burden_ap_days_diff` (mean F1 `0.7161`, mean PR-AUC `0.8038`)
- 최상위 후보의 baseline 대비 mean F1 변화: `+0.0139`
- `val_best_interest_burden_ap_days_diff`가 rolling 평균 F1을 `+0.0139` 개선했습니다. 다만 연도별 변동과 test 성능을 함께 확인한 뒤 후보 모델로만 검토하는 편이 안전합니다.

## 2. Rolling 평균 성능

| Variant | Features | Folds | PR-AUC mean | ROC-AUC mean | Precision mean | Recall mean | F1 mean | F1 min | Total FP | Total FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val_best_interest_burden_ap_days_diff | interest_burden_ratio, ap_days_diff | 4 | 0.8038 | 0.9177 | 0.6206 | 0.8558 | 0.7161 | 0.6938 | 338 | 91 |
| single_ar_days | ar_days | 4 | 0.7954 | 0.9158 | 0.6052 | 0.8481 | 0.7037 | 0.6818 | 356 | 96 |
| balanced_delta_accruals_ppi | delta_accruals_ratio, ppi | 4 | 0.7965 | 0.9158 | 0.6120 | 0.8391 | 0.7037 | 0.6803 | 345 | 102 |
| test_reference_base_rate_treasury_diff | base_rate, treasury_3y_diff | 4 | 0.7950 | 0.9156 | 0.6053 | 0.8442 | 0.7027 | 0.6683 | 353 | 99 |
| baseline_43_native |  | 4 | 0.7955 | 0.9148 | 0.6166 | 0.8256 | 0.7022 | 0.6597 | 332 | 111 |
| single_non_paid_in_equity_ratio | non_paid_in_equity_ratio | 4 | 0.7978 | 0.9154 | 0.6042 | 0.8422 | 0.7006 | 0.6733 | 355 | 99 |

## 3. Baseline 연도별 성능

| Eval Year | Policy Year | Rows | PR-AUC | ROC-AUC | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,019 | 2,018 | 574 | 0.8144 | 0.9189 | 0.5590 | 0.9143 | 0.6938 | 101 | 12 |
| 2,020 | 2,019 | 603 | 0.8142 | 0.9211 | 0.6994 | 0.7550 | 0.7261 | 49 | 37 |
| 2,021 | 2,020 | 673 | 0.7522 | 0.9014 | 0.5773 | 0.7697 | 0.6597 | 93 | 38 |
| 2,022 | 2,021 | 676 | 0.8013 | 0.9178 | 0.6307 | 0.8636 | 0.7290 | 89 | 24 |

## 4. 최상위 후보 연도별 성능

| Eval Year | Policy Year | Rows | PR-AUC | ROC-AUC | Precision | Recall | F1 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,019 | 2,018 | 574 | 0.8006 | 0.9157 | 0.5590 | 0.9143 | 0.6938 | 101 | 12 |
| 2,020 | 2,019 | 603 | 0.8307 | 0.9258 | 0.6982 | 0.7815 | 0.7375 | 51 | 33 |
| 2,021 | 2,020 | 673 | 0.7779 | 0.9099 | 0.6027 | 0.8182 | 0.6941 | 89 | 30 |
| 2,022 | 2,021 | 676 | 0.8059 | 0.9192 | 0.6226 | 0.9091 | 0.7390 | 97 | 16 |

## 5. 해석 주의

- 이 실험은 test 2023~2024를 모델 선택에 쓰지 않기 위한 rolling validation입니다.
- 각 fold의 threshold는 평가 연도 직전 1년에서만 선택했습니다.
- 단일 validation 1년만 보면 우연한 연도 효과나 경기 국면 효과를 후보 변수 성능으로 착각할 수 있습니다.
- rolling 평균은 후보 선별의 1차 기준이고, 운영 반영 여부는 final test와 오류 사례 해석까지 함께 봅니다.
- 후보가 평균에서 좋아도 특정 연도에서 FN이 크게 늘면 조기경보 모델로는 보수적으로 봐야 합니다.