# 43개 vs 44개 변수셋 성능 비교

44개 변수셋은 기존 43개 입력에 `industry_current_ratio_percentile`을 추가한 후보 실험셋이다.
성능 비교 결과 공식 Stage 1 모델은 43개 변수셋으로 되돌리고, 44개 변수셋 artifact는 제거한다.

## Test 성능 비교

| 모델 | 변수 수 | 추가 변수 | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 | FP | FN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 43-feature XGBoost | 43 | - | 0.315 | 0.7930 | 0.9286 | 0.6603 | 0.8522 | 0.7441 | 89 | 30 |
| 44-feature XGBoost | 44 | `industry_current_ratio_percentile` | 0.305 | 0.7912 | 0.9250 | 0.6196 | 0.8424 | 0.7140 | 105 | 32 |

## 해석

- 43개 변수셋이 PR-AUC, ROC-AUC, Precision, Recall, F1에서 모두 더 높았다.
- 44개 변수셋은 FP가 105개로 43개 변수셋의 89개보다 16개 많았다.
- 44개 변수셋은 FN도 32개로 43개 변수셋의 30개보다 2개 많았다.
- 따라서 `industry_current_ratio_percentile`은 Model V1에는 후보 칼럼으로 보존하되, 공식 모델 입력에는 포함하지 않는다.

결론적으로 발표에서는 "산업 내 유동비율 백분위 후보를 추가해봤지만, test 기준 분류 성능과 오류 수가 모두 개선되지 않아 공식 모델은 43개 변수셋으로 유지했다"고 설명한다.
