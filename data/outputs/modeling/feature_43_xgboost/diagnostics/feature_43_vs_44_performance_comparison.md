# 43개 vs 44개 변수셋 성능 비교

44개 변수셋은 기존 43개 입력에 `industry_current_ratio_percentile`을 추가한 후보 실험셋이다.
OpenDART CFS/OFS 보강 후 새 데이터 기준으로 다시 비교해도 공식 Stage 1 모델은
43개 변수셋을 유지하는 것이 더 낫다. 44개 변수셋 artifact는 운영 산출물로
보관하지 않고, 비교 기록만 남긴다.

## Test 성능 비교

| 모델 | 변수 수 | 추가 변수 | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 | FP | FN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 43-feature XGBoost | 43 | - | 0.320 | 0.8329 | 0.9415 | 0.7004 | 0.8522 | 0.7689 | 74 | 30 |
| 44-feature XGBoost | 44 | `industry_current_ratio_percentile` | 0.315 | 0.8244 | 0.9397 | 0.6911 | 0.8374 | 0.7572 | 76 | 33 |

## 해석

- 43개 변수셋이 PR-AUC, ROC-AUC, Precision, Recall, F1에서 모두 더 높았다.
- 44개 변수셋은 FP가 76개로 43개 변수셋의 74개보다 2개 많았다.
- 44개 변수셋은 FN도 33개로 43개 변수셋의 30개보다 3개 많았다.
- 따라서 `industry_current_ratio_percentile`은 Model V1에는 후보 칼럼으로 보존하되, 공식 모델 입력에는 포함하지 않는다.

결론적으로 발표에서는 "산업 내 유동비율 백분위 후보를 추가해봤지만, test 기준 분류 성능과 오류 수가 모두 개선되지 않아 공식 모델은 43개 변수셋으로 유지했다"고 설명한다.
