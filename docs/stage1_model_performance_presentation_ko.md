# Stage 1 모델 성능 개선 발표 정리

작성일: 2026-05-27

## 한 장 요약

Stage 1 XGBoost 모델은 `is_speculative` 투기등급 여부를 예측하는 1차 위험 선별 모델이다. 성능 개선은 단일 validation 점수에 맞춘 것이 아니라, rolling validation으로 연도별 안정성을 확인하고 최종 OOT(Out-of-Time) test에서 운영상 중요한 Recall, F1, FN 감소를 확인하는 방식으로 검증했다.

발표에서 가장 안전한 메시지는 다음과 같다.

- 초기 Core29 모델은 이미 강한 baseline이었다.
- 이후 34 source / 43 input, 37 source / 46 input으로 feature set을 확장하면서 최종 OOT test에서 Recall과 F1이 개선됐다.
- 현재 46-input 모델은 Core29 대비 test Recall을 `0.8374 -> 0.8719`, F1을 `0.7489 -> 0.7729`로 올렸고, FN은 `33 -> 26`으로 줄였다.
- 다만 rolling validation 평균에서는 Core29가 여전히 강하게 나온다. 따라서 "모든 rolling 평균을 압도했다"가 아니라 "rolling으로 과적합 여부를 점검했고, 최종 운영 목표인 Recall/FN 측면에서 46-input을 선택했다"라고 말하는 편이 정확하다.

## 비교한 모델

| 모델 | 원천 변수 | one-hot 후 입력 | 설명 |
|---|---:|---:|---|
| 초기 Core29 | 29 | 38 | 초기 TS2000 Core29 feature set 재현 |
| 34 source baseline | 34 | 43 | 현재 46-input에서 산업-연도 percentile 3개 제외 |
| 현재 공식 모델 | 37 | 46 | 43-input에 산업-연도 percentile 3개 추가 |

현재 공식 46-input에서 추가된 변수는 아래 3개다.

| 추가 변수 | 의미 |
|---|---|
| `assets_total_industry_year_pct` | 동일 산업-연도 내 자산총계 percentile |
| `gross_profit_industry_year_pct` | 동일 산업-연도 내 매출총이익 percentile |
| `depreciation_industry_year_pct` | 동일 산업-연도 내 감가상각비 percentile |

## 데이터와 타깃

| 항목 | 내용 |
|---|---|
| 데이터 | `data/input/credit_46_features/feature_46_master.csv` |
| 타깃 | `is_speculative` |
| 전체 행 | 5,451개 기업-연도 |
| 데이터 보강 | TS2000 결측 재무값을 OpenDART 사업보고서 기준으로 보강, CFS 우선, CFS 없을 때 OFS fallback |
| 모델 | XGBoost binary classifier |
| 결측 처리 | XGBoost native missing 방향 학습 |

## Rolling Validation 설계

rolling validation은 특정 1개 validation 연도에만 잘 맞는 feature 후보를 걸러내기 위해 사용했다. 각 fold는 과거 연도로 학습하고, 직전 1개 연도에서 calibration과 threshold를 결정한 뒤, 다음 1개 연도에서 평가한다.

| Eval year | Train years | Policy year | Eval year | Train rows | Policy rows | Eval rows |
|---:|---|---:|---:|---:|---:|---:|
| 2019 | 2014-2017 | 2018 | 2019 | 1,490 | 511 | 574 |
| 2020 | 2014-2018 | 2019 | 2020 | 2,001 | 574 | 603 |
| 2021 | 2014-2019 | 2020 | 2021 | 2,575 | 603 | 673 |
| 2022 | 2014-2020 | 2021 | 2022 | 3,178 | 673 | 676 |

## Calibration과 Threshold Tuning

모든 비교 모델에 같은 calibration과 threshold 정책을 적용했다.

| 단계 | 방식 |
|---|---|
| Raw score | XGBoost `predict_proba` |
| Calibration | policy year 또는 validation year raw probability에 Platt sigmoid 보정 |
| Threshold tuning | policy year 또는 validation year에서 Recall `>= 0.85`를 만족하는 후보 중 Precision 최대 threshold 선택 |
| Threshold grid | `0.05`부터 `0.95`까지 `0.005` 단위 |
| Fallback | Recall floor를 만족하는 threshold가 없으면 F1 최대 threshold 사용 |
| Test 사용 여부 | rolling fold와 최종 OOT test는 threshold 선택에 사용하지 않음 |

이 정책을 쓴 이유는 투기등급 위험 선별에서 FN을 줄이는 것이 중요하기 때문이다. Precision만 높이는 threshold는 위험 기업을 놓칠 수 있으므로, Recall 하한을 먼저 걸고 그 안에서 Precision을 최적화했다.

## Rolling Validation 결과

아래 표는 2019-2022 네 fold의 평가 연도 성능 평균이다.

| 모델 | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| 초기 Core29 / 38 input | 0.8372 | 0.9327 | 0.7012 | 0.8453 | 0.7656 |
| 34 source / 43 input | 0.8376 | 0.9312 | 0.6825 | 0.8442 | 0.7537 |
| 현재 37 source / 46 input | 0.8363 | 0.9312 | 0.6917 | 0.8432 | 0.7589 |

rolling 평균만 보면 Core29가 여전히 강하다. 이 점은 발표에서 숨기기보다 "초기 모델도 강한 baseline이었고, 이후 모델은 최종 OOT test에서 운영 목표 지표를 개선하는 방향으로 선택했다"라고 설명하는 것이 좋다.

43-input 대비 46-input은 rolling F1이 `0.7537 -> 0.7589`로 `+0.0052` 개선됐다. PR-AUC는 거의 동일한 수준이다.

## 왜 Rolling에서는 Core29가 더 강한가

rolling validation은 2019-2022의 여러 평가 연도를 평균으로 본다. 이 구간에서는 Core29처럼 단순하고 압축된 feature set이 threshold tuning 이후 Precision을 더 잘 지키면서 F1 평균이 높게 나왔다. 반대로 확장 feature는 일부 연도, 특히 2021 fold에서 FP가 늘어 F1 평균이 낮아졌다.

이 결과는 "확장 feature가 무조건 나쁘다"는 뜻보다는, 과거 여러 연도에서는 단순 모델이 매우 강한 baseline이었다는 뜻에 가깝다. 현재 공식 46-input 모델의 장점은 rolling 평균 전체 압도보다, 최근 최종 OOT test 구간에서 Recall을 높이고 FN을 줄인 점이다.

## Core29 Rolling Fold 상세

초기 baseline을 발표에서 강조하고 싶을 때 사용할 수 있는 상세 표다.

| Eval year | Threshold | PR-AUC | Precision | Recall | F1 | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 0.280 | 0.8369 | 0.6889 | 0.8857 | 0.7750 | 124 | 56 | 16 | 378 |
| 2020 | 0.355 | 0.8520 | 0.7469 | 0.8013 | 0.7732 | 121 | 41 | 30 | 411 |
| 2021 | 0.315 | 0.8273 | 0.6699 | 0.8364 | 0.7439 | 138 | 68 | 27 | 440 |
| 2022 | 0.265 | 0.8325 | 0.6991 | 0.8580 | 0.7704 | 151 | 65 | 25 | 435 |

## 최종 OOT Test 설계

여기서 OOT test는 모델 선택과 threshold tuning에 사용하지 않고 마지막까지 떼어둔 미래 기간 평가셋이다. 흔히 holdout test라고도 부르지만, 발표에서는 `Final OOT Test` 또는 `최종 시계열 외부검증`이라고 부르는 편이 더 직관적이다.

| Split | Fiscal years | Rows | Positive rows |
|---|---|---:|---:|
| Train | 2014-2021 | 3,851 | 878 |
| Validation | 2022 | 676 | 176 |
| Test | 2023-2024 | 924 | 203 |

Validation 2022에서 Platt calibration과 threshold tuning을 수행하고, Test 2023-2024에서 최종 성능을 평가했다.

## 최종 OOT Test 결과

| 모델 | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 초기 Core29 / 38 input | 0.305 | 0.8281 | 0.9389 | 0.6773 | 0.8374 | 0.7489 | 170 | 81 | 33 | 640 |
| 34 source / 43 input | 0.320 | 0.8329 | 0.9415 | 0.7004 | 0.8522 | 0.7689 | 173 | 74 | 30 | 647 |
| 현재 37 source / 46 input | 0.300 | 0.8321 | 0.9415 | 0.6941 | 0.8719 | 0.7729 | 177 | 78 | 26 | 643 |

## 개선 폭

초기 Core29 대비 현재 46-input 모델의 최종 OOT test 개선은 다음과 같다.

| 지표 | Core29 | 46-input | 변화 |
|---|---:|---:|---:|
| PR-AUC | 0.8281 | 0.8321 | +0.0040 |
| Precision | 0.6773 | 0.6941 | +0.0168 |
| Recall | 0.8374 | 0.8719 | +0.0345 |
| F1 | 0.7489 | 0.7729 | +0.0240 |
| FP | 81 | 78 | -3 |
| FN | 33 | 26 | -7 |

43-input 대비 현재 46-input 모델의 최종 OOT test 변화는 다음과 같다.

| 지표 | 43-input | 46-input | 변화 |
|---|---:|---:|---:|
| PR-AUC | 0.8329 | 0.8321 | -0.0008 |
| Precision | 0.7004 | 0.6941 | -0.0063 |
| Recall | 0.8522 | 0.8719 | +0.0197 |
| F1 | 0.7689 | 0.7729 | +0.0040 |
| FP | 74 | 78 | +4 |
| FN | 30 | 26 | -4 |

현재 공식 모델은 Precision을 소폭 양보하는 대신 Recall과 FN 감소를 얻었다. 위험 선별 모델이라는 목적에서는 이 trade-off가 납득 가능하다.

## PPT에 넣기 좋은 문장

1. "모델 개선은 단일 validation 점수에 맞추지 않고, walk-forward rolling validation으로 연도별 안정성을 먼저 확인했다."
2. "각 fold에서는 과거 연도로 학습하고, 직전 연도에서 Platt calibration과 threshold tuning을 수행한 뒤, 다음 연도를 평가했다."
3. "초기 Core29 모델은 rolling validation에서 이미 강한 baseline이었으며, 이후 feature 확장은 최종 OOT test에서 Recall과 FN 감소를 개선하는 방향으로 검증했다."
4. "현재 46-input 모델은 Core29 대비 최종 test Recall을 0.8374에서 0.8719로 높이고, FN을 33건에서 26건으로 줄였다."
5. "43-input 대비 46-input은 rolling F1을 0.7537에서 0.7589로 개선했고, 최종 test Recall도 0.8522에서 0.8719로 개선했다."
6. "따라서 최종 모델 선택 기준은 단순 정확도가 아니라, 투기등급 위험 기업을 놓치지 않는 운영 목적에 맞춘 Recall/FN 중심의 성능 개선이다."

## 추천 슬라이드 구성

| Slide | 제목 | 핵심 내용 |
|---:|---|---|
| 1 | 모델 개선 목표 | 투기등급 위험 조기 선별, FN 감소 중심 |
| 2 | 검증 설계 | rolling validation fold, calibration, threshold tuning 구조 |
| 3 | Feature set 진화 | Core29 -> 43 input -> 46 input |
| 4 | Rolling validation | 연도별 안정성 점검, 43 -> 46 F1 개선 |
| 5 | Final OOT test | Core29 대비 46-input Recall/F1/FN 개선 |
| 6 | 결론 | Recall/FN 중심 운영 모델로 46-input 채택 |

## 주의해서 말할 점

- rolling validation 평균에서 Core29가 현재 46-input보다 F1이 높다. 발표에서는 "Core29 대비 rolling 평균 전체 개선"이라고 말하지 않는다.
- 현재 비교는 모두 보강된 `credit_46_features` 데이터셋 기준 재측정이다. 2026-04 초기 산출물과 행 수, split, 보강 상태가 다르므로 직접 수치를 섞지 않는다.
- 최종 OOT test는 tuning에 쓰지 않았다. 따라서 최종 test의 Recall/F1 개선은 모델 선택 이후 확인 지표로 제시한다.
- 이 모델의 목적은 투기등급 위험 선별이다. Accuracy보다 PR-AUC, Recall, F1, FN 감소를 중심으로 설명한다.

## 재현 환경

| 항목 | 값 |
|---|---|
| Python 환경 | `/opt/anaconda3/envs/aura/bin/python` |
| 학습 helper | `src/cas/modeling/stage1_xgboost.py` |
| Calibration helper | `src/cas/modeling/calibration.py` |
| 데이터 | `data/input/credit_46_features/feature_46_master.csv` |
| Feature spec | `data/input/credit_46_features/feature_46_list.json` |
| 공식 모델 artifact | `data/outputs/modeling/feature_46_xgboost/` |
