# 46-Feature Stage2 Trigger Feature Set Experiments

공식 Stage1 `feature_46_xgboost` 판단은 유지하고, Stage2 review aux 모델의 후보 feature set을 바꿔 combined Stage2 trigger 성능을 비교했습니다.

Rolling 평가연도는 `2019, 2020, 2021, 2022`이고, Final Test는 공식 test split인 2023~2024 구간입니다.
각 fold는 `과거 연도 학습 -> 직전 1년 Platt calibration/threshold 선택 -> 다음 1년 평가` 구조입니다.

## 1. 결론

- 기준선 `stage2_aux_48_baseline` rolling Recall/F1: `0.8709` / `0.7571`
- Rolling Recall 최상위 후보: `full_review_trigger_73` (Recall delta `+0.0091`, FN delta `-5`)

## 2. 후보별 Rolling + Final Test 비교

| Candidate | Features | Roll Aux PR | Roll P | Roll R | Roll F1 | Roll dR | Roll FP | Roll FN | Roll dFN | Roll Extra TP | Roll Extra FP | Final Aux PR | Final P | Final R | Final F1 | Final dR | Final FP | Final FN | Final dFN | Final Extra TP | Final Extra FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_review_trigger_73 | 73 | 0.8281 | 0.6410 | 0.8800 | 0.7392 | +0.0091 | 318 | 76 | -5 | 23 | 78 | 0.8186 | 0.6429 | 0.8867 | 0.7453 | +0.0049 | 100 | 23 | -1 | 3 | 22 |
| stage2_aux_plus_fn_rescue_scores_53 | 53 | 0.8356 | 0.6363 | 0.8788 | 0.7360 | +0.0079 | 321 | 77 | -4 | 22 | 81 | 0.8250 | 0.6545 | 0.8867 | 0.7531 | +0.0049 | 95 | 23 | -1 | 3 | 17 |
| fn_rescue_raw_trigger_64 | 63 | 0.8300 | 0.6603 | 0.8733 | 0.7508 | +0.0024 | 288 | 80 | -1 | 19 | 48 | 0.8177 | 0.6487 | 0.8916 | 0.7510 | +0.0099 | 98 | 22 | -2 | 4 | 20 |
| macro_regime_trigger_53 | 53 | 0.8328 | 0.6459 | 0.8727 | 0.7399 | +0.0018 | 309 | 80 | -1 | 19 | 69 | 0.8283 | 0.6716 | 0.8867 | 0.7643 | +0.0049 | 88 | 23 | -1 | 3 | 10 |
| stage2_aux_48_baseline | 48 | 0.8334 | 0.6714 | 0.8709 | 0.7571 | +0.0000 | 274 | 81 | 0 | 18 | 34 | 0.8304 | 0.6654 | 0.8818 | 0.7585 | +0.0000 | 90 | 24 | 0 | 2 | 12 |
| cashflow_turn_trigger_53 | 53 | 0.8391 | 0.6641 | 0.8702 | 0.7522 | -0.0007 | 282 | 82 | 1 | 17 | 42 | 0.8197 | 0.6654 | 0.8916 | 0.7621 | +0.0099 | 91 | 22 | -2 | 4 | 13 |
| working_capital_trigger_54 | 54 | 0.8343 | 0.6750 | 0.8636 | 0.7557 | -0.0074 | 267 | 86 | 5 | 13 | 27 | 0.8300 | 0.6593 | 0.8768 | 0.7526 | -0.0049 | 92 | 25 | 1 | 1 | 14 |
| borrowing_pressure_trigger_52 | 52 | 0.8328 | 0.6572 | 0.8609 | 0.7437 | -0.0101 | 289 | 88 | 7 | 11 | 49 | 0.8228 | 0.6547 | 0.8966 | 0.7568 | +0.0148 | 96 | 21 | -3 | 5 | 18 |

## 3. 참고용 Final Test Recall 순위

| Candidate | Aux PR-AUC | Precision | Recall | F1 | FP | FN | Extra TP | Extra FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| borrowing_pressure_trigger_52 | 0.8228 | 0.6547 | 0.8966 | 0.7568 | 96 | 21 | 5 | 18 |
| cashflow_turn_trigger_53 | 0.8197 | 0.6654 | 0.8916 | 0.7621 | 91 | 22 | 4 | 13 |
| fn_rescue_raw_trigger_64 | 0.8177 | 0.6487 | 0.8916 | 0.7510 | 98 | 22 | 4 | 20 |
| macro_regime_trigger_53 | 0.8283 | 0.6716 | 0.8867 | 0.7643 | 88 | 23 | 3 | 10 |
| stage2_aux_plus_fn_rescue_scores_53 | 0.8250 | 0.6545 | 0.8867 | 0.7531 | 95 | 23 | 3 | 17 |
| full_review_trigger_73 | 0.8186 | 0.6429 | 0.8867 | 0.7453 | 100 | 23 | 3 | 22 |
| stage2_aux_48_baseline | 0.8304 | 0.6654 | 0.8818 | 0.7585 | 90 | 24 | 2 | 12 |
| working_capital_trigger_54 | 0.8300 | 0.6593 | 0.8768 | 0.7526 | 92 | 25 | 1 | 14 |

## 4. 해석 주의

- 이 실험은 공식 Stage1 판정을 덮어쓰지 않는 Stage2 review trigger 후보 비교입니다.
- Recall이 올라가도 review load와 FP가 크게 늘면 운영 기본값으로는 부적합할 수 있습니다.
- 후보 선택은 rolling OOT를 우선 기준으로 보고, Final Test는 사후 확인용입니다.