# 46-Feature Manufacturing/KOSDAQ FN Rescue Gate Experiments

공식 `feature_46_xgboost` Stage1 판단은 유지하고, 낮은 Stage1 확률로 놓치는 KOSDAQ 제조업 후보를 Stage2 에이전트 검토 대상으로 올리는 deterministic gate를 비교했습니다.

Rolling 평가연도는 `2019, 2020, 2021, 2022`이고, Final Test는 공식 test split인 2023~2024 구간입니다.

## 1. 결론

- Baseline rolling F1/Recall: `0.7589` / `0.8432`
- Rolling 기준 FN 최소 후보: `recall_prob030_score045`
- Rolling FN 변화: `-14`
- Rolling FP 변화: `176`
- Final Test FN 변화: `-7`
- 운영 기본 후보: `conservative_group2_prob030_score065` (Rolling FN `-3`, FP `9` / Final FN `-2`, FP `4`)

## 2. 후보별 Rolling + Final Test 비교

| Policy | Prob <=  | Score >= | Groups | Roll P | Roll R | Roll F1 | Roll FP | Roll FN | Roll dFN | Roll extra TP | Roll extra FP | Final P | Final R | Final F1 | Final FP | Final FN | Final dFN | Final extra TP | Final extra FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recall_prob030_score045 | 0.3000 | 0.4500 | 0 | 0.5715 | 0.8654 | 0.6874 | 416 | 85 | -14 | 14 | 176 | 0.6053 | 0.9064 | 0.7258 | 120 | 19 | -7 | 7 | 42 |
| moderate_group1_prob030_score055 | 0.3000 | 0.5500 | 1 | 0.6489 | 0.8494 | 0.7348 | 294 | 95 | -4 | 4 | 54 | 0.6605 | 0.8818 | 0.7553 | 92 | 24 | -2 | 2 | 14 |
| conservative_group2_prob030_score065 | 0.3000 | 0.6500 | 2 | 0.6855 | 0.8479 | 0.7570 | 249 | 96 | -3 | 3 | 9 | 0.6858 | 0.8818 | 0.7716 | 82 | 24 | -2 | 2 | 4 |
| baseline_stage1_only | 0.0000 | 1.0000 | 99 | 0.6917 | 0.8432 | 0.7589 | 240 | 99 | 0 | 0 | 0 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 | 0 | 0 | 0 |
| strict_low_prob_010_score_078 | 0.1000 | 0.7800 | 2 | 0.6917 | 0.8432 | 0.7589 | 240 | 99 | 0 | 0 | 0 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 | 0 | 0 | 0 |

## 3. Baseline 연도별 Rolling 성능

| Eval Year | Threshold | Precision | Recall | F1 | FP | FN | Target FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 0.2600 | 0.6776 | 0.8857 | 0.7678 | 59 | 16 | 7 |
| 2020 | 0.3550 | 0.7407 | 0.7947 | 0.7668 | 42 | 31 | 15 |
| 2021 | 0.3050 | 0.6456 | 0.8061 | 0.7170 | 73 | 32 | 15 |
| 2022 | 0.2400 | 0.7027 | 0.8864 | 0.7839 | 66 | 20 | 9 |

## 4. 최상위 후보 연도별 Rolling 성능

| Eval Year | Threshold | Precision | Recall | F1 | FP | FN | Extra TP | Extra FP | Target FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 0.2600 | 0.5727 | 0.9000 | 0.7000 | 94 | 14 | 2 | 35 | 5 |
| 2020 | 0.3550 | 0.6219 | 0.8278 | 0.7102 | 76 | 26 | 5 | 34 | 10 |
| 2021 | 0.3050 | 0.5112 | 0.8303 | 0.6328 | 131 | 28 | 4 | 58 | 11 |
| 2022 | 0.2400 | 0.5803 | 0.9034 | 0.7067 | 115 | 17 | 3 | 49 | 6 |

## 5. 참고용 Final Test 순위

| Policy | Precision | Recall | F1 | FP | FN | Extra TP | Extra FP | Target Recall | Target FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recall_prob030_score045 | 0.6053 | 0.9064 | 0.7258 | 120 | 19 | 7 | 42 | 0.9154 | 11 |
| conservative_group2_prob030_score065 | 0.6858 | 0.8818 | 0.7716 | 82 | 24 | 2 | 4 | 0.8769 | 16 |
| moderate_group1_prob030_score055 | 0.6605 | 0.8818 | 0.7553 | 92 | 24 | 2 | 14 | 0.8769 | 16 |
| baseline_stage1_only | 0.6941 | 0.8719 | 0.7729 | 78 | 26 | 0 | 0 | 0.8615 | 18 |
| strict_low_prob_010_score_078 | 0.6941 | 0.8719 | 0.7729 | 78 | 26 | 0 | 0 | 0.8615 | 18 |

## 6. Gate 정의

- Policy name: `kosdaq_manufacturing_low_stage1_probability_financial_stress_rescue_gate`
- 대상: `market == KOSDAQ` and `industry_macro_category == manufacturing`
- 공식 Stage1 예측은 정상이고, Stage1 확률이 policy ceiling 이하인 회사만 검토
- Rescue score는 working capital stress, cashflow turn stress, borrowing pressure score를 조합
- 이 gate는 공식 Stage1 판정을 바꾸지 않고 Stage2 에이전트 검토 큐에만 추가합니다.