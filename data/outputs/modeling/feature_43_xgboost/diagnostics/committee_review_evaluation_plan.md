# Committee Review Evaluation Plan

Stage 2 위원회가 모델 판단을 얼마나 보완하는지 평가하기 위한 샘플과 실행 기준입니다.

## 1. Historical Validation Tuning

- 목적: validation 구간에서 위원회가 FN을 보류/부적격으로 끌어올리고, FP를 적격/보류로 완화하도록 에이전트 규칙과 프롬프트를 개선합니다.
- 기준일: 각 행의 `as_of_date = fiscal_year-12-31`입니다.
- 누수 방지: Naver/Tavily는 기준일 이후 결과를 제외하고, 과거 모드에서는 날짜 없는 웹 결과도 제외합니다. OpenDART는 조회 종료일을 기준일로 고정합니다.
- 사용 원칙: 이 샘플은 에이전트 개선용입니다. test 성능을 보면서 규칙을 조정하지 않습니다.

### Validation Tuning Sample Counts

| committee_policy | sample_category | rows |
| --- | --- | --- |
| balanced_current_45_or_near_threshold_0_10 | bbb_minus_bb_plus_boundary | 15 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | 15 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | 15 |
| balanced_current_45_or_near_threshold_0_10 | true_positive_risk_explanation | 15 |
| recall_first_current_45_or_fn_mid_mfg_prob_0_10 | bbb_minus_bb_plus_boundary | 15 |
| recall_first_current_45_or_fn_mid_mfg_prob_0_10 | fn_caught_by_stage2_review | 15 |
| recall_first_current_45_or_fn_mid_mfg_prob_0_10 | fp_needing_committee_mitigation | 15 |
| recall_first_current_45_or_fn_mid_mfg_prob_0_10 | true_positive_risk_explanation | 15 |

### Validation Tuning Sample Preview

| committee_policy | sample_category | corp_name | fiscal_year | eval_year | as_of_date | actual_label_name | model_predicted_label_name | credit_rating | prob_speculative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | 디와이피(주) | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | BB+ | 0.3046390178444226 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)원림 | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | BB | 0.2995454438489445 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | 케이지케미칼(주) | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | BB | 0.2932298848775912 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)온타이드 | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | BB- | 0.2887091403578494 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | 나라엠앤디(주) | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | BB | 0.2623642743922324 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | 엠케이전자(주) | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | BB+ | 0.2623270861019992 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)세원물산 | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | BB+ | 0.2553647996268243 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)이수앱지스 | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | B+ | 0.2334572061085951 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | 미래나노텍(주) | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | BB+ | 0.2316015999868048 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)쏠리드 | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | BB+ | 0.2280001607492164 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)피제이메탈 | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | BB+ | 0.2275349316082501 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)센코 | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | BB+ | 0.2265046961815289 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)브이티 | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | BB+ | 0.2199211796424048 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | 에스넷시스템(주) | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | BB+ | 0.2035608072496871 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | 아이텍(주) | 2022 | 2023 | 2022-12-31 | 투기등급 | 투자적격 | B+ | 0.1483054774318321 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | 제룡산업(주) | 2022 | 2023 | 2022-12-31 | 투자적격 | 투기등급 | BBB+ | 0.3226465472874248 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)아바텍 | 2022 | 2023 | 2022-12-31 | 투자적격 | 투기등급 | A | 0.3226465472874248 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)에이프로 | 2022 | 2023 | 2022-12-31 | 투자적격 | 투기등급 | BBB | 0.3258644531834125 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)대성미생물연구소 | 2022 | 2023 | 2022-12-31 | 투자적격 | 투기등급 | A | 0.3269777841203389 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)파세코 | 2022 | 2023 | 2022-12-31 | 투자적격 | 투기등급 | BBB+ | 0.3269777841203389 |

## 2. Historical Test Holdout

- 목적: validation에서 고정한 에이전트 개선안이 test 구간에서도 유지되는지 마지막에 확인합니다.
- 기준일과 누수 방지 규칙은 validation tuning과 동일합니다.
- 사용 원칙: test 결과는 사후 확인용이며, test 결과를 보고 다시 에이전트 규칙을 고치지 않습니다.

### Test Holdout Sample Counts

| committee_policy | sample_category | rows |
| --- | --- | --- |
| balanced_current_45_or_near_threshold_0_10 | bbb_minus_bb_plus_boundary | 15 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | 11 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | 15 |
| balanced_current_45_or_near_threshold_0_10 | true_positive_risk_explanation | 15 |
| recall_first_current_45_or_fn_mid_mfg_prob_0_10 | bbb_minus_bb_plus_boundary | 15 |
| recall_first_current_45_or_fn_mid_mfg_prob_0_10 | fn_caught_by_stage2_review | 14 |
| recall_first_current_45_or_fn_mid_mfg_prob_0_10 | fp_needing_committee_mitigation | 15 |
| recall_first_current_45_or_fn_mid_mfg_prob_0_10 | true_positive_risk_explanation | 15 |

### Test Holdout Sample Preview

| committee_policy | sample_category | corp_name | fiscal_year | eval_year | as_of_date | actual_label_name | model_predicted_label_name | credit_rating | prob_speculative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | 이원컴포텍(주) | 2023 | 2024 | 2023-12-31 | 투기등급 | 투자적격 | BB | 0.3147086048314541 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)네오크레마 | 2023 | 2024 | 2023-12-31 | 투기등급 | 투자적격 | BB+ | 0.283625267156044 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)마크로젠 | 2023 | 2024 | 2023-12-31 | 투기등급 | 투자적격 | BB+ | 0.2832689977304288 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)네패스 | 2024 | 2025 | 2024-12-31 | 투기등급 | 투자적격 | BB+ | 0.2774527565801999 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | 나라엠앤디(주) | 2023 | 2024 | 2023-12-31 | 투기등급 | 투자적격 | BB | 0.2647707185638435 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)필옵틱스 | 2023 | 2024 | 2023-12-31 | 투기등급 | 투자적격 | BB | 0.2569062583037855 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)세원물산 | 2023 | 2024 | 2023-12-31 | 투기등급 | 투자적격 | BB+ | 0.2412288857051393 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)서플러스글로벌 | 2023 | 2024 | 2023-12-31 | 투기등급 | 투자적격 | B+ | 0.2326359080701688 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)이수앱지스 | 2024 | 2025 | 2024-12-31 | 투기등급 | 투자적격 | B+ | 0.2316820960708796 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)비엠티 | 2023 | 2024 | 2023-12-31 | 투기등급 | 투자적격 | BB | 0.231637033411979 |
| balanced_current_45_or_near_threshold_0_10 | fn_caught_by_stage2_review | (주)아이즈비전 | 2023 | 2024 | 2023-12-31 | 투기등급 | 투자적격 | BB+ | 0.16479115898162 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)코렌텍 | 2023 | 2024 | 2023-12-31 | 투자적격 | 투기등급 | BBB+ | 0.3183565006468821 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | 제이엠아이(주) | 2023 | 2024 | 2023-12-31 | 투자적격 | 투기등급 | BBB | 0.3213278016820243 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)로지시스 | 2023 | 2024 | 2023-12-31 | 투자적격 | 투기등급 | A- | 0.322255304424888 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)아진엑스텍 | 2023 | 2024 | 2023-12-31 | 투자적격 | 투기등급 | BBB+ | 0.3240185597236945 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | 삼성중공업(주) | 2023 | 2024 | 2023-12-31 | 투자적격 | 투기등급 | BBB+ | 0.3243935344809786 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)픽셀플러스 | 2023 | 2024 | 2023-12-31 | 투자적격 | 투기등급 | A- | 0.3268851283594954 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)타이거일렉 | 2023 | 2024 | 2023-12-31 | 투자적격 | 투기등급 | BBB- | 0.3268851283594954 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | (주)켐트로스 | 2023 | 2024 | 2023-12-31 | 투자적격 | 투기등급 | BBB | 0.3268851283594954 |
| balanced_current_45_or_near_threshold_0_10 | fp_needing_committee_mitigation | 성우테크론(주) | 2023 | 2024 | 2023-12-31 | 투자적격 | 투기등급 | BBB+ | 0.3308874730683 |

## 3. Current/2026 External Validation

- 목적: 2026 inference 기업을 현재 시점에서 실제 외부 검증 정답셋과 비교할 준비를 합니다.
- 기준일: 실행일 기준 현재 사용 가능한 뉴스/공시를 사용할 수 있습니다.
- 사용 원칙: validation/test 기반 개선이 끝난 뒤 외부검증셋으로만 사용합니다. 에이전트 규칙 튜닝에는 사용하지 않습니다.

### 2026 Candidate Counts

| model_score_status | actual_label_name | rows |
| --- | --- | --- |
| feature_row_ready_score_not_exported | 투기등급 | 15 |
| feature_row_ready_score_not_exported | 투자적격 | 126 |

## Evaluation Questions

- FN 보완: 실제 투기등급인데 1차 모델이 투자적격으로 본 기업을 위원회가 보류/부적격으로 끌어올리는가?
- FP 완화: 실제 투자적격인데 1차 모델이 위험하다고 본 기업을 위원회가 적격/보류로 완화하는가?
- 근거 신뢰도: veto나 숨은 꼬리위험 판단이 실제 기업 직접 근거에 기반하는가?
- 발표 표현: 과거 validation/test 재현 평가는 look-ahead bias를 막기 위해 기준일 이전 공개 정보만 사용합니다.
