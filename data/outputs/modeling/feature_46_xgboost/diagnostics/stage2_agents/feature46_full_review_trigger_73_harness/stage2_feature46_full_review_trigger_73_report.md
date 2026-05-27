# Stage 2 Feature 46 / full_review_trigger_73 Evaluation Harness

- 생성시각(UTC): `2026-05-27T01:44:54+00:00`
- 기준: 공식 Stage1 `feature_46_xgboost` + Stage2 trigger `full_review_trigger_73`
- Stage2 policy version: `stage2_policy_v1`
- Prompt contract versions: `chair_report=stage2_role_prompt_contract_v2:chair_report, evidence_audit=stage2_role_prompt_contract_v2:evidence_audit, quant_credit=stage2_role_prompt_contract_v2:quant_credit, review_qa=stage2_role_prompt_contract_v2:review_qa, risk_recall_qa=stage2_role_prompt_contract_v2:risk_recall_qa`
- 목적: rolling validation 전체 샘플에서 deterministic/OpenAI/Gemini/multi-role 실행을 같은 지표로 비교한다.

## Sample Pool

| sample_rows | score_rows | policy                           | stage2_policy_version | prompt_contract_versions                                                                                                                                                                                                                                                                                                      | eval_years             | sample_counts_path                                                                                                                                           |
| ----------- | ---------- | -------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 75          | 2526       | feature46_full_review_trigger_73 | stage2_policy_v1      | {'quant_credit': 'stage2_role_prompt_contract_v2:quant_credit', 'evidence_audit': 'stage2_role_prompt_contract_v2:evidence_audit', 'chair_report': 'stage2_role_prompt_contract_v2:chair_report', 'review_qa': 'stage2_role_prompt_contract_v2:review_qa', 'risk_recall_qa': 'stage2_role_prompt_contract_v2:risk_recall_qa'} | 2019, 2020, 2021, 2022 | data/outputs/modeling/feature_46_xgboost/diagnostics/stage2_agents/feature46_full_review_trigger_73_harness/rolling_samples/stage2_harness_sample_counts.csv |

## Provider Summary

| run_id                  | run_status | skip_reason | rows | strict_success_rate | review_safe_success_rate | fn_rescue_success_rate | fp_over_hold_count | stage2_policy_version | stage2_latency_mean_seconds | stage2_latency_p95_seconds | stage2_latency_max_seconds | review_qa_trigger_rate | risk_recall_qa_trigger_rate | any_qa_trigger_rate | llm_cache_hit_rows | any_cache_hit_rows | run_failure_rows |
| ----------------------- | ---------- | ----------- | ---- | ------------------- | ------------------------ | ---------------------- | ------------------ | --------------------- | --------------------------- | -------------------------- | -------------------------- | ---------------------- | --------------------------- | ------------------- | ------------------ | ------------------ | ---------------- |
| deterministic           | completed  |             | 75   | 0.8533              | 0.9333                   | 0.6667                 | 23                 | stage2_policy_v1      |                             |                            |                            | 0.1200                 | 0.0267                      | 0.1467              | 0                  | 0                  | 0                |
| openai_gpt_4_1_mini     | completed  |             | 75   | 0.8533              | 0.9467                   | 0.7333                 | 23                 | stage2_policy_v1      | 17.9127                     | 29.2661                    | 38.8178                    | 0.1200                 | 0.0267                      | 0.1467              | 0                  | 0                  | 0                |
| gemini_gemini_2_5_flash | completed  |             | 75   | 0.8533              | 0.9467                   | 0.7333                 | 23                 | stage2_policy_v1      | 33.6834                     | 59.8867                    | 99.8311                    | 0.1200                 | 0.0267                      | 0.1467              | 0                  | 0                  | 0                |

## Category Summary

| run_id                  | sample_category                        | rows | strict_success_rate | review_safe_success_rate |
| ----------------------- | -------------------------------------- | ---- | ------------------- | ------------------------ |
| deterministic           | bbb_minus_bb_plus_boundary             | 15   | 1.0000              | 1.0000                   |
| deterministic           | fn_caught_by_stage2_review             | 15   | 0.6667              | 0.6667                   |
| deterministic           | fp_needing_committee_mitigation        | 15   | 1.0000              | 1.0000                   |
| deterministic           | true_negative_overescalation_guardrail | 15   | 0.6000              | 1.0000                   |
| deterministic           | true_positive_risk_explanation         | 15   | 1.0000              | 1.0000                   |
| openai_gpt_4_1_mini     | bbb_minus_bb_plus_boundary             | 15   | 1.0000              | 1.0000                   |
| openai_gpt_4_1_mini     | fn_caught_by_stage2_review             | 15   | 0.7333              | 0.7333                   |
| openai_gpt_4_1_mini     | fp_needing_committee_mitigation        | 15   | 1.0000              | 1.0000                   |
| openai_gpt_4_1_mini     | true_negative_overescalation_guardrail | 15   | 0.5333              | 1.0000                   |
| openai_gpt_4_1_mini     | true_positive_risk_explanation         | 15   | 1.0000              | 1.0000                   |
| gemini_gemini_2_5_flash | bbb_minus_bb_plus_boundary             | 15   | 1.0000              | 1.0000                   |
| gemini_gemini_2_5_flash | fn_caught_by_stage2_review             | 15   | 0.7333              | 0.7333                   |
| gemini_gemini_2_5_flash | fp_needing_committee_mitigation        | 15   | 1.0000              | 1.0000                   |
| gemini_gemini_2_5_flash | true_negative_overescalation_guardrail | 15   | 0.5333              | 1.0000                   |
| gemini_gemini_2_5_flash | true_positive_risk_explanation         | 15   | 1.0000              | 1.0000                   |

## Skipped Runs

_No rows._

## Output Directory

`/Users/inji/Documents/금융 데이터 분석/Project/Corporate-Analysis-System/data/outputs/modeling/feature_46_xgboost/diagnostics/stage2_agents/feature46_full_review_trigger_73_harness`

## Metric Notes

- strict success: 오류유형별 기대 최종 라벨을 엄격히 만족한 비율
- review-safe success: 정상기업을 부적격으로 악화시키지 않는 넓은 검토 안전 기준
- FN rescue 성공률: 1차 모델 false negative가 Stage2에서 보류/부적격으로 올라간 비율
- FP over-hold: 1차 모델 false positive가 Stage2에서 부적격은 피했지만 보류로 남은 건수
- latency p95: provider별 긴 꼬리 지연을 보기 위한 95백분위 실행 시간
- QA trigger rate: ReviewQA 또는 RiskRecallQA가 실제로 트리거된 행 비율
- cache hit: Stage2 본 실행, ReviewQA, RiskRecallQA cache hit 중 하나라도 켜진 행을 별도로 집계
