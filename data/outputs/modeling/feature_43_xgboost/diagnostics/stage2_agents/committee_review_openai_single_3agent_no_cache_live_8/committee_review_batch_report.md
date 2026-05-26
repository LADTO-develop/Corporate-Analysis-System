# Committee Review Batch Results

- Rows: 8
- Strict committee success rate: 87.5%
- Review-safe success rate: 100.0%
- Speed: wall `67.6725` sec, mean case `16.8457` sec, throughput `7.093` cases/min
- Stage 2 LLM speed: mean `16.4786` sec, max `19.8059` sec, cache hits `0`

## Result Preview

| sample_category | corp_name | prior_credit_rating | actual_label_name | model_predicted_label_name | final_committee_label | committee_effect | evidence_status | evidence_items |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fn_caught_by_stage2_review | (주)예선테크 |  | 투기등급 | 투자적격 | 보류 | fn_escalated | ready | 6 |
| fn_caught_by_stage2_review | 명신산업(주) |  | 투기등급 | 투자적격 | 보류 | fn_escalated | ready | 6 |
| fp_needing_committee_mitigation | (주)예림당 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | ready | 6 |
| bbb_minus_bb_plus_boundary | (주)라닉스 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | ready | 8 |
| true_positive_risk_explanation | (주)대창솔루션 |  | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | ready | 6 |
| true_negative_overescalation_guardrail | (주)데이타솔루션 |  | 투자적격 | 투자적격 | 적격 | tn_kept_eligible | ready | 6 |
| true_negative_overescalation_guardrail | (주)휴맥스 |  | 투자적격 | 투자적격 | 보류 | tn_escalated | ready | 6 |
| true_negative_overescalation_guardrail | (주)동성화인텍 |  | 투자적격 | 투자적격 | 적격 | tn_kept_eligible | ready | 6 |
