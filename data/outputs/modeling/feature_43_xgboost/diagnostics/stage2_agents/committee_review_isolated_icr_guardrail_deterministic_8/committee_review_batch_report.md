# Committee Review Batch Results

- Rows: 8
- Strict committee success rate: 87.5%
- Review-safe success rate: 100.0%
- Speed: wall `1.8684` sec, mean case `0.4666` sec, throughput `256.9043` cases/min

## Result Preview

| sample_category | corp_name | prior_credit_rating | actual_label_name | model_predicted_label_name | final_committee_label | committee_effect | evidence_status | evidence_items |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fn_caught_by_stage2_review | (주)예선테크 |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 |
| fn_caught_by_stage2_review | 명신산업(주) |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 |
| fp_needing_committee_mitigation | (주)예림당 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 |
| bbb_minus_bb_plus_boundary | (주)라닉스 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 |
| true_positive_risk_explanation | (주)대창솔루션 |  | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | disabled | 0 |
| true_negative_overescalation_guardrail | (주)데이타솔루션 |  | 투자적격 | 투자적격 | 적격 | tn_kept_eligible | disabled | 0 |
| true_negative_overescalation_guardrail | (주)휴맥스 |  | 투자적격 | 투자적격 | 보류 | tn_escalated | disabled | 0 |
| true_negative_overescalation_guardrail | (주)동성화인텍 |  | 투자적격 | 투자적격 | 적격 | tn_kept_eligible | disabled | 0 |
