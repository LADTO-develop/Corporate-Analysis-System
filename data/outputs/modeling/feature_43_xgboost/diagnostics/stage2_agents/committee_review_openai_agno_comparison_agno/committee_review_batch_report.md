# Committee Review Batch Results

- Rows: 4
- Strict committee success rate: 100.0%
- Review-safe success rate: 100.0%
- Speed: wall `0.9084` sec, mean case `0.2269` sec, throughput `264.2008` cases/min

## Result Preview

| sample_category | corp_name | prior_credit_rating | actual_label_name | model_predicted_label_name | final_committee_label | committee_effect | evidence_status | evidence_items |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fn_caught_by_stage2_review | (주)이수앱지스 |  | 투기등급 | 투자적격 | 보류 | fn_escalated | disabled | 0 |
| fp_needing_committee_mitigation | (주)타이거일렉 |  | 투자적격 | 투기등급 | 보류 | fp_mitigated | disabled | 0 |
| bbb_minus_bb_plus_boundary | (주)엠젠솔루션 |  | 투기등급 | 투기등급 | 보류 | tp_risk_supported | disabled | 0 |
| true_positive_risk_explanation | (주)플라즈맵 |  | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | disabled | 0 |
