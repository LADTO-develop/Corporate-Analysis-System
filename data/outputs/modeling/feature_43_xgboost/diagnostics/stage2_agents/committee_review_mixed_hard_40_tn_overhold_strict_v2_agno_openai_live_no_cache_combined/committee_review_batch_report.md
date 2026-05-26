# Committee Review Batch Results

- Rows: 40
- Strict committee success rate: 97.5%
- Review-safe success rate: 100.0%
- Combined from full 40 run plus one-row 제닉 2021 retry
- Stage 2 LLM speed: mean `25.8937` sec, max `61.6745` sec, cache hits `0`

## Result Preview

| sample_category | corp_name | actual_label_name | model_predicted_label_name | final_committee_label | committee_effect | review_safe_effect | evidence_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fn_caught_by_stage2_review | (주)픽셀플러스 | 투기등급 | 투자적격 | 보류 | fn_escalated | review_safe_fn_escalated | ready |
| fn_caught_by_stage2_review | (주)솔디펜스 | 투기등급 | 투자적격 | 보류 | fn_escalated | review_safe_fn_escalated | ready |
| fn_caught_by_stage2_review | 씨아이에스(주) | 투기등급 | 투자적격 | 보류 | fn_escalated | review_safe_fn_escalated | ready |
| fn_caught_by_stage2_review | 아진전자부품(주) | 투기등급 | 투자적격 | 보류 | fn_escalated | review_safe_fn_escalated | ready |
| fn_caught_by_stage2_review | (주)아즈텍더블유비이 | 투기등급 | 투자적격 | 보류 | fn_escalated | review_safe_fn_escalated | ready |
| fn_caught_by_stage2_review | (주)덱스터스튜디오 | 투기등급 | 투자적격 | 보류 | fn_escalated | review_safe_fn_escalated | ready |
| fn_caught_by_stage2_review | (주)에스엠컬처앤콘텐츠 | 투기등급 | 투자적격 | 보류 | fn_escalated | review_safe_fn_escalated | ready |
| fn_caught_by_stage2_review | 핸즈코퍼레이션(주) | 투기등급 | 투자적격 | 보류 | fn_escalated | review_safe_fn_escalated | ready |
| fp_needing_committee_mitigation | (주)파세코 | 투자적격 | 투기등급 | 보류 | fp_mitigated | review_safe_fp_not_rejected | ready |
| fp_needing_committee_mitigation | 현대에이치티(주) | 투자적격 | 투기등급 | 보류 | fp_mitigated | review_safe_fp_not_rejected | ready |
| fp_needing_committee_mitigation | (주)한국큐빅 | 투자적격 | 투기등급 | 보류 | fp_mitigated | review_safe_fp_not_rejected | ready |
| fp_needing_committee_mitigation | 다스코(주) | 투자적격 | 투기등급 | 보류 | fp_mitigated | review_safe_fp_not_rejected | ready |
| fp_needing_committee_mitigation | 와이엠씨(주) | 투자적격 | 투기등급 | 보류 | fp_mitigated | review_safe_fp_not_rejected | ready |
| fp_needing_committee_mitigation | 제이엠티(주) | 투자적격 | 투기등급 | 보류 | fp_mitigated | review_safe_fp_not_rejected | ready |
| fp_needing_committee_mitigation | (주)아이즈비전 | 투자적격 | 투기등급 | 보류 | fp_mitigated | review_safe_fp_not_rejected | ready |
| fp_needing_committee_mitigation | 동일제강(주) | 투자적격 | 투기등급 | 보류 | fp_mitigated | review_safe_fp_not_rejected | ready |
| bbb_minus_bb_plus_boundary | (주)제닉 | 투자적격 | 투기등급 | 보류 | fp_mitigated | review_safe_fp_not_rejected | ready |
| bbb_minus_bb_plus_boundary | 솔트웨어(주) | 투자적격 | 투기등급 | 보류 | fp_mitigated | review_safe_fp_not_rejected | ready |
| bbb_minus_bb_plus_boundary | (주)포바이포 | 투자적격 | 투기등급 | 보류 | fp_mitigated | review_safe_fp_not_rejected | ready |
| bbb_minus_bb_plus_boundary | (주)제닉 | 투자적격 | 투기등급 | 보류 | fp_mitigated | review_safe_fp_not_rejected | ready |
| bbb_minus_bb_plus_boundary | (주)바른손 | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | review_safe_tp_supported | ready |
| bbb_minus_bb_plus_boundary | 씨에스베어링(주) | 투기등급 | 투기등급 | 보류 | tp_risk_supported | review_safe_tp_supported | ready |
| bbb_minus_bb_plus_boundary | (주)제닉 | 투기등급 | 투기등급 | 보류 | tp_risk_supported | review_safe_tp_supported | ready |
| bbb_minus_bb_plus_boundary | (주)아이윈 | 투기등급 | 투기등급 | 보류 | tp_risk_supported | review_safe_tp_supported | ready |
| true_positive_risk_explanation | 휴림로봇(주) | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | review_safe_tp_supported | ready |
| true_positive_risk_explanation | 엔시트론(주) | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | review_safe_tp_supported | ready |
| true_positive_risk_explanation | (주)티에스트릴리온 | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | review_safe_tp_supported | ready |
| true_positive_risk_explanation | (주)에스디생명공학 | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | review_safe_tp_supported | ready |
| true_positive_risk_explanation | (주)국보 | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | review_safe_tp_supported | ready |
| true_positive_risk_explanation | (주)와이투솔루션 | 투기등급 | 투기등급 | 부적격 | tp_risk_supported | review_safe_tp_supported | ready |

## Remaining Strict Miss

- (주)하나투어 2022: 투자적격 / model 투자적격 / final 보류 (risk_hold)
