# Mixed Hard 40 Sample Selection

- 생성 파일: `committee_review_mixed_hard_40_timeout30_speed_gate_v3_samples.csv`
- committee_policy: `mixed_hard_40_timeout30_speed_gate_v3`
- 목적: FN recall, FP 완화, 경계등급 판단, TP 유지, TN 과잉 보류 방지를 한 번에 검증하는 40건 hard sample
- 구성: 각 sample_category 8건씩, 총 40건

## 구성 요약

| sample_category                        | model_error_type   |   rows |
|:---------------------------------------|:-------------------|-------:|
| bbb_minus_bb_plus_boundary             | false_positive     |      4 |
| bbb_minus_bb_plus_boundary             | true_positive      |      4 |
| fn_caught_by_stage2_review             | false_negative     |      8 |
| fp_needing_committee_mitigation        | false_positive     |      8 |
| true_negative_overescalation_guardrail | true_negative      |      8 |
| true_positive_risk_explanation         | true_positive      |      8 |

## 기업 목록

| sample_category                        | model_error_type   | corp_name              |   fiscal_year |   eval_year | actual_label_name   | model_predicted_label_name   | credit_rating   |   prob_speculative |   threshold |
|:---------------------------------------|:-------------------|:-----------------------|--------------:|------------:|:--------------------|:-----------------------------|:----------------|-------------------:|------------:|
| fn_caught_by_stage2_review             | false_negative     | (주)덱스터스튜디오     |          2021 |        2022 | 투기등급            | 투자적격                     | BB+             |           0.288843 |       0.31  |
| fn_caught_by_stage2_review             | false_negative     | (주)에스엠컬처앤콘텐츠 |          2021 |        2022 | 투기등급            | 투자적격                     | BB+             |           0.287783 |       0.31  |
| fn_caught_by_stage2_review             | false_negative     | (주)픽셀플러스         |          2020 |        2021 | 투기등급            | 투자적격                     | BB              |           0.301669 |       0.325 |
| fn_caught_by_stage2_review             | false_negative     | (주)솔디펜스           |          2020 |        2021 | 투기등급            | 투자적격                     | BB-             |           0.301669 |       0.325 |
| fn_caught_by_stage2_review             | false_negative     | 씨아이에스(주)         |          2020 |        2021 | 투기등급            | 투자적격                     | BB              |           0.301669 |       0.325 |
| fn_caught_by_stage2_review             | false_negative     | 아진전자부품(주)       |          2020 |        2021 | 투기등급            | 투자적격                     | BB              |           0.300167 |       0.325 |
| fn_caught_by_stage2_review             | false_negative     | 핸즈코퍼레이션(주)     |          2021 |        2022 | 투기등급            | 투자적격                     | BB+             |           0.284226 |       0.31  |
| fn_caught_by_stage2_review             | false_negative     | (주)아즈텍더블유비이   |          2020 |        2021 | 투기등급            | 투자적격                     | BB+             |           0.29873  |       0.325 |
| fp_needing_committee_mitigation        | false_positive     | (주)파세코             |          2021 |        2022 | 투자적격            | 투기등급                     | A-              |           0.317757 |       0.31  |
| fp_needing_committee_mitigation        | false_positive     | 현대에이치티(주)       |          2021 |        2022 | 투자적격            | 투기등급                     | A               |           0.317757 |       0.31  |
| fp_needing_committee_mitigation        | false_positive     | (주)한국큐빅           |          2022 |        2023 | 투자적격            | 투기등급                     | BBB+            |           0.262084 |       0.25  |
| fp_needing_committee_mitigation        | false_positive     | 다스코(주)             |          2022 |        2023 | 투자적격            | 투기등급                     | BBB             |           0.258313 |       0.25  |
| fp_needing_committee_mitigation        | false_positive     | 와이엠씨(주)           |          2019 |        2020 | 투자적격            | 투기등급                     | BBB+            |           0.23508  |       0.225 |
| fp_needing_committee_mitigation        | false_positive     | 제이엠티(주)           |          2019 |        2020 | 투자적격            | 투기등급                     | BBB-            |           0.234224 |       0.225 |
| fp_needing_committee_mitigation        | false_positive     | (주)아이즈비전         |          2019 |        2020 | 투자적격            | 투기등급                     | BBB             |           0.233606 |       0.225 |
| fp_needing_committee_mitigation        | false_positive     | 동일제강(주)           |          2019 |        2020 | 투자적격            | 투기등급                     | BBB+            |           0.229056 |       0.225 |
| bbb_minus_bb_plus_boundary             | false_positive     | (주)포바이포           |          2022 |        2023 | 투자적격            | 투기등급                     | BBB-            |           0.90436  |       0.25  |
| bbb_minus_bb_plus_boundary             | false_positive     | 솔트웨어(주)           |          2022 |        2023 | 투자적격            | 투기등급                     | BBB-            |           0.90848  |       0.25  |
| bbb_minus_bb_plus_boundary             | false_positive     | (주)제닉               |          2019 |        2020 | 투자적격            | 투기등급                     | BBB-            |           0.891447 |       0.225 |
| bbb_minus_bb_plus_boundary             | false_positive     | (주)제닉               |          2021 |        2022 | 투자적격            | 투기등급                     | BBB-            |           0.949199 |       0.31  |
| bbb_minus_bb_plus_boundary             | true_positive      | (주)바른손             |          2020 |        2021 | 투기등급            | 투기등급                     | BB+             |           0.972037 |       0.325 |
| bbb_minus_bb_plus_boundary             | true_positive      | (주)제닉               |          2022 |        2023 | 투기등급            | 투기등급                     | BB+             |           0.899962 |       0.25  |
| bbb_minus_bb_plus_boundary             | true_positive      | (주)아이윈             |          2019 |        2020 | 투기등급            | 투기등급                     | BB+             |           0.878189 |       0.225 |
| bbb_minus_bb_plus_boundary             | true_positive      | 씨에스베어링(주)       |          2022 |        2023 | 투기등급            | 투기등급                     | BB+             |           0.922837 |       0.25  |
| true_positive_risk_explanation         | true_positive      | 휴림로봇(주)           |          2021 |        2022 | 투기등급            | 투기등급                     | B+              |           0.985682 |       0.31  |
| true_positive_risk_explanation         | true_positive      | 엔시트론(주)           |          2020 |        2021 | 투기등급            | 투기등급                     | B-              |           0.984717 |       0.325 |
| true_positive_risk_explanation         | true_positive      | (주)티에스트릴리온     |          2022 |        2023 | 투기등급            | 투기등급                     | B+              |           0.984603 |       0.25  |
| true_positive_risk_explanation         | true_positive      | (주)에스디생명공학     |          2022 |        2023 | 투기등급            | 투기등급                     | BB              |           0.984226 |       0.25  |
| true_positive_risk_explanation         | true_positive      | (주)국보               |          2021 |        2022 | 투기등급            | 투기등급                     | B               |           0.983267 |       0.31  |
| true_positive_risk_explanation         | true_positive      | (주)와이투솔루션       |          2020 |        2021 | 투기등급            | 투기등급                     | CCC             |           0.982804 |       0.325 |
| true_positive_risk_explanation         | true_positive      | (주)케스피온           |          2022 |        2023 | 투기등급            | 투기등급                     | BB-             |           0.982113 |       0.25  |
| true_positive_risk_explanation         | true_positive      | (주)아티스트컴퍼니     |          2022 |        2023 | 투기등급            | 투기등급                     | B               |           0.982019 |       0.25  |
| true_negative_overescalation_guardrail | true_negative      | 청광건설(주)           |          2019 |        2020 | 투자적격            | 투자적격                     | BBB+            |           0.219776 |       0.225 |
| true_negative_overescalation_guardrail | true_negative      | (주)피제이전자         |          2019 |        2020 | 투자적격            | 투자적격                     | A+              |           0.218459 |       0.225 |
| true_negative_overescalation_guardrail | true_negative      | (주)소프트센           |          2019 |        2020 | 투자적격            | 투자적격                     | BBB             |           0.215034 |       0.225 |
| true_negative_overescalation_guardrail | true_negative      | (주)엔에프씨           |          2020 |        2021 | 투자적격            | 투자적격                     | BBB             |           0.314237 |       0.325 |
| true_negative_overescalation_guardrail | true_negative      | (주)하나투어           |          2022 |        2023 | 투자적격            | 투자적격                     | BBB-            |           0.235341 |       0.25  |
| true_negative_overescalation_guardrail | true_negative      | (주)예림당             |          2019 |        2020 | 투자적격            | 투자적격                     | BBB             |           0.222651 |       0.225 |
| true_negative_overescalation_guardrail | true_negative      | (주)일지테크           |          2019 |        2020 | 투자적격            | 투자적격                     | BBB             |           0.219093 |       0.225 |
| true_negative_overescalation_guardrail | true_negative      | (주)아시아경제         |          2022 |        2023 | 투자적격            | 투자적격                     | BBB             |           0.242221 |       0.25  |
