# Official 43-Feature Error Deep Dive

공식 43개 XGBoost 모델의 test 구간 오답을 중심으로 시장/산업/기업규모/연도별 취약 구간을 진단했습니다.
이 리포트는 새 변수를 바로 추가하기보다, 어떤 구간에서 어떤 방식의 보완이 필요한지 찾기 위한 자료입니다.

## 1. Overall Test Performance

- Rows/positive rate: `924` / `22.0%`
- PR-AUC/ROC-AUC: `0.8329` / `0.9415`
- Precision/Recall/F1: `0.7004` / `0.8522` / `0.7689`
- FP/FN: `74` / `30`

## 2. Market Split

| Market | Rows | Pos rate | PR-AUC | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KOSDAQ | 427 | 38.2% | 0.8427 | 0.7113 | 0.8466 | 0.7731 | 56 | 25 |
| KOSPI | 497 | 8.0% | 0.7908 | 0.6604 | 0.8750 | 0.7527 | 18 | 5 |

## 3. Rating Boundary Split

`/Users/inji/Documents/금융 데이터 분석/Project/Corporate-Analysis-System/data/evaluation/target_label_reference.csv`에서 대표 신용등급을 붙여 경계등급 분석을 수행했습니다. test rows 중 등급이 매칭된 행은 `916`개입니다. 이 등급 정보는 모델 학습에는 쓰지 않고 diagnostics 전용으로만 사용합니다.

BBB-/BB+ 주변은 투자적격과 투기등급이 갈리는 경계라, 모델의 객관적 평가 근거로 따로 보는 것이 좋습니다.

| Boundary group | Rows | Pos rate | PR-AUC | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deep_speculative_B_plus_or_lower | 66 | 98.5% | 1.0000 | 1.0000 | 0.9846 | 0.9922 | 0 | 1 |
| missing_rating | 8 | 0.0% | - | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| near_investment_BBB_plus_to_BBB_minus | 260 | 0.0% | - | 0.0000 | 0.0000 | 0.0000 | 65 | 0 |
| near_speculative_BB_plus_to_BB_minus | 140 | 98.6% | 0.9990 | 1.0000 | 0.7899 | 0.8826 | 0 | 29 |
| upper_investment_A_or_above | 450 | 0.0% | - | 0.0000 | 0.0000 | 0.0000 | 9 | 0 |

### Exact BBB-/BB+ Boundary

| BBB-/BB+ | Rows | Pos rate | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| False | 804 | 18.3% | 0.7253 | 0.8980 | 0.8024 | 50 | 15 |
| True | 120 | 46.7% | 0.6308 | 0.7321 | 0.6777 | 24 | 15 |

### Individual Credit Ratings

| Rating | Rows | Pos rate | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 108 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 2 | 0 |
| A+ | 101 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 1 | 0 |
| A- | 71 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 6 | 0 |
| AA | 43 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| AA+ | 22 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| AA- | 93 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| AAA | 12 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| B | 19 | 100.0% | 1.0000 | 0.9474 | 0.9730 | 0 | 1 |
| B+ | 23 | 95.7% | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| B- | 17 | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| BB | 54 | 100.0% | 1.0000 | 0.7593 | 0.8632 | 0 | 13 |
| BB+ | 58 | 96.6% | 1.0000 | 0.7321 | 0.8454 | 0 | 15 |
| BB- | 28 | 100.0% | 1.0000 | 0.9643 | 0.9818 | 0 | 1 |
| BBB | 94 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 25 | 0 |
| BBB+ | 104 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 16 | 0 |
| BBB- | 62 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 24 | 0 |
| C | 3 | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| CCC | 3 | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| D | 1 | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
|  | 8 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

### Rating Agency Group

| Agency group | Rows | Pos rate | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BIG3 | 482 | 12.4% | 0.7656 | 0.8167 | 0.7903 | 15 | 11 |
| FOREIGN | 3 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| OTHER | 431 | 33.2% | 0.6776 | 0.8671 | 0.7607 | 59 | 19 |
| nan | 8 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

## 4. Weak Recall Segments

실제 투기등급 중 놓친 비율이 높은 구간입니다. positive 표본이 너무 작은 구간은 제외했습니다.

| Industry | Rows | Pos | FN | FN rate | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| it_services | 176 | 21 | 5 | 23.8% | 0.7619 | 0.6957 |
| construction | 44 | 6 | 1 | 16.7% | 0.8333 | 0.7143 |
| manufacturing | 598 | 162 | 24 | 14.8% | 0.8519 | 0.7709 |
| wholesale_retail | 63 | 14 | 0 | 0.0% | 1.0000 | 0.9333 |

| Firm size | Rows | Pos | FN | FN rate | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mid_sized | 434 | 95 | 27 | 28.4% | 0.7158 | 0.7273 |
| small_and_medium | 232 | 107 | 3 | 2.8% | 0.9720 | 0.8157 |

| Fiscal year | Rows | Pos | FN | FN rate | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | 287 | 42 | 7 | 16.7% | 0.8333 | 0.8140 |
| 2023 | 637 | 161 | 23 | 14.3% | 0.8571 | 0.7582 |

## 5. False Positive Concentration

전체 FP 중 비중이 큰 산업 구간입니다. FP가 몰리는 곳은 threshold/Stage 2 과민경고 필터를 우선 검토합니다.

| Industry | Rows | Neg | FP | FP share | FP rate | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| manufacturing | 598 | 436 | 58 | 78.4% | 13.3% | 0.7041 |
| it_services | 176 | 155 | 9 | 12.2% | 5.8% | 0.6400 |
| construction | 44 | 38 | 3 | 4.1% | 7.9% | 0.6250 |
| wholesale_retail | 63 | 49 | 2 | 2.7% | 4.1% | 0.8750 |
| other | 22 | 22 | 1 | 1.4% | 4.5% | 0.0000 |
| transport_storage | 21 | 21 | 1 | 1.4% | 4.8% | 0.0000 |

## 6. Cross-Segment Error Concentration

| Dimension | Segment | Rows | FP | FN | Error count | FP share | FN share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| market_x_industry | KOSDAQ / manufacturing | 302 | 43 | 20 | 63 | 58.1% | 66.7% |
| market_x_firm_size | KOSDAQ / small_and_medium | 223 | 44 | 3 | 47 | 59.5% | 10.0% |
| market_x_firm_size | KOSDAQ / mid_sized | 184 | 12 | 22 | 34 | 16.2% | 73.3% |
| market_x_industry | KOSPI / manufacturing | 296 | 15 | 4 | 19 | 20.3% | 13.3% |
| market_x_firm_size | KOSPI / mid_sized | 250 | 12 | 5 | 17 | 16.2% | 16.7% |
| market_x_industry | KOSDAQ / it_services | 82 | 9 | 5 | 14 | 12.2% | 16.7% |
| market_x_firm_size | KOSPI / large | 237 | 6 | 0 | 6 | 8.1% | 0.0% |
| market_x_industry | KOSPI / construction | 29 | 2 | 1 | 3 | 2.7% | 3.3% |
| market_x_industry | KOSDAQ / wholesale_retail | 22 | 2 | 0 | 2 | 2.7% | 0.0% |
| market_x_industry | KOSDAQ / construction | 15 | 1 | 0 | 1 | 1.4% | 0.0% |
| market_x_industry | KOSDAQ / other | 4 | 1 | 0 | 1 | 1.4% | 0.0% |
| market_x_industry | KOSPI / transport_storage | 19 | 1 | 0 | 1 | 1.4% | 0.0% |
| market_x_industry | KOSDAQ / transport_storage | 2 | 0 | 0 | 0 | 0.0% | 0.0% |
| market_x_industry | KOSPI / it_services | 94 | 0 | 0 | 0 | 0.0% | 0.0% |
| market_x_industry | KOSPI / other | 18 | 0 | 0 | 0 | 0.0% | 0.0% |

## 7. Feature Profile: FN vs TP

FN은 실제 투기등급인데 모델이 안정적으로 본 사례입니다. 아래 변수 차이는 모델이 위험을 낮게 본 이유를 찾는 데 사용합니다.

| Feature | FN median | TP median | Std delta | FN miss | TP miss |
| --- | ---: | ---: | ---: | ---: | ---: |
| net_margin | 0.0270 | -0.0729 | 1.3686 | 0.0% | 0.0% |
| pretax_roe | 0.0661 | -0.0991 | 1.2837 | 0.0% | 0.0% |
| pretax_roa | 0.0336 | -0.0518 | 1.2173 | 0.0% | 0.0% |
| icr_under_1 | 0.0000 | 1.0000 | -1.0000 | 0.0% | 0.0% |
| dividend_payer | 1.0000 | 0.0000 | 1.0000 | 0.0% | 0.0% |
| operating_roa | 0.0264 | -0.0204 | 0.8896 | 0.0% | 0.0% |
| ocf_to_sales | 0.0656 | 0.0025 | 0.6106 | 0.0% | 0.0% |
| ocf_to_total_borrowings | 0.3581 | 0.0183 | 0.5804 | 3.3% | 7.5% |
| capital_impairment_ratio | -21.1356 | -3.6300 | -0.5647 | 0.0% | 0.0% |
| market_to_book | 0.7188 | 1.2655 | -0.5408 | 6.7% | 14.5% |
| ocf_to_total_liabilities | 0.1048 | 0.0033 | 0.5100 | 0.0% | 0.0% |
| interest_coverage_ratio | 2.3524 | -0.7960 | 0.4603 | 0.0% | 0.0% |

## 8. Feature Profile: FP vs TN

FP는 실제 투자적격인데 모델이 위험하다고 본 사례입니다. 아래 변수 차이는 과민경고 원인을 찾는 데 사용합니다.

| Feature | FP median | TN median | Std delta | FP miss | TN miss |
| --- | ---: | ---: | ---: | ---: | ---: |
| icr_under_1 | 1.0000 | 0.0000 | 1.0000 | 0.0% | 0.0% |
| dividend_payer | 0.0000 | 1.0000 | -1.0000 | 0.0% | 0.0% |
| operating_roa | -0.0015 | 0.0409 | -0.8066 | 1.4% | 0.0% |
| ocf_to_sales | 0.0074 | 0.0791 | -0.6930 | 1.4% | 0.2% |
| ocf_to_total_liabilities | 0.0130 | 0.1363 | -0.6195 | 1.4% | 0.0% |
| pretax_roe | -0.0005 | 0.0759 | -0.5930 | 1.4% | 0.0% |
| interest_coverage_ratio | -0.0750 | 3.8056 | -0.5673 | 0.0% | 0.0% |
| ocf_to_total_borrowings | -0.0142 | 0.3115 | -0.5563 | 14.9% | 11.9% |
| pretax_roa | -0.0022 | 0.0369 | -0.5563 | 1.4% | 0.0% |
| short_term_borrowings_share | 0.7693 | 0.4446 | 0.5388 | 14.9% | 11.9% |
| cashflow_coverage_ratio | 0.4221 | 5.4631 | -0.5122 | 0.0% | 0.0% |
| capital_impairment_ratio | -5.8012 | -21.1634 | 0.4956 | 1.4% | 0.2% |

## 9. Threshold Distance

threshold 바로 근처 오류는 threshold 조정으로 개선 여지가 있고, 멀리 떨어진 고확신 오류는 변수/외부근거 보완이 더 필요합니다.

| Distance | Rows | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <=0.02 | 15 | 5 | 0 | 0.4444 | 1.0000 | 0.6154 |
| 0.02-0.05 | 23 | 6 | 5 | 0.4545 | 0.5000 | 0.4762 |
| 0.05-0.10 | 30 | 9 | 5 | 0.1000 | 0.1667 | 0.1250 |
| 0.10-0.20 | 92 | 13 | 6 | 0.6176 | 0.7778 | 0.6885 |
| >0.20 | 764 | 41 | 14 | 0.7760 | 0.9103 | 0.8378 |

## 10. High-Confidence Error Examples

확률상 모델이 자신 있게 틀린 사례입니다. 이 기업들은 뉴스/공시/등급전망 확인 우선순위가 높습니다.

### False Negative

| Company | Market | FY | Industry | Size | Prob | Threshold |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 케이지모빌리티(주) | KOSPI | 2,024 | manufacturing | mid_sized | 0.0183 | 0.3200 |
| 케이지모빌리티(주) | KOSPI | 2,023 | manufacturing | mid_sized | 0.0232 | 0.3200 |
| (주)엑시콘 | KOSDAQ | 2,023 | manufacturing | mid_sized | 0.0250 | 0.3200 |
| (주)톱텍 | KOSDAQ | 2,024 | manufacturing | mid_sized | 0.0327 | 0.3200 |
| 아진산업(주) | KOSDAQ | 2,024 | manufacturing | mid_sized | 0.0341 | 0.3200 |
| 제이엠티(주) | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.0374 | 0.3200 |
| (주)디알비동일 | KOSPI | 2,023 | manufacturing | mid_sized | 0.0421 | 0.3200 |
| (주)서진시스템 | KOSDAQ | 2,024 | manufacturing | mid_sized | 0.0501 | 0.3200 |
| 삼보모터스(주) | KOSDAQ | 2,023 | manufacturing | mid_sized | 0.0536 | 0.3200 |
| 케이지케미칼(주) | KOSPI | 2,023 | manufacturing | mid_sized | 0.0557 | 0.3200 |

### False Positive

| Company | Market | FY | Industry | Size | Prob | Threshold |
| --- | --- | ---: | --- | --- | ---: | ---: |
| (주)라닉스 | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.9715 | 0.3200 |
| 한국맥널티(주) | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.8995 | 0.3200 |
| 유니온머티리얼(주) | KOSPI | 2,023 | manufacturing | mid_sized | 0.8687 | 0.3200 |
| (주)웹스 | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.8567 | 0.3200 |
| (주)푸드나무 | KOSDAQ | 2,023 | wholesale_retail | small_and_medium | 0.8513 | 0.3200 |
| 알에스오토메이션(주) | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.8178 | 0.3200 |
| 한국맥널티(주) | KOSDAQ | 2,024 | manufacturing | small_and_medium | 0.7965 | 0.3200 |
| 브이엠(주) | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.7915 | 0.3200 |
| (주)에스에너지 | KOSDAQ | 2,023 | manufacturing | mid_sized | 0.7723 | 0.3200 |
| (주)엑스페릭스 | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.7711 | 0.3200 |

## 11. Rating Boundary Availability

`/Users/inji/Documents/금융 데이터 분석/Project/Corporate-Analysis-System/data/evaluation/target_label_reference.csv`에서 대표 신용등급을 붙여 경계등급 분석을 수행했습니다. test rows 중 등급이 매칭된 행은 `916`개입니다. 이 등급 정보는 모델 학습에는 쓰지 않고 diagnostics 전용으로만 사용합니다.

## 12. What To Do Next

1. FN이 몰린 산업/규모/연도 조합에서 실제 외부 이벤트가 있었는지 뉴스·공시·등급전망을 먼저 확인합니다.
2. FP가 많은 KOSDAQ/제조업/소형 구간은 전체 threshold를 바꾸기보다 Stage 2 과민경고 필터 또는 구간별 보조 판단으로 완화합니다.
3. FN과 TP의 차이가 큰 변수는 '위험을 숨기는 안정 신호'인지 확인합니다. 특히 규모성/배당/상장연도/절대금액 신호가 실제 위험을 가리는지 봅니다.
4. FP와 TN의 차이가 큰 변수는 단기 악화와 지속 악화를 구분하는 추가 변수 후보로 바꿔봅니다. 예: 2~3년 지속 손실, 현금흐름 악화 지속기간, 운전자본 변화 지속성.
5. BBB-/BB+ 경계 성능은 별도 객관 평가표로 발표 자료에 포함합니다.
