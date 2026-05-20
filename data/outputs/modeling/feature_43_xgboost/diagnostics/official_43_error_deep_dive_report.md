# Official 43-Feature Error Deep Dive

공식 43개 XGBoost 모델의 test 구간 오답을 중심으로 시장/산업/기업규모/연도별 취약 구간을 진단했습니다.
이 리포트는 새 변수를 바로 추가하기보다, 어떤 구간에서 어떤 방식의 보완이 필요한지 찾기 위한 자료입니다.

## 1. Overall Test Performance

- Rows/positive rate: `924` / `22.0%`
- PR-AUC/ROC-AUC: `0.7930` / `0.9286`
- Precision/Recall/F1: `0.6603` / `0.8522` / `0.7441`
- FP/FN: `89` / `30`

## 2. Market Split

| Market | Rows | Pos rate | PR-AUC | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KOSDAQ | 427 | 38.2% | 0.8004 | 0.6587 | 0.8405 | 0.7385 | 71 | 26 |
| KOSPI | 497 | 8.0% | 0.7670 | 0.6667 | 0.9000 | 0.7660 | 18 | 4 |

## 3. Rating Boundary Split

`/Users/inji/Documents/금융 데이터 분석/Project/Corporate-Analysis-System/data/evaluation/target_label_reference.csv`에서 대표 신용등급을 붙여 경계등급 분석을 수행했습니다. test rows 중 등급이 매칭된 행은 `916`개입니다. 이 등급 정보는 모델 학습에는 쓰지 않고 diagnostics 전용으로만 사용합니다.

BBB-/BB+ 주변은 투자적격과 투기등급이 갈리는 경계라, 모델의 객관적 평가 근거로 따로 보는 것이 좋습니다.

| Boundary group | Rows | Pos rate | PR-AUC | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deep_speculative_B_plus_or_lower | 66 | 98.5% | 1.0000 | 1.0000 | 0.9538 | 0.9764 | 0 | 3 |
| missing_rating | 8 | 0.0% | - | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| near_investment_BBB_plus_to_BBB_minus | 260 | 0.0% | - | 0.0000 | 0.0000 | 0.0000 | 71 | 0 |
| near_speculative_BB_plus_to_BB_minus | 140 | 98.6% | 0.9991 | 1.0000 | 0.8043 | 0.8916 | 0 | 27 |
| upper_investment_A_or_above | 450 | 0.0% | - | 0.0000 | 0.0000 | 0.0000 | 18 | 0 |

### Exact BBB-/BB+ Boundary

| BBB-/BB+ | Rows | Pos rate | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| False | 804 | 18.3% | 0.6649 | 0.8776 | 0.7566 | 65 | 18 |
| True | 120 | 46.7% | 0.6471 | 0.7857 | 0.7097 | 24 | 12 |

### Individual Credit Ratings

| Rating | Rows | Pos rate | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 108 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 9 | 0 |
| A+ | 101 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 2 | 0 |
| A- | 71 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |
| AA | 43 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| AA+ | 22 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| AA- | 93 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| AAA | 12 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| B | 19 | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| B+ | 23 | 95.7% | 1.0000 | 0.8636 | 0.9268 | 0 | 3 |
| B- | 17 | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| BB | 54 | 100.0% | 1.0000 | 0.7593 | 0.8632 | 0 | 13 |
| BB+ | 58 | 96.6% | 1.0000 | 0.7857 | 0.8800 | 0 | 12 |
| BB- | 28 | 100.0% | 1.0000 | 0.9286 | 0.9630 | 0 | 2 |
| BBB | 94 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 23 | 0 |
| BBB+ | 104 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 24 | 0 |
| BBB- | 62 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 24 | 0 |
| C | 3 | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| CCC | 3 | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| D | 1 | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
|  | 8 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

### Rating Agency Group

| Agency group | Rows | Pos rate | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BIG3 | 482 | 12.4% | 0.7353 | 0.8333 | 0.7812 | 18 | 10 |
| FOREIGN | 3 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| OTHER | 431 | 33.2% | 0.6340 | 0.8601 | 0.7300 | 71 | 20 |
| nan | 8 | 0.0% | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

## 4. Weak Recall Segments

실제 투기등급 중 놓친 비율이 높은 구간입니다. positive 표본이 너무 작은 구간은 제외했습니다.

| Industry | Rows | Pos | FN | FN rate | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| manufacturing | 598 | 162 | 26 | 16.0% | 0.8395 | 0.7473 |
| it_services | 176 | 21 | 3 | 14.3% | 0.8571 | 0.6545 |
| wholesale_retail | 63 | 14 | 1 | 7.1% | 0.9286 | 0.8387 |
| construction | 44 | 6 | 0 | 0.0% | 1.0000 | 0.8571 |

| Firm size | Rows | Pos | FN | FN rate | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mid_sized | 434 | 95 | 25 | 26.3% | 0.7368 | 0.7216 |
| small_and_medium | 232 | 107 | 5 | 4.7% | 0.9533 | 0.7698 |

| Fiscal year | Rows | Pos | FN | FN rate | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | 287 | 42 | 8 | 19.0% | 0.8095 | 0.8000 |
| 2023 | 637 | 161 | 22 | 13.7% | 0.8634 | 0.7316 |

## 5. False Positive Concentration

전체 FP 중 비중이 큰 산업 구간입니다. FP가 몰리는 곳은 threshold/Stage 2 과민경고 필터를 우선 검토합니다.

| Industry | Rows | Neg | FP | FP share | FP rate | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| manufacturing | 598 | 436 | 66 | 74.2% | 15.1% | 0.6733 |
| it_services | 176 | 155 | 16 | 18.0% | 10.3% | 0.5294 |
| wholesale_retail | 63 | 49 | 4 | 4.5% | 8.2% | 0.7647 |
| construction | 44 | 38 | 2 | 2.2% | 5.3% | 0.7500 |
| other | 22 | 22 | 1 | 1.1% | 4.5% | 0.0000 |
| transport_storage | 21 | 21 | 0 | 0.0% | 0.0% | 0.0000 |

## 6. Cross-Segment Error Concentration

| Dimension | Segment | Rows | FP | FN | Error count | FP share | FN share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| market_x_industry | KOSDAQ / manufacturing | 302 | 51 | 22 | 73 | 57.3% | 73.3% |
| market_x_firm_size | KOSDAQ / small_and_medium | 223 | 56 | 5 | 61 | 62.9% | 16.7% |
| market_x_firm_size | KOSDAQ / mid_sized | 184 | 15 | 21 | 36 | 16.9% | 70.0% |
| market_x_industry | KOSPI / manufacturing | 296 | 15 | 4 | 19 | 16.9% | 13.3% |
| market_x_industry | KOSDAQ / it_services | 82 | 16 | 3 | 19 | 18.0% | 10.0% |
| market_x_firm_size | KOSPI / mid_sized | 250 | 14 | 4 | 18 | 15.7% | 13.3% |
| market_x_industry | KOSDAQ / wholesale_retail | 22 | 3 | 1 | 4 | 3.4% | 3.3% |
| market_x_firm_size | KOSPI / large | 237 | 4 | 0 | 4 | 4.5% | 0.0% |
| market_x_industry | KOSPI / construction | 29 | 2 | 0 | 2 | 2.2% | 0.0% |
| market_x_industry | KOSDAQ / other | 4 | 1 | 0 | 1 | 1.1% | 0.0% |
| market_x_industry | KOSPI / wholesale_retail | 41 | 1 | 0 | 1 | 1.1% | 0.0% |
| market_x_industry | KOSDAQ / construction | 15 | 0 | 0 | 0 | 0.0% | 0.0% |
| market_x_industry | KOSDAQ / transport_storage | 2 | 0 | 0 | 0 | 0.0% | 0.0% |
| market_x_industry | KOSPI / it_services | 94 | 0 | 0 | 0 | 0.0% | 0.0% |
| market_x_industry | KOSPI / other | 18 | 0 | 0 | 0 | 0.0% | 0.0% |

## 7. Feature Profile: FN vs TP

FN은 실제 투기등급인데 모델이 안정적으로 본 사례입니다. 아래 변수 차이는 모델이 위험을 낮게 본 이유를 찾는 데 사용합니다.

| Feature | FN median | TP median | Std delta | FN miss | TP miss |
| --- | ---: | ---: | ---: | ---: | ---: |
| net_margin | 0.0288 | -0.0683 | 1.3873 | 13.3% | 13.3% |
| pretax_roe | 0.0807 | -0.0999 | 1.3647 | 13.3% | 13.3% |
| pretax_roa | 0.0353 | -0.0506 | 1.2418 | 13.3% | 13.3% |
| operating_roa | 0.0325 | -0.0213 | 1.0388 | 13.3% | 13.3% |
| dividend_payer | 1.0000 | 0.0000 | 1.0000 | 0.0% | 0.0% |
| icr_under_1 | 0.0000 | 1.0000 | -1.0000 | 0.0% | 0.0% |
| ocf_to_sales | 0.0860 | 0.0047 | 0.8077 | 13.3% | 13.3% |
| ocf_to_total_liabilities | 0.1398 | 0.0069 | 0.6972 | 13.3% | 13.3% |
| ocf_to_total_borrowings | 0.3882 | 0.0253 | 0.6540 | 13.3% | 17.3% |
| capital_impairment_ratio | -21.5459 | -4.5578 | -0.5200 | 13.3% | 13.3% |
| net_margin_diff | 0.0063 | -0.0237 | 0.4721 | 16.7% | 20.8% |
| market_to_book | 0.8370 | 1.2348 | -0.3935 | 13.3% | 13.3% |

## 8. Feature Profile: FP vs TN

FP는 실제 투자적격인데 모델이 위험하다고 본 사례입니다. 아래 변수 차이는 과민경고 원인을 찾는 데 사용합니다.

| Feature | FP median | TN median | Std delta | FP miss | TN miss |
| --- | ---: | ---: | ---: | ---: | ---: |
| dividend_payer | 0.0000 | 1.0000 | -1.0000 | 0.0% | 0.0% |
| ocf_to_sales | -0.0092 | 0.0797 | -0.8838 | 36.0% | 6.3% |
| net_margin | -0.0259 | 0.0329 | -0.8399 | 36.0% | 6.3% |
| operating_roa | -0.0011 | 0.0406 | -0.8073 | 36.0% | 6.3% |
| ocf_to_total_liabilities | -0.0116 | 0.1298 | -0.7418 | 36.0% | 6.3% |
| pretax_roe | -0.0213 | 0.0764 | -0.7382 | 36.0% | 6.3% |
| pretax_roa | -0.0153 | 0.0355 | -0.7336 | 36.0% | 6.3% |
| ocf_to_total_borrowings | -0.0200 | 0.3060 | -0.5876 | 40.4% | 13.3% |
| listed_year | 2015.0000 | 2003.0000 | 0.5714 | 0.0% | 0.0% |
| short_term_borrowings_share | 0.7776 | 0.4594 | 0.5452 | 40.4% | 13.3% |
| capital_impairment_ratio | -5.8012 | -22.3042 | 0.5052 | 36.0% | 6.3% |
| depreciation | 1696457.0000 | 39725701.0000 | -0.4733 | 0.0% | 0.0% |

## 9. Threshold Distance

threshold 바로 근처 오류는 threshold 조정으로 개선 여지가 있고, 멀리 떨어진 고확신 오류는 변수/외부근거 보완이 더 필요합니다.

| Distance | Rows | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <=0.02 | 27 | 9 | 1 | 0.4000 | 0.8571 | 0.5455 |
| 0.02-0.05 | 23 | 5 | 3 | 0.5455 | 0.6667 | 0.6000 |
| 0.05-0.10 | 41 | 6 | 6 | 0.5000 | 0.5000 | 0.5000 |
| 0.10-0.20 | 100 | 19 | 7 | 0.5000 | 0.7308 | 0.5938 |
| >0.20 | 733 | 50 | 13 | 0.7312 | 0.9128 | 0.8119 |

## 10. High-Confidence Error Examples

확률상 모델이 자신 있게 틀린 사례입니다. 이 기업들은 뉴스/공시/등급전망 확인 우선순위가 높습니다.

### False Negative

| Company | Market | FY | Industry | Size | Prob | Threshold |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 케이지모빌리티(주) | KOSPI | 2,024 | manufacturing | mid_sized | 0.0096 | 0.3150 |
| 제이엠티(주) | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.0123 | 0.3150 |
| (주)엑시콘 | KOSDAQ | 2,023 | manufacturing | mid_sized | 0.0168 | 0.3150 |
| 케이지모빌리티(주) | KOSPI | 2,023 | manufacturing | mid_sized | 0.0307 | 0.3150 |
| (주)톱텍 | KOSDAQ | 2,024 | manufacturing | mid_sized | 0.0314 | 0.3150 |
| (주)서진시스템 | KOSDAQ | 2,024 | manufacturing | mid_sized | 0.0455 | 0.3150 |
| 코리아에프티(주) | KOSDAQ | 2,023 | manufacturing | mid_sized | 0.0488 | 0.3150 |
| 케이지케미칼(주) | KOSPI | 2,023 | manufacturing | mid_sized | 0.0570 | 0.3150 |
| (주)디알비동일 | KOSPI | 2,023 | manufacturing | mid_sized | 0.0610 | 0.3150 |
| (주)에이엘티 | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.0656 | 0.3150 |

### False Positive

| Company | Market | FY | Industry | Size | Prob | Threshold |
| --- | --- | ---: | --- | --- | ---: | ---: |
| (주)라닉스 | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.9559 | 0.3150 |
| (주)푸드나무 | KOSDAQ | 2,023 | wholesale_retail | small_and_medium | 0.9168 | 0.3150 |
| 브이엠(주) | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.8973 | 0.3150 |
| 유니온머티리얼(주) | KOSPI | 2,023 | manufacturing | mid_sized | 0.8523 | 0.3150 |
| 범양건영(주) | KOSPI | 2,023 | construction | mid_sized | 0.8511 | 0.3150 |
| 한국맥널티(주) | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.8425 | 0.3150 |
| (주)웹스 | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.8104 | 0.3150 |
| (주)엑스페릭스 | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.8033 | 0.3150 |
| (주)케이엑스하이텍 | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.7939 | 0.3150 |
| 알에스오토메이션(주) | KOSDAQ | 2,023 | manufacturing | small_and_medium | 0.7819 | 0.3150 |

## 11. Rating Boundary Availability

`/Users/inji/Documents/금융 데이터 분석/Project/Corporate-Analysis-System/data/evaluation/target_label_reference.csv`에서 대표 신용등급을 붙여 경계등급 분석을 수행했습니다. test rows 중 등급이 매칭된 행은 `916`개입니다. 이 등급 정보는 모델 학습에는 쓰지 않고 diagnostics 전용으로만 사용합니다.

## 12. What To Do Next

1. FN이 몰린 산업/규모/연도 조합에서 실제 외부 이벤트가 있었는지 뉴스·공시·등급전망을 먼저 확인합니다.
2. FP가 많은 KOSDAQ/제조업/소형 구간은 전체 threshold를 바꾸기보다 Stage 2 과민경고 필터 또는 구간별 보조 판단으로 완화합니다.
3. FN과 TP의 차이가 큰 변수는 '위험을 숨기는 안정 신호'인지 확인합니다. 특히 규모성/배당/상장연도/절대금액 신호가 실제 위험을 가리는지 봅니다.
4. FP와 TN의 차이가 큰 변수는 단기 악화와 지속 악화를 구분하는 추가 변수 후보로 바꿔봅니다. 예: 2~3년 지속 손실, 현금흐름 악화 지속기간, 운전자본 변화 지속성.
5. BBB-/BB+ 경계 성능은 별도 객관 평가표로 발표 자료에 포함합니다.
