# Feature 43 Error Case SHAP Analysis

이 리포트는 현재 Stage 1 운영 threshold `0.315` 기준의 false positive와
false negative를 `local_shap.csv`와 연결해, 모델이 왜 틀렸는지 확인하기 위한
진단 산출물입니다.

## 1. 오류 규모

- False Positive: `89`개
- False Negative: `30`개
- KOSDAQ False Positive: `71`개
- Manufacturing False Positive: `66`개

## 2. 오류 집중 세그먼트

| Segment | Rows | FP | FP Rate | FN | FN Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| market=KOSDAQ | 427 | 71 | 26.9% | 26 | 16.0% |
| industry_macro_category=manufacturing | 598 | 66 | 15.1% | 26 | 16.0% |
| market=KOSPI | 497 | 18 | 3.9% | 4 | 10.0% |
| industry_macro_category=it_services | 176 | 16 | 10.3% | 3 | 14.3% |

## 3. False Positive 공통 SHAP 패턴

모델이 위험하다고 봤지만 실제 라벨은 투자적격인 사례입니다.

위험을 높인 주요 요인:

| Feature | Direction | Cases | Share | Mean | Rank | 해석 포인트 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 매출총이익 | increase_risk | 76 | 85.4% | 0.3923 | 4.0 | 매출총이익(손실) |
| 배당금 지급 여부 | increase_risk | 63 | 70.8% | 0.2506 | 7.0 | 해당 회계연도에 현금배당을 지급했는지 여부 |
| 기업규모 그룹 | increase_risk | 51 | 57.3% | 0.2934 | 6.0 | 모델링용 기업규모 그룹 |
| 감가상각비 | increase_risk | 51 | 57.3% | 0.1938 | 8.0 | 감가상각비 |
| 자산총계 | increase_risk | 50 | 56.2% | 0.2813 | 5.0 | 총자산 |
| 이자보상배율 | increase_risk | 47 | 52.8% | 0.4316 | 3.0 | 영업이익 대비 이자비용 배수 |
| 자본잠식률 | increase_risk | 46 | 51.7% | 0.2908 | 5.0 | 자본금 대비 자본잠식 정도 |
| 상장연도 | increase_risk | 44 | 49.4% | 0.5119 | 2.0 | 상장일에서 추출한 연도 |
| 현금흐름 커버리지 | increase_risk | 30 | 33.7% | 0.2599 | 7.0 | 이자비용 대비 영업현금흐름 배수 |
| 순이익률 | increase_risk | 29 | 32.6% | 0.4670 | 3.0 | 매출액 대비 당기순이익 비율 |

위험을 낮춘 주요 요인:

| Feature | Direction | Cases | Share | Mean | Rank | 해석 포인트 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 이자보상배율 | decrease_risk | 34 | 38.2% | -0.6497 | 1.0 | 영업이익 대비 이자비용 배수 |
| 현금흐름 커버리지 | decrease_risk | 34 | 38.2% | -0.1839 | 8.0 | 이자비용 대비 영업현금흐름 배수 |
| 자기자본비율 | decrease_risk | 26 | 29.2% | -0.2155 | 9.0 | 총자산 대비 자기자본 비율 |
| 자본잠식률 | decrease_risk | 20 | 22.5% | -0.3516 | 6.0 | 자본금 대비 자본잠식 정도 |
| 상장연도 | decrease_risk | 20 | 22.5% | -0.3444 | 3.5 | 상장일에서 추출한 연도 |
| 산업 대분류 | decrease_risk | 19 | 21.3% | -0.5335 | 3.0 | 모델링용 산업 대분류 |
| 배당금 지급 여부 | decrease_risk | 16 | 18.0% | -0.3828 | 5.0 | 해당 회계연도에 현금배당을 지급했는지 여부 |
| 자산총계 | decrease_risk | 15 | 16.9% | -0.4796 | 5.0 | 총자산 |
| 순이익률 | decrease_risk | 15 | 16.9% | -0.4530 | 4.0 | 매출액 대비 당기순이익 비율 |
| 순이익률 변화 | decrease_risk | 13 | 14.6% | -0.2110 | 9.0 | 전년 대비 순이익률 변화폭 |

## 4. False Negative 공통 SHAP 패턴

모델이 안정적으로 봤지만 실제 라벨은 투기등급인 사례입니다.

위험을 낮춘 주요 요인:

| Feature | Direction | Cases | Share | Mean | Rank | 해석 포인트 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 자본잠식률 | decrease_risk | 20 | 66.7% | -0.4099 | 4.0 | 자본금 대비 자본잠식 정도 |
| 배당금 지급 여부 | decrease_risk | 18 | 60.0% | -0.3837 | 4.0 | 해당 회계연도에 현금배당을 지급했는지 여부 |
| 자기자본비율 | decrease_risk | 16 | 53.3% | -0.2663 | 7.5 | 총자산 대비 자기자본 비율 |
| 자산총계 | decrease_risk | 15 | 50.0% | -0.5761 | 5.0 | 총자산 |
| 매출총이익 | decrease_risk | 14 | 46.7% | -0.7011 | 2.0 | 매출총이익(손실) |
| 총부채회전율 | decrease_risk | 12 | 40.0% | -0.2934 | 7.0 | 총부채 대비 매출액 비율 |
| 상장연도 | decrease_risk | 11 | 36.7% | -0.2988 | 5.0 | 상장일에서 추출한 연도 |
| 감가상각비 | decrease_risk | 10 | 33.3% | -0.9639 | 1.5 | 감가상각비 |
| 순이익률 | decrease_risk | 10 | 33.3% | -0.4016 | 4.0 | 매출액 대비 당기순이익 비율 |
| 현금흐름 커버리지 | decrease_risk | 10 | 33.3% | -0.3053 | 7.0 | 이자비용 대비 영업현금흐름 배수 |

위험을 높인 주요 요인:

| Feature | Direction | Cases | Share | Mean | Rank | 해석 포인트 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 이자보상배율 | increase_risk | 16 | 53.3% | 0.3257 | 6.5 | 영업이익 대비 이자비용 배수 |
| 매출총이익 | increase_risk | 15 | 50.0% | 0.3760 | 4.0 | 매출총이익(손실) |
| 배당금 지급 여부 | increase_risk | 12 | 40.0% | 0.2394 | 7.0 | 해당 회계연도에 현금배당을 지급했는지 여부 |
| 자기자본비율 | increase_risk | 8 | 26.7% | 0.4292 | 4.0 | 총자산 대비 자기자본 비율 |
| 산업 대분류 | increase_risk | 8 | 26.7% | 0.2315 | 8.0 | 모델링용 산업 대분류 |
| 감가상각비 | increase_risk | 8 | 26.7% | 0.1976 | 9.0 | 감가상각비 |
| 자본잠식률 | increase_risk | 7 | 23.3% | 0.3969 | 5.0 | 자본금 대비 자본잠식 정도 |
| 자산총계 | increase_risk | 6 | 20.0% | 0.2752 | 3.5 | 총자산 |
| 순이익률 | increase_risk | 5 | 16.7% | 0.4941 | 3.0 | 매출액 대비 당기순이익 비율 |
| 현금흐름 커버리지 | increase_risk | 5 | 16.7% | 0.2030 | 7.0 | 이자비용 대비 영업현금흐름 배수 |

## 5. 대표 False Positive 사례

| Company | Market | Year | Prob. | Actual | Pred. | Top SHAP |
| --- | --- | ---: | ---: | --- | --- | --- |
| (주)라닉스 | KOSDAQ | 2023 | 0.9559 | 투자적격 | 투기등급 | 총자산증가율(increase_risk, SHAP=+0.568, 값=-0.303645); 순이익률(increase_risk, SHAP=+0.537, 값=-0.465185); 이자보상배율(increase_risk, SHAP=+0.506, 값=-1.922441); 자기자본비율(increase_risk, SHAP=+0.398, 값=0.363743); 매출총이익(increase_risk, SHAP=+0.392, 값=6588483) |
| (주)푸드나무 | KOSDAQ | 2023 | 0.9168 | 투자적격 | 투기등급 | 산업 대분류(increase_risk, SHAP=+0.641, 값=wholesale_retail); 현금비율(decrease_risk, SHAP=-0.532, 값=0.305516); 자기자본비율(increase_risk, SHAP=+0.495, 값=0.199391); 세전계속사업이익 기준 ROE(increase_risk, SHAP=+0.484, 값=-0.889455); 배당금 지급 여부(decrease_risk, SHAP=-0.410, 값=1) |
| 브이엠(주) | KOSDAQ | 2023 | 0.8973 | 투자적격 | 투기등급 | 순이익률(increase_risk, SHAP=+0.972, 값=-0.268192); 이자보상배율(increase_risk, SHAP=+0.559, 값=-16.321037); OCF/매출액(increase_risk, SHAP=+0.451, 값=-0.599392); 매출총이익(increase_risk, SHAP=+0.384, 값=12661220); 자기자본비율(decrease_risk, SHAP=-0.341, 값=0.809501) |
| 유니온머티리얼(주) | KOSPI | 2023 | 0.8523 | 투자적격 | 투기등급 | 이자보상배율(increase_risk, SHAP=+0.585, 값=-0.875495); 총차입금비율(increase_risk, SHAP=+0.523, 값=0.48283); 자기자본비율(increase_risk, SHAP=+0.466, 값=0.340644); 순이익률(increase_risk, SHAP=+0.374, 값=-0.154978); 매출총이익(increase_risk, SHAP=+0.353, 값=9880215) |
| 범양건영(주) | KOSPI | 2023 | 0.8511 | 투자적격 | 투기등급 | 산업 대분류(decrease_risk, SHAP=-0.669, 값=construction); 이자보상배율(increase_risk, SHAP=+0.555, 값=-3.254095); 순이익률(increase_risk, SHAP=+0.456, 값=-0.075445); 자본잠식률(increase_risk, SHAP=+0.413, 값=-1.50217); 매출총이익(increase_risk, SHAP=+0.388, 값=4739925) |

## 6. 대표 False Negative 사례

| Company | Market | Year | Prob. | Actual | Pred. | Top SHAP |
| --- | --- | ---: | ---: | --- | --- | --- |
| 케이지모빌리티(주) | KOSPI | 2024 | 0.0096 | 투기등급 | 투자적격 | 매출총이익(decrease_risk, SHAP=-1.483, 값=345715769); 감가상각비(decrease_risk, SHAP=-1.234, 값=118273864); 자산총계(decrease_risk, SHAP=-1.075, 값=3104452197); 무형자산 비중(decrease_risk, SHAP=-0.507, 값=0.061267); 자본잠식률(increase_risk, SHAP=+0.461, 값=-0.44655) |
| 제이엠티(주) | KOSDAQ | 2023 | 0.0123 | 투기등급 | 투자적격 | 이자보상배율(decrease_risk, SHAP=-1.167, 값=53.294298); OCF/총차입금(decrease_risk, SHAP=-0.629, 값=7.280508); 자기자본비율(decrease_risk, SHAP=-0.585, 값=0.713055); 자본잠식률(decrease_risk, SHAP=-0.537, 값=-17.885609); 총차입금비율(decrease_risk, SHAP=-0.470, 값=0.027052) |
| (주)엑시콘 | KOSDAQ | 2023 | 0.0168 | 투기등급 | 투자적격 | 이자보상배율(decrease_risk, SHAP=-0.749, 값=1000000.0); 순이익률(decrease_risk, SHAP=-0.657, 값=0.059372); 자기자본비율(decrease_risk, SHAP=-0.499, 값=0.879343); 순이익률 변화(decrease_risk, SHAP=-0.398, 값=-0.107497); 매출총이익(increase_risk, SHAP=+0.396, 값=28251051) |
| 케이지모빌리티(주) | KOSPI | 2023 | 0.0307 | 투기등급 | 투자적격 | 매출총이익(decrease_risk, SHAP=-1.283, 값=407465928); 감가상각비(decrease_risk, SHAP=-1.021, 값=116993347); 자산총계(decrease_risk, SHAP=-0.965, 값=2635399999); 자본잠식률(increase_risk, SHAP=+0.804, 값=-0.158239); 무형자산 비중(decrease_risk, SHAP=-0.497, 값=0.070925) |
| (주)톱텍 | KOSDAQ | 2024 | 0.0314 | 투기등급 | 투자적격 | 자본잠식률(decrease_risk, SHAP=-0.532, 값=-20.642183); 순이익률(decrease_risk, SHAP=-0.482, 값=0.068742); 이자보상배율(increase_risk, SHAP=+0.452, 값=2.030646); 배당금 지급 여부(decrease_risk, SHAP=-0.434, 값=1); 자산총계(decrease_risk, SHAP=-0.418, 값=616469709) |

## 7. 개선 후보

- False Positive는 `gross_profit`, `assets_total`, `depreciation` 같은 규모성 원천값과
  `firm_size_group`, `listed_year` 같은 맥락 변수가 반복적으로 위험을 키우는 패턴이
  나타납니다. 규모가 작은 KOSDAQ/제조업 기업에서 절대금액이 위험 신호처럼 작동하는지
  추가 점검이 필요합니다.
- False Negative는 대기업·대규모 자산·배당 지급·자본잠식률 안정 신호가 위험을 강하게
  낮추는 패턴이 나타납니다. 대기업 이벤트성 등급 하락이나 일시적 외부 충격은 재무제표
  기반 모델만으로 놓칠 수 있어 외부근거/공시 신호와 결합하는 편이 좋습니다.
- 다음 변수 실험은 핵심 비율의 산업-연도 내 백분위, 규모 조정 로그 변수,
  최근 악화 속도 변수, 외부 공시 위험 신호 플래그 순서로 진행하는 것이 적절합니다.

## 8. 산출물

- `error_shap_case_details.csv`: 오류 사례별 top SHAP 상세
- `error_shap_feature_summary.csv`: FP/FN별 공통 SHAP 요인 집계
- `error_shap_segment_summary.csv`: 시장/산업별 FP/FN 집중도
- `error_shap_top_cases.csv`: 대표 오류 사례와 상위 SHAP 요인
- `error_shap_analysis_summary.json`: 주요 요약 수치
