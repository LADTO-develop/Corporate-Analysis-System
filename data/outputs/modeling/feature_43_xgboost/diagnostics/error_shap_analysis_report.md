# Feature 43 Error Case SHAP Analysis

이 리포트는 현재 Stage 1 운영 threshold `0.315` 기준의 false positive와
false negative를 `local_shap.csv`와 연결해, 모델이 왜 틀렸는지 확인하기 위한
진단 산출물입니다.

## 1. 오류 규모

- False Positive: `81`개
- False Negative: `23`개
- KOSDAQ False Positive: `70`개
- Manufacturing False Positive: `58`개

## 2. 오류 집중 세그먼트

| Segment | Rows | FP | FP Rate | FN | FN Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| market=KOSDAQ | 384 | 70 | 28.5% | 19 | 13.8% |
| industry_macro_category=manufacturing | 449 | 58 | 18.3% | 19 | 14.4% |
| industry_macro_category=it_services | 131 | 16 | 14.5% | 3 | 14.3% |
| market=KOSPI | 288 | 11 | 4.2% | 4 | 13.8% |

## 3. False Positive 공통 SHAP 패턴

모델이 위험하다고 봤지만 실제 라벨은 투자적격인 사례입니다.

위험을 높인 주요 요인:

| Feature | Direction | Cases | Share | Mean | Rank | 해석 포인트 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 매출총이익 | increase_risk | 72 | 88.9% | 0.3962 | 4.0 | 매출총이익(손실) |
| 배당금 지급 여부 | increase_risk | 58 | 71.6% | 0.2517 | 7.0 | 해당 회계연도에 현금배당을 지급했는지 여부 |
| 기업규모 그룹 | increase_risk | 51 | 63.0% | 0.2934 | 6.0 | 모델링용 기업규모 그룹 |
| 감가상각비 | increase_risk | 48 | 59.3% | 0.1969 | 8.5 | 감가상각비 |
| 자산총계 | increase_risk | 47 | 58.0% | 0.2825 | 5.0 | 총자산 |
| 이자보상배율 | increase_risk | 44 | 54.3% | 0.4366 | 3.0 | 영업이익 대비 이자비용 배수 |
| 자본잠식률 | increase_risk | 41 | 50.6% | 0.2852 | 5.0 | 자본금 대비 자본잠식 정도 |
| 상장연도 | increase_risk | 39 | 48.1% | 0.4974 | 2.0 | 상장일에서 추출한 연도 |
| 현금흐름 커버리지 | increase_risk | 27 | 33.3% | 0.2534 | 7.0 | 이자비용 대비 영업현금흐름 배수 |
| 순이익률 | increase_risk | 26 | 32.1% | 0.4874 | 3.0 | 매출액 대비 당기순이익 비율 |

위험을 낮춘 주요 요인:

| Feature | Direction | Cases | Share | Mean | Rank | 해석 포인트 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 이자보상배율 | decrease_risk | 31 | 38.3% | -0.6459 | 1.0 | 영업이익 대비 이자비용 배수 |
| 현금흐름 커버리지 | decrease_risk | 31 | 38.3% | -0.1840 | 8.0 | 이자비용 대비 영업현금흐름 배수 |
| 자기자본비율 | decrease_risk | 24 | 29.6% | -0.2258 | 9.0 | 총자산 대비 자기자본 비율 |
| 산업 대분류 | decrease_risk | 19 | 23.5% | -0.5335 | 3.0 | 모델링용 산업 대분류 |
| 자본잠식률 | decrease_risk | 19 | 23.5% | -0.3567 | 5.0 | 자본금 대비 자본잠식 정도 |
| 상장연도 | decrease_risk | 19 | 23.5% | -0.3301 | 4.0 | 상장일에서 추출한 연도 |
| 배당금 지급 여부 | decrease_risk | 15 | 18.5% | -0.3856 | 5.0 | 해당 회계연도에 현금배당을 지급했는지 여부 |
| 순이익률 | decrease_risk | 14 | 17.3% | -0.4544 | 3.5 | 매출액 대비 당기순이익 비율 |
| 순이익률 변화 | decrease_risk | 13 | 16.0% | -0.2110 | 9.0 | 전년 대비 순이익률 변화폭 |
| 자산총계 | decrease_risk | 11 | 13.6% | -0.4944 | 5.0 | 총자산 |

## 4. False Negative 공통 SHAP 패턴

모델이 안정적으로 봤지만 실제 라벨은 투기등급인 사례입니다.

위험을 낮춘 주요 요인:

| Feature | Direction | Cases | Share | Mean | Rank | 해석 포인트 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 자본잠식률 | decrease_risk | 16 | 69.6% | -0.4048 | 4.5 | 자본금 대비 자본잠식 정도 |
| 배당금 지급 여부 | decrease_risk | 15 | 65.2% | -0.3853 | 4.0 | 해당 회계연도에 현금배당을 지급했는지 여부 |
| 자기자본비율 | decrease_risk | 14 | 60.9% | -0.2672 | 7.5 | 총자산 대비 자기자본 비율 |
| 총부채회전율 | decrease_risk | 10 | 43.5% | -0.3055 | 7.0 | 총부채 대비 매출액 비율 |
| 자산총계 | decrease_risk | 9 | 39.1% | -0.5775 | 5.0 | 총자산 |
| 매출총이익 | decrease_risk | 8 | 34.8% | -0.7945 | 1.5 | 매출총이익(손실) |
| 순이익률 | decrease_risk | 8 | 34.8% | -0.3948 | 4.5 | 매출액 대비 당기순이익 비율 |
| 현금흐름 커버리지 | decrease_risk | 8 | 34.8% | -0.3077 | 7.0 | 이자비용 대비 영업현금흐름 배수 |
| 상장연도 | decrease_risk | 7 | 30.4% | -0.2735 | 5.0 | 상장일에서 추출한 연도 |
| 감가상각비 | decrease_risk | 6 | 26.1% | -0.8670 | 2.5 | 감가상각비 |

위험을 높인 주요 요인:

| Feature | Direction | Cases | Share | Mean | Rank | 해석 포인트 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 매출총이익 | increase_risk | 14 | 60.9% | 0.3858 | 4.0 | 매출총이익(손실) |
| 이자보상배율 | increase_risk | 10 | 43.5% | 0.3276 | 6.0 | 영업이익 대비 이자비용 배수 |
| 배당금 지급 여부 | increase_risk | 8 | 34.8% | 0.2457 | 7.0 | 해당 회계연도에 현금배당을 지급했는지 여부 |
| 감가상각비 | increase_risk | 7 | 30.4% | 0.1988 | 9.0 | 감가상각비 |
| 산업 대분류 | increase_risk | 6 | 26.1% | 0.2267 | 7.5 | 모델링용 산업 대분류 |
| 자본잠식률 | increase_risk | 5 | 21.7% | 0.4187 | 4.0 | 자본금 대비 자본잠식 정도 |
| 자기자본비율 | increase_risk | 5 | 21.7% | 0.3273 | 5.0 | 총자산 대비 자기자본 비율 |
| 자산총계 | increase_risk | 5 | 21.7% | 0.2775 | 4.0 | 총자산 |
| 순이익률 | increase_risk | 4 | 17.4% | 0.5225 | 2.5 | 매출액 대비 당기순이익 비율 |
| 기업규모 그룹 | increase_risk | 4 | 17.4% | 0.2902 | 7.0 | 모델링용 기업규모 그룹 |

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
| 제이엠티(주) | KOSDAQ | 2023 | 0.0123 | 투기등급 | 투자적격 | 이자보상배율(decrease_risk, SHAP=-1.167, 값=53.294298); OCF/총차입금(decrease_risk, SHAP=-0.629, 값=7.280508); 자기자본비율(decrease_risk, SHAP=-0.585, 값=0.713055); 자본잠식률(decrease_risk, SHAP=-0.537, 값=-17.885609); 총차입금비율(decrease_risk, SHAP=-0.470, 값=0.027052) |
| 에스케이이노베이션(주) | KOSPI | 2024 | 0.0151 | 투기등급 | 투자적격 | 매출총이익(decrease_risk, SHAP=-1.368, 값=4039022328); 기업규모 그룹(decrease_risk, SHAP=-1.168, 값=large); 감가상각비(decrease_risk, SHAP=-0.859, 값=2124635000); 자산총계(decrease_risk, SHAP=-0.803, 값=110530097549); 자기자본비율(increase_risk, SHAP=+0.320, 값=0.358716) |
| (주)엑시콘 | KOSDAQ | 2023 | 0.0168 | 투기등급 | 투자적격 | 이자보상배율(decrease_risk, SHAP=-0.749, 값=1000000.0); 순이익률(decrease_risk, SHAP=-0.657, 값=0.059372); 자기자본비율(decrease_risk, SHAP=-0.499, 값=0.879343); 순이익률 변화(decrease_risk, SHAP=-0.398, 값=-0.107497); 매출총이익(increase_risk, SHAP=+0.396, 값=28251051) |
| 케이지모빌리티(주) | KOSPI | 2023 | 0.0307 | 투기등급 | 투자적격 | 매출총이익(decrease_risk, SHAP=-1.283, 값=407465928); 감가상각비(decrease_risk, SHAP=-1.021, 값=116993347); 자산총계(decrease_risk, SHAP=-0.965, 값=2635399999); 자본잠식률(increase_risk, SHAP=+0.804, 값=-0.158239); 무형자산 비중(decrease_risk, SHAP=-0.497, 값=0.070925) |
| 코리아에프티(주) | KOSDAQ | 2023 | 0.0488 | 투기등급 | 투자적격 | 매출총이익(decrease_risk, SHAP=-0.605, 값=95847992); 총부채회전율(decrease_risk, SHAP=-0.555, 값=2.85753); 감가상각비(decrease_risk, SHAP=-0.536, 값=21185740); 배당금 지급 여부(decrease_risk, SHAP=-0.360, 값=1); 자산총계(decrease_risk, SHAP=-0.322, 값=415160886) |

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
