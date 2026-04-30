# Target Foreign Backfill Summary

## Policy

- 국내 평가사가 있는 회사-평가연도는 기존 규칙 유지
  - `BIG3`가 있으면 `BIG3` 중 최저등급 선택
  - `BIG3`가 없으면 기타 국내 평가사 중 최저등급 선택
- 국내 평가사가 전혀 없는 회사-평가연도만 foreign agency로 backfill
  - `기업신용등급 / ICR / 회사` 성격의 foreign rating 우선
  - 없으면 plain 장기 회사채 계열 foreign rating 사용
- Moody's 계열 등급은 공통 장기등급 스케일로 매핑
  - 예: `A1 -> A+`, `Aa3 -> AA-`, `Baa1 -> BBB+`

## Row Count Changes

- 기존 `Target_Processed.csv`: `5,586` rows
- 변경 후 `Target_Processed.csv`: `5,623` rows
- 증가분: `+37` rows

- 기존 `TS2000_Credit_Model_Dataset.csv`: `5,166` rows
- 변경 후 `TS2000_Credit_Model_Dataset.csv`: `5,202` rows
- 증가분: `+36` rows

- 기존 `TS2000_Credit_Model_Dataset_Model_V1.csv`: `5,166` rows
- 변경 후 `TS2000_Credit_Model_Dataset_Model_V1.csv`: `5,202` rows
- 증가분: `+36` rows

## Foreign Backfill Coverage

- foreign backfill로 최종 선택된 company-year: `23`
- foreign backfill 대상 기업 수: `6`
- fiscal year range: `2014 ~ 2024`

대상 기업과 편입된 company-year 수:

- `삼성전자(주)` (`005930`): `10`
- `현대모비스(주)` (`012330`): `6`
- `한국전력공사(주)` (`015760`): `4`
- `삼성이앤에이(주)` (`028050`): `1`
- `에스케이이노베이션(주)` (`096770`): `1`
- `(주)엘지에너지솔루션` (`373220`): `1`

## Samsung Electronics

- 이전에는 `Target_Processed.csv`에 `0` rows
- 현재는 `10` rows 편입
- fiscal year range: `2014 ~ 2023`

이유:
- raw target에는 `S&P`, `Moody's`, `SP` 등 foreign rating만 존재
- 기존에는 foreign agency를 전부 제외해서 탈락
- 현재는 국내 평가사가 없는 회사-평가연도에 한해 foreign backfill 허용

## One Backfilled Row That Did Not Survive Final Merge

아래 1건은 `Target_Processed.csv`에는 들어왔지만, 재무패널/거시 결합 단계에서 최종 마스터에는 남지 않았습니다.

- `(주)엘지에너지솔루션` (`373220`)
  - `fiscal_year = 2021`
  - `eval_year = 2022`
  - `credit_rating = BBB+`
  - `rating_agency = Moody's (미국) (Aaa~C)`
  - `security_name = 회사`

## Notes

- 이번 변경은 데이터셋과 `Model_V1` 재생성까지 반영했습니다.
- 모델 성능 비교 결과와 대시보드용 예측 산출물은 별도 재학습/재export가 필요합니다.
