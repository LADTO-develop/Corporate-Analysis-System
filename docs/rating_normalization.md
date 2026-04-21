# Rating Normalization

5개 평가사(한기평/한신평/NICE/이크레더블/나이스디앤비)의 등급 기호는 S&P 스타일
22-notch 체계로 일원화됩니다. 이 문서는 정규화 규칙과 편집 절차를 정리합니다.

## 공통 스케일 (22 notches)

```
AAA AA+ AA AA− A+ A A− BBB+ BBB BBB−   (investment grade)
BB+ BB BB− B+ B B− CCC+ CCC CCC− CC C D   (speculative grade)
```

- 정수 encoding: `AAA = 22`, `D = 1` (높을수록 우량). `bfd.ratings.normalize`에서
  `RATING_TO_NOTCH` / `NOTCH_TO_RATING` 사전을 export 합니다.
- `NR` (Not Rated) 은 notch를 갖지 않으며, 이진 타깃 생성 시 drop 됩니다.

## 투자적격 경계

투자적격(0) vs 투기(1) 이진화 경계는 **BBB−/BB+** 입니다
(`INVESTMENT_GRADE_MIN_NOTCH = 13`). 이 경계는 한국 은행·보험사의
위험가중자산 산정 관례와 일치합니다. 경계는 상수로 고정되어 있으며, 변경 시
`tests/unit/test_targets.py`가 반드시 함께 업데이트되어야 합니다.

## 평가사별 매핑 테이블

편집용 CSV는 `data/external/rating_scale_mapping.csv`에 위치합니다. 컬럼은
`agency, rating_raw, rating_normalized` 세 개뿐입니다.

| agency | rating_raw | rating_normalized |
|--------|-----------|-------------------|
| 한국기업평가 | AAA | AAA |
| 한국기업평가 | AA+ | AA+ |
| ... | ... | ... |

대부분의 장기 신용등급은 평가사 간 기호가 동일합니다. 단기 등급(A1, A2, A3, B, C, D) 을
장기로 매핑하거나 평가사별 특수 기호를 처리할 때만 추가 행이 필요합니다.

매핑 테이블이 없으면 `bfd.ratings.normalize._load_mapping_table`이 빈 DataFrame을
반환하고, `normalize_rating`은 입력이 이미 canonical 22-notch 집합에 속하는 경우에만
통과시킵니다 (그렇지 않으면 `ValueError`).

## 평가사 지배력 (Market Dominance)

같은 기업-연도에 여러 평가사의 등급이 동시에 존재할 때의 집계 가중치는
`bfd.ratings.agencies.DEFAULT_AGENCY_WEIGHTS`에 정의되어 있습니다.

| 시장 | 한기평 | 한신평 | NICE | 이크레더블 | 나이스디앤비 |
|------|-------|-------|------|-----------|-------------|
| KOSPI | 0.35 | 0.30 | 0.25 | 0.05 | 0.05 |
| KOSDAQ | 0.15 | 0.10 | 0.20 | 0.30 | 0.25 |

이 가중치는 TS2000 5년 샘플의 커버리지 비율에서 도출된 초기값입니다.
`data/external/agency_weights.csv`가 존재하면 그 값이 우선됩니다 (향후 확장).

## 집계 규칙

`bfd.ratings.targets.aggregate_ratings_per_firm_year`는 다음 단계로 수행합니다:

1. 평가사별 정규화 notch를 얻는다.
2. `(corp_code, rating_year, market)` 별로 `sum(w * notch) / sum(w)`.
3. 반올림하여 정수 notch로 복귀.
4. 그 notch에 해당하는 canonical 기호로 역변환 (`consensus_notch` → `rating_normalized`).

결과 DataFrame은 한 행당 (corp_code, rating_year) 유일성을 보장합니다.

## 이 파일을 수정하려면

1. `data/external/rating_scale_mapping.csv`에 행을 추가.
2. `tests/unit/test_normalize.py`에 새 매핑에 대한 파라미터를 추가.
3. `tests/unit/test_targets.py`에서 경계 로직에 영향이 있는지 확인.

정규화 결과가 바뀌면 기존 학습된 모델의 타깃 분포가 변하므로, 매핑 변경은
모델 재학습과 함께 이루어져야 합니다.
