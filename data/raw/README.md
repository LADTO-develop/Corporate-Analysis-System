# `data/raw/`

**이 디렉토리의 원본 데이터 파일은 git으로 추적되지 않습니다.** (`.gitignore`에서
`data/raw/**`가 제외되어 있음). 각 하위 디렉토리에 두어야 하는 파일의 형식을
기록만 해둡니다.

## `data/raw/ts2000/`

연도별로 아래 다섯 파일이 있어야 합니다. 파일명 패턴은
`configs/data/ts2000.yaml`에서 오버라이드 가능.

```
bs_{year}.csv        # 재무상태표
is_{year}.csv        # 손익계산서
cf_{year}.csv        # 현금흐름표
sce_{year}.csv       # 자본변동표
notes_{year}.csv     # 주석
```

필수 컬럼은 [`docs/data_dictionary_ts2000.md`](../../docs/data_dictionary_ts2000.md)
참조. 기본 인코딩은 CP949 (TS2000 export 표준).

## `data/raw/ecos/`

ECOS API 호출 결과를 캐시하는 용도. 일반적으로는 `bfd.data.loaders.ecos.ECOSClient`가
런타임에 직접 API를 호출하므로 이 디렉토리는 비어 있어도 됩니다.
오프라인/재현성이 필요한 경우 `ECOSClient.fetch_all_configured`의 결과를 parquet로
여기에 dump 해두면 됩니다.

## `data/raw/news/`

기업별 서브디렉토리 구조:

```
data/raw/news/
  ├── 005930/
  │   ├── 2024-03-15_실적발표.txt
  │   └── 2024-04-02_공시.txt
  └── 000660/
      └── 2024-02-10_M&A루머.txt
```

파일명: `YYYY-MM-DD_{title}.txt`. 본문은 UTF-8 평문. 각 파일이 한 기사/공시.

## `data/raw/ratings/`

평가사별 단일 CSV:

```
ratings_kr.csv          # 한국기업평가
ratings_kis.csv         # 한국신용평가
ratings_nice.csv        # NICE신용평가
ratings_ecredible.csv   # 이크레더블
ratings_nicednb.csv     # 나이스디앤비
```

각 파일의 필수 컬럼:

| 컬럼 | 설명 |
|------|------|
| `corp_code` | KRX 6자리 종목코드 |
| `rating_date` | 등급 평가일 (ISO 날짜) |
| `rating_raw` | 평가사 원본 등급 |
| `market` | KOSPI \| KOSDAQ \| KONEX |
| `outlook` (선택) | Positive/Stable/Negative/Developing/NR |

로더가 `rating_date`에서 `rating_year`를 파생하고, 원본 등급은
`data/external/rating_scale_mapping.csv`를 사용해 공통 스케일로 변환합니다.

## 데이터 관리 권장

- 가능하면 **DVC**(https://dvc.org) 로 버전 관리할 것.
- 평가사 데이터는 유료/라이선스 대상이므로 저장소 외부(팀 드라이브 등)에 원본을 두고
  CI/로컬에서는 sync 스크립트로 가져오는 방식을 권장.
- 개인정보·비공개 정보가 섞여 있지 않은지 수령 시점에 확인 (주민등록번호 등).
