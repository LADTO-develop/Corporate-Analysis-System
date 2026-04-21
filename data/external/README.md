# `data/external/`

**이 디렉토리의 파일은 git으로 버전 관리됩니다** (저장소 루트의 `.gitignore` 참조).
여기에는 크기가 작고 변경 이력이 의미 있는 **참조용 매핑 테이블**만 둡니다.
실제 원본 데이터 (TS2000, ECOS, 뉴스, 평가 원본)는 `data/raw/`에 위치하며 git으로
추적되지 않습니다.

## 현재 파일

### `rating_scale_mapping.csv`

5개 평가사의 원본 등급 기호(`rating_raw`)와 공통 22-notch 스케일
(`rating_normalized`) 사이의 매핑입니다. 컬럼:

| 컬럼 | 설명 |
|------|------|
| `agency` | 평가사명 (한국기업평가/한국신용평가/NICE신용평가/이크레더블/나이스디앤비 중 하나) |
| `rating_raw` | 평가사의 원본 등급 기호 |
| `rating_normalized` | 프로젝트 공통 22-notch 기호 |

초기 버전은 5개 평가사 모두 identity mapping으로 채워져 있습니다
(장기 신용등급 기호는 평가사 간 일치). 평가사별 특수 기호나 단기 등급을
장기 스케일에 맞게 매핑해야 할 경우 이 파일에 행을 추가하세요.

자세한 규칙은 [`docs/rating_normalization.md`](../../docs/rating_normalization.md) 참조.

## 파일 추가 시 규칙

1. 크기 **1MB 미만** 유지. 더 크면 DVC/LFS로 분리할 것.
2. 헤더 포함 CSV 또는 사람이 읽을 수 있는 YAML/JSON.
3. 변경 시 PR 설명에 **어떤 데이터로부터 이 값이 도출되었는지**를 반드시 기재.
4. 관련 단위 테스트가 존재하는지 확인 (예: `tests/unit/test_normalize.py`).

## 향후 추가 예정 파일 (플레이스홀더)

- `agency_weights.csv` — 시장별 평가사 커버리지 가중치. 현재는
  `bfd.ratings.agencies.DEFAULT_AGENCY_WEIGHTS` 하드코딩 값을 사용합니다.
- `industry_mapping.csv` — 종목코드 → GICS/KSIC 섹터 매핑 (섹터별 분석에 필요).
- `holiday_calendar_kr.csv` — 한국 증시 휴장일 (ECOS 일별 시리즈 결측 판정용).
