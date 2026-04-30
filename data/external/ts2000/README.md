# TS2000 External Package

이 폴더는 `Corporate-Analysis-System`에서 사용하는 TS2000 공식 데이터 패키지입니다.

목적은 세 가지입니다.
- raw에서 어떻게 전처리했는지 팀원과 AI가 추적할 수 있게 하기
- 공식 학습용 데이터셋(`Model_V1`)과 공식 변수셋(`Core29`)을 함께 제공하기
- 공식 모델 성능, SHAP, 공선성 결과를 바로 대시보드와 설명 레이어에서 재사용하게 하기

## 폴더 구조

### 최상위 핵심 파일
- `TS2000_Credit_Model_Dataset.csv`
  - 최종 전체 마스터 데이터셋입니다.
  - 분석용 라벨, 보류 변수, 파생변수까지 포함한 전체 참조본입니다.
- `TS2000_Credit_Model_Dataset_Model_V1.csv`
  - 공식 모델 학습 기본 데이터셋입니다.
  - 현재 모델 고도화와 대시보드 산출물의 직접 입력 파일입니다.
- `TS2000_Model_V1_Manifest.json`
  - `Model_V1`의 id/time/target/feature 구조를 설명하는 매니페스트입니다.
- `TS2000_Model_Core29_Manifest.json`
  - 현재 공식 핵심 변수셋 Core29 정의입니다.
- `TS2000_Model_Core29_Features.csv`
  - Core29 변수 목록과 변수군 설명입니다.

### `processed/`
- `Target_Processed.csv`
  - 최종 타겟 테이블입니다.
  - 규칙:
    - 국내 등급 우선
    - 국내 등급이 없을 때만 foreign agency backfill
    - `fiscal_year < listed_year`는 제외
    - 상장연도(`fiscal_year == listed_year`)는 포함
- `Target_Processed_audit.csv`
  - 타겟 선택 과정을 더 자세히 담은 audit 버전입니다.
- `ECOS_Macro_Processed.csv`
  - 연도별 거시변수 가공 결과입니다.

### `docs/`
- `00_Build_Process_Model_V1.md`
  - raw → processed → master → `Model_V1` 전체 생성 규칙 설명서
- `00_Build_Process_Model_V1_OnePager.md`
  - 팀원 공유용 1페이지 버전
- `00_Build_Process_Model_V1.json`
  - AI/코드용 구조화 설명
- `00_AI_Handoff_Core29.md`
  - Core29 기반 설명 시스템 handoff 문서
- `00_AI_Handoff_Core29.json`
  - AI 입력용 구조화 버전
- `Target_Foreign_Backfill_Summary.md`
  - foreign backfill 정책과 반영 결과 요약

### `column_dictionary/`
- `TS2000_Credit_Model_Column_Dictionary.xlsx`
  - 전체 컬럼 설명서
- `ts2000_column_dictionary_metadata.json`
  - 변수 역할 메타데이터

### `model_results/`
- `round1_model_comparison/`
  - Logistic / XGBoost / LightGBM 공식 비교 결과
- `xgboost_threshold_shap/`
  - 공식 XGBoost threshold tuning, 예측결과, SHAP 산출물

### `diagnostics/`
- `core29_multicollinearity/`
  - 공식 Core29 기준 상관계수, VIF, 결측 요약

## 현재 공식 규칙 요약

- 타겟: 투자적격 `0` vs 투기등급 `1`
- 타겟 우선순위:
  - `BIG3` 국내 평가사
  - 기타 국내 평가사
  - foreign backfill
- 상장연도 규칙:
  - `fiscal_year < listed_year`만 제외
  - 상장연도는 포함
- 공식 설명 가능한 변수셋:
  - `Core29`

## 추천 사용 순서

1. 학습/추가 실험 시작: `TS2000_Credit_Model_Dataset_Model_V1.csv`
2. 공식 변수셋 확인: `TS2000_Model_Core29_Manifest.json`
3. 전처리 규칙 확인: `docs/00_Build_Process_Model_V1_OnePager.md`
4. 변수 뜻 확인: `column_dictionary/TS2000_Credit_Model_Column_Dictionary.xlsx`
5. 공식 성능 확인: `model_results/round1_model_comparison/performance_summary.csv`

## 참고

- 이 패키지는 raw 원본 전체를 담지 않습니다.
- 대신 raw에서 어떤 규칙으로 `processed`, `Model_V1`, `Core29`가 만들어졌는지 문서와 중간 산출물로 추적 가능하게 구성했습니다.
- 대시보드용 기업별 예측점수와 local SHAP는 `Corporate-Analysis-System/data/outputs/dashboard/ts2000_core29_mvp` 아래에서 사용합니다.
