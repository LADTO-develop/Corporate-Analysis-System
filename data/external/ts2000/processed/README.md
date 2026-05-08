# TS2000 Processed Inputs

이 폴더는 raw 원본을 직접 복사한 것이 아니라, `Model_V1`를 만들기 위해 선행 생성된 중간 전처리 산출물을 보관합니다.

- `Target_Processed.csv`
  - 최종 학습 타겟 테이블
- `Target_Processed_audit.csv`
  - 타겟 선택 근거를 더 자세히 담은 audit 테이블
- `ECOS_Macro_Processed.csv`
  - 연도별 거시변수 전처리 결과

현재 타겟 규칙의 핵심은 다음과 같습니다.
- 국내 등급 우선
- 국내 등급이 없을 때만 foreign agency backfill
- `fiscal_year < listed_year`는 제외
- 상장연도는 포함
