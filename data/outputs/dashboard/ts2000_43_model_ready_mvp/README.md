# TS2000 43-Feature Dashboard Artifacts

이 폴더는 `ts2000_43_model_ready` 입력 파일을
대시보드가 바로 읽을 수 있는 형식으로 변환한 결과입니다.

핵심 파일:
- `company_universe.csv`: 기업-연도 전체 기본값
- `company_latest.csv`: 기업별 최신 행
- `peer_percentiles.csv`: 산업/시장 비교용 백분위
- `feature_dictionary.csv`: 지표 설명 사전
- `prediction_scores.csv`: 기업별 예측확률/판정
- `local_shap.csv`: 기업별 주요 영향 요인
- `industry_*`: 산업 집계 요약
- `model_summary.json`: 성능/기준선 요약
