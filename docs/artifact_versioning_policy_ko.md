# 산출물/데이터 버전 관리 정책

이 저장소는 현재 DVC 또는 Git LFS를 사용하지 않습니다. Git에는 코드 리뷰,
런타임 재현, 팀 결과 공유에 필요한 작은 기준 파일과 핵심 진단 요약을 남기고,
대량 산출물은 로컬에서 재생성하거나 GitHub Release/외부 artifact 저장소에
첨부합니다.

## Git에 보관하는 것

- 코드, 설정, 문서, 테스트
- 작은 기준 입력/참조 데이터
- 현재 운영 baseline 모델 artifact
  - `data/outputs/modeling/feature_46_xgboost/xgboost_model.json`
  - `data/outputs/modeling/feature_46_xgboost/model_artifact_metadata.json`
  - 관련 `README.md`
- 팀 공유용 핵심 diagnostics
  - Stage 1 기준 모델 성능/오류/threshold/SHAP 진단 파일
  - Stage 2 평가 리포트, 요약 JSON, metrics/counts/log 성격의 작은 CSV
  - 결과 해석에 필요한 Markdown 리포트

## Git에 보관하지 않는 것

- dashboard export, report, experiment, cache, log 산출물
- 대량 row-level score 파일
- Stage 2 agent batch 실행 결과, live Agno 반복 실행 디렉터리
- 후보 feature set 모델 artifact와 중간 실험 디렉터리

보관하지 않는 파일들은 재생성 가능한 실행 결과이므로 `.gitignore` 대상입니다.
팀 전체가 같은 기준 결과를 바로 확인해야 하는 작은 진단 파일은 Git에 남기고,
큰 스냅샷은 압축해 release artifact로 올리며 PR에는 생성 명령과 핵심 요약을
남깁니다.

## 새 산출물 추가 기준

1. 앱/CLI가 기본 실행에 직접 읽는 작은 baseline인가?
2. 팀원이 fresh clone에서 바로 확인해야 하는 기준 결과인가?
3. 재생성 비용이 크거나, 결과 해석의 근거로 반복 참조되는가?
4. 파일 크기와 변경 빈도가 코드 리뷰에 부담을 주지 않는가?

네 질문에 모두 답할 수 있을 때만 Git 추적을 검토합니다. 그렇지 않으면
`data/outputs/README.md` 또는 관련 runbook에 재생성 명령을 문서화합니다.
