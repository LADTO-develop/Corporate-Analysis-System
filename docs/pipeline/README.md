# Pipeline Docs

CAS 데이터 파이프라인 관련 문서를 모아 둔 디렉토리다.

## 현재 구현 범위

- 웹 리스팅 기업 선택값을 `company_selection` 계약으로 정규화한다.
- 정규화된 입력을 `AgentState` 시작 상태로 변환한다.
- `data` 노드는 `company_selection`을 feature master/inference row로 해석한다.
- feature master와 inference row는 TS2000 원천에 OpenDART CFS/OFS 보강을 반영한
  공식 46개 입력셋을 기준으로 한다.
- 기존 `company_id` 기반 실행 경로는 유지한다.
- 로컬에 `xgboost`가 없거나 모델 artifact를 읽을 수 없는 경우 Stage 1 deterministic fallback으로 이어진다.

## 구현 위치

- `src/cas/agents/contracts/company_selection.py`: 입력 계약, 정규화, `AgentState` seed 생성
- `src/cas/agents/graph.py`: `run_once(company_selection=...)` 진입점
- `src/cas/agents/nodes/data_node.py`: 계약 입력을 회사-연도 feature snapshot으로 resolve
- `src/cas/cli.py`: `--company-selection-json`, `--company-selection-file` 지원
- `src/cas/dashboard/credit_app.py`: 대시보드 선택 row를 `company_selection`으로 변환

## 문서

- `data_pipeline.md`: 웹 리스팅 입력부터 분석 파이프라인 진입까지의 계약과 실패 규칙
- `../preprocessing_rules_ko.md`: 신용등급 타겟, 재무/거시 결합, 46개 입력셋 전처리 기준

## 검증

- `tests/unit/test_company_selection_contract.py`: 계약 정규화/검증 단위 테스트
- `tests/integration/test_graph_smoke.py`: 기존 `company_id` 경로와 신규 `company_selection` 경로 스모크 테스트
