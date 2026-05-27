# 신용위험 대시보드 실행 안내

## 개요
이 대시보드는 `credit_46_features` 데이터셋과 XGBoost 결과, 2차 에이전트 위원회 검토를 바탕으로 기업별 신용도를 설명형으로 보여주는 Streamlit 앱입니다.
현재 2026 추론 입력은 TS2000 원천에 OpenDART 사업보고서 CFS/OFS 보강을 반영한 기준입니다.

현재 포함된 주요 기능은 다음과 같습니다.
- 기업별 위험확률, 예측 라벨, 위험 밴드 확인
- 주요 설명 변수(SHAP) 확인
- 동종업계 및 시장 중앙값 비교
- 산업별 집계와 연도별 추이 확인
- 시나리오 기반 가정값 조정
- 빠른 deterministic 2차 위원회 검토와 선택형 Agno live 검토
- OpenAI API 기반 AI 심사 요약 생성
- 보고서형/원페이지형 HTML 및 Markdown 내보내기

## 중요한 점
- 이 대시보드는 **GitHub에 푸시했다고 해서 자동으로 웹 링크가 생기지 않습니다.**
- 사용자가 이 저장소를 `pull` 받은 뒤 **자기 로컬 환경에서 실행하면**, 자기 브라우저에서 대시보드를 바로 볼 수 있습니다.
- 즉, 현재는 **각자 로컬에서 실행하는 방식**입니다.

## 실행 방법
프로젝트 루트:

```bash
cd Corporate-Analysis-System
```

운영형 실행:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/run_credit_dashboard.py
```

위 명령은 `data/outputs/dashboard/feature_46_mvp` 아래의 대시보드 입력 파일이
없으면 먼저 생성한 뒤 Streamlit을 실행합니다. 2026 추론 입력까지 최신 OpenDART
보강 기준으로 다시 만들고 싶으면 아래 순서로 갱신한 뒤 실행합니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/import_feature_46_inference_2026_aux.py
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_46_inference_2026.py
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source-kind inference --target-fiscal-year 2025 --fallback-ofs
/opt/anaconda3/envs/aura/bin/python scripts/export_inference_2026_missing_2024_lag_targets.py
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source data/raw/opendart/inference_2026_missing_2024_lag_targets.csv --source-kind inference --target-fiscal-year 2025 --opendart-bsns-year 2024 --fallback-ofs --output-dir data/raw/opendart/lag_2024_tmp
/opt/anaconda3/envs/aura/bin/python scripts/apply_opendart_inference_financial_supplements.py --lag-raw-supplement data/raw/opendart/lag_2024_tmp/financial_statements_inference_2024_cfs_with_ofs_fallback_raw.csv
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_46_inference_2026.py --check-only
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_inference_2026_dashboard_artifacts.py
```

대시보드 산출물을 강제로 다시 만들고 싶으면 다음처럼 실행합니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/run_credit_dashboard.py --rebuild-artifacts
```

Agno live 2차 검토까지 로컬에서 켜고 싶으면 `.env`에 `OPENAI_API_KEY`,
외부 근거 수집용 API 키를 넣은 뒤 아래처럼 실행합니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/run_credit_dashboard.py \
  --stage2-runner agno \
  --stage2-model-provider openai \
  --stage2-model gpt-4.1-mini \
  --stage2-agno-mode single
```

이 모드에서도 `위원회 검토` 탭은 먼저 deterministic 결과를 즉시 표시합니다.
Agno와 외부 뉴스/공시 수집은 `Agno 실행` 버튼을 누를 때 백그라운드에서 돌고,
완료되면 기업-회계연도-모델-외부근거 기준 캐시로 저장됩니다. 기본값은
`CAS_DASHBOARD_STAGE2_TRIGGER_ONLY=1`이라 2차 검토 트리거가 있는 기업만 live
Agno로 보내며, 모든 선택 기업에서 강제로 live 실행을 확인하려면 이 값을 `0`으로
설정합니다. 동시에 실행할 live 작업 수는 `CAS_DASHBOARD_STAGE2_ASYNC_WORKERS`
환경변수로 조정할 수 있습니다.

실행 후 브라우저에서 아래와 같은 로컬 주소로 접속하면 됩니다.
- `http://localhost:8501`
- 실제 포트는 실행 시점에 따라 달라질 수 있습니다.

패키지를 editable로 설치한 환경에서는 같은 launcher를 콘솔 명령으로도 실행할 수 있습니다.

```bash
cas-dashboard
```

## 주요 파일 위치

### 실행 스크립트
- `scripts/export_feature_46_dashboard_artifacts.py`
- `scripts/run_credit_dashboard.py`

### 대시보드 코드
- `src/cas/dashboard/data_loader.py`
- `src/cas/dashboard/llm.py`
- `src/cas/dashboard/credit_app.py`

### 입력 데이터
- `data/input/credit_46_features`

### 대시보드 산출물
- `data/outputs/dashboard/feature_46_mvp`

## 대시보드 구성
- `위원회 검토`
  - 1차 모델 판단을 출발점으로 한 2차 에이전트 위원회 해석
  - 외부 근거, 위험 요인, 완화 요인, 판단 유형 확인
  - API 키 없이 위원회 검토 Markdown 보고서 다운로드
- `기업 기본 정보`
  - 시장, 산업, 규모, 주요 재무 스냅샷
- `주요 요인`
  - 주요 설명 변수(SHAP)
- `동종업계 비교`
  - 선택 기업 vs 산업 중앙값 vs 시장 중앙값
- `산업 집계`
  - 산업 최신 스냅샷
  - 연도별 추이
- `시나리오`
  - 지표 가정 변경에 따른 비교

## AI 요약 사용 방법
- 사이드바의 `AI 요약 설정`에서 OpenAI API 키를 입력합니다.
- 추천 모델 또는 직접 입력 모델명을 선택합니다.
- 출력 형식을 고릅니다.
  - `간단 요약`
  - `기본 심사 메모`
  - `상세 보고서형`
- `AI 요약 생성` 버튼을 누르면 결과가 표시됩니다.

## 주의사항
- API 키는 세션 메모리에서만 사용되며 파일에 저장하지 않습니다.
- 현재 대시보드는 로컬 실행 기준입니다.
- 공용 링크로 바로 공유하려면 별도 배포가 필요합니다.

## 추천 사용 순서
1. 저장소 `pull`
2. `run_credit_dashboard.py` 실행
3. 브라우저에서 로컬 주소 접속
4. 첫 화면에서 시장/산업 필터와 기업명 검색으로 기업 선택
5. `위원회 검토` 탭에서 판단 유형, 위험 신호, 완화 근거, 외부근거를 먼저 확인
6. 필요하면 Markdown 보고서를 다운로드하거나 API 키 기반 요약을 생성
