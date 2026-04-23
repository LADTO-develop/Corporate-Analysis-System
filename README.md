# Corporate Analysis System

기업 재무제표와 한국은행 ECOS 거시지표를 결합해 투자적격성을 판단하는 LangGraph 기반 분석 시스템입니다. 목표는 단순 점수 산출이 아니라, 시장 구분(KOSPI/KOSDAQ), 입력 검증, 특성 생성, 머신러닝 예측, SHAP 기반 설명, 시나리오 재평가, 최종 리포트까지 하나의 일관된 파이프라인으로 연결하는 것입니다.

현재 저장소는 그 전체 구조를 구현하기 위한 실행 가능한 베이스라인입니다. 데이터 수집, 검증, 예측, 설명, 리포트 흐름을 먼저 안정적으로 구축하는 데 집중합니다.

## What This Project Aims To Build

- 재무제표와 ECOS 거시데이터를 결합한 투자적격성 판단 파이프라인
- `KOSPI`와 `KOSDAQ` 시장별 설정과 모델 선택이 가능한 라우팅 구조
- 로지스틱 회귀, XGBoost 등 후보 모델 비교 후 최적 모델 채택
- SHAP 기반 핵심 원인 설명과 시나리오별 위험도 재계산
- 최종 심사 코멘트와 요약 리포트 생성

## Target Workflow

```text
input
-> validate
-> resolve_market
-> load_financials
-> attach_ecos_macro
-> build_features
-> select_model_by_market
-> predict_investment_suitability
-> explain_with_shap
-> run_scenarios
-> generate_report
```

## Why LangGraph

LangGraph는 이 프로젝트에서 모델 자체를 대신하는 도구가 아니라, 여러 처리 단계를 안정적으로 연결하는 오케스트레이션 레이어입니다. 즉, `KOSPI/KOSDAQ` 분기, 입력 경로별 검증, 예측 후 설명 단계, 예외 처리, 리포트 생성 같은 흐름 제어를 담당합니다.

권장 구조는 시장별로 완전히 다른 그래프를 만드는 방식보다, 공통 그래프 안에서 시장별 설정과 모델을 선택하는 방식입니다. 이렇게 하면 전처리와 리포트 구조는 공유하고, 시장별 차이는 피처 정책, 임계값, 모델 파일, 후처리 규칙에서 관리할 수 있어 유지보수가 훨씬 쉬워집니다.

## Current Repository Structure

- `src/cas/agents/graph.py`: LangGraph 파이프라인 구성
- `src/cas/agents/state.py`: 상태 스키마 정의
- `src/cas/agents/nodes/`: 데이터 적재, 피처 생성, 예측, 리포트 노드
- `src/cas/cli.py`: 공식 CLI 진입점
- `configs/agent/graph.yaml`: 그래프 노드와 엣지 설정
- `configs/agent/committee.yaml`: 최종 판단 정책 설정
- `configs/runtime/analysis.yaml`: 피처 범위와 점수 계산 설정
- `data/input/companies/`: 샘플 입력 데이터

## Quick Start

이 프로젝트의 표준 Python 버전은 `3.12`입니다.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Environment Setup

실행 전에 `.env.example`을 복사해서 `.env` 파일을 만든 뒤 필요한 값을 채워주세요.

PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

기본적으로 자주 쓰게 될 항목은 아래입니다.

```env
CAS_ENV=local
CAS_LOG_LEVEL=INFO
CAS_RANDOM_SEED=42
CAS_TORCH_DETERMINISTIC=0
ECOS_API_KEY=
OPENAI_API_KEY=
OPENAI_MODEL=
DATABASE_URL=
EXTERNAL_API_BASE_URL=
OUTPUT_DIR=data/outputs
```

주의:

- 저장소에는 `.env.example`만 커밋합니다.
- 실제 비밀값이 들어 있는 `.env`는 `.gitignore`로 제외되어 있습니다.

## Run

```bash
cas-agent --company-id sample-company
```

결과물은 아래 경로에 생성됩니다.

```text
data/outputs/reports/sample-company/latest.md
data/outputs/reports/sample-company/latest.json
```

## Input Modes We Are Designing For

- 회사 검색 기반 입력
- CSV 업로드 기반 입력
- 최소 직접 입력 기반 입력

현재 베이스라인은 `data/input/companies/sample-company.yaml` 같은 로컬 파일 입력을 기준으로 동작하며, 이후 위 3가지 입력 경로로 확장할 예정입니다.

## Implementation Roadmap

1. 패키지명과 실행 구조를 `cas` 기준으로 통일합니다.
2. 입력 경로를 회사 검색, CSV 업로드, 직접 입력으로 확장합니다.
3. ECOS 및 재무제표 데이터 결합 레이어를 구축합니다.
4. 시장별 피처셋과 모델 선택 로직을 추가합니다.
5. 로지스틱 회귀, XGBoost 등 후보 모델을 비교하고 최적 모델을 채택합니다.
6. SHAP 설명과 시나리오 분석 결과를 리포트에 통합합니다.
7. 최종적으로 투자적격성 판단 리포트를 자동 생성합니다.
