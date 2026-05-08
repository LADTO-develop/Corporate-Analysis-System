# Corporate Analysis System (CAS)

**ML-LLM 하이브리드 기반 기업 신용위험 조기탐지 및 투자 의사결정 지원 시스템**

TS2000(재무), ECOS(거시경제), DART(정성 공시) 데이터를 결합해 국내 상장사의 투자적격 유지 여부와 신용위험 신호를 조기에 탐지하고, 실무자가 곧바로 활용할 수 있는 스크리닝형 의사결정 지원 대시보드를 제공합니다.

## Problem

신용위험 분석과 투자 의사결정 현장에는 세 가지 단절이 존재합니다.

1. 블랙박스 모델의 해석 한계 : 점수는 나오지만 "왜"가 따라오지 않음
2. 정성 정보 반영의 어려움 : 사업보고서의 위험 문구가 정량 모델에 들어오지 못함
3. 예측과 실제 의사결정의 단절 : 모델 결과가 투자 판단과 곧장 연결되지 않음

CAS는 정량 예측 → 정성 보강 → 다중 에이전트 종합 판단의 3단계 흐름으로 이 단절을 메웁니다.

## Architecture

```text
+-----------------------------+
|  1. ML 기반 정량 예측        |   Logistic Regression (baseline)
|   - is_speculative 이진분류   |   XGBoost / LightGBM (core)
|   - SHAP 설명                |   Global + Local feature contribution
+--------------+--------------+
               | 위험 신호 탐지
               v
+-----------------------------+
|  2. LLM 기반 정성 보강        |   DART 사업보고서 핵심 섹션
|   - On-Demand 공시 호출      |   GPT-4o / Claude
|   - 정성 위험 요인 요약       |   Prompt-driven 분석
+--------------+--------------+
               |
               v
+-----------------------------+
|  3. 다중 LLM 위원회 토론      |   LangGraph 멀티 에이전트
|   - 심사역 / 리서치 / 리스크  |   역할 기반 독립 의견 → 토론 → 합의
|   - 최종 투자 의견 도출       |
+--------------+--------------+
               |
               v
+-----------------------------+
|  Streamlit 대시보드           |   기업 선택 → 예측 → SHAP → 정성 요약
|                              |   → 위원회 결론 → 다건 비교 / 시나리오
+-----------------------------+
```

LangGraph는 모델이 아닌 오케스트레이션 레이어입니다. 데이터 적재, 검증, 예측, 설명, 정성 보강, 위원회 토론, 리포트 생성까지의 흐름을 상태 기반으로 연결합니다.

## Dataset

| 항목 | 내용 |
|---|---|
| 분석 범위 | KOSPI·KOSDAQ 상장기업 |
| 기간 | 2014 ~ 2024 (패널 데이터) |
| 관측치 | 4,596개 기업-연도 |
| 변수 수 | 157개 (모델 투입 `feature_x` 136 + 보류 `feature_deferred` 12 + 메타 변수) |
| 타깃 | `is_speculative` — `0 = 투자적격(AAA~BBB-)`, `1 = 투기등급(BB+ 이하)` |
| 결합 키 | `stock_code + fiscal_year` |
| 시점 정렬 | `fiscal_year = t` 재무정보 ↔ `eval_year = t+1` 신용등급 |
| 분할 전략 | Out-of-time (OOT) validation — 미래 누수 차단 |

피처 군 : 수익성 · 안정성 · 활동성 · 성장성 · 현금흐름/상환능력 · 거시지표 · 감사의견 · 추세 · 조기경보
주요 파생변수 : 부채비율, 유동비율, 이자보상배율, 좀비기업 플래그 등
사후 라벨 : 상장폐지일은 예측 변수로 사용하지 않고, 사후 검증 라벨로만 활용

## Pipeline Stages

### 1. 데이터 확보 및 피처 엔지니어링
- TS2000·ECOS·DART 연계 마스터 테이블 구축
- OOT 검증, 결합 키 기반 병합, 시점 일관성 검증

### 2. 머신러닝 모델 학습 및 최적화
- Baseline: Logistic Regression
- Core: XGBoost, LightGBM 등 가장 높은 성능 채택
- 클래스 불균형 대응: 가중치 조정 / SMOTE 검토
- 평가 지표: **ROC-AUC, PR-AUC, F1, Precision, Recall** (Accuracy 지양)
- 설명가능성: **SHAP** (global importance + local attribution)

### 3. LLM 기반 정성 분석
- 1차 ML 예측 결과를 트리거로 **On-Demand** DART 호출
- 입력: 예측 결과 · 주요 변수 기여도 · 공시 핵심 문단
- 출력: 기업별 정성 위험 요인 요약 및 투자 적격성 보조 보고서
- 모델: **GPT-4o** 또는 **Claude** 계열

### 4. 다중 LLM 위원회
- 역할 기반 에이전트: **심사역 · 리서치 팀장 · 리스크 관리자**
- 동일 기업에 대한 독립 의견 → 토론 → 합의 기반 최종 의견
- **LangGraph** 기반 상태 머신 오케스트레이션

### 5. Streamlit 대시보드
- 기업명/종목코드 입력 → ML 결과 → SHAP 설명 → 정성 요약 → 위원회 결론
- **다수 기업 동시 평가**, **산업군 비교**, **거시 충격 시나리오 민감도**
- CSV 일괄 업로드 기반 배치 스크리닝

## Repository Structure

```
.
├── configs/
│   ├── agent/
│   │   ├── graph.yaml            # LangGraph 노드·엣지 정의
│   │   └── committee.yaml        # 다중 LLM 위원회 역할/정책
│   └── runtime/
│       └── analysis.yaml         # 피처 범위, 스코어링 파라미터
├── data/
│   ├── raw/
│   │   ├── ts2000/               # TS2000 재무제표 원본
│   │   ├── ecos/                 # 한국은행 ECOS 거시지표
│   │   ├── dart/                 # DART 사업보고서 (On-Demand 캐시)
│   │   ├── ratings/              # 신용등급 이력
│   │   └── news/                 # (옵션) 뉴스/공시 보조 데이터
│   ├── interim/                  # 전처리 중간 산출물
│   ├── input/companies/          # 샘플/커스텀 기업 입력 YAML
│   └── outputs/reports/          # 기업별 리포트 (.md, .json)
├── docs/                         # 설계 문서
├── notebooks/                    # 피처·모델 실험 노트북
├── scripts/                      # 일회성 스크립트
├── src/cas/                      # 메인 패키지 (import 경로: cas.*)
│   ├── agents/
│   │   ├── graph.py              # LangGraph 파이프라인 조립
│   │   ├── state.py              # AgentState 스키마
│   │   └── nodes/                # 데이터 / 피처 / 예측 / 위원회 / 리포트 노드
│   ├── reporting/                # Markdown/JSON 리포트 생성
│   ├── utils/                    # 로깅, I/O 유틸
│   └── cli.py                    # cas-agent CLI 진입점
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

> **패키지 import 경로는 `cas.*`** 로 통일되어 있습니다. Streamlit 앱(`src/cas/dashboard/`)과 DART 로더는 로드맵 단계에서 추가됩니다.

## Tech Stack

| 영역 | 도구 |
|---|---|
| 데이터 수집·처리 | Python, Pandas, NumPy, Requests, **OpenDartReader** |
| 머신러닝 | scikit-learn, **XGBoost**, **LightGBM** |
| 설명가능성(XAI) | **SHAP** |
| LLM / 오케스트레이션 | OpenAI **GPT-4o**, **Claude**, **LangGraph** |
| 시각화 / 대시보드 | **Plotly**, Matplotlib, **Streamlit** |
| 검증 / 타입 | Pydantic v2, mypy (strict) |
| 품질 | Ruff, pytest, pre-commit |

Python 버전 : `3.12` (`requires-python = ">=3.12,<3.13"`)

## Quick Start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install
```

### 환경 변수

```bash
cp .env.example .env   # Windows: Copy-Item .env.example .env
```

자주 쓰는 키:

```env
CAS_ENV=local
CAS_LOG_LEVEL=INFO
CAS_RANDOM_SEED=42
ECOS_API_KEY=
DART_API_KEY=
OPENAI_API_KEY=
OPENAI_MODEL=
ANTHROPIC_API_KEY=
OUTPUT_DIR=data/outputs
```

`.env.example`만 저장소에 커밋되며, 실제 비밀값이 들어 있는 `.env`는 `.gitignore`로 제외됩니다.

### 파이프라인 실행 (CLI 베이스라인)

```bash
cas-agent --company-id sample-company
```

출력:

```
data/outputs/reports/<company-id>/latest.md
data/outputs/reports/<company-id>/latest.json
```

### Streamlit 대시보드 (로드맵)

```bash
# 구현 완료 시
streamlit run src/cas/dashboard/app.py
```

### 품질 검사

```bash
ruff check --fix . && ruff format .
mypy src/cas
pytest -m "not slow"
pytest -m integration      # 파이프라인 end-to-end
pytest --cov=cas
```

## Input Modes

현재 베이스라인은 `data/input/companies/<id>.yaml` 로컬 파일 입력 기준이며, 아래 3가지로 확장 예정입니다.

1. **회사 검색 기반** — 종목코드/회사명 → DART 최신 공시 자동 수집
2. **CSV 일괄 업로드** — 외부 추출 데이터 배치 분석
3. **최소 직접 입력** — 핵심 지표만 입력해 간이 판단

어느 경로든 동일한 `AgentState`로 정규화되어 파이프라인에 진입합니다.

## Roadmap

1. ✅ 패키지명 `cas` 통일 및 LangGraph 베이스라인 구동
2. ⬜ TS2000 + ECOS 결합 마스터 테이블 구축 (4,596×157)
3. ⬜ OOT 분할 · 클래스 불균형 처리 · 평가 하네스 정비
4. ⬜ LogReg / XGBoost / LightGBM 비교 학습 및 최적 모델 채택
5. ⬜ SHAP 기반 global + local 설명 통합
6. ⬜ DART On-Demand 로더 및 LLM 정성 분석 프롬프트 설계
7. ⬜ 심사역·리서치·리스크 다중 LLM 위원회 토론 구조
8. ⬜ Streamlit 대시보드: 단건/다건/산업비교/거시 시나리오
9. ⬜ 투자적격성 최종 리포트 자동 생성

## Principles

- **시점 일관성** (look-ahead 금지): 모든 노드는 `as_of_date` 이후의 데이터를 조회하지 않습니다.
- **결정론적 리포트**: 같은 입력 → 같은 결과 (LLM 사용 시 `temperature=0`, seed 고정).
- **설정 기반 분기**: 시장/모델/임계값 차이는 코드가 아닌 `configs/**` YAML로 관리합니다.
- **상태 단일화**: 모든 노드는 단일 `AgentState`를 공유하며, 신규 필드는 하위 호환을 유지합니다.

## References

- **Repository**: https://github.com/LADTO-develop/Corporate-Analysis-System
- **License**: Apache-2.0
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **ECOS Open API**: 한국은행 경제통계시스템
- **DART Open API**: 금융감독원 전자공시시스템
- **TS2000**: 한국기업평가 재무 DB
