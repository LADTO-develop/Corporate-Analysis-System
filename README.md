# Borderline Firm Detector

> KOSPI/KOSDAQ 상장사의 **한계기업 징후**를 신용평가과정에 대응되는 AI 에이전트 파이프라인으로 조기에 포착합니다.

[![CI](https://github.com/LADTO-develop/Corporate-Analysis-System/actions/workflows/ci.yml/badge.svg)](https://github.com/LADTO-develop/Corporate-Analysis-System/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

---

## 프로젝트 개요

기존 신용평가등급은 전문기관의 수 개월에 걸친 심사를 거쳐야 하며, 모든 상장사가 평가를 받는 것도 아닙니다.
본 프로젝트는 **TS2000 재무제표 + ECOS 거시경제지표 + 뉴스/공시 LLM 해석**을 결합하여,
다음 두 가지 질문에 빠르게 답하는 *간이 등급*을 생성합니다.

1. 현재 이 기업은 한계기업 징후(흑자도산, 자본잠식, 이자보상배율 악화 등)를 보이고 있는가?
2. 그 판단의 근거는 무엇인가? — **EBM 기반 글로벌/로컬 설명**과 **에이전트 위원회의 토론 로그**를 함께 제공.

실제 신용평가등급을 완전히 대체하는 것이 아니라, 평가 전/평가 불가 상태의 기업에 대한 **의사결정 보조 도구**를 지향합니다.

## 파이프라인 아키텍처

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  TS2000      │   │  ECOS 거시   │   │  뉴스/공시   │
│  재무제표 5본 │   │  경제지표    │   │  DART, 언론  │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       ▼                  ▼                  ▼
   [ 재무 피처 ]     [ 거시 오버레이 ]    [ LLM 피처화 ]
       │                  │                  │
       └──────────┬───────┴──────────────────┘
                  ▼
        ┌─────────────────────┐
        │  시장별 분리 모델   │  KOSPI / KOSDAQ
        │  TabPFN v2.5 (주)   │
        │  + LightGBM         │  (OOF 스태킹)
        │  + EBM (해석 경로)  │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │  위원회 노드        │  악마의 대변인, 6모자,
        │  (LangGraph)        │  스텝래더, 명목집단법
        └──────────┬──────────┘
                   ▼
           [ 이진 판정: borderline / healthy ]
           [ 감사 트레일 + EBM shape functions ]
```

각 컴포넌트의 상세 설명은 [`docs/architecture.md`](docs/architecture.md)를 참고하세요.

## 핵심 설계 원칙

| 원칙 | 구현 위치 |
|------|----------|
| **타깃 누수 차단** — 재무(t년) → 등급(t+1년) 매핑을 코드로 강제 | [`src/bfd/data/splitters.py`](src/bfd/data/splitters.py), [`src/bfd/validation/leakage.py`](src/bfd/validation/leakage.py) |
| **시장별 분리** — KOSPI/KOSDAQ은 구조적으로 달라 동일 모델 금지 | [`configs/market/`](configs/market/), [`scripts/train_market_model.py`](scripts/train_market_model.py) |
| **투명성** — EBM 경로는 항상 학습되어 리포트에 shape function 포함 | [`src/bfd/models/ebm_model.py`](src/bfd/models/ebm_model.py), [`src/bfd/reporting/explanations.py`](src/bfd/reporting/explanations.py) |
| **편향 제거 위원회** — 악마의 대변인 등 다수 기법을 Strategy로 조합 | [`src/bfd/agents/committee/`](src/bfd/agents/committee/) |
| **내부정보 보안** — 기본 비활성, 활성화 시 세션 격리·감사로그 | [`src/bfd/rag/internal_info/`](src/bfd/rag/internal_info/), [`docs/security_policy.md`](docs/security_policy.md) |

## 설치

```bash
# Python 3.11 또는 3.12 권장
git clone https://github.com/LADTO-develop/Corporate-Analysis-System.git
cd Corporate-Analysis-System

# uv 권장 (LangGraph 의존성 resolve 속도 차이가 큼)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,viz]"

# 또는 pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"

# 환경 변수
cp .env.example .env   # ANTHROPIC_API_KEY, ECOS_API_KEY 등을 채워넣기

# pre-commit 훅 설치
pre-commit install
```

### TabPFN v2.5 체크포인트

첫 실행 시 자동으로 HuggingFace에서 체크포인트를 다운로드합니다.
캐시 위치를 바꾸려면 `.env`에 `TABPFN_MODEL_CACHE_DIR`을 설정하세요.

## 사용법

### 1. 데이터셋 빌드

```bash
bfd-build-dataset --config configs/data/ts2000.yaml
```

TS2000 5본(재무상태표/손익계산서/현금흐름표/자본변동표/주석)을 정합성 검증 후
`data/processed/{kospi,kosdaq}/`에 parquet으로 저장합니다.

### 2. 시장별 모델 학습

```bash
bfd-train --market kospi
bfd-train --market kosdaq
```

Walk-forward CV로 TabPFN/LightGBM/EBM을 학습하고 OOF 예측을 `data/processed/oof/`에 저장합니다.

### 3. 단일 종목 평가 (에이전트 실행)

```bash
bfd-agent --corp-code 005930 --fiscal-year 2024
```

LangGraph 파이프라인이 기동되어 데이터 수집 → 피처화 → 예측 → 거시/뉴스 오버레이 →
위원회 토론 → 최종 판정 + 감사 트레일을 출력합니다.

## 프로젝트 구조

```
src/bfd/
├── data/              # TS2000, ECOS, 뉴스 로더 + 스키마 + 누수 방지 스플리터
├── features/          # 재무제표 5본별 파생변수 + ECOS 거시 + 피처 카탈로그
├── ratings/           # 평가사별 등급 정규화 + 투자적격/투기 이진화
├── models/            # TabPFN / LightGBM / EBM + OOF 스태킹 + 캘리브레이션
├── rag/               # 뉴스 RAG + LLM 피처화 + 내부정보 보안 정책
├── agents/            # LangGraph State + 노드 + 위원회 전략 + 툴 카탈로그
├── validation/        # 누수 어서션 + 메트릭 + 백테스트 + 드리프트
├── reporting/         # 감사 트레일 + 설명(EBM) + 최종 리포트 export
└── utils/             # 로깅 / 시드 / I/O / 시간 유틸
```

## 관련 문헌 및 의존 라이브러리

- **TabPFN v2.5** — Hollmann et al., *Nature* (2025). [`PriorLabs/TabPFN`](https://github.com/PriorLabs/TabPFN)
- **Explainable Boosting Machines** — Nori et al. (2019). [`interpretml/interpret`](https://github.com/interpretml/interpret)
- **LangGraph** — LangChain AI. [`langchain-ai/langgraph`](https://github.com/langchain-ai/langgraph)
- **LightGBM** — Ke et al. (2017). [`microsoft/LightGBM`](https://github.com/microsoft/LightGBM)
- **pandera** — tabular schema validation. [`unionai-oss/pandera`](https://github.com/unionai-oss/pandera)

## 라이선스

Apache License 2.0. 세부사항은 [LICENSE](LICENSE) 참조.

## 기여

이 프로젝트는 학습/연구 목적의 프로토타입입니다.
실제 투자/대출 의사결정에 본 시스템의 출력을 단독 근거로 사용해서는 안 됩니다.
신용평가등급의 보조 지표로만 활용하세요.
