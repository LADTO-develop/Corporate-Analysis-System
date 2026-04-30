# TS2000 대시보드 실행 안내

## 개요
이 대시보드는 TS2000 공식 데이터셋과 Core29 모델 결과를 바탕으로 기업별 신용위험을 설명형으로 보여주는 Streamlit 앱입니다.

현재 포함된 주요 기능은 다음과 같습니다.
- 기업별 위험확률, 예측 라벨, 위험 밴드 확인
- 주요 설명 변수(SHAP) 확인
- 동종업계 및 시장 중앙값 비교
- 산업별 집계와 연도별 추이 확인
- 시나리오 기반 가정값 조정
- OpenAI API 기반 AI 심사 요약 생성
- 보고서형/원페이지형 HTML 및 Markdown 내보내기

## 중요한 점
- 이 대시보드는 **GitHub에 푸시했다고 해서 자동으로 웹 링크가 생기지 않습니다.**
- 팀원이 이 저장소를 `pull` 받은 뒤 **자기 로컬 환경에서 실행하면**, 자기 브라우저에서 대시보드를 바로 볼 수 있습니다.
- 즉, 현재는 **각자 로컬에서 실행하는 방식**입니다.

## 실행 방법
프로젝트 루트:

```bash
cd "/Users/inji/Documents/금융 데이터 분석/Project/Corporate-Analysis-System"
```

대시보드 입력 파일 생성:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_dashboard_inputs.py
```

모델 산출물 생성:

```bash
MPLCONFIGDIR='/private/var/folders/6f/82r7vcrd38s90qbm76tw7ml40000gn/T/mpltmp' /opt/anaconda3/envs/aura/bin/python scripts/export_dashboard_model_artifacts.py
```

대시보드 실행:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/run_ts2000_dashboard.py
```

실행 후 브라우저에서 아래와 같은 로컬 주소로 접속하면 됩니다.
- `http://localhost:8501`
- 실제 포트는 실행 시점에 따라 달라질 수 있습니다.

## 주요 파일 위치

### 실행 스크립트
- `scripts/export_dashboard_inputs.py`
- `scripts/export_dashboard_model_artifacts.py`
- `scripts/run_ts2000_dashboard.py`

### 대시보드 코드
- `src/cas/dashboard/data_loader.py`
- `src/cas/dashboard/llm.py`
- `src/cas/dashboard/ts2000_app.py`

### 입력 데이터
- `data/external/ts2000`

### 대시보드 산출물
- `data/outputs/dashboard/ts2000_core29_mvp`

## 대시보드 구성
- `개요`
  - 기업 기본 정보
  - 위험확률
  - 예측 라벨
  - 위험 밴드
  - 핵심 지표
- `AI 심사 요약`
  - OpenAI API 기반 심사 메모 생성
  - HTML/Markdown 다운로드
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
2. `export_dashboard_inputs.py` 실행
3. `export_dashboard_model_artifacts.py` 실행
4. `run_ts2000_dashboard.py` 실행
5. 브라우저에서 로컬 주소 접속

