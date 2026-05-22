# Live Agno/OpenAI External API Runbook

작성일: 2026-05-21

## 문제 원인

현재 Stage 2 Agno 기본 모드는 OpenAI 단일 provider 실행(`CAS_STAGE2_AGNO_MODE=single`, `CAS_STAGE2_MODEL_PROVIDER=openai`)이다. 이 모드는 `OPENAI_API_KEY`만 있으면 preflight가 통과하도록 맞춰져 있다. `single`은 OpenAI 한 provider를 쓰되 QuantCredit/EvidenceAudit/ChairReport 세 역할 agent를 분리 실행하는 모드다. 실제 live latency를 측정할 때는 캐시 재사용을 피하기 위해 `--no-stage2-llm-cache`를 붙인다.

여러 LLM 관점을 비교하는 `multi_llm_committee`는 선택 사항이다. 이 모드는 역할별로 Claude, GPT, Gemini를 함께 쓰므로 `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` 또는 `GEMINI_API_KEY`가 모두 필요할 수 있다.

## Codex 실행 환경 제한

Codex 작업환경에서는 private workspace-derived holdout/evaluation data를 외부 API로 전송하는 실행이 정책상 차단될 수 있다. 사용자가 대화에서 외부 전송을 승인해도, 상위 실행 정책이 차단하면 Codex가 우회할 수 없다.

따라서 실제 기업-회계연도 데이터를 OpenAI API로 보내는 live batch는 로컬 터미널에서 직접 실행한다. 아래 명령은 같은 저장소와 같은 Python 환경을 사용하므로 결과 파일 경로는 Codex에서 만든 산출물과 동일하게 남는다.

## Preflight

```bash
cd "/Users/inji/Documents/금융 데이터 분석/Project/Corporate-Analysis-System"

/usr/bin/env \
  CAS_STAGE2_RUNNER=agno \
  CAS_STAGE2_AGNO_MODE=single \
  CAS_STAGE2_MODEL_PROVIDER=openai \
  CAS_STAGE2_MODEL=gpt-4.1-mini \
  /opt/anaconda3/envs/aura/bin/python scripts/check_agno_stage2.py
```

예상 결과:

```text
Agno Stage 2 preflight passed.
```

## OpenAI API Live Batch

외부 뉴스/공시 API 없이 OpenAI/Agno 위원회만 확인할 때:

```bash
/usr/bin/env \
  CAS_STAGE2_FALLBACK_ON_ERROR=0 \
  CAS_STAGE2_MAX_TOKENS=6000 \
  /opt/anaconda3/envs/aura/bin/python scripts/run_committee_review_evaluation_batch.py \
  --samples data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_holdout_unseen_8_samples.csv \
  --output-dir data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_holdout_unseen_agno_openai_live_batch \
  --policy balanced_current_45_or_near_threshold_0_10 \
  --per-category 2 \
  --max-cases 8 \
  --stage2-runner agno \
  --stage2-agno-mode single \
  --stage2-model-provider openai \
  --stage2-model gpt-4.1-mini \
  --no-stage2-llm-cache \
  --workers 2
```

외부 뉴스/공시 수집까지 함께 켤 때는 `OPENDART_API_KEY`, `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`, `TAVILY_API_KEY`를 `.env`에 설정한 뒤 `--live-external-evidence`를 추가한다.

## Deterministic vs OpenAI Agno 설명 품질 비교

Codex 세션 안에서는 실제 기업-회계연도 평가 맥락을 OpenAI API로 보내는 실행이 차단될 수 있다. 팀/프로젝트 기준으로 외부 전송을 승인한 경우, 아래 명령은 Codex 밖의 로컬 터미널에서 직접 실행한다.

가장 편한 방법:

```bash
cd "/Users/inji/Documents/금융 데이터 분석/Project/Corporate-Analysis-System"

bash scripts/run_openai_agno_comparison_local.sh
```

이 스크립트는 다음을 순서대로 수행한다.

1. `scripts/check_agno_stage2.py`로 OpenAI Agno 설정을 확인한다.
2. 같은 4개 holdout 샘플을 deterministic Stage 2로 실행해 기준선을 만든다.
3. 같은 4개 샘플을 OpenAI Agno 단일 모델(`gpt-4.1-mini`)로 실행한다.
4. `scripts/export_stage2_agno_explanation_comparison.py`로 최종 라벨 변화, 성공 여부 변화, 설명 품질 점수를 비교한다.
5. `scripts/export_stage2_evaluation_report.py`로 통합 Stage 2 평가 리포트를 다시 생성한다.

생성되는 주요 파일:

- `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_openai_agno_comparison_deterministic/committee_review_batch_results.csv`
- `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_openai_agno_comparison_agno/committee_review_batch_results.csv`
- `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_openai_agno_explanation_comparison.md`
- `data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_evaluation_report.md`

수동으로 나눠 실행하고 싶을 때:

```bash
cd "/Users/inji/Documents/금융 데이터 분석/Project/Corporate-Analysis-System"

/opt/anaconda3/envs/aura/bin/python scripts/run_committee_review_evaluation_batch.py \
  --samples data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_holdout_unseen_8_samples.csv \
  --output-dir data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_openai_agno_comparison_deterministic \
  --policy balanced_current_45_or_near_threshold_0_10 \
  --per-category 1 \
  --max-cases 4 \
  --stage2-runner deterministic \
  --workers 1

/usr/bin/env \
  CAS_STAGE2_FALLBACK_ON_ERROR=0 \
  /opt/anaconda3/envs/aura/bin/python scripts/run_committee_review_evaluation_batch.py \
  --samples data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_holdout_unseen_8_samples.csv \
  --output-dir data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_openai_agno_comparison_agno \
  --policy balanced_current_45_or_near_threshold_0_10 \
  --per-category 1 \
  --max-cases 4 \
  --stage2-runner agno \
  --stage2-agno-mode single \
  --stage2-model-provider openai \
  --stage2-model gpt-4.1-mini \
  --workers 1

/opt/anaconda3/envs/aura/bin/python scripts/export_stage2_agno_explanation_comparison.py
/opt/anaconda3/envs/aura/bin/python scripts/export_stage2_evaluation_report.py
```

해석 기준:

- `deterministic_success_rate`와 `agno_success_rate`는 같은 샘플에서 Stage 2 판단이 FN/FP/TP/TN 목적에 맞게 작동했는지 비교한다.
- `quality_delta_mean`이 양수면 Agno 설명이 deterministic 설명보다 메모 길이, 핵심 용어, 수치 포함 측면에서 더 풍부해졌다는 뜻이다.
- 최종 판단 라벨은 더 좋아졌는데 설명 품질이 낮아질 수도 있고, 반대로 라벨은 같지만 설명 품질만 좋아질 수도 있으므로 두 지표를 함께 본다.
- 이 비교는 4건 smoke test이므로 성능 일반화 지표가 아니라 “실제 LLM 설명 품질 확인용”으로 해석한다.

## Multi-LLM Committee를 꼭 쓸 때

Claude/Gemini 역할까지 포함하려면 다음이 필요하다.

```bash
/opt/anaconda3/envs/aura/bin/python -m pip install -e ".[agent]"
```

그리고 `.env`에 `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` 또는 `GEMINI_API_KEY`를 추가한 뒤 `--stage2-agno-mode multi_llm_committee`로 실행한다.

## 속도 운영 원칙

Agno/OpenAI live batch는 deterministic guardrail보다 느리다. 실제 대시보드에서는 전체 기업을 보내지 않고 다음 조건 중 하나를 만족하는 기업만 live Agno로 보낸다.

- `stage2_review_trigger=True`
- `stage2_secondary_trigger=True`
- `stage2_review_priority`가 `medium` 또는 `high`
- 직접 관련 외부근거에서 치명 리스크 후보가 확인됨
