#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/aura/bin/python}"
MODEL_PROVIDER="${CAS_STAGE2_MODEL_PROVIDER:-openai}"
MODEL_NAME="${CAS_STAGE2_MODEL:-gpt-4.1-mini}"
AGNO_MODE="${CAS_STAGE2_AGNO_MODE:-single}"
LLM_CACHE_ENABLED="${CAS_STAGE2_LLM_CACHE_ENABLED:-1}"
LLM_CACHE_NORMALIZED="$(printf '%s' "$LLM_CACHE_ENABLED" | tr '[:upper:]' '[:lower:]')"
if [[ "$LLM_CACHE_NORMALIZED" =~ ^(0|false|no|off)$ ]]; then
  CACHE_ARGS=(--no-stage2-llm-cache)
else
  CACHE_ARGS=(--stage2-llm-cache)
fi
SAMPLES_PATH="data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_holdout_unseen_8_samples.csv"
POLICY="balanced_current_45_or_near_threshold_0_10"
DETERMINISTIC_DIR="data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_openai_agno_comparison_deterministic"
AGNO_DIR="data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/committee_review_openai_agno_comparison_agno"

cd "$ROOT_DIR"

echo "[1/5] Checking OpenAI Agno runtime"
CAS_STAGE2_RUNNER=agno \
CAS_STAGE2_AGNO_MODE="$AGNO_MODE" \
CAS_STAGE2_MODEL_PROVIDER="$MODEL_PROVIDER" \
CAS_STAGE2_MODEL="$MODEL_NAME" \
CAS_STAGE2_LLM_CACHE_ENABLED="$LLM_CACHE_ENABLED" \
"$PYTHON_BIN" scripts/check_agno_stage2.py

echo "[2/5] Running deterministic comparison baseline"
"$PYTHON_BIN" scripts/run_committee_review_evaluation_batch.py \
  --samples "$SAMPLES_PATH" \
  --output-dir "$DETERMINISTIC_DIR" \
  --policy "$POLICY" \
  --per-category 1 \
  --max-cases 4 \
  --stage2-runner deterministic \
  --workers 1

echo "[3/5] Running OpenAI Agno comparison batch"
CAS_STAGE2_FALLBACK_ON_ERROR=0 \
"$PYTHON_BIN" scripts/run_committee_review_evaluation_batch.py \
  --samples "$SAMPLES_PATH" \
  --output-dir "$AGNO_DIR" \
  --policy "$POLICY" \
  --per-category 1 \
  --max-cases 4 \
  --stage2-runner agno \
  --stage2-agno-mode "$AGNO_MODE" \
  --stage2-model-provider "$MODEL_PROVIDER" \
  --stage2-model "$MODEL_NAME" \
  "${CACHE_ARGS[@]}" \
  --workers 1

echo "[4/5] Exporting deterministic vs OpenAI Agno explanation comparison"
"$PYTHON_BIN" scripts/export_stage2_agno_explanation_comparison.py

echo "[5/5] Refreshing consolidated Stage 2 evaluation report"
"$PYTHON_BIN" scripts/export_stage2_evaluation_report.py

echo "[Done] Open these reports:"
echo "- data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_openai_agno_explanation_comparison.md"
echo "- data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents/stage2_evaluation_report.md"
