# Data Layout

This directory keeps the project datasets and generated artifacts in one
reproducible flow.

## Canonical Flow

1. `data/raw/ts2000/`
   - Canonical TS2000 model source dataset.
   - Current baseline: `TS2000_Credit_Model_Dataset_Model_V1.csv`.
   - This is the dataset produced after the unified target preprocessing rules.
   - CFS-missing company-years are supplemented from OpenDART annual reports
     with a CFS-first, OFS-fallback rule.

2. `data/raw/opendart/`
   - OpenDART financial-statement raw rows, summary files, and supplement
     audit trails used to repair TS2000 CFS gaps.

3. `data/input/credit_46_features/`
   - Model-ready 46-feature input tables derived from the raw TS2000 Model V1
     source.
   - Contains the master table, train/valid/test matrices, split identifiers,
     and feature metadata.

4. `data/outputs/dashboard/feature_46_mvp/`
   - Dashboard-ready exports derived from `credit_46_features`.
   - Contains company lists, prediction scores, SHAP summaries, peer
     percentiles, and the dashboard manifest.

5. `data/outputs/modeling/feature_46_xgboost/`
   - Stage 1 XGBoost model artifacts used by runtime inference and team
     handoff.
   - Contains the model JSON and model metadata.

6. `data/outputs/reports/`
   - Generated per-company report outputs.
   - These are run artifacts, not source datasets.

## Regeneration

Rebuild the 46-feature input tables from the canonical raw source:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source-kind model-v1 --all-years --fallback-ofs
/opt/anaconda3/envs/aura/bin/python scripts/apply_opendart_financial_supplements.py
/opt/anaconda3/envs/aura/bin/python scripts/rebuild_feature_46_dataset.py
```

Repair and validate the 2026 inference table:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/import_feature_46_inference_2026_aux.py
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_46_inference_2026.py
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source-kind inference --target-fiscal-year 2025 --fallback-ofs
/opt/anaconda3/envs/aura/bin/python scripts/apply_opendart_inference_financial_supplements.py
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_46_inference_2026.py --check-only
```

Rebuild dashboard/model artifacts from the 46-feature input tables:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_dashboard_artifacts.py
```

## Current Baseline Checks

- Canonical TS2000 Model V1 rows: 5,451
- `feature_46_master.csv` rows: 5,451
- Model V1 financial-source missing rows after OpenDART supplement: 73
- 2026 inference financial-source missing rows after OpenDART supplement: 2
- 삼성전자(주): 10 company-year rows
- (주)토마토시스템: 1 company-year row (`2023 -> 2024`)
- Current tuned-threshold test performance: PR-AUC 0.8321, ROC-AUC 0.9415,
  Precision 0.6941, Recall 0.8719, F1 0.7729
