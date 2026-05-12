# Data Layout

This directory keeps the project datasets and generated artifacts in one
reproducible flow.

## Canonical Flow

1. `data/raw/ts2000/`
   - Canonical TS2000 model source dataset.
   - Current baseline: `TS2000_Credit_Model_Dataset_Model_V1.csv`.
   - This is the dataset produced after the unified target preprocessing rules.

2. `data/input/credit_43_features/`
   - Model-ready 43-feature input tables derived from the raw TS2000 Model V1
     source.
   - Contains the master table, train/valid/test matrices, split identifiers,
     and feature metadata.

3. `data/external/model_artifacts/feature_43_xgboost/`
   - Tracked Stage 1 XGBoost model artifacts used by dashboard inference.
   - Contains the model file and model metadata.

4. `data/outputs/dashboard/feature_43_mvp/`
   - Dashboard-ready exports derived from `credit_43_features`.
   - Contains company lists, prediction scores, SHAP summaries, peer
     percentiles, and the dashboard manifest.

5. `data/outputs/reports/`
   - Generated per-company report outputs.
   - These are run artifacts, not source datasets.

## Regeneration

Rebuild the 43-feature input tables from the canonical raw source:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/rebuild_feature_43_dataset.py
```

Rebuild dashboard/model artifacts from the 43-feature input tables:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_dashboard_artifacts.py
```

## Current Baseline Checks

- Canonical TS2000 Model V1 rows: 5,199
- `feature_43_master.csv` rows: 5,199
- 삼성전자(주): 10 company-year rows
- (주)토마토시스템: 1 company-year row (`2023 -> 2024`)
