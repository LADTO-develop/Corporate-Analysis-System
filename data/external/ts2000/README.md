## TS2000 Dataset Package

This folder contains the official TS2000 dataset package copied into the repository for dashboard development and LLM-based explanation workflows.

### What is included

- `TS2000_Credit_Model_Dataset.csv`
  - Full master dataset.
  - Includes all processed company-year level features kept in the official export.

- `TS2000_Credit_Model_Dataset_Model_V1.csv`
  - Model-ready baseline dataset.
  - This is the main table to use for further model development.
  - It is included on purpose even though Core29 is defined separately, because future modeling may use a broader feature set than Core29.

- `TS2000_Model_V1_Manifest.json`
  - Metadata for the model-ready dataset.
  - Defines target column, id/time columns, feature groups, and dataset usage rules.

- `TS2000_Model_Core29_Manifest.json`
  - Official compact feature-set definition used for the current explainable baseline model.

- `TS2000_Model_Core29_Features.csv`
  - Flat list of the 29 official Core29 features.

### Supporting folders

- `column_dictionary/`
  - Variable dictionary workbook and machine-readable metadata.

- `docs/`
  - Human-readable and AI-readable handoff documents.
  - Includes the raw-to-Model_V1 preprocessing and merge rules.

- `model_results/`
  - Official model comparison outputs.
  - Includes Logistic Regression, XGBoost, LightGBM comparison and XGBoost threshold/SHAP outputs.

- `diagnostics/`
  - Multicollinearity and correlation diagnostics for the official Core29 feature set.

### Recommended usage

For continued modeling:
- start from `TS2000_Credit_Model_Dataset_Model_V1.csv`
- use `TS2000_Model_V1_Manifest.json` for dataset structure
- use `TS2000_Model_Core29_Manifest.json` only when you want to reproduce the official compact Core29 baseline

For LLM explanation:
- use `docs/00_AI_Handoff_Core29.md`
- use `docs/00_Build_Process_Model_V1_OnePager.md`
- use `model_results/xgboost_threshold_shap/` for SHAP-based explanation context

### Important note

This package intentionally includes both:
- the broader `Model_V1` dataset for future feature expansion
- the narrower `Core29` definition for the current official explainable baseline

So `Core29` should be treated as a named feature subset, not as the only dataset available in this folder.
