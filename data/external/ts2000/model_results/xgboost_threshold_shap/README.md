# XGBoost Threshold Tuning and SHAP

- Dataset: `TS2000_Credit_Model_Dataset_Model_V1.csv`
- Feature set manifest: `TS2000_Model_Core29_Manifest.json`
- Split: `train=2014~2020`, `valid=2021~2022`, `test=2023~2024`
- Imputation: `marketwise_train_stats`
- Threshold tuning rule: `maximize precision subject to recall >= 0.80`
- Selected threshold: `0.5437`

## Threshold Summary

- Default 0.5 test: PR-AUC 0.7812, ROC-AUC 0.9088, Precision 0.6250, Recall 0.8434, F1 0.7179
- Tuned test: PR-AUC 0.7812, ROC-AUC 0.9088, Precision 0.6308, Recall 0.8133, F1 0.7105

## Top Grouped SHAP Features

- `gross_profit`: 0.621492
- `assets_total`: 0.577762
- `firm_size_group`: 0.516678
- `interest_coverage_ratio`: 0.448434
- `industry_macro_category`: 0.447916
- `net_margin`: 0.388058
- `depreciation`: 0.386978
- `capital_impairment_ratio`: 0.379419
- `dividend_payer`: 0.369981
- `listed_year`: 0.340678
