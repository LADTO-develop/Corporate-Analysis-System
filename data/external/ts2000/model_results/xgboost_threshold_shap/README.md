# XGBoost Threshold Tuning and SHAP

- Dataset: `TS2000_Credit_Model_Dataset_Model_V1.csv`
- Feature set manifest: `TS2000_Model_Core29_Manifest.json`
- Split: `train=2014~2020`, `valid=2021~2022`, `test=2023~2024`
- Imputation: `marketwise_train_stats`
- Threshold tuning rule: `maximize precision subject to recall >= 0.80`
- Selected threshold: `0.5375`

## Threshold Summary

- Default 0.5 test: PR-AUC 0.7621, ROC-AUC 0.8998, Precision 0.6233, Recall 0.8323, F1 0.7128
- Tuned test: PR-AUC 0.7621, ROC-AUC 0.8998, Precision 0.6233, Recall 0.8024, F1 0.7016

## Top Grouped SHAP Features

- `gross_profit`: 0.625359
- `assets_total`: 0.567099
- `firm_size_group`: 0.507796
- `interest_coverage_ratio`: 0.454133
- `industry_macro_category`: 0.427883
- `net_margin`: 0.425991
- `depreciation`: 0.415514
- `capital_impairment_ratio`: 0.397672
- `dividend_payer`: 0.345990
- `listed_year`: 0.326776
