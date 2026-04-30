# Core29 Multicollinearity Summary

- Dataset: `TS2000_Credit_Model_Dataset_Model_V1.csv`
- Feature set: `TS2000_Model_Core29_Manifest.json`
- Core29 feature count: 29
- Numeric features analyzed for correlation/VIF: 26
- Categorical features analyzed for association: 3

## Suggested interpretation rules

- `|correlation| >= 0.70`: high correlation
- `|correlation| >= 0.85`: very strong overlap
- `VIF >= 5`: caution
- `VIF >= 10`: severe multicollinearity

## Quick counts

- High-correlation pairs (`|r| >= 0.70`): 3
- Severe VIF variables (`>= 10`): 0
- Caution VIF variables (`5 ~ <10`): 1

## Top 10 absolute correlations

- `current_ratio` vs `cash_ratio`: 0.835
- `pretax_roa` vs `operating_roa`: 0.813
- `gross_profit` vs `depreciation`: 0.723
- `assets_total` vs `gross_profit`: 0.685
- `assets_total` vs `depreciation`: 0.676
- `equity_ratio` vs `total_borrowings_ratio`: -0.667
- `pretax_roa` vs `accruals_ratio`: 0.594
- `equity_ratio` vs `total_debt_turnover`: 0.586
- `operating_roa` vs `ocf_to_total_liabilities`: 0.524
- `net_margin` vs `pretax_roa`: 0.523

## Top 10 VIF

- `pretax_roa`: VIF 7.991 (caution)
- `current_ratio`: VIF 4.751 (ok)
- `operating_roa`: VIF 3.883 (ok)
- `cash_ratio`: VIF 3.734 (ok)
- `accruals_ratio`: VIF 3.423 (ok)
- `ocf_to_total_liabilities`: VIF 2.789 (ok)
- `gross_profit`: VIF 2.696 (ok)
- `equity_ratio`: VIF 2.651 (ok)
- `depreciation`: VIF 2.597 (ok)
- `assets_total`: VIF 2.533 (ok)
