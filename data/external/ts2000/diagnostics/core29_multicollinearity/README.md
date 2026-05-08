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

- High-correlation pairs (`|r| >= 0.70`): 5
- Severe VIF variables (`>= 10`): 0
- Caution VIF variables (`5 ~ <10`): 3

## Top 10 absolute correlations

- `gross_profit` vs `depreciation`: 0.903
- `assets_total` vs `depreciation`: 0.843
- `current_ratio` vs `cash_ratio`: 0.835
- `assets_total` vs `gross_profit`: 0.814
- `pretax_roa` vs `operating_roa`: 0.814
- `equity_ratio` vs `total_borrowings_ratio`: -0.669
- `pretax_roa` vs `accruals_ratio`: 0.592
- `equity_ratio` vs `total_debt_turnover`: 0.587
- `operating_roa` vs `ocf_to_total_liabilities`: 0.525
- `net_margin` vs `pretax_roa`: 0.522

## Top 10 VIF

- `pretax_roa`: VIF 7.951 (caution)
- `depreciation`: VIF 6.855 (caution)
- `gross_profit`: VIF 6.093 (caution)
- `current_ratio`: VIF 4.747 (ok)
- `assets_total`: VIF 4.396 (ok)
- `operating_roa`: VIF 3.888 (ok)
- `cash_ratio`: VIF 3.732 (ok)
- `accruals_ratio`: VIF 3.389 (ok)
- `ocf_to_total_liabilities`: VIF 2.802 (ok)
- `equity_ratio`: VIF 2.659 (ok)
