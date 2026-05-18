# TS2000 Model Source

This directory stores the canonical TS2000 model source dataset used to
rebuild the 43-feature credit-risk inputs.

- `TS2000_Credit_Model_Dataset_Model_V1.csv`: unified preprocessing baseline
  stored as the CAS canonical source after the target-rating preprocessing
  rules were consolidated.
- `feature_43_inference_2026_aux.csv`: minimal 2025 profile/market auxiliary
  source used to repair `feature_43_inference_2026.csv`. It carries only
  repository-local inference support fields such as latest available
  `firm_size_group` and 2025 `market_to_book` source values.

Baseline checks:
- Rows: 5,199
- 삼성전자(주): 10 company-year rows
- (주)토마토시스템: 1 company-year row (`2023 -> 2024`)

The 43-feature inputs in `data/input/credit_43_features/` can be regenerated
from this file with:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/rebuild_feature_43_dataset.py
```

If the 2026 inference auxiliary source needs to be refreshed from the local
TS2000 raw extracts, run:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/import_feature_43_inference_2026_aux.py
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_43_inference_2026.py
```
