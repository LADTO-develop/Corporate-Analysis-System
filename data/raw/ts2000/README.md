# TS2000 Model Source

This directory stores the canonical TS2000 model source dataset used to
rebuild the 43-feature credit-risk inputs.

- `TS2000_Credit_Model_Dataset_Model_V1.csv`: unified preprocessing baseline
  stored as the CAS canonical source after the target-rating preprocessing
  rules were consolidated.

Baseline checks:
- Rows: 5,199
- 삼성전자(주): 10 company-year rows
- (주)토마토시스템: 1 company-year row (`2023 -> 2024`)

The 43-feature inputs in `data/input/credit_43_features/` can be regenerated
from this file with:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/rebuild_feature_43_dataset.py
```
