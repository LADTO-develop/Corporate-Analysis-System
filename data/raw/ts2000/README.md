# TS2000 Model Source

This directory stores the canonical TS2000 model source dataset used to
rebuild the CAS credit-risk model inputs.

- `TS2000_Credit_Model_Dataset_Model_V1.csv`: unified preprocessing baseline
  stored as the CAS canonical source after the target-rating preprocessing
  rules were consolidated. The 2025 domestic credit-rating disclosures that
  can be paired with fiscal-year 2024 feature rows are merged into this file.
  Company-year rows with missing TS2000 consolidated financial statement values
  are supplemented from OpenDART annual filings before the 43-feature inputs are
  rebuilt. The supplement rule uses CFS first and falls back to OFS only when CFS
  is unavailable.
  `industry_current_ratio_percentile` is retained here as a candidate feature,
  but the official Stage 1 runtime input remains the 43-feature set. The
  rejected 44-feature performance comparison is summarized under the 43-feature
  diagnostics folder.
- `feature_46_inference_2026_aux.csv`: minimal 2025 profile/market auxiliary
  source used to repair `feature_46_inference_2026.csv`. It carries only
  repository-local inference support fields such as latest available
  `firm_size_group` and 2025 `market_to_book` source values.

PDF/XLS disclosure exports and staging CSVs are not retained here. The usable
2025 disclosure labels have already been reflected in the canonical Model_V1
dataset above, so downstream feature generation should start from this file.

Baseline checks:
- Rows: 5,451
- 2025 evaluation-year rows: 287
- OpenDART financial-source supplement: 669 of 741 missing-source rows filled
  (`CFS=5`, `OFS=664`), leaving 73 rows without supplementable OpenDART values.
- 삼성전자(주): 10 company-year rows
- (주)토마토시스템: 1 company-year row (`2023 -> 2024`)

The official 43-feature inputs can be regenerated from this file with the
following sequence:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source-kind model-v1 --all-years --fallback-ofs
/opt/anaconda3/envs/aura/bin/python scripts/apply_opendart_financial_supplements.py
/opt/anaconda3/envs/aura/bin/python scripts/rebuild_feature_43_dataset.py
```

If the 2026 inference auxiliary source needs to be refreshed from the local
TS2000 raw extracts and OpenDART supplements, run:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/import_feature_43_inference_2026_aux.py
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_43_inference_2026.py
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source-kind inference --target-fiscal-year 2025 --fallback-ofs
/opt/anaconda3/envs/aura/bin/python scripts/apply_opendart_inference_financial_supplements.py
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_43_inference_2026.py --check-only
```
