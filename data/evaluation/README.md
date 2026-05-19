# Evaluation Datasets

This directory stores external evaluation labels that are not used for model
training, validation, threshold tuning, or test-set construction.

## `target_label_reference.csv`

Diagnostic target-label reference for the 2015-2025 Model V1 universe.

This file exists so model diagnostics can reproduce credit-rating boundary
analysis without putting rating strings back into model input data.

- Source inputs: legacy `Target_Processed_audit` plus 2025 disclosure labels
- Unit: one representative row per company and evaluation year
- Model input leakage policy: `credit_rating` and `credit_rating_rank` are for
  diagnostics only and must not be used as feature columns
- Selection rule: choose the worst credit rating within available disclosure
  candidates; if tied, choose the latest rating date

Current row counts:

- Model V1 rows: 5,451
- Label-reference rows: 5,443
- 2025 Model V1 rows: 287
- 2025 label-reference rows: 279
- 2025 rows without preserved rating string: 8

Key diagnostic fields:

- `credit_rating`, `credit_rating_rank`
- `rating_agency`, `rating_agency_group`
- `rating_target`, `rating_date`
- `source_label_set`

Use this file for BBB-/BB+ boundary analysis and error diagnostics only.

## `credit_rating_labels_2026.csv`

External validation labels for the 2026 prediction task.

- Source agencies currently included: 한국기업평가, 한국신용평가, NICE신용평가
- Source periods currently included:
  - 한국기업평가: 2026-01-01 to 2026-03-31, 2026-04-01 to 2026-05-19
  - 한국신용평가: 2026-01-01 to 2026-05-19
  - NICE신용평가: 2026-01-01 to 2026-05-19
- Source rating targets currently included: 회사채, 기업신용평가
- Input universe: `data/input/credit_43_features/feature_43_inference_2026.csv`
- Unit: one representative row per company and evaluation year
- Schema: Model_V1 key columns plus representative credit-rating fields only

Selection rule:

1. Keep only disclosures whose company name matches the CAS 2026 inference
   universe.
2. Remove cancelled disclosures.
3. Use only company-bond and issuer-rating style disclosures.
4. If a company has multiple disclosures, choose the worst credit rating by
   `credit_rating_rank`; if tied, choose the latest `rating_date`.

Current row counts:

- Source-level matched rows: 399
- Representative company-year labels: 141
- KOSPI labels: 121
- KOSDAQ labels: 20
- Speculative-grade labels: 15
- Investment-grade labels: 126

Final columns:

- `market`, `stock_code`, `corp_name`, `fiscal_year`, `eval_year`
- `is_speculative`
- `credit_rating`, `credit_rating_rank`
- `rating_agency`, `rating_agency_code`, `rating_target`, `rating_date`,
  `current_outlook`

## Cleanup Policy

PDF/XLS disclosure exports and source-level staging CSVs are not retained in
the final repository state. The final evaluation labels above are the dataset
used for external validation, and `credit_rating_labels_2026_summary.json`
keeps aggregate source counts for audit at the project-summary level.
