# Evaluation Datasets

This directory stores external evaluation labels that are not used for model
training, validation, threshold tuning, or test-set construction.

## `credit_rating_labels_2026.csv`

External validation labels for the 2026 prediction task.

- Source agency currently included: 한국기업평가
- Source period currently included: 2026-01-01 to 2026-03-31
- Source rating targets currently included: 회사채, 기업신용평가
- Input universe: `data/input/credit_44_features/feature_44_inference_2026.csv`
- Unit: one representative row per company and evaluation year

Selection rule:

1. Keep only disclosures whose company name matches the CAS 2026 inference
   universe.
2. Remove cancelled disclosures.
3. Use only company-bond and issuer-rating style disclosures.
4. If a company has multiple disclosures, choose the worst credit rating by
   `credit_rating_rank`; if tied, choose the latest `rating_date`.

Current row counts:

- Source-level matched rows: 60
- Representative company-year labels: 49
- KOSPI labels: 47
- KOSDAQ labels: 2
- Speculative-grade labels: 1
- Investment-grade labels: 48

## Staging

`staging/korea_ratings_2026_q1_matched_rows.csv` keeps the source-level matched
rows used to build the representative labels. This file is intentionally kept
for audit and manual review while the 2026 external validation set is being
assembled.

