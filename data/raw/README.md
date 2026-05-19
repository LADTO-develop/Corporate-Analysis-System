# `data/raw/`

Reserved for raw and canonical source datasets that downstream processing
scripts need in order to be reproducible from a fresh repository checkout.

Current contents:
- `ts2000/`: canonical TS2000 model source dataset for rebuilding
  `data/input/credit_43_features/`, plus a minimal 2025 auxiliary source for
  repairing the 2026 inference feature table.

PDF/XLS disclosure downloads and intermediate rating-staging CSVs are not kept
after their usable labels are reflected in the canonical TS2000 dataset.
