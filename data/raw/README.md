# `data/raw/`

Reserved for raw and canonical source datasets that downstream processing
scripts need in order to be reproducible from a fresh repository checkout.

Current contents:
- `ts2000/`: canonical TS2000 model source dataset for rebuilding
  `data/input/credit_43_features/`, plus a minimal 2025 auxiliary source for
  repairing the 2026 inference feature table.
- `dart/`, `ecos/`, `ratings/`, `news/`: reserved raw-source folders for
  future ingestion.
