# `data/raw/`

Reserved for raw and canonical source datasets that downstream processing
scripts need in order to be reproducible from a fresh repository checkout.

Current contents:
- `ts2000/`: canonical TS2000 model source dataset for rebuilding
  `data/input/credit_46_features/`, plus a minimal 2025 auxiliary source for
  repairing the 2026 inference feature table.
- `opendart/`: OpenDART annual-report financial statement rows, summaries,
  and supplement audit files used to fill TS2000 CFS gaps. The supplement
  policy is CFS first and OFS fallback only when CFS is unavailable.

PDF/XLS disclosure downloads and intermediate rating-staging CSVs are not kept
after their usable labels are reflected in the canonical TS2000 dataset.
