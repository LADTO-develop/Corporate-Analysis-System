# `data/input/`

Model input datasets live here. Files in this directory are derived from
canonical raw sources and should be regenerated through scripts rather than
edited by hand.

Current datasets:
- `credit_46_features/`: current 46-feature XGBoost input set used by the runtime.
  It is regenerated from the canonical TS2000 Model V1 source after OpenDART
  CFS/OFS financial-statement supplementation.
- `companies/`: reserved for sample or custom company input payloads.

Candidate feature experiments are not retained as separate input folders. Keep
small comparison summaries under `data/outputs/modeling/feature_46_xgboost/diagnostics/`
when a rejected candidate needs to be explained.
