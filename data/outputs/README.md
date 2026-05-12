# `data/outputs/`

Generated outputs live here. These files are derived artifacts used by the
dashboard, reports, and local inspection workflows.

Current outputs:
- `dashboard/feature_43_mvp/`: dashboard-ready company lists, prediction
  scores, SHAP outputs, peer comparisons, and manifests.
- `modeling/feature_43_xgboost/`: team-facing 43-feature XGBoost model JSON
  and metadata synced from the current CAS Model V1 baseline.
- `reports/`: generated per-company Markdown and JSON reports.

To rebuild dashboard outputs:

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_dashboard_artifacts.py
```
