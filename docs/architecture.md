# Architecture

## Intent

CAS is organized around the runtime structure in the system diagram:

- Offline preparation builds a selectable company universe, a company feature store,
  and a model registry.
- Online service code accepts a selected company, loads features,
  runs realtime XGBoost inference, applies deterministic rules, runs role-fixed
  multi-agent explanation, validates a strict JSON response, and renders the web
  dashboard payload.

The current repository uses local YAML and exported CSV/JSON artifacts as the
deterministic stand-ins for external stores. The node boundaries match the target
service architecture so each stand-in can be replaced without reshaping the graph.
The web-to-pipeline input contract starts in `docs/pipeline/data_pipeline.md`.

## Offline Preparation Area

```text
----------------------+      +-------------------------+
| Source data          | ---> | Preprocessing pipeline  |
| finance/market/info  |      +-------------------------+
+----------------------+                 |
                                         v
                              +-------------------------+
                              | Processed company list  |
                              +-------------------------+
                                         |
                                         v
                              +-------------------------+
                              | Company Feature Store   |
                              +-------------------------+
                                         |
                                         v
                              +-------------------------+
                              | Model Registry          |
                              | XGBoost model metadata  |
                              +-------------------------+

+----------------------+
| News/crawling slot   |
| placeholder only     |
+----------------------+
```

Local equivalents:

- `data/input/companies/*.yaml`: processed company selection records
- `configs/runtime/analysis.yaml`: feature ranges, registry metadata, rules
- `data/outputs/dashboard/*`: exported dashboard feature/model artifacts
- `src/cas/agents/nodes/news_overlay_node.py`: placeholder file for future crawling/news cache work

## Online Service Flow

```text
START
  -> data
  -> feature_store
  -> news_cache
  -> xgboost_inference
  -> rule_engine
  -> agno_agents
  -> json_schema
  -> report
  -> END
```

Node responsibilities:

- `data`: resolves the selected company from the processed company list.
- `feature_store`: builds the feature-store snapshot and records the active model registry ref.
- `news_cache`: collects optional external evidence when `CAS_ENABLE_EXTERNAL_EVIDENCE=1`;
  otherwise emits a deterministic disabled snapshot for CI and offline runs.
- `xgboost_inference`: emits realtime model output shaped as the production XGBoost result.
- `rule_engine`: converts model output plus cached context into the service risk band.
- `agno_agents`: runs fixed Stage 2 roles: QuantCreditAgent, EvidenceAuditAgent, ChairReportAgent.
  Role contracts live in `src/cas/agents/stage2_specs.py`, state is normalized
  through `src/cas/agents/stage2_bundle.py`, role-specific outputs are validated
  in `src/cas/agents/stage2_outputs.py`, execution goes through
  `src/cas/agents/stage2_runner.py`, EvidenceAuditAgent signal logic is split
  under `src/cas/agents/signals/`, the strict `committee_view` schema lives in
  `src/cas/agents/committee_schema.py`, veto rules are read from
  `configs/agent/committee.yaml`, and the payload is assembled in
  `src/cas/agents/committee_view.py`.
- `json_schema`: validates the strict dashboard response JSON.
- `report`: writes `latest.json` and `latest.md`.

## Strict Response JSON

The online service emits exactly these dashboard sections before rendering:

```json
{
  "company_overview": {},
  "model_result": {},
  "news_analysis": {},
  "agent_summary": {},
  "committee_view": {
    "final_committee_label": "적격 | 보류 | 부적격",
    "veto_triggered": false,
    "conflict_resolution": "",
    "key_risk_factors": [],
    "mitigating_factors": [],
    "evidence_summary": [],
    "final_review_memo": ""
  }
}
```

The Pydantic contract lives in `src/cas/agents/response_schema.py`. The report
writer stores the validated response as `data/outputs/reports/<company-id>/latest.json`
and keeps Markdown as a companion artifact.
