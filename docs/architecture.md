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
- `news_cache`: placeholder node only; crawling/news-cache implementation is intentionally absent.
- `xgboost_inference`: emits realtime model output shaped as the production XGBoost result.
- `rule_engine`: converts model output plus cached context into the service risk band.
- `agno_agents`: runs fixed roles: news summary, model interpretation, risk review, synthesis/format.
- `json_schema`: validates the strict dashboard response JSON.
- `report`: writes `latest.json` and `latest.md`.

## Strict Response JSON

The online service emits exactly these dashboard sections before rendering:

```json
{
  "company_overview": {},
  "model_result": {},
  "news_analysis": {},
  "agent_summary": {}
}
```

The Pydantic contract lives in `src/cas/agents/response_schema.py`. The report
writer stores the validated response as `data/outputs/reports/<company-id>/latest.json`
and keeps Markdown as a companion artifact.
