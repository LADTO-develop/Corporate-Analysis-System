# Architecture

## Intent

This project is now positioned as a generic corporate analysis system scaffold.

The design keeps LangGraph as the orchestration backbone and makes it easy to add
market-specific policies, ML models, and report-generation steps over time.

## Runtime Flow

```text
START
  -> data
  -> feature
  -> base_prediction
  -> market_overlay
  -> news_overlay
  -> committee
  -> report
  -> END
```

## State Design

The state keeps four categories of information:

- identity: company id, market, analysis year
- evidence: raw profile, normalized features, overlay outputs
- decisions: lens scores, committee reviews, final recommendation
- traceability: audit entries and generated artifact paths

## Configuration

The runtime is driven by:

- `configs/agent/graph.yaml`
- `configs/agent/committee.yaml`
- `configs/runtime/analysis.yaml`

## Output

Every run writes:

- a machine-readable JSON report
- a Markdown summary
- an append-only audit trail embedded in the report
