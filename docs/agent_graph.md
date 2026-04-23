# Agent Graph

```mermaid
graph TD
    START --> data
    data --> feature
    feature --> base_prediction
    base_prediction --> market_overlay
    market_overlay --> news_overlay
    news_overlay --> committee
    committee --> report
    report --> END
```
