# Fraud Risk Modeling

Three projects covering the main fraud surfaces a lender or payment institution faces:

1. **Transaction Fraud** — real-time scoring of card / payment transactions. XGBoost on heavy class imbalance + Isolation Forest as a complementary unsupervised detector.
2. **Application Fraud** — flagging fraudulent applications at origination, with a focus on synthetic-identity detection.
3. **Network Analysis** — graph-based fraud ring detection. Surfaces shared-attribute clusters that look unrelated row-by-row.

## How they fit together

A real fraud-risk stack uses all three in concert:

```
Application arrives ──► Application-fraud model     (tabular ML, identity signals)
                  │
                  └──► Network features (graph)     (do they belong to a known ring?)
                                                    
Account opened ──► Transactions begin ──► Transaction fraud model (tabular ML + iForest)
                                       │
                                       └──► Network features (shared device / IP velocity)
```

The graph features from the network analysis feed *back into* the supervised models — that's how the rings get caught. Component fraud rate and component size are usually two of the highest-IV features in a mature fraud stack.

## Where this differs from credit risk

| | Credit risk | Fraud risk |
|---|---|---|
| Class imbalance | 1–10% | 0.1–1% |
| Time horizon | Months | Seconds |
| Attacker adapts | Slowly (macro cycles) | Continuously |
| False positive cost | Lost revenue | Lost customer experience |
| Validation focus | Calibration, KS, Gini | Precision @ K, recall at threshold |
| Champion model | Logistic regression | Gradient boosting |

Both rely on the same statistical machinery, but the metrics and operational rhythm are quite different. Models in `01-credit-risk-modeling/` use Gini and calibration; models here use precision @ K and tuned-threshold recall.
