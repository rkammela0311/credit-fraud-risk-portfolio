# Credit Risk Modeling

Four projects covering the lending lifecycle from underwriting through impairment accounting:

1. **PD Modeling** — the backbone: probability the borrower defaults in 12 months. Logistic regression baseline + XGBoost challenger.
2. **LGD Modeling** — how much of a defaulted loan you actually lose, after recoveries. Two-stage cure + severity model.
3. **Credit Scoring** — application scorecard scaled to FICO-style points. The format underwriters and adverse-action notices actually need.
4. **IFRS 9 ECL** — wires PD × LGD × EAD together with stage classification to produce the loan-loss provision that lands in the income statement.

Each subfolder has its own README walking through the methodology, validation results, and what would need to change for a real production deployment.

## Suggested reading order

If you're new to the field, work through them in this order:
1. `pd-modeling/` (the model itself)
2. `credit-scoring/` (how the model gets deployed in underwriting)
3. `lgd-modeling/` (the second leg of the loss equation)
4. `ifrs9-ecl/` (how it all rolls up into accounting numbers)
