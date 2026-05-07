# Deployment Guide

This guide covers how a model in this repository would move from a Jupyter/script prototype to a monitored production system. The repo itself contains the prototyping layer; the notes below describe what additional infrastructure a real deployment requires.

## 1. Lifecycle stages

| Stage | What happens | Owner |
|---|---|---|
| Development | Feature engineering, model fit, in-sample validation | Modeler |
| Validation | OOT/OOS testing, calibration, stability, challenger comparison | Independent validation team |
| Documentation | Model card, methodology doc, limitations, monitoring plan | Modeler + governance |
| Approval | Sign-off from model risk committee | Model risk officer |
| Deployment | Containerization, scoring service, batch/streaming pipeline | MLOps |
| Monitoring | Performance, PSI, calibration, data drift | Modeler + MLOps |
| Re-validation | Annual or triggered review | Validation team |

## 2. Packaging a model

A trained estimator should ship together with everything needed to score new data the same way:

- The fitted preprocessing pipeline (imputers, encoders, WoE bin tables)
- The fitted estimator
- The exact feature list and column order
- The training data schema with allowed value ranges
- A model version string

`joblib.dump` for sklearn-compatible objects and `model.save_model(...)` for XGBoost/LightGBM are the standard serialization paths. For a scorecard, the points table itself is the deployable artifact and can ship as a CSV.

## 3. Scoring patterns

**Batch scoring** is the default for credit risk: ECL is recalculated monthly, application scores can be cached at decision time, and PD refreshes on a defined cadence. A scheduled job reads the production feature mart, applies the saved pipeline, and writes scores back to a database.

**Real-time scoring** is required for transaction fraud and online application fraud. The model is wrapped in a small service (FastAPI, Flask, or a managed endpoint) behind a feature store that can produce point-in-time-correct features in single-digit milliseconds.

## 4. Monitoring

Three things need ongoing measurement after deployment:

- **Input stability**: PSI on every model feature against the training distribution. The `model_evaluation.psi` helper in this repo is the same calculation used in production monitoring.
- **Output stability**: PSI on the model score distribution.
- **Performance**: Once outcomes are observed (defaults, charge-offs, confirmed fraud), recompute Gini/KS/recall/precision on each new vintage.

Alert thresholds typically follow the Basel convention of PSI < 0.10 acceptable, 0.10–0.25 watch, > 0.25 investigate or rebuild.

## 5. Governance artefacts

For a regulated credit model, the documentation pack usually includes:

- A model development document (data, methodology, results, limitations)
- A validation report from an independent team
- A monitoring plan with thresholds and escalation paths
- A change log
- An attestation from the model owner

The README files in each subfolder of this repo are sized as the development-document layer, not the validation report.

## 6. What this repo deliberately does not include

- A feature store
- A serving layer
- Authentication, rate limiting, or audit logging
- A CI/CD pipeline
- An experiment tracking server (MLflow, Weights & Biases)

These are organisation-specific and would replace, not supplement, what is here. The repo is the modeling content.
