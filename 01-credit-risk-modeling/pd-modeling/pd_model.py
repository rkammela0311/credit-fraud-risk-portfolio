"""
Probability of Default (PD) Model
==================================

End-to-end PD model on the synthetic loan book, with two stacked models:

    1. Logistic regression — interpretable, regulatory-friendly baseline
    2. XGBoost             — non-linear challenger, with SHAP explainability

Validation is OOT (out-of-time): we hold out the most recent 6 months as the
test set, mimicking how the model will actually be used in production.

Run:
    python pd_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# Make the shared utilities importable regardless of where we run from
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03-shared-utilities"))

from model_evaluation import (  # noqa: E402
    calibration_table,
    decile_lift_table,
    evaluate_binary_classifier,
    population_stability_index,
    print_metrics_block,
)

DATA_PATH = ROOT / "data" / "credit_loans.csv"


# ---------------------------------------------------------------------------
# Data loading and time-based split
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_synthetic_data.py` first."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["origination_date"])
    return df


def time_based_split(
    df: pd.DataFrame, oot_months: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """In-time development sample vs. out-of-time validation sample."""
    cutoff = df.origination_date.max() - pd.DateOffset(months=oot_months)
    dev = df[df.origination_date <= cutoff].copy()
    oot = df[df.origination_date > cutoff].copy()
    print(f"  Dev sample : {len(dev):>7,} rows  (up to {cutoff.date()})")
    print(f"  OOT sample : {len(oot):>7,} rows  (after {cutoff.date()})")
    print(f"  Dev default rate : {dev.default_flag.mean():.2%}")
    print(f"  OOT default rate : {oot.default_flag.mean():.2%}")
    return dev, oot


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "age", "annual_income", "employment_years", "loan_amount", "interest_rate",
    "term_months", "dti_ratio", "credit_history_years", "num_open_accounts",
    "num_delinquencies_2y", "revolving_utilization",
]
CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose"]
TARGET = "default_flag"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_FEATURES),
    ])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_logistic(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train) -> Pipeline:
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ---------------------------------------------------------------------------
# Validation suite
# ---------------------------------------------------------------------------
def run_validation_suite(
    name: str,
    model: Pipeline,
    X_train, y_train, X_val, y_val, X_oot, y_oot,
) -> None:
    p_train = model.predict_proba(X_train)[:, 1]
    p_val   = model.predict_proba(X_val)[:, 1]
    p_oot   = model.predict_proba(X_oot)[:, 1]

    print(f"\n========== {name} ==========")
    for label, y, p in [("train", y_train, p_train),
                        ("validation", y_val, p_val),
                        ("out-of-time", y_oot, p_oot)]:
        m = evaluate_binary_classifier(y, p, label=f"{name} / {label}")
        print_metrics_block(m)

    # Drift check: PSI on score distribution between train and OOT
    psi = population_stability_index(p_train, p_oot)
    print(f"\n  Score PSI (train vs OOT) : {psi:.4f}  "
          f"({'stable' if psi < 0.10 else 'moderate drift' if psi < 0.25 else 'major drift'})")

    # Calibration on OOT
    print(f"\n  OOT calibration (decile-level):")
    cal = calibration_table(y_oot, p_oot, n_bins=10)
    print(cal.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Decile lift on OOT
    print(f"\n  OOT decile lift:")
    lift = decile_lift_table(y_oot, p_oot)
    print(lift.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading data…")
    df = load_data()

    print("\nTime-based split:")
    dev, oot = time_based_split(df, oot_months=6)

    X_dev = dev[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_dev = dev[TARGET]
    X_oot = oot[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_oot = oot[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.20, random_state=42, stratify=y_dev
    )

    print("\nTraining logistic regression baseline…")
    logit = train_logistic(X_train, y_train)
    run_validation_suite("Logistic", logit, X_train, y_train, X_val, y_val, X_oot, y_oot)

    print("\nTraining XGBoost challenger…")
    xgb = train_xgboost(X_train, y_train)
    run_validation_suite("XGBoost",  xgb,   X_train, y_train, X_val, y_val, X_oot, y_oot)

    # Quick feature-importance summary for the challenger
    print("\nXGBoost top features by gain:")
    booster = xgb.named_steps["clf"]
    feat_names = (
        NUMERIC_FEATURES
        + list(xgb.named_steps["prep"].named_transformers_["cat"]
               .get_feature_names_out(CATEGORICAL_FEATURES))
    )
    importances = pd.Series(booster.feature_importances_, index=feat_names)
    print(importances.sort_values(ascending=False).head(10).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
