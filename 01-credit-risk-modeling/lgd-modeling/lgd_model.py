"""
Loss Given Default (LGD) Model
================================

Predicts the loss severity on defaulted accounts — the second leg of the
Basel/IFRS 9 loss formula:

    Expected Loss = PD × LGD × EAD

LGD is bounded in [0, 1], usually U-shaped (lots of full recoveries and lots
of total losses, less mass in the middle), and traditionally hard to model
well. We use a two-stage approach:

    Stage 1 (cure model)     : predict P(LGD == 0)  — accounts that fully cure
    Stage 2 (severity model) : among non-cured accounts, predict the LGD value

Final LGD = (1 - P(cure)) × E[LGD | not cured]

This decomposition is standard in IFRS 9 / Basel A-IRB submissions and almost
always beats a single regressor on raw LGD.

Run:
    python lgd_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03-shared-utilities"))

DATA_PATH = ROOT / "data" / "credit_loans.csv"

NUMERIC_FEATURES = [
    "annual_income", "loan_amount", "interest_rate", "term_months",
    "dti_ratio", "credit_history_years", "revolving_utilization",
]
CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose"]


def load_defaulted() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_synthetic_data.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    return df[df.default_flag == 1].copy()


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_FEATURES),
    ])


def main() -> None:
    print("Loading defaulted accounts only…")
    df = load_defaulted()
    print(f"  N defaulted accounts : {len(df):,}")
    print(f"  Mean LGD             : {df.loss_given_default.mean():.4f}")
    print(f"  Median LGD           : {df.loss_given_default.median():.4f}")
    print(f"  P(cure, LGD=0)       : {(df.loss_given_default <= 0.05).mean():.4f}")

    # ------------------------------------------------------------------
    # Stage 1: cure model — binary classifier for "fully/largely cured" accounts.
    # Threshold of 0.25 means we treat LGD <= 25% as a "low-loss / cured"
    # outcome — matches how most banks operationally define a workout cure.
    # ------------------------------------------------------------------
    df["cured"] = (df.loss_given_default <= 0.25).astype(int)

    train, test = train_test_split(df, test_size=0.25, random_state=42, stratify=df.cured)

    print("\n--- Stage 1: cure model (logistic regression) ---")
    cure_model = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    cure_model.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train.cured)

    p_cure_test = cure_model.predict_proba(
        test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )[:, 1]
    from sklearn.metrics import roc_auc_score
    print(f"  Cure model AUC (test): {roc_auc_score(test.cured, p_cure_test):.4f}")

    # ------------------------------------------------------------------
    # Stage 2: severity model on non-cured accounts
    # ------------------------------------------------------------------
    train_nc = train[train.cured == 0].copy()
    test_nc = test[test.cured == 0].copy()

    print(f"\n--- Stage 2: severity model on {len(train_nc):,} non-cured accounts ---")
    severity_model = Pipeline([
        ("prep", build_preprocessor()),
        ("reg", XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        )),
    ])
    severity_model.fit(
        train_nc[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        train_nc.loss_given_default,
    )

    sev_pred_test = severity_model.predict(
        test_nc[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )
    print(f"  Severity MAE  (test, non-cured): "
          f"{mean_absolute_error(test_nc.loss_given_default, sev_pred_test):.4f}")
    print(f"  Severity RMSE (test, non-cured): "
          f"{np.sqrt(mean_squared_error(test_nc.loss_given_default, sev_pred_test)):.4f}")
    print(f"  Severity R²   (test, non-cured): "
          f"{r2_score(test_nc.loss_given_default, sev_pred_test):.4f}")

    # ------------------------------------------------------------------
    # Combined two-stage prediction on full test set
    # ------------------------------------------------------------------
    p_cure_full = cure_model.predict_proba(
        test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )[:, 1]
    sev_full = severity_model.predict(
        test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    ).clip(0, 1)
    lgd_pred = (1 - p_cure_full) * sev_full
    lgd_pred = np.clip(lgd_pred, 0, 1)

    print(f"\n--- Combined two-stage LGD on full test set ---")
    print(f"  MAE  : {mean_absolute_error(test.loss_given_default, lgd_pred):.4f}")
    print(f"  RMSE : {np.sqrt(mean_squared_error(test.loss_given_default, lgd_pred)):.4f}")
    print(f"  R²   : {r2_score(test.loss_given_default, lgd_pred):.4f}")

    # Bucketed calibration: predicted LGD decile vs. actual mean LGD
    cal = pd.DataFrame({
        "actual": test.loss_given_default.values,
        "predicted": lgd_pred,
    })
    cal["decile"] = pd.qcut(cal.predicted, q=10, labels=False, duplicates="drop")
    cal_table = cal.groupby("decile").agg(
        n=("actual", "size"),
        actual_lgd=("actual", "mean"),
        predicted_lgd=("predicted", "mean"),
    ).reset_index()
    cal_table["abs_error"] = (cal_table.actual_lgd - cal_table.predicted_lgd).abs()
    print("\nDecile calibration (predicted LGD vs. actual):")
    print(cal_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
