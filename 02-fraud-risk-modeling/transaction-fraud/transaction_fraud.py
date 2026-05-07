"""
Transaction Fraud Detection
=============================

Real-time fraud scoring for card / payment transactions. The dataset has
~0.5% fraud rate — realistic class imbalance for card-not-present + card-
present mixed traffic.

Two models, complementary:

    1. Supervised XGBoost — learns from confirmed fraud labels. The workhorse
                            of every fraud team in 2026.
    2. Isolation Forest    — unsupervised anomaly detector. Catches novel
                            patterns that haven't been labeled yet
                            (zero-day fraud), and runs in parallel as a
                            second-line check.

Threshold is tuned to a target *recall* rather than a default 0.5 cutoff —
fraud teams pick the threshold that lets them work the right number of
alerts per day given their analyst capacity.

Run:
    python transaction_fraud.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03-shared-utilities"))

from model_evaluation import (  # noqa: E402
    evaluate_binary_classifier,
    find_threshold_for_recall,
    print_metrics_block,
)

DATA_PATH = ROOT / "data" / "card_transactions.csv"

NUMERIC_FEATURES = [
    "amount", "hour_of_day", "txn_count_1h", "txn_count_24h", "amount_sum_24h",
]
CATEGORICAL_FEATURES = [
    "merchant_category", "channel", "country", "device_type",
]
TARGET = "is_fraud"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_synthetic_data.py` first."
        )
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """A few cheap features the raw data doesn't have yet."""
    df = df.copy()
    df["is_night"] = (df.hour_of_day < 6).astype(int)
    df["high_amount"] = (df.amount > df.amount.quantile(0.95)).astype(int)
    df["amount_log"] = np.log1p(df.amount)
    df["velocity_score"] = df.txn_count_1h * 3 + df.txn_count_24h
    return df


def time_split(
    df: pd.DataFrame, oot_days: int = 14
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last 14 days as the OOT set."""
    cutoff = df.timestamp.max() - pd.Timedelta(days=oot_days)
    dev = df[df.timestamp <= cutoff].copy()
    oot = df[df.timestamp > cutoff].copy()
    return dev, oot


def build_xgb_pipeline(scale_pos_weight: float) -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_FEATURES + ["amount_log", "velocity_score"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_FEATURES),
        ])),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="aucpr",   # better than AUC-ROC under heavy imbalance
            tree_method="hist",
            n_jobs=-1,
        )),
    ])


def main() -> None:
    print("Loading transactions…")
    df = add_engineered_features(load_data())
    print(f"  Total rows : {len(df):,}")
    print(f"  Fraud rate : {df[TARGET].mean():.4%}")
    print(f"  Date range : {df.timestamp.min().date()} → {df.timestamp.max().date()}")

    print("\nTime-based split (last 14 days held out as OOT)…")
    dev, oot = time_split(df, oot_days=14)
    print(f"  Dev rows : {len(dev):>7,}   fraud rate {dev[TARGET].mean():.4%}")
    print(f"  OOT rows : {len(oot):>7,}   fraud rate {oot[TARGET].mean():.4%}")

    feats = NUMERIC_FEATURES + ["amount_log", "velocity_score"] + CATEGORICAL_FEATURES
    X_dev, y_dev = dev[feats], dev[TARGET]
    X_oot, y_oot = oot[feats], oot[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.20, random_state=42, stratify=y_dev
    )

    # ------------------------------------------------------------------
    # Supervised XGBoost
    # ------------------------------------------------------------------
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"\nTraining XGBoost (scale_pos_weight={pos_weight:.1f})…")
    xgb = build_xgb_pipeline(pos_weight)
    xgb.fit(X_train, y_train)

    p_val = xgb.predict_proba(X_val)[:, 1]
    p_oot = xgb.predict_proba(X_oot)[:, 1]

    # Tune threshold on validation set to hit 80% recall
    thr = find_threshold_for_recall(y_val.values, p_val, target_recall=0.80)
    print(f"\nThreshold tuned on validation for 80% recall : {thr:.6f}")

    print_metrics_block(evaluate_binary_classifier(
        y_val.values, p_val, threshold=thr, label="XGBoost / validation"
    ))
    print_metrics_block(evaluate_binary_classifier(
        y_oot.values, p_oot, threshold=thr, label="XGBoost / OOT"
    ))

    # Top-20 alerts as a precision-at-k example
    print("\nPrecision @ top-K alerts on OOT (key fraud-ops metric):")
    oot_sorted = pd.DataFrame({"y": y_oot.values, "p": p_oot}).sort_values("p", ascending=False)
    for k in [50, 100, 500, 1000, 5000]:
        top_k = oot_sorted.head(k)
        prec = top_k.y.mean()
        cap = top_k.y.sum() / max(y_oot.sum(), 1)
        print(f"  Top {k:>5}  precision {prec:.4f}   captured {cap:.4f} of OOT fraud")

    # ------------------------------------------------------------------
    # Unsupervised Isolation Forest
    # ------------------------------------------------------------------
    print("\nTraining Isolation Forest (unsupervised anomaly detector)…")
    # Use a numeric subset; iForest doesn't need labels
    numeric_for_iso = NUMERIC_FEATURES + ["amount_log", "velocity_score"]
    iso = Pipeline([
        ("scale", StandardScaler()),
        ("iso", IsolationForest(
            n_estimators=200,
            contamination=0.01,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    iso.fit(X_train[numeric_for_iso])

    # Higher anomaly score = more anomalous; flip sign for ranking
    iso_score_oot = -iso.named_steps["iso"].score_samples(
        iso.named_steps["scale"].transform(X_oot[numeric_for_iso])
    )

    iso_metrics = evaluate_binary_classifier(
        y_oot.values, iso_score_oot,
        threshold=np.quantile(iso_score_oot, 0.99),
        label="IsolationForest / OOT (top 1% flagged)",
    )
    print_metrics_block(iso_metrics)

    # ------------------------------------------------------------------
    # Ensemble: alert if either model flags
    # ------------------------------------------------------------------
    print("\nEnsemble alert rule: flag if XGBoost prob > tuned threshold")
    print("                     OR isolation forest in top 1% anomaly score")
    xgb_alert = (p_oot >= thr).astype(int)
    iso_alert = (iso_score_oot >= np.quantile(iso_score_oot, 0.99)).astype(int)
    ensemble_alert = ((xgb_alert + iso_alert) >= 1).astype(int)

    captured = ((ensemble_alert == 1) & (y_oot.values == 1)).sum()
    total_fraud = y_oot.sum()
    alert_rate = ensemble_alert.mean()
    precision = captured / max(ensemble_alert.sum(), 1)
    recall = captured / max(total_fraud, 1)
    print(f"  Alert rate : {alert_rate:.4%}  ({ensemble_alert.sum():,} of {len(ensemble_alert):,} txns)")
    print(f"  Recall     : {recall:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Captured   : {captured} / {total_fraud} OOT fraud")

    print("\nDone.")


if __name__ == "__main__":
    main()
