"""
Application Fraud Detection
=============================

Flags fraudulent loan / card applications at origination — primarily targeted
at synthetic-identity fraud and first-party fraud (bust-out, never-pay).

Synthetic identity is the fastest-growing fraud loss in US lending. The
classic signals are thin-file applicants where one or more identity
elements (SSN, email, phone, device, address) is too new to match the
applicant's stated age and history.

Run:
    python application_fraud.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03-shared-utilities"))

from model_evaluation import (  # noqa: E402
    evaluate_binary_classifier,
    find_threshold_for_recall,
    print_metrics_block,
)

DATA_PATH = ROOT / "data" / "loan_applications.csv"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Identity-consistency features — the heart of synthetic-identity detection."""
    df = df.copy()

    # Thin-file signals: ID elements that are suspiciously young
    df["new_email"] = (df.email_age_days < 30).astype(int)
    df["new_phone"] = (df.phone_age_days < 60).astype(int)
    df["new_address"] = (df.months_at_address < 3).astype(int)
    df["new_employer"] = (df.months_at_employer < 3).astype(int)

    # Bureau / inquiry burst
    df["high_inquiries"] = (df.num_inquiries_6m > 8).astype(int)

    # Income vs. age sanity check (high income for young applicants is suspicious)
    df["income_age_ratio"] = df.stated_income / df.age.clip(lower=18)
    df["young_high_income"] = (
        (df.age < 30) & (df.stated_income > 150_000)
    ).astype(int)

    # Composite identity-mismatch score (rule-based, fed as a feature)
    df["identity_mismatch_score"] = (
        df.new_email + df.new_phone + df.new_address +
        (1 - df.ip_country_matches_address) + (1 - df.ssn_age_consistent)
    )

    df["log_income"] = np.log1p(df.stated_income)
    return df


FEATURES = [
    "age", "log_income", "bureau_score", "months_at_address",
    "months_at_employer", "num_inquiries_6m", "email_age_days",
    "phone_age_days", "device_seen_before", "ip_country_matches_address",
    "ssn_age_consistent",
    # Engineered:
    "new_email", "new_phone", "new_address", "new_employer", "high_inquiries",
    "income_age_ratio", "young_high_income", "identity_mismatch_score",
]
TARGET = "fraud_flag"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_synthetic_data.py` first."
        )

    print("Loading applications…")
    df = pd.read_csv(DATA_PATH, parse_dates=["application_date"])
    df = add_engineered_features(df)
    print(f"  Total applications : {len(df):,}")
    print(f"  Fraud rate         : {df[TARGET].mean():.4%}")

    # OOT split — last 60 days
    cutoff = df.application_date.max() - pd.Timedelta(days=60)
    dev = df[df.application_date <= cutoff]
    oot = df[df.application_date > cutoff]
    print(f"  Dev rows : {len(dev):>6,}")
    print(f"  OOT rows : {len(oot):>6,}")

    X_dev, y_dev = dev[FEATURES], dev[TARGET]
    X_oot, y_oot = oot[FEATURES], oot[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.20, random_state=42, stratify=y_dev
    )

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    print(f"\nTraining XGBoost (scale_pos_weight={pos_weight:.1f})…")
    model = Pipeline([
        ("prep", ColumnTransformer([("num", StandardScaler(), FEATURES)])),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)

    p_val = model.predict_proba(X_val)[:, 1]
    p_oot = model.predict_proba(X_oot)[:, 1]

    # Application fraud teams typically run at higher recall (90%+) than
    # transaction fraud — false positives at origination cost an underwriter
    # 10 minutes of manual review, not a blocked transaction.
    thr = find_threshold_for_recall(y_val.values, p_val, target_recall=0.90)
    print(f"\nThreshold tuned for 90% recall : {thr:.6f}")

    print_metrics_block(evaluate_binary_classifier(
        y_val.values, p_val, threshold=thr, label="App-fraud / validation"
    ))
    print_metrics_block(evaluate_binary_classifier(
        y_oot.values, p_oot, threshold=thr, label="App-fraud / OOT"
    ))

    # Operational view: alert volume vs. fraud capture
    print("\nManual review queue sizing — what % of book do you pull, what fraud do you catch?")
    oot_sorted = pd.DataFrame({"y": y_oot.values, "p": p_oot}).sort_values("p", ascending=False)
    n_oot = len(oot_sorted)
    for pct in [0.5, 1, 2, 5, 10]:
        k = int(n_oot * pct / 100)
        top_k = oot_sorted.head(k)
        prec = top_k.y.mean()
        cap = top_k.y.sum() / max(y_oot.sum(), 1)
        print(f"  Review top {pct:>4}% ({k:>5} apps)  precision {prec:.4f}   "
              f"captured {cap:.4f} of fraud")

    # Feature importance
    print("\nTop features by gain:")
    booster = model.named_steps["clf"]
    importances = pd.Series(booster.feature_importances_, index=FEATURES) \
        .sort_values(ascending=False)
    print(importances.head(10).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
