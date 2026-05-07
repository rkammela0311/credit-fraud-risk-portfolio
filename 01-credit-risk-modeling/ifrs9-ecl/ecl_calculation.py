"""
IFRS 9 Expected Credit Loss (ECL) Calculation
==============================================

Combines PD, LGD, and EAD into the IFRS 9 / CECL provisions used in the
financial statements of any IFRS-reporting bank.

Three-stage classification per the standard:
    Stage 1 — performing       : 12-month ECL,   no SICR (significant increase
                                                  in credit risk)
    Stage 2 — under-performing : lifetime ECL,   SICR triggered, not yet
                                                  credit-impaired
    Stage 3 — non-performing   : lifetime ECL,   credit-impaired (defaulted)

ECL formulas:
    Stage 1 :  ECL = PD_12m       × LGD × EAD
    Stage 2 :  ECL = Σ_t  PD_t    × LGD × EAD_t × DF_t      (lifetime)
    Stage 3 :  ECL =      LGD     × EAD                      (default occurred)

This module is a worked numerical example, not a full model. PD/LGD/EAD
are taken as inputs (you'd source them from the dedicated PD and LGD models).

Run:
    python ecl_calculation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03-shared-utilities"))

DATA_PATH = ROOT / "data" / "credit_loans.csv"


# ---------------------------------------------------------------------------
# Stage classification
# ---------------------------------------------------------------------------
def classify_stage(
    df: pd.DataFrame,
    pd_12m: np.ndarray,
    pd_origination: np.ndarray,
    sicr_threshold: float = 2.0,
) -> np.ndarray:
    """
    Stage assignment per IFRS 9.

    Logic:
        Stage 3 — defaulted today
        Stage 2 — SICR triggered: PD has at least doubled vs origination,
                  OR account is 30+ DPD (we proxy DPD with delinquencies>=2)
        Stage 1 — everything else (performing, no SICR)
    """
    stage = np.ones(len(df), dtype=int)  # default to Stage 1

    # SICR via PD ratio (12-month forward / at-origination PD)
    pd_ratio = pd_12m / np.maximum(pd_origination, 1e-6)
    sicr = pd_ratio >= sicr_threshold

    # Backstop: 30+ DPD ≈ recent delinquencies
    backstop = df.num_delinquencies_2y.values >= 2

    stage[(sicr | backstop)] = 2
    stage[df.default_flag.values == 1] = 3
    return stage


# ---------------------------------------------------------------------------
# Lifetime PD curve
# ---------------------------------------------------------------------------
def lifetime_pd_curve(pd_12m: np.ndarray, term_months: np.ndarray) -> np.ndarray:
    """
    Convert a 12-month PD into a sequence of marginal monthly PDs over the
    loan's remaining life, using a geometric decay assumption.

    Industry shortcut (used widely as a starting point before macro overlays):
        monthly hazard λ_m  such that (1-λ_m)^12 = 1 - PD_12m
    """
    monthly_hazard = 1 - (1 - pd_12m) ** (1 / 12)
    n_loans = len(pd_12m)
    max_term = int(term_months.max())

    # Marginal default probability at month t = survival(t-1) * λ
    surv = np.ones(n_loans)
    marginal = np.zeros((n_loans, max_term))
    for t in range(max_term):
        marginal[:, t] = surv * monthly_hazard
        surv = surv * (1 - monthly_hazard)
        # Clip to actual loan term — payments stop after term ends
        active = (t + 1) <= term_months
        marginal[~active, t] = 0
    return marginal


# ---------------------------------------------------------------------------
# Amortization-based EAD path
# ---------------------------------------------------------------------------
def amortizing_ead(
    loan_amount: np.ndarray,
    interest_rate_pct: np.ndarray,
    term_months: np.ndarray,
) -> np.ndarray:
    """Outstanding balance month-by-month for a fully-amortizing loan."""
    n = len(loan_amount)
    max_term = int(term_months.max())
    monthly_rate = interest_rate_pct / 100 / 12

    # Standard amortization: payment = P * r / (1 - (1+r)^-n)
    payment = np.where(
        monthly_rate > 0,
        loan_amount * monthly_rate / (1 - (1 + monthly_rate) ** -term_months),
        loan_amount / term_months,
    )

    bal = np.zeros((n, max_term))
    bal[:, 0] = loan_amount
    for t in range(1, max_term):
        interest = bal[:, t - 1] * monthly_rate
        principal = np.minimum(payment - interest, bal[:, t - 1])
        bal[:, t] = np.maximum(bal[:, t - 1] - principal, 0)
        bal[:, t] = np.where((t + 1) > term_months, 0, bal[:, t])
    return bal


# ---------------------------------------------------------------------------
# ECL calculation
# ---------------------------------------------------------------------------
def compute_ecl(
    df: pd.DataFrame,
    pd_12m: np.ndarray,
    pd_origination: np.ndarray,
    lgd: np.ndarray,
    discount_rate_annual: float = 0.05,
) -> pd.DataFrame:
    """
    Compute stage-aware ECL for every loan and return a per-loan summary.
    """
    stage = classify_stage(df, pd_12m, pd_origination)

    loan_amount = df.loan_amount.values
    interest_rate = df.interest_rate.values
    term_months = df.term_months.values.astype(int)

    # Lifetime PD curve
    marginal_pd = lifetime_pd_curve(pd_12m, term_months)
    # EAD path (declining balance)
    ead_path = amortizing_ead(loan_amount, interest_rate, term_months)

    # Discount factors per month
    monthly_disc = (1 + discount_rate_annual) ** (-np.arange(1, ead_path.shape[1] + 1) / 12)

    # Lifetime ECL = sum over t of marginal_PD_t × LGD × EAD_t × DF_t
    lifetime_ecl = np.sum(marginal_pd * ead_path * monthly_disc, axis=1) * lgd

    # 12-month ECL = sum over first 12 months
    twelve_m_ecl = np.sum(marginal_pd[:, :12] * ead_path[:, :12] * monthly_disc[:12], axis=1) * lgd

    # Stage 3: full LGD × current EAD (already defaulted)
    stage3_ecl = lgd * loan_amount

    # Apply stage logic
    ecl = np.where(stage == 1, twelve_m_ecl,
          np.where(stage == 2, lifetime_ecl, stage3_ecl))

    return pd.DataFrame({
        "loan_id": df.loan_id.values,
        "stage": stage,
        "pd_12m": pd_12m,
        "lgd": lgd,
        "exposure": loan_amount,
        "ecl_12m": twelve_m_ecl,
        "ecl_lifetime": lifetime_ecl,
        "ecl": ecl,
    })


# ---------------------------------------------------------------------------
# Main: train a quick PD and LGD, then compute ECL on the book
# ---------------------------------------------------------------------------
def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_synthetic_data.py` first."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["origination_date"])

    print(f"Book size : {len(df):,} loans")
    print(f"Total exposure : ${df.loan_amount.sum():,.0f}")

    # -- Fit a quick PD model just to get probabilities for ECL --
    feat_num = ["age", "annual_income", "employment_years", "loan_amount",
                "interest_rate", "term_months", "dti_ratio",
                "credit_history_years", "num_open_accounts",
                "num_delinquencies_2y", "revolving_utilization"]
    feat_cat = ["home_ownership", "loan_purpose"]

    prep = ColumnTransformer([
        ("num", StandardScaler(), feat_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), feat_cat),
    ])
    pd_pipe = Pipeline([("prep", prep),
                        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

    train, _ = train_test_split(df, test_size=0.20, random_state=42, stratify=df.default_flag)
    pd_pipe.fit(train[feat_num + feat_cat], train.default_flag)

    pd_12m = pd_pipe.predict_proba(df[feat_num + feat_cat])[:, 1]

    # PD at origination — proxy: same model, but with all delinquency features zeroed
    df_orig = df.copy()
    df_orig["num_delinquencies_2y"] = 0
    df_orig["revolving_utilization"] = df_orig.revolving_utilization.median()
    pd_origination = pd_pipe.predict_proba(df_orig[feat_num + feat_cat])[:, 1]

    # LGD — for this demo, use a portfolio-average LGD by purpose
    avg_lgd_by_purpose = (
        df[df.default_flag == 1].groupby("loan_purpose").loss_given_default.mean()
    )
    overall_lgd = df[df.default_flag == 1].loss_given_default.mean()
    lgd = df.loan_purpose.map(avg_lgd_by_purpose).fillna(overall_lgd).values

    # -- Compute ECL --
    print("\nComputing IFRS 9 ECL with 5% annual discount rate…")
    ecl_df = compute_ecl(df, pd_12m, pd_origination, lgd, discount_rate_annual=0.05)

    # -- Portfolio summary --
    summary = ecl_df.groupby("stage").agg(
        n_loans=("loan_id", "size"),
        exposure=("exposure", "sum"),
        ecl=("ecl", "sum"),
    ).reset_index()
    summary["coverage_ratio"] = summary.ecl / summary.exposure
    summary["pct_of_book"] = summary.n_loans / len(ecl_df)

    print("\n--- IFRS 9 stage distribution ---")
    stage_names = {1: "Stage 1 (performing)", 2: "Stage 2 (SICR)", 3: "Stage 3 (defaulted)"}
    summary["stage_name"] = summary.stage.map(stage_names)
    cols = ["stage_name", "n_loans", "pct_of_book", "exposure", "ecl", "coverage_ratio"]
    print(summary[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:,.4f}" if abs(x) < 100 else f"{x:,.0f}",
    ))

    total_ecl = ecl_df.ecl.sum()
    total_exposure = ecl_df.exposure.sum()
    print(f"\nTotal ECL provision    : ${total_ecl:,.0f}")
    print(f"Total exposure         : ${total_exposure:,.0f}")
    print(f"Portfolio coverage     : {total_ecl / total_exposure:.4%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
