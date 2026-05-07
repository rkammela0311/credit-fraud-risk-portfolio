"""
Synthetic data generator for the portfolio.

Produces three datasets that mimic realistic credit-risk and fraud distributions:
  1. credit_loans.csv         — loan-level data with default flags (for PD/scoring/ECL)
  2. card_transactions.csv    — payment transactions with fraud flags
  3. loan_applications.csv    — application-level data with application-fraud flags

Run once before any other script:
    python data/generate_synthetic_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)
OUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# 1. Credit loan dataset (for PD, scoring, IFRS 9)
# ---------------------------------------------------------------------------
def generate_credit_loans(n: int = 50_000) -> pd.DataFrame:
    """Loan-level book with a 12-month default flag."""
    age = RNG.normal(42, 12, n).clip(18, 80).astype(int)
    income = np.exp(RNG.normal(10.8, 0.6, n)).clip(15_000, 500_000)
    employment_years = RNG.gamma(2, 3, n).clip(0, 45)
    loan_amount = np.exp(RNG.normal(9.5, 0.8, n)).clip(1_000, 200_000)
    interest_rate = RNG.normal(8.5, 3.5, n).clip(2, 30)
    term_months = RNG.choice([12, 24, 36, 48, 60, 72, 84], n)
    dti = RNG.beta(2, 5, n) * 0.7  # debt-to-income, 0–0.7
    credit_history_years = (age - 18 - RNG.exponential(2, n)).clip(0, 60)
    num_open_accounts = RNG.poisson(4, n).clip(0, 30)
    num_delinquencies_2y = RNG.poisson(0.3, n).clip(0, 15)
    revolving_utilization = RNG.beta(2, 3, n)
    home_ownership = RNG.choice(
        ["RENT", "MORTGAGE", "OWN", "OTHER"], n, p=[0.4, 0.45, 0.13, 0.02]
    )
    purpose = RNG.choice(
        ["debt_consolidation", "credit_card", "home_improvement",
         "major_purchase", "small_business", "other"],
        n, p=[0.45, 0.20, 0.10, 0.10, 0.05, 0.10],
    )

    # Latent default-risk score — drives the binary outcome
    z = (
        -2.8
        + 0.025 * (40 - age)
        + 1.8 * dti
        + 1.2 * revolving_utilization
        + 0.25 * num_delinquencies_2y
        - 0.04 * employment_years
        - 0.000003 * income
        + 0.04 * (interest_rate - 8)
        + 0.6 * (purpose == "small_business").astype(float)
        + 0.3 * (home_ownership == "RENT").astype(float)
        + RNG.normal(0, 0.5, n)
    )
    p_default = 1 / (1 + np.exp(-z))
    default_flag = (RNG.uniform(0, 1, n) < p_default).astype(int)

    # Origination dates spread over 3 years (for OOT splits)
    origination_date = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        RNG.integers(0, 1095, n), unit="D"
    )

    # For LGD: among defaults, generate a recovery rate
    recovery_rate = np.where(
        default_flag == 1,
        RNG.beta(2, 3, n) * (1 - 0.3 * (purpose == "small_business").astype(float)),
        np.nan,
    )
    loss_given_default = np.where(default_flag == 1, 1 - recovery_rate, np.nan)

    df = pd.DataFrame({
        "loan_id": [f"L{i:07d}" for i in range(n)],
        "origination_date": origination_date,
        "age": age,
        "annual_income": income.round(2),
        "employment_years": employment_years.round(1),
        "loan_amount": loan_amount.round(2),
        "interest_rate": interest_rate.round(2),
        "term_months": term_months,
        "dti_ratio": dti.round(4),
        "credit_history_years": credit_history_years.round(1),
        "num_open_accounts": num_open_accounts,
        "num_delinquencies_2y": num_delinquencies_2y,
        "revolving_utilization": revolving_utilization.round(4),
        "home_ownership": home_ownership,
        "loan_purpose": purpose,
        "default_flag": default_flag,
        "loss_given_default": loss_given_default,
    })
    return df.sort_values("origination_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Card transaction dataset (for transaction fraud)
# ---------------------------------------------------------------------------
def generate_card_transactions(n: int = 200_000) -> pd.DataFrame:
    """Card transactions, ~0.5% fraud — realistic class imbalance."""
    n_customers = 5_000
    customer_id = RNG.integers(1, n_customers + 1, n)
    merchant_id = RNG.integers(1, 2_000, n)

    # Time
    start = pd.to_datetime("2025-01-01")
    timestamps = start + pd.to_timedelta(RNG.uniform(0, 90 * 24 * 3600, n), unit="s")
    hour = pd.Series(timestamps).dt.hour.to_numpy()

    # Amount
    amount = np.exp(RNG.normal(3.5, 1.3, n)).clip(0.5, 10_000)

    # Categorical features
    merchant_category = RNG.choice(
        ["grocery", "restaurant", "gas", "online_retail",
         "travel", "atm", "entertainment", "other"],
        n, p=[0.20, 0.18, 0.12, 0.20, 0.05, 0.05, 0.10, 0.10],
    )
    channel = RNG.choice(
        ["chip", "swipe", "online", "contactless", "manual"],
        n, p=[0.35, 0.10, 0.30, 0.20, 0.05],
    )
    country = RNG.choice(["US", "MX", "CA", "GB", "OTHER"], n, p=[0.85, 0.04, 0.05, 0.03, 0.03])
    device_type = RNG.choice(["mobile", "desktop", "tablet", "pos", "atm"], n,
                             p=[0.35, 0.15, 0.05, 0.40, 0.05])

    # Velocity features (simulated): same-card txns last 1h / 24h
    txn_count_1h = RNG.poisson(0.4, n)
    txn_count_24h = RNG.poisson(3, n)
    amount_sum_24h = RNG.gamma(2, 100, n)

    # Fraud signal — drives ~0.5% positives
    z = (
        -7.0
        + 0.0008 * amount
        + 1.8 * (channel == "online").astype(float)
        + 1.5 * (channel == "manual").astype(float)
        + 1.2 * (country == "OTHER").astype(float)
        + 0.9 * (country == "MX").astype(float)
        + 1.0 * (hour < 5).astype(float)
        + 0.4 * txn_count_1h
        + 0.05 * txn_count_24h
        + 0.6 * (merchant_category == "online_retail").astype(float)
        + RNG.normal(0, 0.3, n)
    )
    p_fraud = 1 / (1 + np.exp(-z))
    is_fraud = (RNG.uniform(0, 1, n) < p_fraud).astype(int)

    df = pd.DataFrame({
        "transaction_id": [f"T{i:09d}" for i in range(n)],
        "timestamp": timestamps,
        "customer_id": customer_id,
        "merchant_id": merchant_id,
        "amount": amount.round(2),
        "merchant_category": merchant_category,
        "channel": channel,
        "country": country,
        "device_type": device_type,
        "hour_of_day": hour,
        "txn_count_1h": txn_count_1h,
        "txn_count_24h": txn_count_24h,
        "amount_sum_24h": amount_sum_24h.round(2),
        "is_fraud": is_fraud,
    })
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Loan application dataset (for application fraud)
# ---------------------------------------------------------------------------
def generate_loan_applications(n: int = 30_000) -> pd.DataFrame:
    """Application-level data with synthetic-identity / first-party fraud flags."""
    age = RNG.normal(38, 12, n).clip(18, 75).astype(int)
    stated_income = np.exp(RNG.normal(10.7, 0.7, n)).clip(15_000, 600_000)
    bureau_score = RNG.normal(680, 80, n).clip(300, 850).astype(int)
    months_at_address = RNG.gamma(2, 18, n).clip(0, 360)
    months_at_employer = RNG.gamma(2, 24, n).clip(0, 480)
    num_inquiries_6m = RNG.poisson(2, n).clip(0, 30)
    email_age_days = RNG.gamma(2, 400, n).clip(0, 4_000)
    phone_age_days = RNG.gamma(2, 600, n).clip(0, 5_000)
    device_seen_before = RNG.choice([0, 1], n, p=[0.7, 0.3])
    ip_country_matches_address = RNG.choice([0, 1], n, p=[0.05, 0.95])
    ssn_age_consistent = RNG.choice([0, 1], n, p=[0.03, 0.97])

    # Fraud signal: thin file + new email/phone + IP mismatch + SSN inconsistency
    z = (
        -4.5
        + 1.5 * (email_age_days < 30).astype(float)
        + 1.3 * (phone_age_days < 60).astype(float)
        + 1.8 * (1 - ssn_age_consistent)
        + 1.2 * (1 - ip_country_matches_address)
        + 0.6 * (months_at_address < 3).astype(float)
        + 0.5 * (num_inquiries_6m > 8).astype(float)
        + 0.4 * (stated_income > 250_000).astype(float)
        + RNG.normal(0, 0.4, n)
    )
    p_fraud = 1 / (1 + np.exp(-z))
    fraud_flag = (RNG.uniform(0, 1, n) < p_fraud).astype(int)

    application_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        RNG.integers(0, 730, n), unit="D"
    )

    df = pd.DataFrame({
        "application_id": [f"A{i:07d}" for i in range(n)],
        "application_date": application_date,
        "age": age,
        "stated_income": stated_income.round(2),
        "bureau_score": bureau_score,
        "months_at_address": months_at_address.round(0).astype(int),
        "months_at_employer": months_at_employer.round(0).astype(int),
        "num_inquiries_6m": num_inquiries_6m,
        "email_age_days": email_age_days.round(0).astype(int),
        "phone_age_days": phone_age_days.round(0).astype(int),
        "device_seen_before": device_seen_before,
        "ip_country_matches_address": ip_country_matches_address,
        "ssn_age_consistent": ssn_age_consistent,
        "fraud_flag": fraud_flag,
    })
    return df.sort_values("application_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating synthetic datasets…")

    loans = generate_credit_loans()
    loans.to_csv(OUT_DIR / "credit_loans.csv", index=False)
    print(f"  credit_loans.csv          rows={len(loans):>7,}  "
          f"default_rate={loans.default_flag.mean():.2%}")

    txns = generate_card_transactions()
    txns.to_csv(OUT_DIR / "card_transactions.csv", index=False)
    print(f"  card_transactions.csv     rows={len(txns):>7,}  "
          f"fraud_rate={txns.is_fraud.mean():.2%}")

    apps = generate_loan_applications()
    apps.to_csv(OUT_DIR / "loan_applications.csv", index=False)
    print(f"  loan_applications.csv     rows={len(apps):>7,}  "
          f"fraud_rate={apps.fraud_flag.mean():.2%}")

    print(f"\nDone. Files written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
