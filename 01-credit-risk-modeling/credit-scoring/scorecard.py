"""
Application Credit Scorecard
=============================

End-to-end application scorecard, scaled to FICO-style points.

Pipeline:
    1. WoE binning of all candidate features
    2. Information Value ranking — drop weak features, flag suspicious ones
    3. Logistic regression on WoE-transformed inputs
    4. Scale to points using the standard PDO / odds construction
    5. Validate: KS, Gini, calibration, score-band default rates

The output is a points table you could literally hand to underwriters and
deploy as a lookup. This is the format every consumer-credit scorecard ships
in, from FICO to Vantage to internal bank-built models.

Run:
    python scorecard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03-shared-utilities"))

from model_evaluation import (  # noqa: E402
    evaluate_binary_classifier,
    print_metrics_block,
)
from woe_iv import (  # noqa: E402
    compute_iv_table,
    woe_iv_categorical,
    woe_iv_numeric,
)
from plotting import (  # noqa: E402
    plot_roc_curve,
    plot_score_distribution,
    PALETTE,
)
import matplotlib.pyplot as plt  # noqa: E402

DATA_PATH = ROOT / "data" / "credit_loans.csv"
CHARTS_DIR = Path(__file__).resolve().parent / "charts"

NUMERIC_FEATURES = [
    "age", "annual_income", "employment_years", "loan_amount", "interest_rate",
    "dti_ratio", "credit_history_years", "num_open_accounts",
    "num_delinquencies_2y", "revolving_utilization",
]
CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose"]
TARGET = "default_flag"


# ---------------------------------------------------------------------------
# Step 1+2: WoE binning and IV ranking
# ---------------------------------------------------------------------------
def build_woe_tables(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Build a WoE table for every feature; return them plus the IV ranking.
    """
    woe_tables = {}
    for feat in NUMERIC_FEATURES:
        table, _ = woe_iv_numeric(df, feat, TARGET, n_bins=8)
        woe_tables[feat] = ("numeric", table)

    for feat in CATEGORICAL_FEATURES:
        table, _ = woe_iv_categorical(df, feat, TARGET)
        woe_tables[feat] = ("categorical", table)

    iv_rank = compute_iv_table(
        df, TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES, n_bins=8
    )
    return woe_tables, iv_rank


def transform_to_woe(
    df: pd.DataFrame, woe_tables: dict, features: list[str]
) -> pd.DataFrame:
    """Replace each raw feature value with its bin's WoE."""
    out = pd.DataFrame(index=df.index)
    for feat in features:
        kind, table = woe_tables[feat]
        if kind == "categorical":
            mapping = dict(zip(table["bin"].astype(str), table.woe))
            out[feat] = df[feat].astype(str).map(mapping).fillna(0.0)
        else:
            intervals = pd.IntervalIndex(table["bin"])
            woes = table.woe.values
            col = pd.Series(0.0, index=df.index)
            for i, interval in enumerate(intervals):
                mask = df[feat].between(interval.left, interval.right, inclusive="right")
                col.loc[mask] = woes[i]
            out[feat] = col
    return out


# ---------------------------------------------------------------------------
# Step 4: scale logistic-regression output to points
# ---------------------------------------------------------------------------
def build_points_table(
    model: LogisticRegression,
    woe_tables: dict,
    features: list[str],
    base_score: int = 600,
    base_odds: float = 50.0,
    pdo: int = 20,
) -> pd.DataFrame:
    """
    Convert a fitted logistic-regression-on-WoE into a classic points table.

    Standard scorecard scaling:
        Score = offset + factor * ln(odds)
        factor = pdo / ln(2)
        offset = base_score - factor * ln(base_odds)

    Each bin's contribution is:  -(beta_i * woe_i + alpha/n) * factor
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    intercept = model.intercept_[0]
    coefs = dict(zip(features, model.coef_[0]))
    n_features = len(features)

    rows = []
    for feat in features:
        kind, table = woe_tables[feat]
        beta = coefs[feat]
        for _, r in table.iterrows():
            points = -(beta * r.woe + intercept / n_features) * factor
            rows.append({
                "feature": feat,
                "bin": str(r.bin),
                "n": int(r.n),
                "bad_rate": r.n_bad / max(r.n, 1),
                "woe": float(r.woe),
                "points": int(round(points)),
            })

    points_df = pd.DataFrame(rows)
    points_df["base_score"] = base_score
    return points_df


def score_applicants(
    df: pd.DataFrame,
    model: LogisticRegression,
    woe_tables: dict,
    features: list[str],
    base_score: int = 600,
    base_odds: float = 50.0,
    pdo: int = 20,
) -> np.ndarray:
    """Apply the model and return integer scorecard points per applicant."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    X_woe = transform_to_woe(df, woe_tables, features)
    log_odds = model.decision_function(X_woe)
    scores = offset + factor * (-log_odds)
    return scores.round().astype(int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_synthetic_data.py` first."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["origination_date"])

    train, test = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df[TARGET]
    )

    print("Building WoE tables on training data…")
    woe_tables, iv_rank = build_woe_tables(train)

    print("\nInformation Value ranking:")
    print(iv_rank.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Keep only features with IV >= 0.02 (anything weaker is noise)
    keep = iv_rank[iv_rank.iv >= 0.02].feature.tolist()
    print(f"\nFeatures kept (IV >= 0.02): {keep}")

    # WoE transform
    X_train_woe = transform_to_woe(train, woe_tables, keep)
    X_test_woe = transform_to_woe(test, woe_tables, keep)
    y_train, y_test = train[TARGET], test[TARGET]

    # Fit logistic regression
    print("\nFitting logistic regression on WoE features…")
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_train_woe, y_train)

    coefs = pd.DataFrame({
        "feature": keep,
        "coefficient": model.coef_[0],
    }).sort_values("coefficient", ascending=False)
    print("\nLogit coefficients on WoE inputs (should all be positive — "
          "WoE is constructed so higher = lower risk):")
    print(coefs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Validation
    p_train = model.predict_proba(X_train_woe)[:, 1]
    p_test = model.predict_proba(X_test_woe)[:, 1]
    print_metrics_block(evaluate_binary_classifier(y_train, p_train, label="Scorecard / train"))
    print_metrics_block(evaluate_binary_classifier(y_test, p_test, label="Scorecard / test"))

    # Build the points table
    print("\nBuilding points table (PDO=20, base score=600 @ 50:1 odds)…")
    points_table = build_points_table(model, woe_tables, keep)
    print("\n--- Scorecard points table (sample) ---")
    print(points_table.to_string(index=False))

    # Score the test set and report band-level default rates
    test_scores = score_applicants(test, model, woe_tables, keep)
    band = pd.cut(
        test_scores,
        bins=[-np.inf, 540, 580, 620, 660, 700, np.inf],
        labels=["<540", "540-579", "580-619", "620-659", "660-699", "700+"],
    )
    band_summary = pd.DataFrame({"band": band, "default": y_test.values}) \
        .groupby("band", observed=True) \
        .agg(n=("default", "size"), default_rate=("default", "mean")) \
        .reset_index()
    print("\n--- Default rate by score band (test) ---")
    print(band_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- charts ----
    CHARTS_DIR.mkdir(exist_ok=True)
    print(f"\nSaving charts to {CHARTS_DIR}/ …")

    plot_roc_curve(y_test.values, p_test,
                   "Scorecard — ROC Curve (Test)",
                   str(CHARTS_DIR / "roc_curve.png"))
    plot_score_distribution(test_scores[y_test.values == 1],
                            test_scores[y_test.values == 0],
                            "Scorecard — Score Distribution (Test)",
                            str(CHARTS_DIR / "score_distribution.png"),
                            pos_label="Defaulters", neg_label="Non-defaulters")

    # IV bar chart
    fig, ax = plt.subplots(figsize=(7, max(4, len(iv_rank) * 0.32)))
    iv_sorted = iv_rank.sort_values("iv", ascending=True)
    iv_colors = [PALETTE["good"] if v >= 0.10 else
                 PALETTE["accent"] if v >= 0.02 else PALETTE["neutral"]
                 for v in iv_sorted.iv]
    ax.barh(iv_sorted.feature, iv_sorted.iv, color=iv_colors, edgecolor="white")
    ax.axvline(0.02, color=PALETTE["neutral"], ls="--", lw=1, label="Weak (0.02)")
    ax.axvline(0.10, color=PALETTE["good"], ls="--", lw=1, label="Medium (0.10)")
    ax.set_xlabel("Information Value")
    ax.set_title("Scorecard — Information Value by Feature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "information_value.png")
    plt.close(fig)

    # Default rate by score band
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(band_summary["band"].astype(str), band_summary["default_rate"],
                  color=PALETTE["primary"], edgecolor="white")
    bars[0].set_color(PALETTE["bad"])
    bars[-1].set_color(PALETTE["good"])
    for bar, val, n in zip(bars, band_summary["default_rate"], band_summary["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f"{val:.1%}\n(n={n:,})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Default rate")
    ax.set_xlabel("Score band")
    ax.set_title("Scorecard — Default Rate by Score Band (Test)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "default_rate_by_band.png")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
