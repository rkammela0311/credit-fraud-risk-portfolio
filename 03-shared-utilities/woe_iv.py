"""
Weight of Evidence (WoE) and Information Value (IV) utilities.

These are the workhorses of credit scorecard development:
    - WoE transforms a feature into a log-odds scale, making logistic
      regression coefficients interpretable as "how much does this bin
      shift the score".
    - IV summarises the predictive power of a feature in a single number.

IV interpretation (industry convention):
    < 0.02   useless
    0.02-0.1 weak
    0.1-0.3  medium
    0.3-0.5  strong
    > 0.5    suspiciously strong (check for leakage)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float, fallback: float = 1e-6) -> float:
    return a / b if b != 0 else fallback


def woe_iv_numeric(
    df: pd.DataFrame,
    feature: str,
    target: str,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """
    Compute WoE table and IV for a numeric feature using quantile bins.

    Returns
    -------
    table : DataFrame with bin, n, n_good, n_bad, dist_good, dist_bad, woe, iv
    iv    : total Information Value (sum of bin IVs)
    """
    x = df[feature]
    y = df[target]
    bins = pd.qcut(x, q=n_bins, duplicates="drop")

    table = pd.DataFrame({"bin": bins, "y": y})
    grp = table.groupby("bin", observed=True).agg(
        n=("y", "size"),
        n_bad=("y", "sum"),
    )
    grp["n_good"] = grp.n - grp.n_bad

    total_good = max(grp.n_good.sum(), 1)
    total_bad = max(grp.n_bad.sum(), 1)
    grp["dist_good"] = grp.n_good / total_good
    grp["dist_bad"] = grp.n_bad / total_bad

    # Avoid log(0) and div-by-zero
    grp["dist_good"] = grp.dist_good.replace(0, 1e-6)
    grp["dist_bad"] = grp.dist_bad.replace(0, 1e-6)

    grp["woe"] = np.log(grp.dist_good / grp.dist_bad)
    grp["iv_bin"] = (grp.dist_good - grp.dist_bad) * grp.woe
    iv_total = float(grp.iv_bin.sum())

    grp = grp.reset_index()
    grp.insert(0, "feature", feature)
    return grp, iv_total


def woe_iv_categorical(
    df: pd.DataFrame, feature: str, target: str
) -> tuple[pd.DataFrame, float]:
    """WoE table and IV for a categorical feature (no binning needed)."""
    grp = df.groupby(feature).agg(
        n=(target, "size"),
        n_bad=(target, "sum"),
    )
    grp["n_good"] = grp.n - grp.n_bad

    total_good = max(grp.n_good.sum(), 1)
    total_bad = max(grp.n_bad.sum(), 1)
    grp["dist_good"] = (grp.n_good / total_good).replace(0, 1e-6)
    grp["dist_bad"] = (grp.n_bad / total_bad).replace(0, 1e-6)

    grp["woe"] = np.log(grp.dist_good / grp.dist_bad)
    grp["iv_bin"] = (grp.dist_good - grp.dist_bad) * grp.woe
    iv_total = float(grp.iv_bin.sum())

    grp = grp.reset_index().rename(columns={feature: "bin"})
    grp.insert(0, "feature", feature)
    return grp, iv_total


def compute_iv_table(
    df: pd.DataFrame,
    target: str,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    One-shot IV ranking across many features.

    Returns a DataFrame: feature, iv, strength_label
    """
    rows = []
    for feat in numeric_features or []:
        _, iv = woe_iv_numeric(df, feat, target, n_bins=n_bins)
        rows.append({"feature": feat, "iv": iv, "type": "numeric"})
    for feat in categorical_features or []:
        _, iv = woe_iv_categorical(df, feat, target)
        rows.append({"feature": feat, "iv": iv, "type": "categorical"})

    out = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def _label(iv: float) -> str:
        if iv < 0.02: return "useless"
        if iv < 0.10: return "weak"
        if iv < 0.30: return "medium"
        if iv < 0.50: return "strong"
        return "suspicious (check leakage)"

    out["strength"] = out.iv.apply(_label)
    return out


def apply_woe(
    df: pd.DataFrame, woe_table: pd.DataFrame, feature: str, kind: str
) -> pd.Series:
    """
    Map a feature to its WoE values using a previously-built WoE table.

    kind : "numeric" or "categorical"
    """
    if kind == "categorical":
        mapping = dict(zip(woe_table.bin.astype(str), woe_table.woe))
        return df[feature].astype(str).map(mapping).fillna(0.0)

    # numeric — bin edges are stored implicitly in the IntervalIndex on the table
    intervals = pd.IntervalIndex(woe_table.bin)
    woes = woe_table.woe.values
    out = pd.Series(0.0, index=df.index)
    for i, interval in enumerate(intervals):
        mask = df[feature].between(interval.left, interval.right, inclusive="right")
        out.loc[mask] = woes[i]
    return out
