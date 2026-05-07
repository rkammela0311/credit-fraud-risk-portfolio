"""
Reusable model evaluation utilities for credit risk and fraud risk models.

Includes the metrics you'll find on every credit-risk validation report:
    - KS statistic (Kolmogorov-Smirnov)
    - Gini coefficient (= 2*AUC - 1)
    - Population Stability Index (PSI)
    - Calibration / reliability
    - Lift and capture rate at deciles
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Core rank-ordering metrics
# ---------------------------------------------------------------------------
def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: max separation between cumulative
    distribution of goods and bads. The single most-cited number on a
    credit-risk model card.
    """
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score")
    df["cum_bad"] = (df.y == 1).cumsum() / max((df.y == 1).sum(), 1)
    df["cum_good"] = (df.y == 0).cumsum() / max((df.y == 0).sum(), 1)
    return float((df.cum_good - df.cum_bad).abs().max())


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2 * AUC - 1. Range [-1, 1]; > 0.4 is typically considered strong."""
    return 2 * roc_auc_score(y_true, y_score) - 1


# ---------------------------------------------------------------------------
# Population Stability Index — the standard drift metric in credit risk
# ---------------------------------------------------------------------------
def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    PSI between an expected (development / baseline) and actual (recent) sample.

    Industry rules of thumb:
        PSI < 0.10   no significant change
        0.10 - 0.25  moderate change, investigate
        > 0.25       major shift, model likely needs to be redeveloped
    """
    breakpoints = np.quantile(
        expected, np.linspace(0, 1, n_bins + 1)
    )
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)  # safeguard duplicate edges

    expected_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, breakpoints)[0] / len(actual)

    # Avoid division-by-zero / log(0)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Decile-level calibration: predicted vs. actual default rate."""
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bucket"] = pd.qcut(df.p, q=n_bins, duplicates="drop")
    out = df.groupby("bucket", observed=True).agg(
        n=("y", "size"),
        actual_rate=("y", "mean"),
        predicted_rate=("p", "mean"),
    ).reset_index(drop=True)
    out["abs_error"] = (out.actual_rate - out.predicted_rate).abs()
    return out


# ---------------------------------------------------------------------------
# Lift / decile analysis (the regulator's friend)
# ---------------------------------------------------------------------------
def decile_lift_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """How concentrated are bads in the worst-scored deciles?"""
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df["decile"] = pd.qcut(
        df.score, q=n_deciles, labels=False, duplicates="drop"
    )
    # decile 0 = lowest score (best); flip to put riskiest at top
    df["decile"] = df.decile.max() - df.decile

    base_rate = df.y.mean()
    out = df.groupby("decile").agg(
        n=("y", "size"),
        bads=("y", "sum"),
        bad_rate=("y", "mean"),
    ).reset_index()
    out["lift"] = out.bad_rate / base_rate
    out["cum_bads_pct"] = out.bads.cumsum() / out.bads.sum()
    out["cum_pop_pct"] = out.n.cumsum() / out.n.sum()
    return out


# ---------------------------------------------------------------------------
# One-shot summary
# ---------------------------------------------------------------------------
def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label: str = "model",
) -> dict:
    """All the standard metrics in one dict — handy for logging."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "label": label,
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "gini": float(gini_coefficient(y_true, y_prob)),
        "ks": float(ks_statistic(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": threshold,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    return metrics


def find_threshold_for_recall(
    y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.80
) -> float:
    """Return the lowest threshold that still hits the target recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns thresholds shorter by 1 than precisions/recalls
    eligible = np.where(recalls[:-1] >= target_recall)[0]
    if len(eligible) == 0:
        return 0.0
    return float(thresholds[eligible[-1]])


def print_metrics_block(metrics: dict) -> None:
    """Pretty-print the metrics dict from evaluate_binary_classifier."""
    print(f"\n--- {metrics['label']} ---")
    print(f"  N                  : {metrics['n']:,}")
    print(f"  Positive rate      : {metrics['positive_rate']:.4f}")
    print(f"  AUC-ROC            : {metrics['auc_roc']:.4f}")
    print(f"  Gini               : {metrics['gini']:.4f}")
    print(f"  KS                 : {metrics['ks']:.4f}")
    print(f"  AUC-PR             : {metrics['auc_pr']:.4f}")
    print(f"  Brier score        : {metrics['brier']:.4f}")
    print(f"  Threshold          : {metrics['threshold']:.4f}")
    print(f"  Precision / Recall : {metrics['precision']:.4f} / {metrics['recall']:.4f}")
    print(f"  F1                 : {metrics['f1']:.4f}")
    print(f"  TP/FP/TN/FN        : {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}")
