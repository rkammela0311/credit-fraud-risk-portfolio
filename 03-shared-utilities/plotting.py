"""
plotting.py
-----------
Shared plotting utilities used across all models in the portfolio. Each function
produces a single figure with consistent styling and saves it to disk as a PNG.

These are designed to be called at the end of a model script so the charts land
next to the script itself and can be embedded in the project's README.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix


# ---------- consistent styling ----------

PALETTE = {
    "primary":   "#1f4e79",
    "secondary": "#c0504d",
    "accent":    "#4f81bd",
    "neutral":   "#7f7f7f",
    "good":      "#548235",
    "bad":       "#a52a2a",
    "grid":      "#dddddd",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.frameon": False,
    "font.family": "DejaVu Sans",
})


# ---------- chart functions ----------

def plot_roc_curve(y_true, y_score, title: str, out_path: str) -> float:
    """Plot ROC curve and return AUC. Saves to out_path."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    gini = 2 * roc_auc - 1

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2.2,
            label=f"Model (AUC = {roc_auc:.3f}, Gini = {gini:.3f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1, ls="--",
            label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return roc_auc


def plot_calibration(y_true, y_prob, title: str, out_path: str, n_bins: int = 10):
    """Reliability diagram: predicted probability vs observed event rate."""
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    cal = df.groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"),
        actual=("y", "mean"),
        n=("y", "size"),
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, cal["mean_pred"].max() * 1.05],
            [0, cal["mean_pred"].max() * 1.05],
            color=PALETTE["neutral"], lw=1, ls="--", label="Perfectly calibrated")
    ax.scatter(cal["mean_pred"], cal["actual"],
               s=cal["n"] / cal["n"].max() * 200 + 30,
               color=PALETTE["primary"], alpha=0.85, edgecolor="white",
               label="Decile (size = n)")
    ax.plot(cal["mean_pred"], cal["actual"],
            color=PALETTE["primary"], lw=1.2, alpha=0.5)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_decile_lift(y_true, y_score, title: str, out_path: str):
    """Decile lift chart sorted from highest predicted risk to lowest."""
    df = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s", ascending=False)
    df["decile"] = pd.qcut(df["s"].rank(method="first", ascending=False),
                           10, labels=False)
    base_rate = df["y"].mean()
    lift = df.groupby("decile").agg(rate=("y", "mean"), n=("y", "size")).reset_index()
    lift["lift"] = lift["rate"] / base_rate

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(lift["decile"].astype(int) + 1, lift["lift"],
                  color=PALETTE["primary"], edgecolor="white")
    bars[0].set_color(PALETTE["secondary"])  # highlight top decile
    ax.axhline(1.0, color=PALETTE["neutral"], ls="--", lw=1,
               label="Base rate (lift = 1)")
    ax.set_xlabel("Decile (1 = highest predicted risk)")
    ax.set_ylabel("Lift over base rate")
    ax.set_title(title)
    ax.set_xticks(range(1, 11))
    ax.legend()
    for bar, val in zip(bars, lift["lift"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                f"{val:.1f}x", ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(features, importances, title: str, out_path: str,
                            top_n: int = 15):
    """Horizontal bar chart of top-N feature importances."""
    df = pd.DataFrame({"feature": features, "importance": importances})
    df = df.sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.32)))
    ax.barh(df["feature"], df["importance"], color=PALETTE["accent"],
            edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_score_distribution(scores_pos, scores_neg, title: str, out_path: str,
                            pos_label: str = "Bad / fraud",
                            neg_label: str = "Good / legit"):
    """Overlapping histogram of model scores split by class."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(scores_neg, bins=40, color=PALETTE["primary"], alpha=0.6,
            label=neg_label, density=True)
    ax.hist(scores_pos, bins=40, color=PALETTE["secondary"], alpha=0.6,
            label=pos_label, density=True)
    ax.set_xlabel("Model score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_psi_bars(psi_dict: dict, title: str, out_path: str):
    """Bar chart of PSI by feature with reference thresholds."""
    df = pd.DataFrame(
        sorted(psi_dict.items(), key=lambda kv: kv[1], reverse=True),
        columns=["feature", "psi"],
    )

    def _color(v):
        if v < 0.10:
            return PALETTE["good"]
        if v < 0.25:
            return "#e0a020"
        return PALETTE["bad"]

    fig, ax = plt.subplots(figsize=(7, max(4, len(df) * 0.3)))
    ax.barh(df["feature"][::-1], df["psi"][::-1],
            color=[_color(v) for v in df["psi"][::-1]], edgecolor="white")
    ax.axvline(0.10, color=PALETTE["neutral"], ls="--", lw=1)
    ax.axvline(0.25, color=PALETTE["bad"], ls="--", lw=1)
    ax.set_xlabel("PSI")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pr_curve(y_true, y_score, title: str, out_path: str):
    """Precision-recall curve, useful for highly imbalanced fraud problems."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    base_rate = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=PALETTE["primary"], lw=2.2, label="Model")
    ax.axhline(base_rate, color=PALETTE["neutral"], ls="--", lw=1,
               label=f"Base rate = {base_rate:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion(y_true, y_pred, title: str, out_path: str,
                   labels=("Legit", "Fraud")):
    """Confusion matrix heatmap with counts and rates."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ecl_stage_breakdown(stage_counts: dict, stage_ecl: dict,
                             title: str, out_path: str):
    """Side-by-side bar chart of account count and ECL by IFRS-9 stage."""
    stages = ["Stage 1", "Stage 2", "Stage 3"]
    counts = [stage_counts.get(s, 0) for s in stages]
    ecls   = [stage_ecl.get(s, 0)   for s in stages]
    colors = [PALETTE["good"], "#e0a020", PALETTE["bad"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(stages, counts, color=colors, edgecolor="white")
    axes[0].set_title("Accounts by Stage")
    axes[0].set_ylabel("Number of accounts")
    for i, v in enumerate(counts):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(stages, ecls, color=colors, edgecolor="white")
    axes[1].set_title("ECL ($) by Stage")
    axes[1].set_ylabel("Expected Credit Loss")
    for i, v in enumerate(ecls):
        axes[1].text(i, v, f"${v/1e6:.2f}M", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
