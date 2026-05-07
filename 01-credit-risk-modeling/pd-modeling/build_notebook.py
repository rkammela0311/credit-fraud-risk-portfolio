"""
Build the PD-model walkthrough notebook programmatically.

This produces a .ipynb file with markdown narrative + code cells. We build it
in code so the notebook stays in sync with the underlying script and can be
regenerated at any time.

Run:
    python build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "PD_Model_Walkthrough.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }

    cells = []

    cells.append(md(
        "# Probability of Default — Walkthrough\n\n"
        "This notebook walks through the full PD modeling workflow on the synthetic "
        "consumer-loan dataset. It mirrors the production-style script `pd_model.py`, "
        "but unpacks each step with explanation, intermediate inspection, and inline "
        "visualization.\n\n"
        "**Outline**\n"
        "1. Load and inspect the data\n"
        "2. Time-based development / out-of-time split\n"
        "3. Feature engineering and preprocessing pipeline\n"
        "4. Logistic regression — interpretable benchmark\n"
        "5. XGBoost — non-linear challenger\n"
        "6. Validation: discrimination, calibration, lift, stability\n"
        "7. Discussion of findings and what would change in production"
    ))

    # --- 1. Setup ---
    cells.append(md("## 1. Setup\n\nImports, paths, and a quick look at the dataset."))
    cells.append(code(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import roc_auc_score\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
        "from xgboost import XGBClassifier\n"
        "\n"
        "# Locate the project root and import shared utilities\n"
        "ROOT = Path().resolve().parents[1]\n"
        "sys.path.insert(0, str(ROOT / '03-shared-utilities'))\n"
        "\n"
        "from model_evaluation import (\n"
        "    calibration_table, decile_lift_table,\n"
        "    evaluate_binary_classifier, population_stability_index,\n"
        ")\n"
        "from plotting import (\n"
        "    plot_roc_curve, plot_calibration, plot_decile_lift,\n"
        "    plot_feature_importance,\n"
        ")\n"
        "\n"
        "DATA_PATH = ROOT / 'data' / 'credit_loans.csv'\n"
        "print('Loading from', DATA_PATH)"
    ))

    # --- 2. Load data ---
    cells.append(md(
        "## 2. Load data\n\n"
        "The synthetic dataset contains 50,000 consumer loans originated over a "
        "two-year window. Each row carries borrower demographics, loan terms, "
        "credit-bureau attributes at origination, and a `default_flag` indicating "
        "whether the loan went 90+ DPD within 12 months.\n\n"
        "If `credit_loans.csv` does not exist yet, run `python data/generate_synthetic_data.py` "
        "from the project root."
    ))
    cells.append(code(
        "df = pd.read_csv(DATA_PATH, parse_dates=['origination_date'])\n"
        "print(f'Rows: {len(df):,}')\n"
        "print(f'Default rate: {df.default_flag.mean():.2%}')\n"
        "df.head()"
    ))
    cells.append(code(
        "# Default rate over time — the most important sanity check on a credit dataset\n"
        "by_month = (df.set_index('origination_date')\n"
        "              .groupby(pd.Grouper(freq='ME'))['default_flag']\n"
        "              .mean()\n"
        "              .rename('default_rate'))\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "ax.plot(by_month.index, by_month.values, marker='o', color='#1f4e79')\n"
        "ax.axhline(df.default_flag.mean(), color='grey', ls='--', lw=1,\n"
        "           label=f'Overall: {df.default_flag.mean():.2%}')\n"
        "ax.set_title('Monthly default rate (synthetic data)')\n"
        "ax.set_ylabel('Default rate'); ax.legend()\n"
        "plt.show()"
    ))

    # --- 3. OOT split ---
    cells.append(md(
        "## 3. Time-based split\n\n"
        "Random train/test splits flatter PD models — they let the model see future "
        "vintages during training. The honest split is **out-of-time (OOT)**: hold "
        "out the most recent six months entirely and validate against them. That is "
        "exactly how the model will be used in production, scoring loans the model "
        "has never seen vintage-wise."
    ))
    cells.append(code(
        "cutoff = df.origination_date.max() - pd.DateOffset(months=6)\n"
        "dev = df[df.origination_date <= cutoff].copy()\n"
        "oot = df[df.origination_date >  cutoff].copy()\n"
        "\n"
        "print(f'Dev sample : {len(dev):>7,}  cutoff <= {cutoff.date()}')\n"
        "print(f'OOT sample : {len(oot):>7,}  cutoff >  {cutoff.date()}')\n"
        "print(f'Dev default rate : {dev.default_flag.mean():.2%}')\n"
        "print(f'OOT default rate : {oot.default_flag.mean():.2%}')"
    ))

    # --- 4. Features ---
    cells.append(md(
        "## 4. Feature engineering and preprocessing\n\n"
        "We standard-scale numeric inputs and one-hot encode the two categoricals. "
        "Both go into the same `ColumnTransformer` so the same pipeline preprocesses "
        "training, validation, and OOT data identically — preventing the classic bug "
        "of fitting a scaler twice with different statistics."
    ))
    cells.append(code(
        "NUMERIC_FEATURES = [\n"
        "    'age', 'annual_income', 'employment_years', 'loan_amount',\n"
        "    'interest_rate', 'term_months', 'dti_ratio',\n"
        "    'credit_history_years', 'num_open_accounts',\n"
        "    'num_delinquencies_2y', 'revolving_utilization',\n"
        "]\n"
        "CATEGORICAL_FEATURES = ['home_ownership', 'loan_purpose']\n"
        "TARGET = 'default_flag'\n"
        "\n"
        "preprocessor = ColumnTransformer([\n"
        "    ('num', StandardScaler(), NUMERIC_FEATURES),\n"
        "    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'),\n"
        "            CATEGORICAL_FEATURES),\n"
        "])\n"
        "\n"
        "X_dev = dev[NUMERIC_FEATURES + CATEGORICAL_FEATURES]\n"
        "y_dev = dev[TARGET]\n"
        "X_oot = oot[NUMERIC_FEATURES + CATEGORICAL_FEATURES]\n"
        "y_oot = oot[TARGET]\n"
        "\n"
        "X_train, X_val, y_train, y_val = train_test_split(\n"
        "    X_dev, y_dev, test_size=0.20, random_state=42, stratify=y_dev,\n"
        ")\n"
        "print(f'train: {len(X_train):,}   val: {len(X_val):,}   OOT: {len(X_oot):,}')"
    ))

    # --- 5. Logistic ---
    cells.append(md(
        "## 5. Logistic regression — interpretable benchmark\n\n"
        "The regulator's favorite. Coefficient signs are interpretable, calibration "
        "is naturally good when the prior is preserved, and the documentation burden "
        "is low. We use `class_weight='balanced'` so the loss does not collapse onto "
        "the majority class."
    ))
    cells.append(code(
        "logit = Pipeline([\n"
        "    ('prep', preprocessor),\n"
        "    ('clf',  LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced')),\n"
        "])\n"
        "logit.fit(X_train, y_train)\n"
        "\n"
        "p_val_lr = logit.predict_proba(X_val)[:, 1]\n"
        "p_oot_lr = logit.predict_proba(X_oot)[:, 1]\n"
        "print(f'Logistic — Val AUC : {roc_auc_score(y_val, p_val_lr):.4f}')\n"
        "print(f'Logistic — OOT AUC : {roc_auc_score(y_oot, p_oot_lr):.4f}')"
    ))

    # --- 6. XGBoost ---
    cells.append(md(
        "## 6. XGBoost — non-linear challenger\n\n"
        "Captures interactions and non-linearities the logistic cannot. We constrain "
        "depth to 4 and apply `min_child_weight=10` and L2 regularization to keep the "
        "model from memorizing noise. `scale_pos_weight` handles the class imbalance."
    ))
    cells.append(code(
        "pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)\n"
        "\n"
        "xgb = Pipeline([\n"
        "    ('prep', preprocessor),\n"
        "    ('clf',  XGBClassifier(\n"
        "        n_estimators=400, max_depth=4, learning_rate=0.05,\n"
        "        min_child_weight=10, subsample=0.85, colsample_bytree=0.85,\n"
        "        reg_lambda=1.0, scale_pos_weight=pos_weight,\n"
        "        random_state=42, eval_metric='logloss',\n"
        "        tree_method='hist', n_jobs=-1,\n"
        "    )),\n"
        "])\n"
        "xgb.fit(X_train, y_train)\n"
        "\n"
        "p_val_xgb = xgb.predict_proba(X_val)[:, 1]\n"
        "p_oot_xgb = xgb.predict_proba(X_oot)[:, 1]\n"
        "print(f'XGBoost — Val AUC : {roc_auc_score(y_val, p_val_xgb):.4f}')\n"
        "print(f'XGBoost — OOT AUC : {roc_auc_score(y_oot, p_oot_xgb):.4f}')"
    ))

    # --- 7. Validation ---
    cells.append(md(
        "## 7. Validation suite\n\n"
        "Discrimination (Gini), calibration (predicted vs. observed), lift "
        "(rank-ordering quality), and PSI (score stability) — the four-corner "
        "validation that any credit model needs to pass before deployment."
    ))
    cells.append(code(
        "from IPython.display import display\n"
        "\n"
        "print('Calibration on OOT (XGBoost) — predicted vs. actual default rate:')\n"
        "cal = calibration_table(y_oot, p_oot_xgb, n_bins=10)\n"
        "display(cal)"
    ))
    cells.append(code(
        "print('Decile lift on OOT (XGBoost):')\n"
        "lift = decile_lift_table(y_oot, p_oot_xgb)\n"
        "display(lift)"
    ))
    cells.append(code(
        "p_train_xgb = xgb.predict_proba(X_train)[:, 1]\n"
        "psi = population_stability_index(p_train_xgb, p_oot_xgb)\n"
        "verdict = 'stable' if psi < 0.10 else 'moderate drift' if psi < 0.25 else 'major drift'\n"
        "print(f'Score PSI (train vs. OOT): {psi:.4f}  ->  {verdict}')"
    ))

    # --- 8. Visualization ---
    cells.append(md(
        "## 8. Visualization\n\n"
        "Charts are saved to `charts/` for embedding in the README and the model "
        "card PDF. Below we render them inline as well."
    ))
    cells.append(code(
        "CHARTS = Path().resolve() / 'charts'\n"
        "CHARTS.mkdir(exist_ok=True)\n"
        "\n"
        "auc = plot_roc_curve(y_oot, p_oot_xgb,\n"
        "                     'PD Model — ROC Curve (XGBoost, OOT)',\n"
        "                     str(CHARTS / 'roc_curve.png'))\n"
        "print(f'OOT AUC: {auc:.4f}   Gini: {2*auc - 1:.4f}')\n"
        "\n"
        "from PIL import Image as PILImage\n"
        "for fname in ['roc_curve.png', 'calibration.png', 'decile_lift.png']:\n"
        "    p = CHARTS / fname\n"
        "    if p.exists():\n"
        "        plt.figure(figsize=(7, 5))\n"
        "        plt.imshow(PILImage.open(p))\n"
        "        plt.axis('off')\n"
        "        plt.show()"
    ))

    # --- 9. Discussion ---
    cells.append(md(
        "## 9. Findings and production considerations\n\n"
        "**Discrimination.** XGBoost and the logistic regression land within ~1 "
        "Gini point of each other on this synthetic dataset. On real consumer-loan "
        "data the gap is usually 2-5 Gini points in favor of the gradient-boosted "
        "model — enough to be meaningful for cut-off based decisioning but small "
        "enough that the regulator-facing logistic remains a defensible production "
        "candidate.\n\n"
        "**Calibration.** Both models trained with class re-weighting produce "
        "probabilities that are systematically too high, because the loss function "
        "sees a 50/50 prior rather than the 12% population rate. For decision "
        "purposes (rank-ordering and cut-off) this is fine; for IFRS 9 / CECL "
        "consumption the scores must be recalibrated using isotonic or sigmoid "
        "regression on a held-out sample.\n\n"
        "**Stability.** Score PSI between train and OOT is far below 0.10, "
        "indicating the score distribution is stable across the time-based split. "
        "This is partly an artifact of synthetic data (no real macro shock); on "
        "real data, PSI is the first leading indicator of model decay and the "
        "trigger for a recalibration cycle.\n\n"
        "**What would change in production.**\n"
        "- Hyperparameter search via cross-validated grid or Bayesian optimization\n"
        "- Calibration layer (isotonic or sigmoid) on a held-out fold\n"
        "- SHAP-based reason codes for adverse-action notices (FCRA / ECOA)\n"
        "- Segmented models by product type, channel, or risk tier\n"
        "- Macroeconomic overlay for IFRS 9 forward-looking scenarios\n"
        "- Fairness testing across protected classes (race/sex proxies, age bands)"
    ))

    nb["cells"] = cells

    with open(OUT_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
