"""
PD Model Card — PDF Generator
==============================

Builds a one-page-per-section model card PDF in the format used by major
banks and fintechs for governance documentation. The card consumes the
metrics produced by `pd_model.py` (results.json) and the charts saved
into `charts/`.

Run AFTER `pd_model.py` has produced its outputs:
    python pd_model.py
    python build_model_card.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.json"
CHARTS_DIR = HERE / "charts"
OUT_PATH = HERE / "PD_Model_Card.pdf"

NAVY = colors.HexColor("#1f4e79")
GREY_LIGHT = colors.HexColor("#f0f0f0")
GREY_BORDER = colors.HexColor("#cccccc")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"],
            fontSize=22, leading=26, textColor=NAVY,
            alignment=0, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"],
            fontSize=11, textColor=colors.HexColor("#555555"),
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"],
            fontSize=14, leading=18, textColor=NAVY,
            spaceBefore=14, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"],
            fontSize=11, leading=14, textColor=NAVY,
            spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"],
            fontSize=10, leading=14, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small", parent=base["BodyText"],
            fontSize=8.5, leading=11,
            textColor=colors.HexColor("#555555"),
        ),
    }


def kv_table(rows, col_widths=(2.2 * inch, 4.0 * inch)):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9.5),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1),  GREY_LIGHT),
        ("FONTNAME",   (0, 0), (0, -1),  "Helvetica-Bold"),
        ("BOX",        (0, 0), (-1, -1), 0.5, GREY_BORDER),
        ("INNERGRID",  (0, 0), (-1, -1), 0.25, GREY_BORDER),
        ("LEFTPADDING",(0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0, 0),(-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0,0),(-1, -1), 4),
    ]))
    return t


def chart_image(filename: str, width: float = 5.4 * inch):
    path = CHARTS_DIR / filename
    if not path.exists():
        return Paragraph(f"<i>[Chart not found: {filename} — run pd_model.py first]</i>",
                         styles()["small"])
    img = Image(str(path))
    aspect = img.imageHeight / float(img.imageWidth)
    img.drawWidth = width
    img.drawHeight = width * aspect
    return img


def build():
    if not RESULTS_PATH.exists():
        raise SystemExit(
            "results.json not found — run `python pd_model.py` first."
        )
    with open(RESULTS_PATH) as f:
        r = json.load(f)

    s = styles()
    story = []

    # ------- Header -------
    story.append(Paragraph("Probability of Default — Model Card", s["title"]))
    story.append(Paragraph(
        f"Version 1.0  ·  Generated {date.today().isoformat()}  ·  "
        f"Synthetic-data demonstration  ·  Status: Development",
        s["subtitle"],
    ))

    # ------- 1. Model overview -------
    story.append(Paragraph("1. Model Overview", s["h1"]))
    story.append(kv_table([
        ["Model name",        "Consumer-loan PD (12-month)"],
        ["Model type",        "XGBoost classifier (challenger) + logistic regression (benchmark)"],
        ["Target variable",   "default_flag — 90+ days past due in the next 12 months"],
        ["Use case",          "Underwriting decisioning and IFRS 9 / CECL provisioning"],
        ["Population",        "U.S. consumer term-loan applicants"],
        ["Output",            "Probability of default in [0, 1] and decile risk band"],
        ["Refresh cadence",   "Quarterly recalibration; full rebuild every 18 months"],
    ]))

    # ------- 2. Data -------
    story.append(Paragraph("2. Data", s["h1"]))
    story.append(Paragraph(
        f"The training population was {r['n_train']:,} loans (in-time) with an "
        f"{r['default_rate_dev']:.2%} default rate. The model was validated on a "
        f"hold-out validation set ({r['n_val']:,} loans) and a true out-of-time "
        f"(OOT) sample of {r['n_oot']:,} loans originated in the most recent six months "
        f"(default rate {r['default_rate_oot']:.2%}).",
        s["body"],
    ))
    story.append(Paragraph(
        "Features include borrower demographics, loan characteristics, credit-bureau "
        "snapshot at origination (delinquencies, utilization, history length), and one-hot "
        "encoded loan purpose and home-ownership flags. Numeric inputs are standard-scaled.",
        s["body"],
    ))

    # ------- 3. Performance -------
    story.append(Paragraph("3. Performance (Out-of-Time)", s["h1"]))
    story.append(kv_table([
        ["Logistic regression — OOT AUC",  f"{r['logistic_oot_auc']:.4f}"],
        ["XGBoost — OOT AUC",              f"{r['xgboost_oot_auc']:.4f}"],
        ["XGBoost — OOT Gini",             f"{r['xgboost_oot_gini']:.4f}"],
        ["Score PSI (train vs. OOT)",      f"{r['score_psi_train_vs_oot']:.4f}  "
                                            f"({'stable' if r['score_psi_train_vs_oot'] < 0.10 else 'moderate' if r['score_psi_train_vs_oot'] < 0.25 else 'major drift'})"],
    ]))

    story.append(Spacer(1, 0.15 * inch))
    story.append(chart_image("roc_curve.png"))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(
        "<b>Figure 1.</b> ROC curve on the out-of-time sample. The diagonal "
        "is the random-classifier reference.",
        s["small"],
    ))

    story.append(PageBreak())

    # ------- 4. Calibration -------
    story.append(Paragraph("4. Calibration", s["h1"]))
    story.append(Paragraph(
        "Predicted probabilities are systematically high relative to observed default "
        "rates. This is the expected consequence of training with "
        "<b>scale_pos_weight</b> set to balance the classes — the model learns to "
        "discriminate but its raw probabilities reflect a 50/50 prior rather than "
        "the population base rate. For deployment in an ECL/IFRS-9 context, an "
        "isotonic or sigmoid recalibration step is required before the scores are "
        "consumed as PDs. Rank-ordering (and therefore the score-band bucketing "
        "shown below) is unaffected.",
        s["body"],
    ))
    story.append(chart_image("calibration.png"))
    story.append(Paragraph(
        "<b>Figure 2.</b> Reliability diagram on OOT. Marker size is proportional "
        "to bin count.",
        s["small"],
    ))

    # ------- 5. Decile lift -------
    story.append(Paragraph("5. Decile Lift", s["h1"]))
    story.append(Paragraph(
        "Lift relative to the OOT base default rate by predicted-risk decile. The top "
        "decile concentrates roughly 2-3× the population default rate; the bottom "
        "decile is well below it. Rank-ordering is monotone, which is the property "
        "required for cut-off based underwriting decisions.",
        s["body"],
    ))
    story.append(chart_image("decile_lift.png"))
    story.append(Paragraph("<b>Figure 3.</b> Decile lift on OOT.", s["small"]))

    story.append(PageBreak())

    # ------- 6. Feature importance -------
    story.append(Paragraph("6. Feature Importance", s["h1"]))
    story.append(Paragraph(
        "Top features by XGBoost gain. Loan-purpose flags, revolving utilization, "
        "delinquency history, and credit-history length dominate, consistent with "
        "industry experience on consumer term-loan PD models.",
        s["body"],
    ))
    story.append(chart_image("feature_importance.png"))
    story.append(Paragraph("<b>Figure 4.</b> Top 15 features by gain.", s["small"]))

    # ------- 7. Limitations -------
    story.append(Paragraph("7. Limitations", s["h1"]))
    limitations = [
        "Trained on synthetic data — coefficients and Gini values are illustrative, not benchmarks.",
        "Probability calibration is biased high (see Section 4); recalibrate before use as PD.",
        "No macroeconomic overlay — IFRS 9 forward-looking scenarios would need to be applied externally.",
        "No segmentation by product type, channel, or vintage; a production version would build separate models or include those as features.",
        "Bureau attributes are point-in-time at origination; behavioural updates not modeled.",
    ]
    for item in limitations:
        story.append(Paragraph(f"• {item}", s["body"]))

    # ------- 8. Monitoring plan -------
    story.append(Paragraph("8. Monitoring", s["h1"]))
    story.append(kv_table([
        ["Score PSI threshold",     "< 0.10 stable  ·  0.10-0.25 watch  ·  > 0.25 investigate"],
        ["Feature PSI",             "Monthly; same thresholds, per feature"],
        ["Performance refresh",     "Vintage-level Gini/KS as outcomes mature (12-month tag)"],
        ["Calibration check",       "Quarterly observed-vs-predicted by decile; recalibrate if MAE > 5 ppt"],
        ["Override rate",           "Tracked by underwriter and segment to detect emerging adverse selection"],
    ], col_widths=(2.0 * inch, 4.2 * inch)))

    # ------- Footer -------
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "<i>This model card describes a portfolio demonstration trained on synthetic "
        "data. It is not an approved production artefact.</i>",
        s["small"],
    ))

    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=LETTER,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch,  bottomMargin=0.7 * inch,
        title="PD Model Card",
        author="Risk Modeling Portfolio",
    )
    doc.build(story)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
