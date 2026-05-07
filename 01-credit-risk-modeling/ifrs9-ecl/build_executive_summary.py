"""
IFRS 9 ECL — Executive Summary PDF
=====================================

Generates a concise executive summary of the ECL run for distribution to a
risk committee or finance audience. Consumes results.json from
ecl_calculation.py.

Run AFTER ecl_calculation.py:
    python ecl_calculation.py
    python build_executive_summary.py
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
OUT_PATH = HERE / "IFRS9_ECL_Executive_Summary.pdf"

NAVY = colors.HexColor("#1f4e79")
GREEN = colors.HexColor("#548235")
AMBER = colors.HexColor("#e0a020")
RED = colors.HexColor("#a52a2a")
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
        "kpi_label": ParagraphStyle(
            "kpi_label", parent=base["Normal"],
            fontSize=9, textColor=colors.HexColor("#555555"),
            alignment=1,
        ),
        "kpi_value": ParagraphStyle(
            "kpi_value", parent=base["Normal"],
            fontSize=18, leading=22, textColor=NAVY,
            alignment=1, fontName="Helvetica-Bold",
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


def fmt_money(x: float) -> str:
    if abs(x) >= 1e9:
        return f"${x/1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:.0f}K"
    return f"${x:,.0f}"


def kpi_row(items, s):
    """Build a row of KPI cells. items = [(label, value), ...]"""
    cells = []
    for label, value in items:
        inner = Table(
            [[Paragraph(value, s["kpi_value"])],
             [Paragraph(label, s["kpi_label"])]],
            colWidths=[1.55 * inch], hAlign="CENTER",
        )
        inner.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        cells.append(inner)

    outer = Table([cells], colWidths=[1.6 * inch] * len(cells))
    outer.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ("BOX",    (0, 0), (-1, -1), 0.5, GREY_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, GREY_BORDER),
        ("BACKGROUND", (0, 0), (-1, -1), GREY_LIGHT),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    return outer


def stage_table(stages):
    header = ["Stage", "# Loans", "% of Book", "Exposure", "ECL", "Coverage"]
    rows = [header]
    color_map = {1: GREEN, 2: AMBER, 3: RED}
    for st in stages:
        rows.append([
            st["stage_name"],
            f"{st['n_loans']:,}",
            f"{st['pct_of_book']:.1%}",
            fmt_money(st["exposure"]),
            fmt_money(st["ecl"]),
            f"{st['coverage_ratio']:.2%}",
        ])

    t = Table(rows, hAlign="LEFT",
              colWidths=[1.7 * inch, 0.85 * inch, 0.85 * inch,
                         1.0 * inch, 1.0 * inch, 0.85 * inch])
    style = [
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0),  colors.whitesmoke),
        ("ALIGN",      (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN",      (0, 0), (0, -1),  "LEFT"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("BOX",        (0, 0), (-1, -1), 0.5, GREY_BORDER),
        ("INNERGRID",  (0, 0), (-1, -1), 0.25, GREY_BORDER),
        ("LEFTPADDING",(0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0, 0),(-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0),(-1, -1), 5),
    ]
    for i, st in enumerate(stages, start=1):
        c = color_map.get(st["stage"], GREY_LIGHT)
        style.append(("BACKGROUND", (0, i), (0, i), c))
        style.append(("TEXTCOLOR",  (0, i), (0, i), colors.whitesmoke))
        style.append(("FONTNAME",   (0, i), (0, i), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


def chart_image(filename: str, width: float = 6.5 * inch):
    path = CHARTS_DIR / filename
    if not path.exists():
        return Paragraph(f"<i>[Chart not found: {filename}]</i>",
                         styles()["small"])
    img = Image(str(path))
    aspect = img.imageHeight / float(img.imageWidth)
    img.drawWidth = width
    img.drawHeight = width * aspect
    return img


def build():
    if not RESULTS_PATH.exists():
        raise SystemExit(
            "results.json not found — run `python ecl_calculation.py` first."
        )
    with open(RESULTS_PATH) as f:
        r = json.load(f)

    s = styles()
    story = []

    # ----- Header -----
    story.append(Paragraph("IFRS 9 ECL — Executive Summary", s["title"]))
    story.append(Paragraph(
        f"Reporting date: {date.today().isoformat()}  ·  "
        f"Discount rate: {r['discount_rate']:.1%}  ·  "
        f"Synthetic-data demonstration",
        s["subtitle"],
    ))

    # ----- KPIs -----
    story.append(kpi_row([
        ("Total Loans",        f"{r['total_loans']:,}"),
        ("Total Exposure",     fmt_money(r["total_exposure"])),
        ("Total ECL",          fmt_money(r["total_ecl"])),
        ("Portfolio Coverage", f"{r['portfolio_coverage']:.2%}"),
    ], s))
    story.append(Spacer(1, 0.2 * inch))

    # ----- Headline narrative -----
    story.append(Paragraph("Headline", s["h1"]))
    s1 = next((x for x in r["stages"] if x["stage"] == 1), None)
    s2 = next((x for x in r["stages"] if x["stage"] == 2), None)
    s3 = next((x for x in r["stages"] if x["stage"] == 3), None)
    narrative = (
        f"The portfolio of {r['total_loans']:,} loans carries a total exposure "
        f"of {fmt_money(r['total_exposure'])} against which an ECL provision of "
        f"{fmt_money(r['total_ecl'])} has been calculated under IFRS 9, equivalent "
        f"to a coverage ratio of <b>{r['portfolio_coverage']:.2%}</b>."
    )
    if s3:
        narrative += (
            f" Stage 3 (credit-impaired) accounts for {s3['pct_of_book']:.1%} "
            f"of the book and contributes {fmt_money(s3['ecl'])} of the provision "
            f"({s3['ecl']/r['total_ecl']:.0%} of total ECL)."
        )
    if s2:
        narrative += (
            f" Stage 2 (significant increase in credit risk) covers "
            f"{s2['pct_of_book']:.1%} of accounts and "
            f"{fmt_money(s2['ecl'])} of provision."
        )
    story.append(Paragraph(narrative, s["body"]))

    # ----- Stage breakdown table -----
    story.append(Paragraph("Provision by IFRS 9 Stage", s["h1"]))
    story.append(stage_table(r["stages"]))

    story.append(Spacer(1, 0.2 * inch))
    story.append(chart_image("stage_breakdown.png"))
    story.append(Paragraph(
        "<b>Figure 1.</b> Account counts (left) and ECL (right) by IFRS-9 stage.",
        s["small"],
    ))

    story.append(PageBreak())

    # ----- Coverage chart -----
    story.append(Paragraph("Coverage Ratio by Stage", s["h1"]))
    story.append(Paragraph(
        "Coverage ratio is the share of stage exposure carried as ECL provision. "
        "Stage 1 carries 12-month ECL only and therefore shows the lowest coverage; "
        "Stage 2 books lifetime ECL on assets that have not yet defaulted and shows "
        "an intermediate ratio; Stage 3 is credit-impaired so coverage tracks "
        "expected loss-given-default directly.",
        s["body"],
    ))
    story.append(chart_image("coverage_ratio.png"))
    story.append(Paragraph(
        "<b>Figure 2.</b> ECL / exposure ratio by stage.",
        s["small"],
    ))

    # ----- Lifetime PD curve -----
    story.append(Paragraph("Lifetime PD Construction (Stage 2)", s["h1"]))
    story.append(Paragraph(
        "Lifetime ECL for Stage 2 accounts requires projecting default probability "
        "across the full remaining contractual life. Term-structure construction "
        "uses a constant-hazard transformation of the 12-month PD: "
        "h = -ln(1 - PD<sub>12m</sub>) / 12, with cumulative PD at month t equal "
        "to 1 - e<sup>-h·t</sup>. Each period's marginal PD is multiplied by the "
        "discounted exposure and LGD, and the resulting cash flows are summed.",
        s["body"],
    ))
    story.append(chart_image("lifetime_pd_curve.png"))
    story.append(Paragraph(
        "<b>Figure 3.</b> Cumulative PD over a 60-month horizon for three risk levels.",
        s["small"],
    ))

    # ----- Methodology box -----
    story.append(Paragraph("Methodology Summary", s["h1"]))
    story.append(Paragraph(
        "<b>Stage classification</b> follows the IFRS 9 standard: Stage 1 covers "
        "performing loans with no significant increase in credit risk since "
        "origination; Stage 2 captures loans whose lifetime PD has materially "
        "deteriorated (here proxied by a 2.5× ratio of current to origination PD, "
        "or 30+ DPD); Stage 3 contains credit-impaired loans (defaulted or 90+ DPD). "
        "ECL is computed as PD × LGD × EAD, with Stage 1 on a 12-month horizon and "
        f"Stages 2-3 on a lifetime horizon discounted at "
        f"{r['discount_rate']:.0%} per annum.",
        s["body"],
    ))
    story.append(Paragraph(
        "<i>This summary is generated from a synthetic-data demonstration model. "
        "Figures are illustrative, not a benchmark.</i>",
        s["small"],
    ))

    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.6 * inch,  bottomMargin=0.6 * inch,
        title="IFRS 9 ECL Executive Summary",
        author="Risk Modeling Portfolio",
    )
    doc.build(story)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
