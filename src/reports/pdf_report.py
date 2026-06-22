"""
pdf_report.py
-------------
Renders an UnderwriterReport to a professional PDF using ReportLab.

The PDF layout includes:
  - Header with platform name and report metadata
  - Colour-coded risk tier and rule verdict badges
  - Each report section as a formatted content block
  - Guardrail disclaimer footer on every page

Usage
-----
    from src.reports.pdf_report import PDFReportRenderer
    from src.reports.llm_report import UnderwriterReport

    renderer = PDFReportRenderer()
    path = renderer.save(report, output_path="reports/underwriter_report.pdf")
    print(f"PDF saved to: {path}")
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    HRFlowable,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

NAVY      = colors.HexColor("#1A2744")
DARK_BLUE = colors.HexColor("#1E3A5F")
MID_BLUE  = colors.HexColor("#2C5F8A")
LIGHT_BLUE= colors.HexColor("#E8F1FA")
WHITE     = colors.white

RED       = colors.HexColor("#C0392B")
ORANGE    = colors.HexColor("#E67E22")
YELLOW    = colors.HexColor("#F1C40F")
GREEN     = colors.HexColor("#27AE60")
GREY_LIGHT= colors.HexColor("#F5F5F5")
GREY_MID  = colors.HexColor("#BDC3C7")
GREY_DARK = colors.HexColor("#7F8C8D")
BLACK     = colors.HexColor("#1C1C1C")

# Verdict colour map
VERDICT_COLOURS = {
    "DECLINE":       (RED,    WHITE),
    "REFER":         (ORANGE, WHITE),
    "MANUAL_REVIEW": (YELLOW, BLACK),
    "AUTO_APPROVE":  (GREEN,  WHITE),
}

# Risk tier colour map
TIER_COLOURS = {
    "Low":       (GREEN,  WHITE),
    "Medium":    (YELLOW, BLACK),
    "High":      (ORANGE, WHITE),
    "Very High": (RED,    WHITE),
}


# ---------------------------------------------------------------------------
# Style sheet
# ---------------------------------------------------------------------------

def _build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            fontName="Helvetica",
            fontSize=10,
            textColor=GREY_MID,
            alignment=TA_LEFT,
        ),
        "section_head": ParagraphStyle(
            "SectionHead",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=DARK_BLUE,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName="Helvetica",
            fontSize=9,
            textColor=BLACK,
            leading=14,
            spaceAfter=4,
        ),
        "meta_label": ParagraphStyle(
            "MetaLabel",
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=GREY_DARK,
        ),
        "meta_value": ParagraphStyle(
            "MetaValue",
            fontName="Helvetica",
            fontSize=9,
            textColor=BLACK,
        ),
        "guardrail": ParagraphStyle(
            "Guardrail",
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            textColor=GREY_DARK,
            leading=11,
            alignment=TA_CENTER,
        ),
        "footer": ParagraphStyle(
            "Footer",
            fontName="Helvetica",
            fontSize=7,
            textColor=GREY_MID,
            alignment=TA_RIGHT,
        ),
    }
    return styles


# ---------------------------------------------------------------------------
# Page callbacks
# ---------------------------------------------------------------------------

def _make_header_footer(applicant: str, timestamp: str):
    """Return an onPage callback that draws header/footer on every page."""

    def _draw(canvas, doc):
        canvas.saveState()
        w, h = A4

        # --- Header bar ---
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - 40 * mm, w, 40 * mm, fill=1, stroke=0)

        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(15 * mm, h - 18 * mm, "AI-ASSISTED CREDIT RISK PLATFORM")

        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(GREY_MID)
        canvas.drawString(15 * mm, h - 26 * mm, "CONFIDENTIAL UNDERWRITER REPORT")

        # Timestamp top-right
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 15 * mm, h - 18 * mm, timestamp)
        canvas.drawRightString(w - 15 * mm, h - 26 * mm, applicant)

        # --- Footer ---
        canvas.setFillColor(GREY_LIGHT)
        canvas.rect(0, 0, w, 12 * mm, fill=1, stroke=0)
        canvas.setStrokeColor(GREY_MID)
        canvas.setLineWidth(0.5)
        canvas.line(15 * mm, 12 * mm, w - 15 * mm, 12 * mm)

        canvas.setFillColor(GREY_DARK)
        canvas.setFont("Helvetica-Oblique", 6.5)
        canvas.drawString(
            15 * mm, 4 * mm,
            "This document is generated by an AI system and does not constitute "
            "a loan approval, rejection, or recommendation. Human underwriter sign-off required."
        )
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(
            w - 15 * mm, 4 * mm,
            f"Page {doc.page}"
        )

        canvas.restoreState()

    return _draw


# ---------------------------------------------------------------------------
# PDFReportRenderer
# ---------------------------------------------------------------------------

class PDFReportRenderer:
    """
    Renders an UnderwriterReport to a formatted A4 PDF.

    Usage
    -----
        renderer = PDFReportRenderer()
        path = renderer.save(report, "reports/my_report.pdf")
    """

    def save(self, report: Any, output_path: str, rule_verdict: Optional[str] = None) -> str:
        """
        Render the UnderwriterReport to a PDF file.

        Parameters
        ----------
        report       : UnderwriterReport instance from llm_report.py.
        output_path  : Destination file path (creates dirs if needed).
        rule_verdict : The explicit string routing decision from the rule engine.
                       If provided, overrides text-parsing logic.

        Returns
        -------
        str : Absolute path to the saved PDF.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        styles   = _build_styles()
        w, h     = A4
        margin   = 15 * mm
        top_gap  = 45 * mm   # space for header bar
        bottom_gap = 12 * mm  # space for footer (disclaimer area)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=margin,
            rightMargin=margin,
            topMargin=top_gap,
            bottomMargin=margin + bottom_gap,
        )

        on_page = _make_header_footer(
            applicant=report.applicant_name,
            timestamp=report.report_timestamp,
        )

        story = []

        # ------------------------------------------------------------------
        # 1. Report metadata summary table
        # ------------------------------------------------------------------
        meta_data = report.to_dict()
        risk_tier     = "N/A"

        if rule_verdict is None:
            # Try to extract verdict and tier from section text (best-effort fallback)
            rule_verdict = "N/A"
            routing_text = meta_data["sections"].get("ROUTING DECISION SUMMARY", "")
            # Split and clean words to look for exact keyword matches
            words = [w.strip(".,:;()[]{}'\"").upper() for w in routing_text.split()]
            # Check for matches in precedence order
            for verdict in ("AUTO_APPROVE", "MANUAL_REVIEW", "DECLINE", "REFER"):
                if verdict in words:
                    rule_verdict = verdict
                    break
                # Substring check fallback
                elif verdict in routing_text.upper():
                    rule_verdict = verdict
                    break

        risk_text = meta_data["sections"].get("RISK MODEL ANALYSIS", "")

        # Verdict badge colour
        v_bg, v_fg = VERDICT_COLOURS.get(rule_verdict, (GREY_MID, BLACK))

        meta_rows = [
            [
                Paragraph("Applicant", styles["meta_label"]),
                Paragraph(report.applicant_name, styles["meta_value"]),
                Paragraph("Routing Decision", styles["meta_label"]),
                Paragraph(
                    f'<font color="white"><b> {rule_verdict} </b></font>',
                    ParagraphStyle(
                        "badge",
                        fontName="Helvetica-Bold",
                        fontSize=9,
                        textColor=v_fg,
                        backColor=v_bg,
                        borderPadding=(3, 6, 3, 6),
                    ),
                ),
            ],
            [
                Paragraph("Risk Model", styles["meta_label"]),
                Paragraph("XGBoost Binary Classifier", styles["meta_value"]),
                Paragraph("Explainability", styles["meta_label"]),
                Paragraph("SHAP TreeExplainer", styles["meta_value"]),
            ],
            [
                Paragraph("Model AUC", styles["meta_label"]),
                Paragraph("0.742", styles["meta_value"]),
                Paragraph("Fraud Engine", styles["meta_label"]),
                Paragraph("Deterministic Engine", styles["meta_value"]),
            ],
            [
                Paragraph("Report Generator", styles["meta_label"]),
                Paragraph(f"{report.llm_provider} (Advisory Only)", styles["meta_value"]),
                Paragraph("Generated At", styles["meta_label"]),
                Paragraph(report.report_timestamp, styles["meta_value"]),
            ],
        ]

        col_widths = [35 * mm, 55 * mm, 40 * mm, 50 * mm]
        meta_table = Table(meta_rows, colWidths=col_widths)
        meta_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), LIGHT_BLUE),
            ("GRID",         (0, 0), (-1, -1), 0.5, GREY_MID),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))

        story.append(Spacer(1, 4 * mm))
        story.append(meta_table)
        story.append(Spacer(1, 6 * mm))

        # ------------------------------------------------------------------
        # 2. Report sections
        # ------------------------------------------------------------------
        section_order = [
            "APPLICANT OVERVIEW",
            "RISK MODEL ANALYSIS",
            "FRAUD & ANOMALY ASSESSMENT",
            "ROUTING DECISION SUMMARY",
        ]

        for title in section_order:
            content = meta_data["sections"].get(title, "")
            if not content:
                continue

            # Section header
            story.append(HRFlowable(
                width="100%", thickness=1.5,
                color=MID_BLUE, spaceAfter=2,
            ))
            story.append(Paragraph(f"◆  {title}", styles["section_head"]))
            story.append(Spacer(1, 2 * mm))

            # Section body — preserve line breaks
            for line in content.split("\n"):
                text = line.strip()
                if text:
                    story.append(Paragraph(text, styles["body"]))
                else:
                    story.append(Spacer(1, 2 * mm))

            story.append(Spacer(1, 4 * mm))

        # Handle any extra sections not in the standard order
        for title, content in meta_data["sections"].items():
            if title not in section_order and content.strip():
                story.append(HRFlowable(width="100%", thickness=1, color=GREY_MID))
                story.append(Paragraph(title, styles["section_head"]))
                story.append(Paragraph(content, styles["body"]))
                story.append(Spacer(1, 4 * mm))

        # ------------------------------------------------------------------
        # 3. Guardrail box
        # ------------------------------------------------------------------
        story.append(Spacer(1, 4 * mm))
        story.append(HRFlowable(width="100%", thickness=1, color=GREY_MID))
        guardrail_table = Table(
            [[Paragraph(report.guardrail, styles["guardrail"])]],
            colWidths=[w - 2 * margin],
        )
        guardrail_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), GREY_LIGHT),
            ("BOX",          (0, 0), (-1, -1), 0.5, GREY_MID),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(KeepTogether(guardrail_table))

        doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
        return str(output_path.resolve())
