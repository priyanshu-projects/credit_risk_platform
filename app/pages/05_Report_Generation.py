"""
05_Report_Generation.py
Full underwriter report: Rule Engine verdict + LLM narrative + PDF download.
"""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.fraud.fraud_engine import FraudEngine
from src.reports.llm_report import LLMReportGenerator
from src.reports.pdf_report import PDFReportRenderer
from src.rules.rule_engine import RuleEngine

st.set_page_config(page_title="Report Generation", page_icon="📝", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.page-header {
    background: linear-gradient(135deg, #2c1a00, #6a3e00);
    border-radius: 12px; padding: 28px 32px; margin-bottom: 28px; color: white;
}
.page-header h2 { margin: 0; font-size: 1.8rem; font-weight: 700; }
.page-header p  { margin: 6px 0 0; color: #f0c87a; font-size: 0.95rem; }

.verdict-box {
    border-radius: 16px; padding: 32px 28px; text-align: center;
    color: white; box-shadow: 0 4px 20px rgba(0,0,0,0.2); margin-bottom: 20px;
}
.verdict-DECLINE       { background: linear-gradient(135deg, #c0392b, #e74c3c); }
.verdict-AUTO_APPROVE  { background: linear-gradient(135deg, #27ae60, #2ecc71); }
.verdict-MANUAL_REVIEW { background: linear-gradient(135deg, #d68910, #f1c40f); color: #2c1a00; }
.verdict-REFER         { background: linear-gradient(135deg, #e67e22, #f39c12); }

.verdict-name { font-size: 2.8rem; font-weight: 800; letter-spacing: -1px; }
.verdict-sub  { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }

.rule-tag {
    background: rgba(255,255,255,0.2); border-radius: 6px;
    padding: 4px 12px; font-size: 0.78rem; font-family: monospace;
    display: inline-block; margin: 3px;
}

.section-block {
    background: white; border-radius: 12px; padding: 22px 26px;
    border: 1px solid #e8edf4; box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    margin-bottom: 18px;
}
.section-title {
    color: #1a2744; font-size: 1rem; font-weight: 700;
    margin: 0 0 14px 0; padding-bottom: 10px;
    border-bottom: 2px solid #e8edf4;
}
.section-body {
    color: #3a4a5e; font-size: 0.92rem; line-height: 1.7;
    white-space: pre-wrap;
}
.guardrail-strip {
    background: #fff8e1; border: 1px solid #ffe082;
    border-radius: 10px; padding: 16px 20px; margin-top: 20px;
}
.guardrail-strip p { color: #7a5800; font-size: 0.88rem; margin: 0; }

.input-summary {
    background: #f8fafd; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #e0e8f0; margin-bottom: 20px;
}
.input-row {
    display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid #f0f2f5; font-size: 0.88rem;
}
.input-row:last-child { border-bottom: none; }
.in-key { color: #6b7c93; font-weight: 600; }
.in-val { color: #1a2744; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <h2>📝 Report Generation</h2>
  <p>Rule Engine applies credit policy → Gemini Flash writes the structured underwriter report.</p>
</div>""", unsafe_allow_html=True)

# ── Require prior steps ───────────────────────────────────────────────────────
missing = []
if "extraction_result" not in st.session_state: missing.append("Document Extraction")
if "prob_default"       not in st.session_state: missing.append("Risk Assessment")
if missing:
    st.warning(f"⚠️ Please complete these steps first: **{', '.join(missing)}**")
    st.stop()

extraction_result = st.session_state["extraction_result"]
fields            = extraction_result.get("fields", {})
prob_default      = st.session_state["prob_default"]
risk_tier         = st.session_state.get("risk_tier", "Medium")

# Use cached fraud report or re-run
fraud_report = st.session_state.get("fraud_report")
if fraud_report is None:
    fraud_report = FraudEngine().evaluate(extraction_result)
    st.session_state["fraud_report"] = fraud_report

# Use cached SHAP or default
shap_factors = st.session_state.get("shap_factors", {
    "base_value": 0.0,
    "risk_drivers": [],
    "mitigators": [],
})

# ── Run Rule Engine ───────────────────────────────────────────────────────────
rule_engine  = RuleEngine()
rule_decision = rule_engine.evaluate(
    prob_default=prob_default,
    fraud_risk_level=fraud_report.fraud_risk_level,
    fields=fields,
)
st.session_state["rule_decision"] = rule_decision

verdict = rule_decision.verdict

# ── Verdict display ───────────────────────────────────────────────────────────
col_verdict, col_inputs = st.columns([1, 1.4])

with col_verdict:
    verdict_emoji = {
        "DECLINE":       "🚫",
        "AUTO_APPROVE":  "✅",
        "MANUAL_REVIEW": "👤",
        "REFER":         "📋",
    }.get(verdict, "⚠️")

    tags_html = " ".join(
        f'<span class="rule-tag">{r}</span>'
        for r in rule_decision.triggered_rules
    )
    st.markdown(f"""
    <div class="verdict-box verdict-{verdict}">
      <div class="verdict-sub">RULE ENGINE DECISION</div>
      <div class="verdict-name">{verdict_emoji} {verdict.replace("_", " ")}</div>
      <div class="verdict-sub" style="margin-top:12px;">{rule_decision.rationale[:120]}...</div>
      <div style="margin-top:14px;">{tags_html}</div>
    </div>""", unsafe_allow_html=True)

with col_inputs:
    st.markdown("**📥 Inputs to Rule Engine:**")
    st.markdown('<div class="input-summary">', unsafe_allow_html=True)
    rows = {
        "P(Default)":         f"{prob_default:.2%}",
        "Risk Tier":          risk_tier,
        "Fraud Risk Level":   fraud_report.fraud_risk_level,
        "FICO Score":         str(fields.get("fico_avg", "N/A")),
        "DTI":                f"{fields.get('dti', 0):.1f}%" if fields.get("dti") else "N/A",
        "Loan Amount":        f"${fields.get('loan_amount', 0):,.0f}" if fields.get("loan_amount") else "N/A",
    }
    for k, v in rows.items():
        st.markdown(f'<div class="input-row"><span class="in-key">{k}</span><span class="in-val">{v}</span></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Generate LLM report ───────────────────────────────────────────────────────
st.subheader("📄 Underwriter Report")

generate_btn = st.button("🤖 Generate LLM Underwriter Report", type="primary", use_container_width=True)

if "llm_report" in st.session_state:
    generate_btn = True  # Auto-show if already generated

if generate_btn:
    if "llm_report" not in st.session_state:
        with st.spinner("✍️ Generating underwriter report with Gemini Flash..."):
            try:
                generator = LLMReportGenerator()
                report    = generator.generate(
                    fields=fields,
                    prob_default=prob_default,
                    risk_tier=risk_tier,
                    shap_factors=shap_factors,
                    fraud_report=fraud_report,
                    rule_decision=rule_decision,
                )
                st.session_state["llm_report"] = report
            except Exception as e:
                st.error(f"Report generation error: {e}")
                st.stop()

    report = st.session_state["llm_report"]

    # ── Display sections ──────────────────────────────────────────────────────
    section_icons = {
        "APPLICANT OVERVIEW":          "👤",
        "RISK MODEL ANALYSIS":         "📊",
        "FRAUD & ANOMALY ASSESSMENT":  "🚨",
        "ROUTING DECISION SUMMARY":    "🔀",
    }

    st.caption(f"Generated by: **{report.llm_provider}** | {report.report_timestamp}")

    for title, content in report.sections.items():
        icon = section_icons.get(title, "📋")
        st.markdown(f"""
        <div class="section-block">
          <div class="section-title">{icon} {title}</div>
          <div class="section-body">{content}</div>
        </div>""", unsafe_allow_html=True)

    # ── Guardrail ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="guardrail-strip">
      <p>⚠️ <strong>Guardrail Notice:</strong> {report.guardrail}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PDF download ──────────────────────────────────────────────────────────
    st.subheader("💾 Download PDF Report")

    col_dl, col_info = st.columns([1, 1.5])

    with col_dl:
        if st.button("📄 Generate PDF", type="secondary", use_container_width=True):
            with st.spinner("Rendering PDF..."):
                try:
                    renderer = PDFReportRenderer()
                    applicant = (fields.get("applicant_name") or "unknown").replace(" ", "_")
                    pdf_path  = f"artifacts/reports/underwriter_{applicant}.pdf"
                    saved     = renderer.save(report, pdf_path, rule_verdict=verdict)
                    st.success(f"✅ PDF saved to: `{saved}`")

                    with open(saved, "rb") as f:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=f.read(),
                            file_name=Path(saved).name,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"PDF error: {e}")

    with col_info:
        st.markdown("""
        **PDF includes:**
        - Professional header with platform branding
        - Colour-coded rule verdict badge
        - All four report sections
        - Guardrail disclaimer on every page
        """)

    # ── Raw JSON ──────────────────────────────────────────────────────────────
    with st.expander("🔎 Raw report JSON"):
        st.json(report.to_dict())
