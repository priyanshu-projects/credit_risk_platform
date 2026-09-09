"""
03_Fraud_Analysis.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.fraud.fraud_engine import FraudEngine

st.set_page_config(page_title="Fraud Analysis", page_icon="🚨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.page-header {
    background: linear-gradient(135deg, #2c1a44, #4a2c7a);
    border-radius: 12px; padding: 28px 32px; margin-bottom: 28px; color: white;
}
.page-header h2 { margin: 0; font-size: 1.8rem; font-weight: 700; }
.page-header p  { margin: 6px 0 0; color: #c8a8f0; font-size: 0.9rem; }

.level-clear  { background: linear-gradient(135deg, #27ae60, #2ecc71); }
.level-low    { background: linear-gradient(135deg, #3498db, #5dade2); }
.level-medium { background: linear-gradient(135deg, #f39c12, #f1c40f); }
.level-high   { background: linear-gradient(135deg, #c0392b, #e74c3c); }

.risk-level-box {
    border-radius: 16px; padding: 36px 24px; text-align: center; color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2); margin-bottom: 20px;
}
.level-name { font-size: 3rem; font-weight: 800; letter-spacing: -1px; }
.level-sub  { font-size: 0.88rem; opacity: 0.85; margin-top: 4px; }

.flag-card {
    border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;
    border-left: 5px solid; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.flag-high   { background: #fdecea; border-color: #c0392b; }
.flag-medium { background: #fff8e1; border-color: #f39c12; }
.flag-low    { background: #e8f4fd; border-color: #3498db; }

.flag-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.flag-name   { font-weight: 700; color: #1a2744; font-size: 0.9rem; }
.severity-tag { border-radius: 12px; padding: 3px 12px; font-size: 0.72rem; font-weight: 700; }
.tag-high   { background: #c0392b; color: white; }
.tag-medium { background: #f39c12; color: white; }
.tag-low    { background: #3498db; color: white; }

.flag-desc { color: #4a5a6e; font-size: 0.86rem; line-height: 1.5; margin-bottom: 8px; }
.evidence  { background: rgba(255,255,255,0.6); border-radius: 6px; padding: 8px 12px; }
.ev-row    { display: flex; justify-content: space-between; font-size: 0.8rem; }
.ev-key    { color: #6b7c93; }
.ev-val    { color: #1a2744; font-weight: 600; }

.check-row { padding: 4px 0; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <h2>Fraud Analysis</h2>
  <p>13 checks run against the application for income and behaviour anomalies.</p>
</div>""", unsafe_allow_html=True)

if "extraction_result" not in st.session_state:
    st.warning("Please complete Document Extraction first.")
    st.stop()

extraction_result = st.session_state["extraction_result"]
fields            = extraction_result.get("fields", {})

if "fraud_report" not in st.session_state:
    with st.spinner("Running checks..."):
        engine = FraudEngine()
        report = engine.evaluate(extraction_result)
        st.session_state["fraud_report"] = report
        st.switch_page("pages/03_Fraud_Analysis.py")
else:
    report = st.session_state["fraud_report"]

level  = report.fraud_risk_level.lower()
flags  = report.flags
counts = report.flag_counts

col_level, col_counts = st.columns([1, 1.6])

with col_level:
    emoji = {"clear": "✅", "low": "🔵", "medium": "🟡", "high": "🔴"}.get(level, "")
    st.markdown(f"""
    <div class="risk-level-box level-{level}">
      <div class="level-name">{emoji} {report.fraud_risk_level}</div>
      <div class="level-sub">{len(flags)} flag(s) raised</div>
    </div>""", unsafe_allow_html=True)

with col_counts:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("High", counts["high"])
    with c2:
        st.metric("Medium", counts["medium"])
    with c3:
        st.metric("Low", counts["low"])

    st.markdown("<br>", unsafe_allow_html=True)

    check_name_map = {
        "dti_income_mismatch":                "DTI vs income",
        "high_installment_to_income":         "Installment to income ratio",
        "implausible_income_too_low":         "Income too low",
        "implausible_income_too_high":        "Income too high",
        "high_util_high_income_inconsistency":"Utilization vs income",
        "round_number_income":                "Round number income",
        "round_number_loan_amount":           "Round number loan amount",
        "fico_dti_inconsistency":             "FICO vs DTI",
        "new_credit_surge":                   "New credit surge",
        "delinquency_despite_high_fico":      "Delinquency despite high FICO",
        "public_record_high_fico_mismatch":   "Public record vs FICO",
        "revolving_balance_over_limit":       "Revolving balance over limit",
        "excessive_loan_to_income_leverage":  "Loan to income leverage",
        "synthetic_identity_thin_file":       "Thin file identity",
        "ocr_confidence_fallback":            "OCR confidence fallback",
        "high_missing_fields_anomaly":        "High missing fields",
    }
    triggered_names = {f.check_name for f in flags}
    for check_name, display_name in check_name_map.items():
        if check_name in triggered_names:
            st.markdown(f'<div class="check-row" style="color:#c0392b; font-weight:600;">🔴 {display_name}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="check-row" style="color:#6b7c93;">🟢 {display_name}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if not flags:
    st.success("No issues found.")
else:
    st.subheader(f"{len(flags)} issue(s) found")

    severity_order = {"high": 0, "medium": 1, "low": 2}
    sorted_flags = sorted(flags, key=lambda f: severity_order.get(f.severity, 3))

    for flag in sorted_flags:
        sev = flag.severity.lower()
        ev_rows = "".join(
            f'<div class="ev-row"><span class="ev-key">{k}</span><span class="ev-val">{v}</span></div>'
            for k, v in flag.evidence.items()
        )
        st.markdown(f"""
        <div class="flag-card flag-{sev}">
          <div class="flag-header">
            <span class="flag-name">{flag.check_name.replace("_", " ").title()}</span>
            <span class="severity-tag tag-{sev}">{flag.severity.upper()}</span>
          </div>
          <div class="flag-desc">{flag.description}</div>
          <div class="evidence">{ev_rows}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Next: Explainability", type="primary", use_container_width=True):
    st.switch_page("pages/04_Explainability.py")
