"""
03_Fraud_Analysis.py
Run the Fraud Engine and display all flags raised on this application.
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
.page-header p  { margin: 6px 0 0; color: #c8a8f0; font-size: 0.95rem; }

.level-clear  { background: linear-gradient(135deg, #27ae60, #2ecc71); }
.level-low    { background: linear-gradient(135deg, #3498db, #5dade2); }
.level-medium { background: linear-gradient(135deg, #f39c12, #f1c40f); }
.level-high   { background: linear-gradient(135deg, #c0392b, #e74c3c); }

.risk-level-box {
    border-radius: 16px; padding: 36px 24px; text-align: center; color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2); margin-bottom: 20px;
}
.level-name { font-size: 3rem; font-weight: 800; letter-spacing: -1px; }
.level-sub  { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }

.flag-card {
    border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;
    border-left: 5px solid; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.flag-high   { background: #fdecea; border-color: #c0392b; }
.flag-medium { background: #fff8e1; border-color: #f39c12; }
.flag-low    { background: #e8f4fd; border-color: #3498db; }

.flag-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.flag-name   { font-weight: 700; color: #1a2744; font-size: 0.95rem; font-family: monospace; }
.severity-tag {
    border-radius: 12px; padding: 3px 12px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;
}
.tag-high   { background: #c0392b; color: white; }
.tag-medium { background: #f39c12; color: white; }
.tag-low    { background: #3498db; color: white; }

.flag-desc  { color: #4a5a6e; font-size: 0.88rem; line-height: 1.5; margin-bottom: 8px; }
.evidence   { background: rgba(255,255,255,0.6); border-radius: 6px; padding: 8px 12px; }
.ev-row     { display: flex; justify-content: space-between; font-size: 0.82rem; }
.ev-key     { color: #6b7c93; font-family: monospace; }
.ev-val     { color: #1a2744; font-weight: 600; }

.count-pill {
    display: inline-block; border-radius: 20px; padding: 6px 18px; margin: 4px;
    font-size: 0.9rem; font-weight: 700; text-align: center;
}
.pill-high   { background: #fdecea; color: #c0392b; border: 2px solid #c0392b; }
.pill-medium { background: #fff8e1; color: #e67e22; border: 2px solid #e67e22; }
.pill-low    { background: #e8f4fd; color: #2980b9; border: 2px solid #2980b9; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <h2>🚨 Fraud Analysis</h2>
  <p>10 deterministic rules check for income inflation, behavioural anomalies, and data inconsistencies.</p>
</div>""", unsafe_allow_html=True)

# ── Require extraction ────────────────────────────────────────────────────────
if "extraction_result" not in st.session_state:
    st.warning("⚠️ No document extracted yet. Please go to **Document Extraction** first.")
    st.stop()

extraction_result = st.session_state["extraction_result"]
fields            = extraction_result.get("fields", {})

# ── Run fraud engine ──────────────────────────────────────────────────────────
if "fraud_report" not in st.session_state:
    with st.spinner("🔍 Running fraud checks..."):
        engine = FraudEngine()
        report = engine.evaluate(extraction_result)
        st.session_state["fraud_report"] = report
        st.switch_page("pages/03_Fraud_Analysis.py")
else:
    report = st.session_state["fraud_report"]

level     = report.fraud_risk_level.lower()
flags     = report.flags
counts    = report.flag_counts

# ── Level display ─────────────────────────────────────────────────────────────
col_level, col_counts = st.columns([1, 1.6])

with col_level:
    emoji = {"clear": "✅", "low": "🔵", "medium": "🟡", "high": "🔴"}.get(level, "⚠️")
    st.markdown(f"""
    <div class="risk-level-box level-{level}">
      <div class="level-sub">FRAUD RISK LEVEL</div>
      <div class="level-name">{emoji} {report.fraud_risk_level}</div>
      <div class="level-sub">{len(flags)} flag(s) raised</div>
    </div>""", unsafe_allow_html=True)

    # Guardrail
    st.info("🛡️ Fraud signals are **advisory only**. No loan decisions are made here.")

with col_counts:
    st.subheader("📊 Flags by Severity")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="count-pill pill-high">🔴 HIGH<br>{counts["high"]}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="count-pill pill-medium">🟡 MEDIUM<br>{counts["medium"]}</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="count-pill pill-low">🔵 LOW<br>{counts["low"]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Checks Run")

    # check_name from FraudFlag maps to these display names
    check_name_map = {
        "dti_income_mismatch":                "DTI income mismatch",
        "high_installment_to_income":         "Installment-to-income ratio",
        "implausible_income_too_low":         "Implausible income (too low)",
        "implausible_income_too_high":        "Implausible income (too high)",
        "high_util_high_income_inconsistency":"Utilization vs income",
        "round_number_income":                "Round number income",
        "round_number_loan_amount":           "Round number loan amount",
        "fico_dti_inconsistency":             "FICO vs DTI inconsistency",
        "new_credit_surge":                   "New credit surge",
        "delinquency_despite_high_fico":      "Delinquency despite high FICO",
        "public_record_high_fico_mismatch":   "Public record vs FICO",
        "revolving_balance_over_limit":       "Revolving balance over limit",
        "excessive_loan_to_income_leverage":  "Excessive loan-to-income leverage",
        "synthetic_identity_thin_file":       "Synthetic identity thin-file",
        "ocr_confidence_fallback":            "OCR confidence fallback",
        "high_missing_fields_anomaly":         "High count of missing fields",
    }
    triggered_names = {f.check_name for f in flags}
    for check_name, display_name in check_name_map.items():
        if check_name in triggered_names:
            icon = "🔴"
            style = "color:#c0392b; font-weight:600;"
        else:
            icon = "🟢"
            style = "color:#4a5a6e;"
        st.markdown(f'<span style="{style}">{icon} {display_name}</span>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Individual flags ──────────────────────────────────────────────────────────
if not flags:
    st.success("✅ No fraud flags raised — clean application profile.")
else:
    st.subheader(f"🚩 {len(flags)} Flag(s) Raised")

    # Sort: high first
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
            <span class="flag-name">{flag.check_name}</span>
            <span class="severity-tag tag-{sev}">{flag.severity.upper()}</span>
          </div>
          <div class="flag-desc">{flag.description}</div>
          <div class="evidence">{ev_rows}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔍 Proceed to Explainability (SHAP)", type="primary", use_container_width=True):
    st.switch_page("pages/04_Explainability.py")
