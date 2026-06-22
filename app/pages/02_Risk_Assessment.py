"""
02_Risk_Assessment.py
Run the ML risk model on extracted fields and display the probability of default.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.feature_engineering import FeatureEngineer
from src.risk_models.predict import RiskModelPredictor

st.set_page_config(page_title="Risk Assessment", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.page-header {
    background: linear-gradient(135deg, #1a2744, #2c4a7a);
    border-radius: 12px; padding: 28px 32px; margin-bottom: 28px; color: white;
}
.page-header h2 { margin: 0; font-size: 1.8rem; font-weight: 700; }
.page-header p  { margin: 6px 0 0; color: #a8c8f0; font-size: 0.95rem; }

.risk-gauge {
    border-radius: 16px; padding: 36px 24px; text-align: center;
    margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.risk-prob  { font-size: 4rem; font-weight: 800; letter-spacing: -2px; line-height: 1; }
.risk-label { font-size: 0.9rem; opacity: 0.8; margin-top: 4px; }
.risk-tier  { font-size: 1.4rem; font-weight: 700; margin-top: 12px; border-radius: 8px; display: inline-block; padding: 4px 20px; }

.tier-low       { background: #e8f5e9; color: #27ae60; }
.tier-medium    { background: #fff8e1; color: #f39c12; }
.tier-high      { background: #fff3e0; color: #e67e22; }
.tier-very-high { background: #fdecea; color: #c0392b; }

.gauge-low       { background: linear-gradient(135deg, #27ae60, #2ecc71); color: white; }
.gauge-medium    { background: linear-gradient(135deg, #f39c12, #f1c40f); color: white; }
.gauge-high      { background: linear-gradient(135deg, #e67e22, #f39c12); color: white; }
.gauge-very-high { background: linear-gradient(135deg, #c0392b, #e74c3c); color: white; }

.feature-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; border-bottom: 1px solid #f0f2f5;
}
.feature-row:last-child { border-bottom: none; }
.feature-row:hover { background: #f8fafd; }
.feat-name  { color: #4a5a6e; font-size: 0.85rem; }
.feat-val   { color: #1a2744; font-weight: 600; font-size: 0.9rem; }

.info-card {
    background: white; border-radius: 12px; padding: 20px 24px;
    border: 1px solid #e8edf4; box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    margin-bottom: 16px;
}
.info-card h4 { color: #1a2744; margin: 0 0 12px 0; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <h2>📊 Risk Assessment</h2>
  <p>XGBoost model predicts the probability of default using 123 engineered features.</p>
</div>""", unsafe_allow_html=True)

# ── Require extraction ────────────────────────────────────────────────────────
if "extraction_result" not in st.session_state:
    st.warning("⚠️ No document extracted yet. Please go to **Document Extraction** first.")
    st.stop()

extraction_result = st.session_state["extraction_result"]
fields            = extraction_result.get("fields", {})

# ── Run model ─────────────────────────────────────────────────────────────────
if "risk_result" not in st.session_state or "features_df" not in st.session_state:
    with st.spinner("⚙️ Running feature engineering and risk model..."):
        try:
            engineer  = FeatureEngineer()
            predictor = RiskModelPredictor(
                model_path="models/xgboost_baseline.joblib",
                metadata_path="data/processed/features/lendingclub_baseline_metadata.json",
            )
            # FeatureEngineer produces ~129 cols; model was trained on 123.
            # Align to exact model feature set so SHAP page gets correct input.
            raw_df = engineer.transform(extraction_result)

            if hasattr(predictor.model, "feature_names_in_"):
                model_cols = list(predictor.model.feature_names_in_)
            else:
                model_cols = predictor.expected_features

            for col in model_cols:
                if col not in raw_df.columns:
                    raw_df[col] = 0.0
            features_df = raw_df[model_cols].copy()

            result = predictor.predict_risk(features_df.iloc[0].to_dict())

            st.session_state["risk_result"]  = result
            st.session_state["features_df"] = features_df   # 123 cols, model-aligned
            st.session_state["prob_default"] = result["probability_of_default"]
            st.session_state["risk_tier"]    = result["risk_tier"]
            st.switch_page("pages/02_Risk_Assessment.py")

        except Exception as e:
            st.error(f"Model error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
else:
    result = st.session_state["risk_result"]
    features_df = st.session_state["features_df"]

prob    = result["probability_of_default"]
tier    = result["risk_tier"]
tier_key = tier.lower().replace(" ", "-")

# ── Gauge display ─────────────────────────────────────────────────────────────
col_gauge, col_detail = st.columns([1, 1.6])

with col_gauge:
    st.markdown(f"""
    <div class="risk-gauge gauge-{tier_key}">
      <div class="risk-label">PROBABILITY OF DEFAULT</div>
      <div class="risk-prob">{prob:.1%}</div>
      <div>
        <span class="risk-tier tier-{tier_key}">{tier} Risk</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Thresholds explainer
    st.markdown("""
    <div class="info-card">
      <h4>🎯 Risk Tier (P(Default) Bands)</h4>
      <p style="color:#6b7c93;font-size:0.8rem;margin:0 0 10px;">What probability of default range puts an applicant in each tier:</p>
    """, unsafe_allow_html=True)

    thresholds = [
        ("Low",       "< 10%",  "🟢"),
        ("Medium",    "10–25%", "🟡"),
        ("High",      "25–50%", "🟠"),
        ("Very High", "> 50%",  "🔴"),
    ]
    for t_name, t_range, icon in thresholds:
        active = "font-weight:700;" if t_name == tier else "color:#888;"
        st.markdown(f'<div class="feature-row"><span class="feat-name" style="{active}">{icon} {t_name}</span><span class="feat-val" style="{active}">{t_range}</span></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col_detail:
    st.subheader("📋 Key Application Metrics")

    display_fields = {
        "Applicant Name":        fields.get("applicant_name"),
        "Loan Amount":           f"${fields.get('loan_amount', 0):,.0f}" if fields.get("loan_amount") else None,
        "Annual Income":         f"${fields.get('annual_income', 0):,.0f}" if fields.get("annual_income") else None,
        "DTI":                   f"{fields.get('dti', 0):.1f}%" if fields.get("dti") else None,
        "FICO Score":            f"{fields.get('fico_avg', 0):.0f}" if fields.get("fico_avg") else None,
        "Interest Rate":         f"{fields.get('interest_rate', 0):.2f}%" if fields.get("interest_rate") else None,
        "Revolving Utilization": f"{fields.get('revolving_utilization', 0):.1f}%" if fields.get("revolving_utilization") else None,
        "Employment Length":     fields.get("employment_length"),
        "Home Ownership":        fields.get("home_ownership"),
        "Loan Purpose":          fields.get("purpose"),
        "Term":                  f"{fields.get('term_months', 0):.0f} months" if fields.get("term_months") else None,
        "Delinquencies (2yr)":   fields.get("delinquencies_2yrs"),
    }

    st.markdown('<div class="info-card"><h4>Application Fields</h4>', unsafe_allow_html=True)
    for label, val in display_fields.items():
        if val is not None:
            st.markdown(f'<div class="feature-row"><span class="feat-name">{label}</span><span class="feat-val">{val}</span></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="info-card"><h4>⚙️ Model Details</h4>', unsafe_allow_html=True)
    model_meta = [
        ("Model",         result.get("model_version", "tuned_xgboost_v1")),
        ("Features Used", f"{features_df.shape[1]}"),   # now 123 after alignment
        ("Training Rows", "391,164"),
        ("Test ROC-AUC",  "0.742"),
    ]
    for label, val in model_meta:
        st.markdown(f'<div class="feature-row"><span class="feat-name">{label}</span><span class="feat-val">{val}</span></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_nav1, col_nav2 = st.columns(2)
with col_nav1:
    if st.button("🚨 Proceed to Fraud Analysis", type="secondary", use_container_width=True):
        st.switch_page("pages/03_Fraud_Analysis.py")
with col_nav2:
    if st.button("🔍 Proceed to Explainability (SHAP)", type="primary", use_container_width=True):
        st.switch_page("pages/04_Explainability.py")
