"""
04_Explainability.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.risk_models.predict import RiskModelPredictor

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.page-header {
    background: linear-gradient(135deg, #1a3a1a, #2a5a2a);
    border-radius: 12px; padding: 28px 32px; margin-bottom: 28px; color: white;
}
.page-header h2 { margin: 0; font-size: 1.8rem; font-weight: 700; }
.page-header p  { margin: 6px 0 0; color: #a8f0b0; font-size: 0.9rem; }

.driver-row {
    display: flex; align-items: center; padding: 10px 14px;
    border-bottom: 1px solid #f0f2f5; gap: 12px;
}
.driver-row:hover { background: #f8fafd; }
.bar-fill { height: 10px; border-radius: 5px; min-width: 4px; }
.bar-pos { background: linear-gradient(90deg, #e74c3c, #c0392b); }
.bar-neg { background: linear-gradient(90deg, #27ae60, #2ecc71); }
.driver-label { color: #1a2744; font-size: 0.85rem; min-width: 200px; }
.driver-val   { color: #6b7c93; font-size: 0.8rem; font-family: monospace; min-width: 80px; }
.driver-shap  { font-weight: 700; font-size: 0.88rem; min-width: 70px; text-align: right; }
.shap-pos { color: #c0392b; }
.shap-neg { color: #27ae60; }

.info-card {
    background: white; border-radius: 12px; padding: 20px 24px;
    border: 1px solid #e8edf4; box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}
.info-card h4 { color: #1a2744; margin: 0 0 12px 0; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <h2>Explainability</h2>
  <p>Which factors pushed the risk score up or down for this applicant.</p>
</div>""", unsafe_allow_html=True)

if "extraction_result" not in st.session_state:
    st.warning("Please complete Document Extraction first.")
    st.stop()

if "features_df" not in st.session_state:
    st.warning("Please complete Risk Assessment first.")
    st.stop()

features_df = st.session_state["features_df"]
prob        = st.session_state.get("prob_default", 0.0)
tier        = st.session_state.get("risk_tier", "Unknown")

with st.spinner("Computing SHAP values..."):
    try:
        predictor = RiskModelPredictor(
            model_path="models/xgboost_baseline.joblib",
            metadata_path="data/processed/features/lendingclub_baseline_metadata.json",
        )
        model = predictor.model

        # Align features_df to exactly the columns the model was trained on
        if hasattr(model, "feature_names_in_"):
            model_cols = list(model.feature_names_in_)
        elif hasattr(model, "get_booster") and model.get_booster().feature_names:
            model_cols = model.get_booster().feature_names
        else:
            model_cols = predictor.expected_features

        for col in model_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0
        shap_input_df = features_df[model_cols].copy()

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer(shap_input_df)

        bv = shap_values.base_values
        base_val = float(bv[0, 1] if bv.ndim == 2 else bv[0])

        sv = shap_values.values
        shap_vec = sv[0, :, 1] if sv.ndim == 3 else sv[0]

        feat_names = shap_input_df.columns.tolist()
        feat_vals  = shap_input_df.iloc[0].tolist()

        pairs = sorted(
            zip(feat_names, feat_vals, shap_vec),
            key=lambda x: abs(x[2]),
            reverse=True,
        )
        top_n     = 15
        top_pairs = pairs[:top_n]

        drivers    = [(n, v, s) for n, v, s in top_pairs if s > 0]
        mitigators = [(n, v, s) for n, v, s in top_pairs if s <= 0]

        if "shap_factors" not in st.session_state:
            st.session_state["shap_factors"] = {
                "base_value":   base_val,
                "risk_drivers": [
                    {"feature": n, "value": float(v), "shap_contribution": float(s)}
                    for n, v, s in drivers[:5]
                ],
                "mitigators": [
                    {"feature": n, "value": float(v), "shap_contribution": float(s)}
                    for n, v, s in mitigators[:5]
                ],
            }
            st.switch_page("pages/04_Explainability.py")
    except Exception as e:
        st.error(f"SHAP error: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

st.subheader("Waterfall Chart")
st.caption("Red bars increase the risk score. Green bars reduce it.")

with st.spinner("Rendering..."):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#fafbfc")
        ax.set_facecolor("#fafbfc")
        exp_slice = shap_values[0, :, 1] if shap_values.values.ndim == 3 else shap_values[0]
        shap.plots.waterfall(exp_slice, max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Chart could not render: {e}")
        plt.close("all")

st.markdown("<br>", unsafe_allow_html=True)

col_d, col_m = st.columns(2)
MAX_BAR = max(abs(s) for _, _, s in top_pairs) if top_pairs else 1

def _bar(shap_val: float, max_val: float) -> str:
    width = max(4, int(160 * abs(shap_val) / max_val))
    cls   = "bar-pos" if shap_val > 0 else "bar-neg"
    return f'<div class="bar-fill {cls}" style="width:{width}px;"></div>'

with col_d:
    st.markdown('<div class="info-card"><h4>Risk Drivers</h4>', unsafe_allow_html=True)
    if drivers:
        for name, val, shap_val in sorted(drivers, key=lambda x: -x[2])[:8]:
            display_val = f"{val:.2f}" if isinstance(val, float) else str(val)
            st.markdown(f"""
            <div class="driver-row">
              <div class="driver-label" title="{name}">{name[:28]}</div>
              <div class="driver-val">{display_val}</div>
              {_bar(shap_val, MAX_BAR)}
              <div class="driver-shap shap-pos">+{shap_val:.4f}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#888;padding:12px;">None found.</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_m:
    st.markdown('<div class="info-card"><h4>Mitigating Factors</h4>', unsafe_allow_html=True)
    if mitigators:
        for name, val, shap_val in sorted(mitigators, key=lambda x: x[2])[:8]:
            display_val = f"{val:.2f}" if isinstance(val, float) else str(val)
            st.markdown(f"""
            <div class="driver-row">
              <div class="driver-label" title="{name}">{name[:28]}</div>
              <div class="driver-val">{display_val}</div>
              {_bar(shap_val, MAX_BAR)}
              <div class="driver-shap shap-neg">{shap_val:.4f}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#888;padding:12px;">None found.</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("How to read this"):
    st.markdown(f"""
    - Base value `{base_val:.4f}` is the model's average output across all training loans
    - Red bars show features that increased this applicant's default probability
    - Green bars show features that decreased it
    - Final prediction: **{prob:.1%}**
    """)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Next: Report Generation", type="primary", use_container_width=True):
    st.switch_page("pages/05_Report_Generation.py")
