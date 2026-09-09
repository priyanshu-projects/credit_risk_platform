"""
01_Document_Extraction.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.document_ai.loan_extractor import LoanExtractor

st.set_page_config(page_title="Document Extraction", page_icon="📄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.page-header {
    background: linear-gradient(135deg, #0f2744, #1a4a7a);
    border-radius: 12px; padding: 28px 32px; margin-bottom: 28px; color: white;
}
.page-header h2 { margin: 0; font-size: 1.8rem; font-weight: 700; }
.page-header p  { margin: 6px 0 0; color: #a8c8f0; font-size: 0.9rem; }

.field-card {
    background: white; border-radius: 10px; padding: 12px 16px;
    margin-bottom: 6px; border: 1px solid #e8edf4;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    display: flex; justify-content: space-between; align-items: center;
}
.field-name  { color: #4a5a6e; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; }
.field-value { color: #1a2744; font-size: 0.9rem; font-weight: 500; text-align: right; }
.field-null  { color: #c0c8d4; font-size: 0.82rem; font-style: italic; }

.method-badge {
    display: inline-block; border-radius: 20px; padding: 3px 12px;
    font-size: 0.76rem; font-weight: 600; letter-spacing: 0.4px; margin-bottom: 12px;
}
.badge-gemini { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
.badge-regex  { background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }

.stat-box {
    background: #f0f6ff; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #c5d9f0; text-align: center;
}
.stat-num { font-size: 1.8rem; font-weight: 700; color: #1a4a7a; }
.stat-lbl { color: #6b7c93; font-size: 0.8rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
  <h2>Document Extraction</h2>
  <p>Upload a loan application PDF to extract all structured fields.</p>
</div>""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

use_gemini = st.checkbox("Use Gemini for extraction", value=True,
                         help="Uncheck to use regex fallback instead")

if uploaded:
    file_key = f"uploaded_{uploaded.name}_{uploaded.size}"
    is_new = st.session_state.get("last_uploaded_key") != file_key

    if is_new or "extraction_result" not in st.session_state:
        for key in ["risk_result", "features_df", "prob_default", "risk_tier", "fraud_report", "shap_factors", "llm_report", "rule_decision"]:
            st.session_state.pop(key, None)

        tmp_path = Path("artifacts/tmp_upload.pdf")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.read())

        with st.spinner("Reading document..."):
            try:
                extractor = LoanExtractor(use_gemini=use_gemini)
                result = extractor.extract_from_pdf(str(tmp_path))
                st.session_state["extraction_result"] = result
                st.session_state["extracted_fields"]  = result.get("fields", {})
                st.session_state["last_uploaded_key"] = file_key
                st.switch_page("pages/01_Document_Extraction.py")
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                st.stop()

    result = st.session_state["extraction_result"]
    st.success("Done.")

    method = result.get("extraction_method", "unknown")
    badge_cls = "badge-gemini" if "gemini" in method else "badge-regex"
    badge_txt = "Gemini" if "gemini" in method else "Regex"
    st.markdown(f'<span class="method-badge {badge_cls}">{badge_txt}</span>', unsafe_allow_html=True)

    fields = result.get("fields", {})
    total  = len(fields)
    filled = sum(1 for v in fields.values() if v is not None and v != "")
    missing_list = result.get("missing_fields", [])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{total}</div><div class="stat-lbl">Total Fields</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{filled}</div><div class="stat-lbl">Extracted</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{len(missing_list)}</div><div class="stat-lbl">Missing</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    field_items = list(fields.items())
    half = (len(field_items) + 1) // 2

    for col, chunk in [(col_a, field_items[:half]), (col_b, field_items[half:])]:
        with col:
            for name, value in chunk:
                if value is not None and value != "":
                    display = f"{value:,.2f}" if isinstance(value, float) else str(value)
                    st.markdown(f"""
                    <div class="field-card">
                      <span class="field-name">{name.replace('_', ' ')}</span>
                      <span class="field-value">{display}</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="field-card">
                      <span class="field-name">{name.replace('_', ' ')}</span>
                      <span class="field-null">not found</span>
                    </div>""", unsafe_allow_html=True)

    with st.expander("Raw JSON"):
        st.json(result)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Next: Risk Assessment", type="primary", use_container_width=True):
        st.switch_page("pages/02_Risk_Assessment.py")

else:
    st.markdown("Upload a loan application PDF above to get started.")

    sample_dir = Path("sample_documents/loan_forms")
    if sample_dir.exists():
        samples = list(sample_dir.glob("*.pdf"))
        if samples:
            st.markdown("**Sample files available:**")
            for s in samples:
                st.markdown(f"- `{s.name}`")
