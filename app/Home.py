"""
Home.py — Credit Risk Platform Landing Page
"""

import streamlit as st

st.set_page_config(
    page_title="Credit Risk Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

def show_home():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0f2744 0%, #1a4a7a 50%, #0d3060 100%);
        border-radius: 16px;
        padding: 48px 40px;
        margin-bottom: 32px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0 0 10px 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #a8c8f0;
        font-size: 1.05rem;
        margin: 0;
        line-height: 1.7;
        max-width: 640px;
    }
    .pipeline-step {
        background: #f8fafd;
        border-left: 4px solid #1a4a7a;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .pipeline-step:hover {
        background: #e8f1fa;
        border-left-color: #2e7bcf;
        transform: translateX(3px);
    }
    .pipeline-step .step-title {
        color: #1a2744;
        font-weight: 600;
        font-size: 0.97rem;
        margin: 0 0 2px 0;
    }
    .pipeline-step .step-desc {
        color: #5a6a7e;
        font-size: 0.83rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border: 1px solid #e8edf4;
        text-align: center;
        transition: box-shadow 0.2s;
    }
    .metric-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.12); }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #1a4a7a;
        line-height: 1;
    }
    .metric-label {
        color: #6b7c93;
        font-size: 0.82rem;
        margin-top: 6px;
        font-weight: 500;
    }
    .nav-hint {
        background: linear-gradient(90deg, #e8f1fa, #f0f6ff);
        border-radius: 10px;
        padding: 14px 18px;
        color: #1a4a7a;
        font-size: 0.9rem;
        margin-top: 20px;
        border: 1px solid #c5d9f0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <h1>Credit Risk Platform</h1>
      <p>
        An end-to-end loan underwriting pipeline — from raw PDF documents to a
        structured underwriter report — combining machine learning, fraud detection,
        and explainable AI.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Platform Metrics ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-val">0.742</div>
          <div class="metric-label">Model ROC-AUC</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-val">391K</div>
          <div class="metric-label">Training Loans</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-val">123</div>
          <div class="metric-label">Model Features</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-val">13</div>
          <div class="metric-label">Fraud Checks</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline overview + Navigation ───────────────────────────────────────────
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.subheader("How it works")

        steps = [
            ("Document Processing",    "PyMuPDF + EasyOCR — reads native PDFs or scanned fallback"),
            ("Field Extraction",       "Gemini Flash pulls 35+ structured fields from the document text"),
            ("Feature Engineering",    "Maps extracted fields to the 123 features the model expects"),
            ("Risk Scoring",           "XGBoost predicts the probability of default"),
            ("Fraud Detection",        "13 rule-based checks flag income or behaviour anomalies"),
            ("Policy Decision",        "Credit policy engine routes: Decline / Refer / Review / Approve"),
            ("Explainability",         "SHAP shows which factors drove the score up or down"),
            ("Underwriter Report",     "GPT-4.1 mini writes the final narrative for the underwriter"),
        ]

        for title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
              <div class="step-title">{title}</div>
              <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.subheader("Pages")
        st.markdown("""
        | Page | What you can do |
        |---|---|
        | 📄 Document Extraction | Upload a loan PDF and extract fields |
        | 📊 Risk Assessment | Run the model and see the risk score |
        | 🚨 Fraud Analysis | View any fraud flags raised |
        | 🔍 Explainability | SHAP waterfall — why this score? |
        | 📝 Report Generation | Generate the full underwriter report |
        """)

        st.markdown("""
        <div class="nav-hint">
          Start by uploading a loan application PDF on the <strong>Document Extraction</strong> page.
          The platform runs each step in sequence automatically.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Tech Stack")

        tech = {
            "Risk Model":     "XGBoost",
            "Explainability": "SHAP TreeExplainer",
            "Document AI":    "Gemini Flash",
            "Report Writer":  "GPT-4.1 mini / Gemini Flash",
            "OCR":            "EasyOCR + PyMuPDF",
            "Frontend":       "Streamlit",
            "PDF Export":     "ReportLab",
        }
        for k, v in tech.items():
            st.markdown(f"**{k}:** {v}")


# ── Navigation ─────────────────────────────────────────────────────────────────
home_page    = st.Page(show_home, title="Home", icon="🏠", default=True)
doc_page     = st.Page("pages/01_Document_Extraction.py", title="Document Extraction", icon="📄")
risk_page    = st.Page("pages/02_Risk_Assessment.py",     title="Risk Assessment",     icon="📊")
fraud_page   = st.Page("pages/03_Fraud_Analysis.py",      title="Fraud Analysis",      icon="🚨")
explain_page = st.Page("pages/04_Explainability.py",      title="Explainability",      icon="🔍")
report_page  = st.Page("pages/05_Report_Generation.py",   title="Report Generation",   icon="📝")

try:
    active_pages = [home_page, doc_page]

    if "extraction_result" in st.session_state:
        active_pages.append(risk_page)
    if "features_df" in st.session_state:
        active_pages.append(fraud_page)
    if "risk_result" in st.session_state:
        active_pages.append(explain_page)
    if "shap_factors" in st.session_state:
        active_pages.append(report_page)

    pg = st.navigation(active_pages)
    pg.run()
except AttributeError:
    show_home()
