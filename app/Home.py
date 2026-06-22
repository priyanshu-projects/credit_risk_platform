"""
Home.py — Credit Risk Platform Landing Page
"""

import streamlit as st

st.set_page_config(
    page_title="AI Credit Risk Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Wrapped Landing Page ──────────────────────────────────────────────────────
def show_home():
    # ── Custom CSS ────────────────────────────────────────────────────────────────
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
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #a8c8f0;
        font-size: 1.1rem;
        margin: 0;
        line-height: 1.6;
    }
    .badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        color: #7ec8f0;
        border: 1px solid rgba(126,200,240,0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 4px 4px 12px 0;
        letter-spacing: 0.5px;
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
    .pipeline-step .step-num {
        color: #2e7bcf;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .pipeline-step .step-title {
        color: #1a2744;
        font-weight: 600;
        font-size: 1rem;
        margin: 2px 0;
    }
    .pipeline-step .step-desc {
        color: #5a6a7e;
        font-size: 0.85rem;
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
    .guardrail-box {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 24px;
    }
    .guardrail-box p {
        color: #7a5800;
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.6;
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
      <div>
        <span class="badge">🏦 PHASE 1 — COMPLETE</span>
        <span class="badge">🤖 GEMINI FLASH</span>
        <span class="badge">⚡ XGBOOST</span>
        <span class="badge">🔍 SHAP</span>
      </div>
      <h1>AI-Assisted Credit Risk Platform</h1>
      <p>
        Industry-grade loan underwriting pipeline combining Document AI,
        ML Risk Modelling, Fraud Detection, Rule-based Decisioning,
        Explainable AI, and LLM-generated Underwriter Reports.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Platform Metrics ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-val">0.742</div>
          <div class="metric-label">XGBoost Test ROC-AUC</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-val">391K</div>
          <div class="metric-label">Training Samples</div>
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
          <div class="metric-val">10</div>
          <div class="metric-label">Fraud Detection Rules</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline overview ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.subheader("📋 Processing Pipeline")

        steps = [
            ("01", "Document Processing",    "PyMuPDF + EasyOCR — native PDF or scanned fallback"),
            ("02", "Document AI Extraction", "Gemini Flash extracts 35+ structured loan fields"),
            ("03", "Feature Engineering",    "Maps extracted fields → 123 XGBoost features"),
            ("04", "ML Risk Model",          "Tuned XGBoost predicts probability of default"),
            ("05", "Fraud Detection",        "10 deterministic rules flag income / behaviour anomalies"),
            ("06", "Rule Engine",            "Credit policy: DECLINE / REFER / REVIEW / AUTO-APPROVE"),
            ("07", "SHAP Explainability",    "Top risk drivers and mitigators for each decision"),
            ("08", "LLM Underwriter Report", "GPT-4.1 mini / Gemini Flash writes the final narrative"),
        ]

        for num, title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
              <div class="step-num">Step {num}</div>
              <div class="step-title">{title}</div>
              <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.subheader("🗺️ Navigation")
        st.markdown("""
        Use the **sidebar** to navigate between pages:

        | Page | What you can do |
        |---|---|
        | 📄 Document Extraction | Upload a loan PDF and extract fields |
        | 📊 Risk Assessment | Run the ML model and see risk score |
        | 🚨 Fraud Analysis | View all fraud flags raised |
        | 🔍 Explainability | SHAP waterfall — why this score? |
        | 📝 Report Generation | Generate the full underwriter report |
        """)

        st.markdown("""
        <div class="nav-hint">
          💡 <strong>Quick Start:</strong> Upload a loan application PDF on the
          <em>Document Extraction</em> page — the platform will run the full
          pipeline automatically.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🛠️ Tech Stack")

        tech = {
            "Risk Model":    "XGBoost (GridSearchCV tuned)",
            "Explainability":"SHAP TreeExplainer",
            "Document AI":  "Gemini Flash (extraction)",
            "LLM Reports":  "GPT-4.1 mini / Gemini Flash",
            "OCR":          "EasyOCR + PyMuPDF",
            "Frontend":     "Streamlit",
            "PDF Output":   "ReportLab",
        }
        for k, v in tech.items():
            st.markdown(f"**{k}:** {v}")

    # ── Guardrail ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="guardrail-box">
      <p>
        ⚠️ <strong>Guardrail Notice:</strong>
        This platform is designed so that <strong>no LLM ever approves or rejects a loan.</strong>
        The ML model and rule engine produce risk signals and routing decisions.
        The LLM summarises findings only. All final credit decisions require a
        qualified human underwriter.
      </p>
    </div>""", unsafe_allow_html=True)


# ── Progressive Navigation Setup (Streamlit 1.35.0+) ──────────────────────────
home_page = st.Page(show_home, title="Home", icon="🏠", default=True)
doc_page = st.Page("pages/01_Document_Extraction.py", title="Document Extraction", icon="📄")
risk_page = st.Page("pages/02_Risk_Assessment.py", title="Risk Assessment", icon="📊")
fraud_page = st.Page("pages/03_Fraud_Analysis.py", title="Fraud Analysis", icon="🚨")
explain_page = st.Page("pages/04_Explainability.py", title="Explainability", icon="🔍")
report_page = st.Page("pages/05_Report_Generation.py", title="Report Generation", icon="📝")

try:
    # Build list of active pages based on state
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
    # Fallback for older Streamlit versions
    show_home()
