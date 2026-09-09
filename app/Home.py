"""
Home.py
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
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0 0 10px 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #a8c8f0;
        font-size: 1rem;
        margin: 0;
        line-height: 1.7;
        max-width: 600px;
    }
    .pipeline-step {
        background: #f8fafd;
        border-left: 4px solid #1a4a7a;
        border-radius: 0 10px 10px 0;
        padding: 12px 18px;
        margin-bottom: 8px;
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
        font-size: 0.95rem;
        margin: 0 0 2px 0;
    }
    .pipeline-step .step-desc {
        color: #5a6a7e;
        font-size: 0.82rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border: 1px solid #e8edf4;
        text-align: center;
    }
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
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
      <h1>Credit Risk Platform</h1>
      <p>Upload a loan application and get a full risk assessment, fraud check, and underwriter report.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-val">0.742</div><div class="metric-label">Model AUC</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-val">391K</div><div class="metric-label">Training Loans</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-val">123</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-val">13</div><div class="metric-label">Fraud Checks</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.subheader("How it works")

        steps = [
            ("Document Processing",  "Reads the uploaded PDF, extracts raw text via OCR if needed"),
            ("Field Extraction",     "Pulls structured loan fields from the document"),
            ("Feature Engineering", "Prepares the 123 inputs the model needs"),
            ("Risk Scoring",        "Predicts probability of default"),
            ("Fraud Detection",     "Runs 13 checks for income and behaviour anomalies"),
            ("Policy Decision",     "Routes to: Decline / Refer / Manual Review / Approve"),
            ("Explainability",      "Shows which factors pushed the score up or down"),
            ("Underwriter Report",  "Writes a structured narrative for the underwriter"),
        ]

        for title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
              <div class="step-title">{title}</div>
              <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.subheader("Getting started")
        st.markdown("""
        Go to **Document Extraction** in the sidebar and upload a loan application PDF.
        Each subsequent page unlocks automatically as you move through the steps.
        """)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Pages")
        st.markdown("""
        - **Document Extraction** - upload and parse the PDF
        - **Risk Assessment** - view the default probability
        - **Fraud Analysis** - see any flags raised
        - **Explainability** - understand the score
        - **Report Generation** - generate and download the report
        """)


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
