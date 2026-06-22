# AI-Assisted Credit Risk Platform - Architecture and Workflows

## Objective
Build an industry-grade AI-Assisted Credit Risk Platform that demonstrates Document AI, Credit Risk Modeling, Fraud Detection, Rule-based Decisioning, Explainable AI, and LLM-generated Underwriter Reports.

## Core Philosophy
**Never allow the LLM to approve or reject loans.** The LLM only summarizes findings and generates underwriting reports.

**Correct Flow:**
Loan Documents -> OCR / Document Processing -> Document AI -> Structured Financial Data -> Feature Engineering -> ML Risk Model -> Fraud Detection -> Rule Engine -> Explainability -> LLM Underwriter Report -> Human Decision

## Tech Stack
- **Language:** Python 3.12 (Conda env: `loan_ai`)
- **IDE:** VS Code
- **Version Control:** Git, GitHub
- **Frontend:** Streamlit
- **Risk Models:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Document Processing:** PyMuPDF, EasyOCR
- **LLMs:** Gemini Flash (Extraction, classification), GPT-4.1 mini (Reports)
- **Reports:** ReportLab

## Fixed Development Order
STRICTLY FOLLOW THIS ORDER:
1. LendingClub Risk Model
2. SHAP Explainability
3. Document Processing
4. Document AI
5. Feature Engineering
6. Fraud Engine
7. Rule Engine
8. LLM Underwriter Report
9. Streamlit Dashboard

## Current Development Phase
**PHASE 1:** LendingClub Risk Model

## Dataset & State
- **Dataset:** `accepted_2007_to_2018Q4.csv` (Location: `data/lendingclub/raw/`)
- **Working dataframe:** `df_clean` (Rows: 391,164, Columns: 151)
- **Target variable:** `target` (Fully Paid -> 0, Charged Off -> 1)
- **Class Distribution:** Fully Paid: 79.85%, Charged Off: 20.15%
- **Excluded loan statuses:** Current, Late (31-120 days), Late (16-30 days), In Grace Period, Default (outcomes unknown/rare).

