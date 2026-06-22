"""
feature_engineering.py
----------------------
Bridges the output of LoanExtractor (Document AI) to the exact feature
vector expected by the trained XGBoost risk model.

Responsibilities:
  1. Map extracted field names -> LendingClub training column names
  2. Derive engineered features (ratios, FICO average, etc.)
  3. One-hot encode categorical fields using the exact dummy columns
     seen during training
  4. Impute missing values using training-time medians
  5. Drop leakage columns
  6. Return a clean 1-row DataFrame aligned to model feature order

This module never approves or rejects loans. It only prepares data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Columns the model was trained on (order matters for XGBoost)
# Source: data/processed/features/lendingclub_baseline_metadata.json
# ---------------------------------------------------------------------------
MODEL_FEATURE_COLUMNS: list[str] = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "policy_code", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths",
    "avg_cur_bal", "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths",
    "delinq_amnt", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
    "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
    "num_rev_accts", "num_rev_tl_bal_gt_0", "num_tl_120dpd_2m", "num_tl_30dpd",
    "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
    "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
    "issue_year", "issue_month", "credit_history_months", "term_months",
    "emp_length_years",
    # --- one-hot: grade ---
    "grade_B", "grade_C", "grade_D", "grade_E", "grade_F", "grade_G",
    # --- one-hot: sub_grade ---
    "sub_grade_A2", "sub_grade_A3", "sub_grade_A4", "sub_grade_A5",
    "sub_grade_B1", "sub_grade_B2", "sub_grade_B3", "sub_grade_B4", "sub_grade_B5",
    "sub_grade_C1", "sub_grade_C2", "sub_grade_C3", "sub_grade_C4", "sub_grade_C5",
    "sub_grade_D1", "sub_grade_D2", "sub_grade_D3", "sub_grade_D4", "sub_grade_D5",
    "sub_grade_E1", "sub_grade_E2", "sub_grade_E3", "sub_grade_E4", "sub_grade_E5",
    "sub_grade_F1", "sub_grade_F2", "sub_grade_F3", "sub_grade_F4", "sub_grade_F5",
    "sub_grade_G1", "sub_grade_G2", "sub_grade_G3", "sub_grade_G4", "sub_grade_G5",
    # --- one-hot: home_ownership ---
    "home_ownership_MORTGAGE", "home_ownership_OWN", "home_ownership_RENT",
    # --- one-hot: verification_status ---
    "verification_status_Source Verified", "verification_status_Verified",
    # --- one-hot: purpose ---
    "purpose_credit_card", "purpose_debt_consolidation", "purpose_educational",
    "purpose_home_improvement", "purpose_house", "purpose_major_purchase",
    "purpose_medical", "purpose_moving", "purpose_other",
    "purpose_renewable_energy", "purpose_small_business", "purpose_vacation",
    "purpose_wedding",
    # --- one-hot: misc ---
    "initial_list_status_w", "application_type_Joint App",
    "disbursement_method_DirectPay",
    # --- engineered features ---
    "fico_avg",
    "loan_to_income",
    "installment_to_income",
    "credit_utilization_ratio",
    "recent_credit_ratio",
]

# Training-time median imputation values
IMPUTE_MEDIANS: Dict[str, float] = {
    "loan_amnt": 12500.0, "int_rate": 12.29, "installment": 377.04,
    "annual_inc": 65000.0, "dti": 18.32, "delinq_2yrs": 0.0,
    "inq_last_6mths": 0.0, "mths_since_last_delinq": 31.0, "open_acc": 11.0,
    "pub_rec": 0.0, "revol_bal": 11407.0, "revol_util": 52.8, "total_acc": 24.0,
    "policy_code": 1.0, "acc_now_delinq": 0.0, "tot_coll_amt": 0.0,
    "tot_cur_bal": 75893.0, "total_rev_hi_lim": 24100.0,
    "acc_open_past_24mths": 4.0, "avg_cur_bal": 7010.0, "bc_open_to_buy": 4500.0,
    "bc_util": 64.1, "chargeoff_within_12_mths": 0.0, "delinq_amnt": 0.0,
    "mo_sin_old_il_acct": 129.0, "mo_sin_old_rev_tl_op": 168.0,
    "mo_sin_rcnt_rev_tl_op": 8.0, "mo_sin_rcnt_tl": 5.0, "mort_acc": 1.0,
    "mths_since_recent_bc": 13.0, "mths_since_recent_inq": 5.0,
    "num_accts_ever_120_pd": 0.0, "num_actv_bc_tl": 3.0, "num_actv_rev_tl": 5.0,
    "num_bc_sats": 4.0, "num_bc_tl": 7.0, "num_il_tl": 7.0, "num_op_rev_tl": 7.0,
    "num_rev_accts": 13.0, "num_rev_tl_bal_gt_0": 5.0, "num_sats": 11.0,
    "num_tl_120dpd_2m": 0.0, "num_tl_30dpd": 0.0, "num_tl_90g_dpd_24m": 0.0,
    "num_tl_op_past_12m": 2.0, "pct_tl_nvr_dlq": 97.4, "percent_bc_gt_75": 50.0,
    "pub_rec_bankruptcies": 0.0, "tax_liens": 0.0, "tot_hi_cred_lim": 107400.0,
    "total_bal_ex_mort": 38529.0, "total_bc_limit": 15000.0,
    "total_il_high_credit_limit": 32342.5, "issue_year": 2015.0,
    "issue_month": 7.0, "credit_history_months": 181.0, "term_months": 36.0,
    "emp_length_years": 6.0,
    "fico_avg": 687.0,
    "loan_to_income": 0.1923,
    "installment_to_income": 0.0058,
    "credit_utilization_ratio": 0.4733,
    "recent_credit_ratio": 0.0833,
}

# Leakage columns excluded from the model (post-origination outcomes)
LEAKAGE_COLS: set[str] = {
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "last_fico_range_high", "last_fico_range_low",
}

# ---------------------------------------------------------------------------
# Field name mapping: LoanExtractor keys -> LendingClub column names
# ---------------------------------------------------------------------------
EXTRACTOR_TO_LC_MAP: Dict[str, str] = {
    "loan_amount":              "loan_amnt",
    "interest_rate":            "int_rate",
    "annual_income":            "annual_inc",
    "dti":                      "dti",
    "open_accounts":            "open_acc",
    "total_accounts":           "total_acc",
    "revolving_balance":        "revol_bal",
    "revolving_utilization":    "revol_util",
    "mortgage_accounts":        "mort_acc",
    "total_current_balance":    "tot_cur_bal",
    "total_credit_limit":       "total_rev_hi_lim",
    "bc_utilization":           "bc_util",
    "acc_open_past_24m":        "acc_open_past_24mths",
    "months_since_recent_inquiry": "mths_since_recent_inq",
    "months_since_recent_bc":   "mths_since_recent_bc",
    "delinquencies_2yrs":       "delinq_2yrs",
    "public_records":           "pub_rec",
    "issue_year":               "issue_year",
    "issue_month":              "issue_month",
    "installment":              "installment",
    "term_months":              "term_months",
    "fico_avg":                 "fico_avg",           # model trained on avg directly
}

# Categorical field: one-hot dummy column prefix mapping
# key = extractor field name, value = LC dummy prefix
CATEGORICAL_MAP: Dict[str, str] = {
    "grade":               "grade",
    "sub_grade":           "sub_grade",
    "home_ownership":      "home_ownership",
    "verification_status": "verification_status",
    "purpose":             "purpose",
    "application_type":    "application_type",
    "disbursement_method": "disbursement_method",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_emp_length(value: Any) -> Optional[float]:
    """
    Convert employment_length strings to numeric years.
    Examples: '10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3
    """
    import re
    if value is None:
        return None
    text = str(value).strip().lower()
    if "10+" in text:
        return 10.0
    if "< 1" in text:
        return 0.0
    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Transforms a LoanExtractor result dict into a 1-row DataFrame
    aligned to the 129-column XGBoost model feature space.

    Usage:
        engineer = FeatureEngineer()
        features_df = engineer.transform(extractor_result)
    """

    def transform(self, extractor_result: Dict[str, Any]) -> pd.DataFrame:
        """
        Main entry point.

        Parameters
        ----------
        extractor_result : dict
            The dict returned by LoanExtractor.extract() or .extract_from_pdf().
            Expected to have a 'fields' key containing extracted field values.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with exactly the columns in MODEL_FEATURE_COLUMNS,
            ready to be passed to RiskModelPredictor or SHAPExplainer.
        """
        fields: Dict[str, Any] = extractor_result.get("fields", extractor_result)

        # Step 1: Build a flat working dict with LC column names
        row: Dict[str, Any] = {}
        self._map_numeric_fields(fields, row)
        self._derive_engineered_features(fields, row)
        self._encode_categoricals(fields, row)

        # Step 2: Build DataFrame with all model columns, fill missing with medians
        df = self._build_aligned_dataframe(row)

        return df

    # ------------------------------------------------------------------
    # Step 1a: Map numeric fields using EXTRACTOR_TO_LC_MAP
    # ------------------------------------------------------------------

    def _map_numeric_fields(
        self, fields: Dict[str, Any], row: Dict[str, Any]
    ) -> None:
        for extractor_key, lc_col in EXTRACTOR_TO_LC_MAP.items():
            raw_value = fields.get(extractor_key)
            numeric = _safe_float(raw_value)
            if numeric is not None:
                row[lc_col] = numeric

        # employment_length needs special parsing
        emp_years = _parse_emp_length(fields.get("employment_length"))
        if emp_years is not None:
            row["emp_length_years"] = emp_years

    # ------------------------------------------------------------------
    # Step 1b: Derive engineered / derived features
    # ------------------------------------------------------------------

    def _derive_engineered_features(
        self, fields: Dict[str, Any], row: Dict[str, Any]
    ) -> None:
        # fico_avg: model was trained on the average directly (notebook 03 dropped
        # fico_range_low / fico_range_high and created fico_avg instead).
        # FeatureEngineer maps extractor 'fico_avg' -> 'fico_avg' directly.
        # No split needed here.

        # funded_amnt and funded_amnt_inv default to loan_amnt when not extracted
        # NOTE: these were DROPPED in notebook 03 training, so the model's
        # feature_names_in_ won't include them — alignment step will discard them.
        loan_amnt = row.get("loan_amnt")
        if loan_amnt is not None:
            row.setdefault("funded_amnt", loan_amnt)
            row.setdefault("funded_amnt_inv", loan_amnt)

        # policy_code is always 1 for standard loans on LendingClub
        row.setdefault("policy_code", 1.0)

        # acc_now_delinq, delinq_amnt default to 0 if not present
        row.setdefault("acc_now_delinq", 0.0)
        row.setdefault("delinq_amnt", 0.0)

        # ── Ratio features added in notebook 03 ──────────────────────────────
        # loan_to_income
        annual_inc = row.get("annual_inc")
        if loan_amnt and annual_inc and annual_inc != 0:
            row["loan_to_income"] = loan_amnt / annual_inc

        # installment_to_income
        installment = row.get("installment")
        if installment and annual_inc and annual_inc != 0:
            row["installment_to_income"] = installment / annual_inc

        # credit_utilization_ratio  (revol_bal / total_rev_hi_lim)
        revol_bal        = row.get("revol_bal")
        total_rev_hi_lim = row.get("total_rev_hi_lim")
        if revol_bal is not None and total_rev_hi_lim and total_rev_hi_lim != 0:
            row["credit_utilization_ratio"] = revol_bal / total_rev_hi_lim

        # recent_credit_ratio  (num_tl_op_past_12m / total_acc)
        num_tl_op = row.get("num_tl_op_past_12m")
        total_acc  = row.get("total_acc")
        if num_tl_op is not None and total_acc and total_acc != 0:
            row["recent_credit_ratio"] = num_tl_op / total_acc

    # ------------------------------------------------------------------
    # Step 1c: One-hot encode categoricals
    # ------------------------------------------------------------------

    def _encode_categoricals(
        self, fields: Dict[str, Any], row: Dict[str, Any]
    ) -> None:
        for extractor_key, lc_prefix in CATEGORICAL_MAP.items():
            raw_value = fields.get(extractor_key)
            if raw_value is None:
                continue

            # Normalize: strip whitespace, preserve original casing for matching
            value_str = str(raw_value).strip()

            # Build the dummy column name as it appears in MODEL_FEATURE_COLUMNS
            dummy_col = f"{lc_prefix}_{value_str}"

            if dummy_col in MODEL_FEATURE_COLUMNS:
                row[dummy_col] = 1.0
            # If the value doesn't match any known dummy, all dummies stay 0
            # (i.e., it maps to the dropped reference category — correct behaviour)

        # Special: initial_list_status — 'w' was the non-reference category
        init_status = fields.get("initial_list_status")
        if str(init_status).strip().lower() == "w":
            row["initial_list_status_w"] = 1.0

    # ------------------------------------------------------------------
    # Step 2: Assemble aligned DataFrame
    # ------------------------------------------------------------------

    def _build_aligned_dataframe(self, row: Dict[str, Any]) -> pd.DataFrame:
        """
        Build a 1-row DataFrame with MODEL_FEATURE_COLUMNS in the correct order.
        Missing numeric columns are filled with training-time medians.
        Missing dummy columns default to 0.
        Leakage columns are set to their median (they exist in feature list
        but will be handled if the model was saved without them — predict.py
        already drops them via feature_names_in_).
        """
        record: Dict[str, Any] = {}
        for col in MODEL_FEATURE_COLUMNS:
            if col in row:
                record[col] = row[col]
            elif col in IMPUTE_MEDIANS:
                record[col] = IMPUTE_MEDIANS[col]
            else:
                # One-hot dummy column not present -> 0 (reference category)
                record[col] = 0.0

        df = pd.DataFrame([record])

        # Coerce all columns to numeric; any stray non-numeric becomes NaN -> 0
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        return df


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def engineer_features(extractor_result: Dict[str, Any]) -> pd.DataFrame:
    """
    Module-level shortcut for FeatureEngineer().transform(extractor_result).

    Parameters
    ----------
    extractor_result : dict
        Output of LoanExtractor.extract() or .extract_from_pdf().

    Returns
    -------
    pd.DataFrame
        Model-ready single-row feature DataFrame.
    """
    return FeatureEngineer().transform(extractor_result)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Simulate a typical LoanExtractor output
    mock_extractor_result = {
        "document_type": "loan_application",
        "extraction_method": "regex_fallback",
        "guardrail": "Extraction only. No loan approval, rejection, or recommendation is produced.",
        "fields": {
            "loan_amount": 15000,
            "interest_rate": 13.5,
            "installment": 508.0,
            "annual_income": 72000,
            "dti": 19.4,
            "term_months": 36,
            "grade": "C",
            "sub_grade": "C3",
            "home_ownership": "RENT",
            "verification_status": "Verified",
            "purpose": "debt_consolidation",
            "employment_length": "5 years",
            "fico_avg": 695.0,
            "open_accounts": 9,
            "total_accounts": 22,
            "revolving_balance": 8500,
            "revolving_utilization": 58.0,
            "delinquencies_2yrs": 0,
            "public_records": 0,
            "issue_year": 2024,
            "issue_month": 6,
            "application_type": "Individual",
            "disbursement_method": "Cash",
        },
        "missing_fields": [],
        "raw_text_char_count": 3200,
    }

    engineer = FeatureEngineer()
    features_df = engineer.transform(mock_extractor_result)

    print(f"Output shape : {features_df.shape}")
    print(f"Columns      : {len(features_df.columns)}")
    print(f"\nSample values (non-zero):")
    non_zero = features_df.loc[:, (features_df != 0).any()].T
    print(non_zero.to_string())
