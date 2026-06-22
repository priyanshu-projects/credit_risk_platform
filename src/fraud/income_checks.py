"""
income_checks.py
----------------
Deterministic, rule-based checks that detect potential income fraud or
income inflation on a loan application.

These checks use only data available at application time (extracted by
LoanExtractor). They never approve or reject loans — they produce
structured flag objects for the FraudEngine to aggregate.

Each check function receives a dict of extracted fields and returns a
FraudFlag (or None if the check passes cleanly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# FraudFlag dataclass
# ---------------------------------------------------------------------------

@dataclass
class FraudFlag:
    """
    Represents a single fraud or anomaly signal.

    Attributes
    ----------
    check_name  : Short identifier for the check that raised this flag.
    severity    : 'low' | 'medium' | 'high'
    description : Human-readable explanation of why this was flagged.
    evidence    : Key-value pairs with the actual values that triggered the flag.
    """
    check_name: str
    severity: str          # 'low' | 'medium' | 'high'
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe(fields: Dict[str, Any], key: str) -> Optional[float]:
    """Return fields[key] as float, or None if missing/non-numeric."""
    val = fields.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Individual income checks
# ---------------------------------------------------------------------------

def check_dti_income_mismatch(fields: Dict[str, Any]) -> Optional[FraudFlag]:
    """
    Flag if the stated DTI is implausibly low given loan amount and income.

    A very low DTI (<= 1%) on a non-trivial loan is a classic inflation signal:
    the applicant may have overstated income to make the ratio look healthy.
    """
    dti = _safe(fields, "dti")
    loan_amount = _safe(fields, "loan_amount")
    annual_income = _safe(fields, "annual_income")

    if dti is None or loan_amount is None or annual_income is None:
        return None

    # Only flag when the loan is non-trivial (>= $5,000) and income is stated
    if loan_amount >= 5_000 and annual_income > 0 and dti <= 1.0:
        return FraudFlag(
            check_name="dti_income_mismatch",
            severity="high",
            description=(
                f"DTI of {dti:.2f}% is suspiciously low for a loan of "
                f"${loan_amount:,.0f} with stated annual income of "
                f"${annual_income:,.0f}. Possible income inflation."
            ),
            evidence={
                "dti": dti,
                "loan_amount": loan_amount,
                "annual_income": annual_income,
            },
        )
    return None


def check_income_vs_installment(fields: Dict[str, Any]) -> Optional[FraudFlag]:
    """
    Flag if monthly installment exceeds 50% of monthly income.

    Installment-to-income > 0.5 is a strong affordability signal.
    When the stated income is very high but DTI is also very high,
    it suggests a strained applicant who may have inflated income.
    """
    installment = _safe(fields, "installment")
    annual_income = _safe(fields, "annual_income")

    if installment is None or annual_income is None or annual_income <= 0:
        return None

    monthly_income = annual_income / 12
    ratio = installment / monthly_income

    if ratio > 0.50:
        severity = "high" if ratio > 0.75 else "medium"
        return FraudFlag(
            check_name="high_installment_to_income",
            severity=severity,
            description=(
                f"Monthly installment of ${installment:,.0f} is "
                f"{ratio * 100:.1f}% of stated monthly income "
                f"(${monthly_income:,.0f}). High affordability risk."
            ),
            evidence={
                "installment": installment,
                "monthly_income": round(monthly_income, 2),
                "installment_to_income_ratio": round(ratio, 4),
            },
        )
    return None


def check_implausible_income(fields: Dict[str, Any]) -> Optional[FraudFlag]:
    """
    Flag incomes that are statistically extreme.

    - Below $6,000 / year (~$500/month) is below US federal minimum wage
      for full-time work — may indicate fabricated or incomplete data.
    - Above $1,000,000 / year on a LendingClub-style personal loan is
      unusual and warrants scrutiny.
    """
    annual_income = _safe(fields, "annual_income")
    if annual_income is None:
        return None

    if annual_income < 6_000:
        return FraudFlag(
            check_name="implausible_income_too_low",
            severity="medium",
            description=(
                f"Stated annual income of ${annual_income:,.0f} is below "
                "the US federal minimum wage threshold for full-time work. "
                "Data may be incomplete or fabricated."
            ),
            evidence={"annual_income": annual_income},
        )

    if annual_income > 1_000_000:
        return FraudFlag(
            check_name="implausible_income_too_high",
            severity="low",
            description=(
                f"Stated annual income of ${annual_income:,.0f} is unusually "
                "high for a personal loan application. Verify income documentation."
            ),
            evidence={"annual_income": annual_income},
        )

    return None


def check_revolving_utilization_vs_income(
    fields: Dict[str, Any],
) -> Optional[FraudFlag]:
    """
    Flag if revolving utilization is very high while income is also high.

    High utilization (>= 90%) combined with high stated income (>= $80k)
    is inconsistent — a high earner should not typically be maxed out.
    Could indicate income inflation or hidden liabilities.
    """
    revol_util = _safe(fields, "revolving_utilization")
    annual_income = _safe(fields, "annual_income")

    if revol_util is None or annual_income is None:
        return None

    if revol_util >= 90.0 and annual_income >= 80_000:
        return FraudFlag(
            check_name="high_util_high_income_inconsistency",
            severity="medium",
            description=(
                f"Revolving utilization of {revol_util:.1f}% is very high "
                f"for an applicant with stated annual income of "
                f"${annual_income:,.0f}. Possible income inflation or "
                "hidden debt obligations."
            ),
            evidence={
                "revolving_utilization": revol_util,
                "annual_income": annual_income,
            },
        )
    return None


def check_revolving_balance_over_limit(fields: Dict[str, Any]) -> Optional[FraudFlag]:
    """
    Flag if revolving balance exceeds the total credit limit.
    This represents an over-limit credit utilization anomaly.
    """
    revol_bal = _safe(fields, "revolving_balance")
    total_limit = _safe(fields, "total_credit_limit") or _safe(fields, "total_rev_hi_lim")

    if revol_bal is not None and total_limit is not None and total_limit > 0:
        if revol_bal > total_limit:
            return FraudFlag(
                check_name="revolving_balance_over_limit",
                severity="high",
                description=(
                    f"Revolving balance of ${revol_bal:,.0f} exceeds the "
                    f"total credit limit of ${total_limit:,.0f}. "
                    "Over-limit credit utilization represents a critical credit risk anomaly."
                ),
                evidence={"revolving_balance": revol_bal, "total_credit_limit": total_limit},
            )
    return None


def check_excessive_leverage(fields: Dict[str, Any]) -> Optional[FraudFlag]:
    """
    Flag if the requested loan amount exceeds stated annual income.
    Extremely high leverage indicates severe default and possible profile fraud.
    """
    loan_amount = _safe(fields, "loan_amount")
    annual_income = _safe(fields, "annual_income")

    if loan_amount is not None and annual_income is not None and annual_income > 0:
        if loan_amount > annual_income:
            return FraudFlag(
                check_name="excessive_loan_to_income_leverage",
                severity="high",
                description=(
                    f"Requested loan amount of ${loan_amount:,.0f} exceeds the "
                    f"stated annual income of ${annual_income:,.0f}. "
                    "Extreme leverage indicates severe default and capacity risk."
                ),
                evidence={"loan_amount": loan_amount, "annual_income": annual_income},
            )
    return None


# ---------------------------------------------------------------------------
# Aggregated runner
# ---------------------------------------------------------------------------

def run_income_checks(fields: Dict[str, Any]) -> List[FraudFlag]:
    """
    Run all income-related fraud checks against extracted loan fields.

    Parameters
    ----------
    fields : dict
        The 'fields' dict from a LoanExtractor result.

    Returns
    -------
    list[FraudFlag]
        All flags raised (may be empty if no anomalies detected).
    """
    checks = [
        check_dti_income_mismatch,
        check_income_vs_installment,
        check_implausible_income,
        check_revolving_utilization_vs_income,
        check_revolving_balance_over_limit,
        check_excessive_leverage,
    ]

    flags: List[FraudFlag] = []
    for check_fn in checks:
        result = check_fn(fields)
        if result is not None:
            flags.append(result)

    return flags
