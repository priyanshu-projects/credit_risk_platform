"""
behavior_checks.py
------------------
Deterministic, rule-based checks that detect suspicious application
behaviour patterns — things that are statistically unlikely to occur
in honest applications.

These checks use only data available at application time (extracted by
LoanExtractor). They never approve or reject loans — they produce
structured FraudFlag objects for the FraudEngine to aggregate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.fraud.income_checks import FraudFlag, _safe


# ---------------------------------------------------------------------------
# Individual behaviour checks
# ---------------------------------------------------------------------------

def check_round_number_income(fields: Dict[str, Any]) -> Optional[FraudFlag]:
    """
    Flag suspiciously round income figures.

    Fabricated income is often entered as a perfectly round number
    (e.g., $50,000 exactly). We flag incomes that are exact multiples
    of $10,000 above $30,000, which is a well-known fraud signal.
    """
    annual_income = _safe(fields, "annual_income")
    if annual_income is None or annual_income < 30_000:
        return None

    if annual_income % 10_000 == 0:
        return FraudFlag(
            check_name="round_number_income",
            severity="low",
            description=(
                f"Stated annual income of ${annual_income:,.0f} is a "
                "perfectly round number. This is a weak but known fraud "
                "signal — verify with income documentation."
            ),
            evidence={"annual_income": annual_income},
        )
    return None


def check_round_number_loan(fields: Dict[str, Any]) -> Optional[FraudFlag]:
    """
    Flag very round loan amounts (multiples of $5,000 above $10,000).

    While round loan amounts are common, extremely round requests
    (e.g., $40,000 exactly) combined with other flags can be a signal.
    This is a weak signal only — low severity.
    """
    loan_amount = _safe(fields, "loan_amount")
    if loan_amount is None or loan_amount < 10_000:
        return None

    if loan_amount % 5_000 == 0:
        return FraudFlag(
            check_name="round_number_loan_amount",
            severity="low",
            description=(
                f"Loan amount of ${loan_amount:,.0f} is a perfectly round "
                "figure. Weak signal — note in conjunction with other flags."
            ),
            evidence={"loan_amount": loan_amount},
        )
    return None


def check_fico_dti_inconsistency(fields: Dict[str, Any]) -> Optional[FraudFlag]:
    """
    Flag applicants with excellent FICO but very high DTI.

    A FICO >= 750 combined with DTI >= 40% is inconsistent:
    bureaus penalise high debt loads, so a very clean score with heavy
    debt suggests the stated income (used to compute DTI) may be inflated.
    """
    fico_avg = _safe(fields, "fico_avg")
    dti = _safe(fields, "dti")

    if fico_avg is None or dti is None:
        return None

    if fico_avg >= 750 and dti >= 40.0:
        return FraudFlag(
            check_name="fico_dti_inconsistency",
            severity="medium",
            description=(
                f"FICO score of {fico_avg:.0f} is excellent, but DTI of "
                f"{dti:.1f}% is very high. This combination is statistically "
                "unusual and may indicate income inflation."
            ),
            evidence={"fico_avg": fico_avg, "dti": dti},
        )
    return None


def check_new_credit_surge(fields: Dict[str, Any]) -> Optional[FraudFlag]:
    """
    Flag a surge in newly opened accounts.

    Opening many accounts in the past 24 months (>= 8) relative to
    total account history (total_accounts) can indicate an applicant
    rapidly loading up on credit before a potential default.
    """
    acc_open_past_24m = _safe(fields, "acc_open_past_24m")
    total_accounts = _safe(fields, "total_accounts")

    if acc_open_past_24m is None or total_accounts is None or total_accounts == 0:
        return None

    recent_ratio = acc_open_past_24m / total_accounts

    if acc_open_past_24m >= 8 and recent_ratio >= 0.40:
        return FraudFlag(
            check_name="new_credit_surge",
            severity="medium",
            description=(
                f"{int(acc_open_past_24m)} accounts opened in the last 24 months "
                f"({recent_ratio * 100:.1f}% of {int(total_accounts)} total accounts). "
                "Rapid credit accumulation may indicate pre-default loading."
            ),
            evidence={
                "acc_open_past_24m": acc_open_past_24m,
                "total_accounts": total_accounts,
                "recent_ratio": round(recent_ratio, 4),
            },
        )
    return None


def check_delinquency_despite_high_fico(
    fields: Dict[str, Any],
) -> Optional[FraudFlag]:
    """
    Flag recent delinquencies on a high FICO score.

    A FICO >= 720 paired with delinquencies in the last 2 years (>= 2)
    is inconsistent — delinquencies strongly drag FICO scores down.
    This suggests possible FICO score manipulation or data error.
    """
    fico_avg = _safe(fields, "fico_avg")
    delinquencies = _safe(fields, "delinquencies_2yrs")

    if fico_avg is None or delinquencies is None:
        return None

    if fico_avg >= 720 and delinquencies >= 2:
        return FraudFlag(
            check_name="delinquency_despite_high_fico",
            severity="high",
            description=(
                f"FICO score of {fico_avg:.0f} is high, but {int(delinquencies)} "
                "delinquencies were reported in the last 2 years. This is "
                "statistically inconsistent — verify bureau data."
            ),
            evidence={
                "fico_avg": fico_avg,
                "delinquencies_2yrs": delinquencies,
            },
        )
    return None


def check_public_record_mismatch(fields: Dict[str, Any]) -> Optional[FraudFlag]:
    """
    Flag public records (bankruptcies, judgements) on a high FICO score.

    A FICO >= 700 with >= 1 public record is highly unusual — public
    records are severely derogatory marks that typically drop scores by
    100+ points.
    """
    fico_avg = _safe(fields, "fico_avg")
    public_records = _safe(fields, "public_records")

    if fico_avg is None or public_records is None:
        return None

    if fico_avg >= 700 and public_records >= 1:
        return FraudFlag(
            check_name="public_record_high_fico_mismatch",
            severity="high",
            description=(
                f"FICO score of {fico_avg:.0f} is good, but {int(public_records)} "
                "public record(s) are listed. Public records typically cause "
                "large FICO drops — possible data inconsistency or fraud."
            ),
            evidence={
                "fico_avg": fico_avg,
                "public_records": public_records,
            },
        )
    return None



def check_synthetic_identity_thin_file(fields: Dict[str, Any]) -> Optional[FraudFlag]:
    """
    Flag if FICO score is exceptionally high but the credit file is extremely thin (few accounts).
    This is a standard indicator for synthetic identity profiles.
    """
    fico_avg = _safe(fields, "fico_avg")
    total_acc = _safe(fields, "total_accounts")

    if fico_avg is not None and total_acc is not None:
        if fico_avg >= 740 and total_acc <= 3:
            return FraudFlag(
                check_name="synthetic_identity_thin_file",
                severity="medium",
                description=(
                    f"Applicant has an excellent FICO score of {fico_avg:.0f} "
                    f"but only {int(total_acc)} total accounts on file. "
                    "Thin-file, high-FICO score profiles are a key synthetic identity anomaly."
                ),
                evidence={"fico_avg": fico_avg, "total_accounts": total_acc},
            )
    return None


# ---------------------------------------------------------------------------
# Aggregated runner
# ---------------------------------------------------------------------------

def run_behavior_checks(fields: Dict[str, Any]) -> List[FraudFlag]:
    """
    Run all behaviour-pattern fraud checks against extracted loan fields.

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
        check_round_number_income,
        check_round_number_loan,
        check_fico_dti_inconsistency,
        check_new_credit_surge,
        check_delinquency_despite_high_fico,
        check_public_record_mismatch,
        check_synthetic_identity_thin_file,
    ]

    flags: List[FraudFlag] = []
    for check_fn in checks:
        result = check_fn(fields)
        if result is not None:
            flags.append(result)

    return flags
