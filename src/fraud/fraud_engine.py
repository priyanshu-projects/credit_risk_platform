"""
fraud_engine.py
---------------
Orchestrates all fraud detection checks and produces a unified
fraud assessment for a single loan application.

Pipeline position:
    LoanExtractor -> FeatureEngineer -> FraudEngine -> RuleEngine

The FraudEngine:
  1. Runs all registered check modules (income, behaviour, employer)
  2. Aggregates all FraudFlag objects into a single FraudReport
  3. Computes an overall fraud risk level based on flag severity counts
  4. Never approves or rejects loans — provides signals for underwriters

Fraud Risk Levels
-----------------
  CLEAR   : No flags raised
  LOW     : Only low-severity flags
  MEDIUM  : At least one medium-severity flag, no high flags
  HIGH    : At least one high-severity flag
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.fraud.behavior_checks import run_behavior_checks
from src.fraud.employer_checks import run_employer_checks
from src.fraud.income_checks import FraudFlag, run_income_checks


# ---------------------------------------------------------------------------
# FraudReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class FraudReport:
    """
    Unified fraud assessment for a single loan application.

    Attributes
    ----------
    fraud_risk_level : 'CLEAR' | 'LOW' | 'MEDIUM' | 'HIGH'
    flags            : All FraudFlag objects raised across all checks.
    flag_counts      : Summary count by severity.
    guardrail        : Reminder that this report does not make loan decisions.
    """
    fraud_risk_level: str
    flags: List[FraudFlag]
    flag_counts: Dict[str, int]
    guardrail: str = (
        "Fraud signals are advisory only. "
        "No loan approval or rejection is produced by this engine."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a plain dict (JSON-safe)."""
        return {
            "fraud_risk_level": self.fraud_risk_level,
            "flag_counts": self.flag_counts,
            "flags": [
                {
                    "check_name": f.check_name,
                    "severity": f.severity,
                    "description": f.description,
                    "evidence": f.evidence,
                }
                for f in self.flags
            ],
            "guardrail": self.guardrail,
        }

    def has_flags(self) -> bool:
        return len(self.flags) > 0

    def highest_severity(self) -> str:
        """Return the worst severity level present, or 'none'."""
        if self.flag_counts.get("high", 0) > 0:
            return "high"
        if self.flag_counts.get("medium", 0) > 0:
            return "medium"
        if self.flag_counts.get("low", 0) > 0:
            return "low"
        return "none"


# ---------------------------------------------------------------------------
# FraudEngine
# ---------------------------------------------------------------------------

class FraudEngine:
    """
    Runs all registered fraud check modules and produces a FraudReport.

    Usage
    -----
        engine = FraudEngine()
        report = engine.evaluate(extractor_result)
        print(report.fraud_risk_level)  # 'CLEAR', 'LOW', 'MEDIUM', 'HIGH'
    """

    def evaluate(self, extractor_result: Dict[str, Any]) -> FraudReport:
        """
        Run all fraud checks and return a FraudReport.

        Parameters
        ----------
        extractor_result : dict
            The dict returned by LoanExtractor.extract() or extract_from_pdf().
            Expected to have a 'fields' key.

        Returns
        -------
        FraudReport
        """
        fields: Dict[str, Any] = extractor_result.get("fields", extractor_result)

        # --- Gather flags from all modules ---
        all_flags: List[FraudFlag] = []
        all_flags.extend(run_income_checks(fields))
        all_flags.extend(run_behavior_checks(fields))
        all_flags.extend(run_employer_checks(fields))

        # --- OCR & Document AI Specific Checks ---
        ext_method = extractor_result.get("extraction_method", "unknown")
        missing = extractor_result.get("missing_fields", [])
        
        if ext_method == "regex_fallback":
            all_flags.append(
                FraudFlag(
                    check_name="ocr_confidence_fallback",
                    severity="low",
                    description=(
                        "Structured data was extracted using deterministic regex fallback "
                        "instead of LLM extraction. Verify raw document formatting and scan quality."
                    ),
                    evidence={"extraction_method": ext_method},
                )
            )
            
        if len(missing) > 5:
            all_flags.append(
                FraudFlag(
                    check_name="high_missing_fields_anomaly",
                    severity="medium",
                    description=(
                        f"Application document is missing {len(missing)} critical data fields. "
                        "Incomplete document data indicates high validation risk."
                    ),
                    evidence={"missing_fields_count": len(missing), "missing_fields": missing},
                )
            )

        # --- Count by severity ---
        counts = {"low": 0, "medium": 0, "high": 0}
        for flag in all_flags:
            severity = flag.severity.lower()
            if severity in counts:
                counts[severity] += 1

        # --- Determine overall risk level ---
        if counts["high"] > 0:
            risk_level = "HIGH"
        elif counts["medium"] > 0:
            risk_level = "MEDIUM"
        elif counts["low"] > 0:
            risk_level = "LOW"
        else:
            risk_level = "CLEAR"

        return FraudReport(
            fraud_risk_level=risk_level,
            flags=all_flags,
            flag_counts=counts,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def evaluate_fraud(extractor_result: Dict[str, Any]) -> FraudReport:
    """
    Module-level shortcut for FraudEngine().evaluate(extractor_result).
    """
    return FraudEngine().evaluate(extractor_result)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # --- Test Case 1: Clean application ---
    clean_application = {
        "fields": {
            "loan_amount": 10000,
            "annual_income": 75000,
            "dti": 15.0,
            "installment": 320.0,
            "fico_avg": 720.0,
            "revolving_utilization": 35.0,
            "delinquencies_2yrs": 0,
            "public_records": 0,
            "total_accounts": 15,
            "acc_open_past_24m": 2,
        }
    }

    # --- Test Case 2: Suspicious application ---
    suspicious_application = {
        "fields": {
            "loan_amount": 30000,
            "annual_income": 200000,  # Round number + very high
            "dti": 0.5,               # Suspiciously low
            "installment": 900.0,
            "fico_avg": 760.0,        # High FICO...
            "revolving_utilization": 92.0,  # ...but maxed out cards
            "delinquencies_2yrs": 3,  # ...and recent delinquencies
            "public_records": 1,
            "total_accounts": 10,
            "acc_open_past_24m": 8,   # Opened 80% of accounts recently
        }
    }

    engine = FraudEngine()

    print("=" * 60)
    print("TEST CASE 1: Clean Application")
    print("=" * 60)
    report1 = engine.evaluate(clean_application)
    print(json.dumps(report1.to_dict(), indent=2))

    print("\n" + "=" * 60)
    print("TEST CASE 2: Suspicious Application")
    print("=" * 60)
    report2 = engine.evaluate(suspicious_application)
    print(json.dumps(report2.to_dict(), indent=2))
