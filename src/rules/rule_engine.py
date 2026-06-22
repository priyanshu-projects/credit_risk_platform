"""
rule_engine.py
--------------
Applies deterministic credit policy rules to produce a routing decision
for each loan application.

Pipeline position:
    FraudEngine -> RuleEngine -> LLM Underwriter Report (if needed)

The Rule Engine combines:
  - ML model output   : probability of default (0.0 – 1.0)
  - Fraud report      : fraud_risk_level ('CLEAR'|'LOW'|'MEDIUM'|'HIGH')
  - Extracted fields  : dti, fico_avg, loan_amount, annual_income, etc.

Output: a RuleDecision with one of four verdicts:
  DECLINE        : Hard policy violation — do not proceed
  AUTO_APPROVE   : All green lights — fast-track (low risk, clean)
  MANUAL_REVIEW  : Borderline — send to human underwriter
  REFER          : Refer for additional document verification

IMPORTANT: This engine never approves or rejects loans autonomously.
AUTO_APPROVE means "eligible for approval" — a human still signs off.
DECLINE means "policy breach" — a human still reviews the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Thresholds (credit policy parameters — easy to tune)
# ---------------------------------------------------------------------------

# Hard decline thresholds
HARD_DECLINE_FICO_MIN          = 580      # Below this: always decline
HARD_DECLINE_DTI_MAX           = 50.0     # Above this: always decline
HARD_DECLINE_PROB_DEFAULT_MAX  = 0.70     # Above this: always decline
HARD_DECLINE_FRAUD_LEVELS      = {"HIGH"} # Fraud level that triggers decline

# Auto-approve thresholds (ALL must be satisfied simultaneously)
AUTO_APPROVE_PROB_DEFAULT_MAX  = 0.10     # Model must predict <= 10% default
AUTO_APPROVE_FICO_MIN          = 720      # FICO must be >= 720
AUTO_APPROVE_DTI_MAX           = 35.0     # DTI must be <= 35%
AUTO_APPROVE_FRAUD_LEVELS      = {"CLEAR", "LOW"}  # Only clean fraud signals

# Manual review band
MANUAL_REVIEW_PROB_DEFAULT_MIN = 0.20     # Between 20–70% → manual review
MANUAL_REVIEW_FRAUD_LEVELS     = {"MEDIUM"}

# Refer thresholds
REFER_FICO_BAND_LOW            = 580      # FICO in [580, 650) → refer
REFER_FICO_BAND_HIGH           = 650


# ---------------------------------------------------------------------------
# RuleDecision dataclass
# ---------------------------------------------------------------------------

@dataclass
class RuleDecision:
    """
    The routing decision produced by the Rule Engine.

    Attributes
    ----------
    verdict      : 'DECLINE' | 'AUTO_APPROVE' | 'MANUAL_REVIEW' | 'REFER'
    triggered_rules : List of rule names that influenced this verdict.
    rationale    : Human-readable explanation of the decision.
    inputs_used  : Key input values used to reach this verdict.
    guardrail    : Reminder that this is a routing decision, not a loan decision.
    """
    verdict: str
    triggered_rules: List[str]
    rationale: str
    inputs_used: Dict[str, Any] = field(default_factory=dict)
    guardrail: str = (
        "This is a routing decision only. Loan approval or rejection "
        "requires a human underwriter."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "triggered_rules": self.triggered_rules,
            "rationale": self.rationale,
            "inputs_used": self.inputs_used,
            "guardrail": self.guardrail,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe(fields: Dict[str, Any], key: str) -> Optional[float]:
    val = fields.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# RuleEngine
# ---------------------------------------------------------------------------

class RuleEngine:
    """
    Applies credit policy rules and routes applications to the correct
    downstream handler.

    Usage
    -----
        engine = RuleEngine()
        decision = engine.evaluate(
            prob_default=0.18,
            fraud_risk_level="MEDIUM",
            fields=extractor_result["fields"],
        )
        print(decision.verdict)  # 'MANUAL_REVIEW'
    """

    def evaluate(
        self,
        prob_default: float,
        fraud_risk_level: str,
        fields: Dict[str, Any],
    ) -> RuleDecision:
        """
        Evaluate policy rules and return a RuleDecision.

        Parameters
        ----------
        prob_default      : float, 0.0–1.0 from RiskModelPredictor.
        fraud_risk_level  : str, from FraudEngine ('CLEAR'|'LOW'|'MEDIUM'|'HIGH').
        fields            : dict, extracted loan fields from LoanExtractor.

        Returns
        -------
        RuleDecision
        """
        fico    = _safe(fields, "fico_avg")
        dti     = _safe(fields, "dti")
        loan_amnt = _safe(fields, "loan_amount")

        inputs = {
            "prob_default": round(prob_default, 4),
            "fraud_risk_level": fraud_risk_level,
            "fico_avg": fico,
            "dti": dti,
            "loan_amount": loan_amnt,
        }

        # ------------------------------------------------------------------
        # Priority 1: Hard Decline — checked first, highest priority
        # ------------------------------------------------------------------
        decline_rules: List[str] = []

        if fraud_risk_level in HARD_DECLINE_FRAUD_LEVELS:
            decline_rules.append(
                f"fraud_risk_HIGH: fraud level is '{fraud_risk_level}'"
            )

        if fico is not None and fico < HARD_DECLINE_FICO_MIN:
            decline_rules.append(
                f"fico_below_minimum: FICO {fico:.0f} < {HARD_DECLINE_FICO_MIN}"
            )

        if dti is not None and dti > HARD_DECLINE_DTI_MAX:
            decline_rules.append(
                f"dti_exceeds_maximum: DTI {dti:.1f}% > {HARD_DECLINE_DTI_MAX}%"
            )

        if prob_default > HARD_DECLINE_PROB_DEFAULT_MAX:
            decline_rules.append(
                f"model_prob_too_high: P(default)={prob_default:.2%} "
                f"> {HARD_DECLINE_PROB_DEFAULT_MAX:.0%}"
            )

        if decline_rules:
            return RuleDecision(
                verdict="DECLINE",
                triggered_rules=decline_rules,
                rationale=(
                    "Application breaches one or more hard credit policy thresholds. "
                    "Routing to decline queue for human review. "
                    f"Rules triggered: {'; '.join(decline_rules)}."
                ),
                inputs_used=inputs,
            )

        # ------------------------------------------------------------------
        # Priority 2: REFER — FICO in borderline band [580, 650)
        # ------------------------------------------------------------------
        refer_rules: List[str] = []

        if fico is not None and REFER_FICO_BAND_LOW <= fico < REFER_FICO_BAND_HIGH:
            refer_rules.append(
                f"fico_borderline: FICO {fico:.0f} in [{REFER_FICO_BAND_LOW}, "
                f"{REFER_FICO_BAND_HIGH})"
            )

        if refer_rules:
            return RuleDecision(
                verdict="REFER",
                triggered_rules=refer_rules,
                rationale=(
                    "FICO score is in the borderline band. "
                    "Additional verification documents required before underwriting. "
                    f"Rules triggered: {'; '.join(refer_rules)}."
                ),
                inputs_used=inputs,
            )

        # ------------------------------------------------------------------
        # Priority 3: Manual Review
        # ------------------------------------------------------------------
        review_rules: List[str] = []

        if fraud_risk_level in MANUAL_REVIEW_FRAUD_LEVELS:
            review_rules.append(
                f"fraud_risk_MEDIUM: fraud level is '{fraud_risk_level}'"
            )

        if prob_default >= MANUAL_REVIEW_PROB_DEFAULT_MIN:
            review_rules.append(
                f"model_prob_elevated: P(default)={prob_default:.2%} "
                f">= {MANUAL_REVIEW_PROB_DEFAULT_MIN:.0%}"
            )

        if review_rules:
            return RuleDecision(
                verdict="MANUAL_REVIEW",
                triggered_rules=review_rules,
                rationale=(
                    "Application does not meet hard decline thresholds but "
                    "has elevated risk or fraud signals requiring human review. "
                    f"Rules triggered: {'; '.join(review_rules)}."
                ),
                inputs_used=inputs,
            )

        # ------------------------------------------------------------------
        # Priority 4: Auto-Approve — ALL conditions must pass
        # ------------------------------------------------------------------
        approve_conditions = []

        if prob_default <= AUTO_APPROVE_PROB_DEFAULT_MAX:
            approve_conditions.append("low_model_risk")
        if fico is not None and fico >= AUTO_APPROVE_FICO_MIN:
            approve_conditions.append("fico_above_threshold")
        if dti is None or dti <= AUTO_APPROVE_DTI_MAX:
            approve_conditions.append("dti_within_limit")
        if fraud_risk_level in AUTO_APPROVE_FRAUD_LEVELS:
            approve_conditions.append("clean_fraud_signal")

        all_approve = (
            prob_default <= AUTO_APPROVE_PROB_DEFAULT_MAX
            and (fico is None or fico >= AUTO_APPROVE_FICO_MIN)
            and (dti is None or dti <= AUTO_APPROVE_DTI_MAX)
            and fraud_risk_level in AUTO_APPROVE_FRAUD_LEVELS
        )

        if all_approve:
            return RuleDecision(
                verdict="AUTO_APPROVE",
                triggered_rules=approve_conditions,
                rationale=(
                    "All fast-track approval criteria satisfied: low model risk, "
                    "strong FICO, manageable DTI, and clean fraud signal. "
                    "Eligible for accelerated underwriting."
                ),
                inputs_used=inputs,
            )

        # ------------------------------------------------------------------
        # Default: Manual Review (catches everything not explicitly routed)
        # ------------------------------------------------------------------
        return RuleDecision(
            verdict="MANUAL_REVIEW",
            triggered_rules=["default_routing"],
            rationale=(
                "Application does not meet auto-approve criteria and has no "
                "hard decline triggers. Routing to standard underwriting review."
            ),
            inputs_used=inputs,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def evaluate_rules(
    prob_default: float,
    fraud_risk_level: str,
    fields: Dict[str, Any],
) -> RuleDecision:
    """Module-level shortcut for RuleEngine().evaluate(...)."""
    return RuleEngine().evaluate(prob_default, fraud_risk_level, fields)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    engine = RuleEngine()

    test_cases = [
        {
            "label": "Hard Decline — FICO too low",
            "prob_default": 0.35,
            "fraud_risk_level": "CLEAR",
            "fields": {"fico_avg": 540, "dti": 22.0, "loan_amount": 15000},
        },
        {
            "label": "Hard Decline — Fraud HIGH",
            "prob_default": 0.18,
            "fraud_risk_level": "HIGH",
            "fields": {"fico_avg": 700, "dti": 18.0, "loan_amount": 20000},
        },
        {
            "label": "Hard Decline — Model prob too high",
            "prob_default": 0.75,
            "fraud_risk_level": "CLEAR",
            "fields": {"fico_avg": 640, "dti": 30.0, "loan_amount": 10000},
        },
        {
            "label": "Refer — Borderline FICO",
            "prob_default": 0.15,
            "fraud_risk_level": "CLEAR",
            "fields": {"fico_avg": 615, "dti": 25.0, "loan_amount": 12000},
        },
        {
            "label": "Manual Review — Fraud MEDIUM",
            "prob_default": 0.12,
            "fraud_risk_level": "MEDIUM",
            "fields": {"fico_avg": 690, "dti": 28.0, "loan_amount": 18000},
        },
        {
            "label": "Manual Review — Elevated model risk",
            "prob_default": 0.28,
            "fraud_risk_level": "CLEAR",
            "fields": {"fico_avg": 680, "dti": 32.0, "loan_amount": 25000},
        },
        {
            "label": "Auto-Approve — All green",
            "prob_default": 0.06,
            "fraud_risk_level": "CLEAR",
            "fields": {"fico_avg": 755, "dti": 14.0, "loan_amount": 10000},
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"  {tc['label']}")
        print(f"{'='*60}")
        decision = engine.evaluate(
            prob_default=tc["prob_default"],
            fraud_risk_level=tc["fraud_risk_level"],
            fields=tc["fields"],
        )
        print(json.dumps(decision.to_dict(), indent=2))
