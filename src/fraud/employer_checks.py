"""
employer_checks.py
------------------
Employer and income-source verification checks.

NOTE: These checks require bank statement transaction data, which is
not yet available in this pipeline. The functions are defined as stubs
and will return no flags until bank statement parsing is implemented
(statement_extractor.py + transaction_classifier.py).

When bank statement data is available, this module will detect:
  - Stated employer not matching salary credits in bank statement
  - Income frequency mismatch (e.g., stated monthly salary but credits
    appear bi-weekly or irregularly)
  - Salary credits much lower than stated annual income
  - Missing salary credits in recent months (employment gap)
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.fraud.income_checks import FraudFlag


def run_employer_checks(fields: Dict[str, Any]) -> List[FraudFlag]:
    """
    Run employer/income-source checks against extracted loan fields.

    Currently returns no flags — requires bank statement transaction
    data which is not yet available in the pipeline.

    Parameters
    ----------
    fields : dict
        The 'fields' dict from a LoanExtractor result.

    Returns
    -------
    list[FraudFlag]
        Always empty until bank statement parsing is implemented.
    """
    # TODO: Implement when statement_extractor.py is complete.
    # Planned checks:
    #   - check_employer_name_vs_salary_credits(fields, transactions)
    #   - check_income_frequency_consistency(fields, transactions)
    #   - check_salary_amount_vs_stated_income(fields, transactions)
    return []
