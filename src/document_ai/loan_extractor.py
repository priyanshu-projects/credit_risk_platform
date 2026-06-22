import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LoanExtractor:
    """
    Extracts structured loan application fields from raw document text.

    Gemini Flash is used when GEMINI_API_KEY is configured. A deterministic
    regex fallback is included so Phase 4 can be tested locally without an API
    call. This class extracts facts only; it never approves or rejects loans.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    FIELD_NAMES = [
        "applicant_name",
        "dob",
        "pan",
        "address",
        "phone",
        "email",
        "employment_type",
        "employment_length",
        "home_ownership",
        "loan_amount",
        "term_months",
        "interest_rate",
        "installment",
        "purpose",
        "issue_year",
        "issue_month",
        "verification_status",
        "application_type",
        "disbursement_method",
        "grade",
        "sub_grade",
        "fico_avg",
        "annual_income",
        "dti",
        "open_accounts",
        "total_accounts",
        "revolving_balance",
        "revolving_utilization",
        "mortgage_accounts",
        "total_current_balance",
        "total_credit_limit",
        "bc_utilization",
        "acc_open_past_24m",
        "months_since_recent_inquiry",
        "months_since_recent_bc",
        "delinquencies_2yrs",
        "public_records",
    ]

    LABEL_MAP = {
        "applicant_name": "Applicant Name",
        "dob": "DOB",
        "pan": "PAN",
        "address": "Address",
        "phone": "Phone",
        "email": "Email",
        "employment_type": "Employment Type",
        "employment_length": "Employment Length",
        "home_ownership": "Home Ownership",
        "loan_amount": "Loan Amount",
        "term_months": "Term",
        "interest_rate": "Interest Rate",
        "installment": "Installment",
        "purpose": "Purpose",
        "issue_year": "Issue Year",
        "issue_month": "Issue Month",
        "verification_status": "Verification Status",
        "application_type": "Application Type",
        "disbursement_method": "Disbursement Method",
        "grade": "Grade",
        "sub_grade": "Sub Grade",
        "fico_avg": "FICO Avg",
        "annual_income": "Annual Income",
        "dti": "DTI",
        "open_accounts": "Open Accounts",
        "total_accounts": "Total Accounts",
        "revolving_balance": "Revolving Balance",
        "revolving_utilization": "Revolving Utilization",
        "mortgage_accounts": "Mortgage Accounts",
        "total_current_balance": "Total Current Balance",
        "total_credit_limit": "Total Credit Limit",
        "bc_utilization": "BC Utilization",
        "acc_open_past_24m": "Acc Open Past 24 Months",
        "months_since_recent_inquiry": "Months Since Recent Inquiry",
        "months_since_recent_bc": "Months Since Recent BC",
        "delinquencies_2yrs": "Delinquencies (2 yrs)",
        "public_records": "Public Records",
    }

    LABEL_ALIASES = {
        "applicant_name": ["Applicant", "Applicant Name"],
        "pan": ["PAN", "PAV"],
        "interest_rate": ["Interest Rate", "Interest Rale", "Intetest Rate", "Intetest Rale"],
        "revolving_utilization": ["Revolving Utilization", "Revolving Utilisation"],
        "delinquencies_2yrs": ["Delinquencies (2 yrs)", "Delinquencies 2 yrs", "Delinquencies"],
        "acc_open_past_24m": ["Acc Open Past 24 Months", "Accounts Open Past 24 Months"],
    }

    NUMERIC_FIELDS = {
        "loan_amount",
        "term_months",
        "interest_rate",
        "installment",
        "issue_year",
        "issue_month",
        "fico_avg",
        "annual_income",
        "dti",
        "open_accounts",
        "total_accounts",
        "revolving_balance",
        "revolving_utilization",
        "mortgage_accounts",
        "total_current_balance",
        "total_credit_limit",
        "bc_utilization",
        "acc_open_past_24m",
        "months_since_recent_inquiry",
        "months_since_recent_bc",
        "delinquencies_2yrs",
        "public_records",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_gemini: bool = True,
        api_key: Optional[str] = None,
    ):
        load_dotenv()
        self.model_name = model_name or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.use_gemini = use_gemini and bool(self.api_key)
        self.client = None

        if self.use_gemini:
            try:
                from google import genai

                self.client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                logger.warning(
                    "Gemini initialization failed; using regex fallback. Error: %s",
                    exc,
                )
                self.use_gemini = False

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract structured loan fields from raw document text.
        """
        if not text or not text.strip():
            raise ValueError("Cannot extract loan fields from empty text.")

        if self.use_gemini and self.client is not None:
            try:
                extracted = self._extract_with_gemini(text)
            except Exception as exc:
                logger.warning(
                    "Gemini extraction failed; using regex fallback. Error: %s",
                    exc,
                )
                extracted = self._extract_with_regex(text)
        else:
            extracted = self._extract_with_regex(text)

        return self._normalize_result(extracted, source_text=text)

    def extract_from_pdf(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Extract raw text via DocumentRouter, then convert it into structured data.
        """
        from src.documents.document_router import DocumentRouter

        text = DocumentRouter().process_document(file_path)
        if not text:
            raise ValueError(f"No text could be extracted from {file_path}")
        return self.extract(text)

    def _extract_with_gemini(self, text: str) -> Dict[str, Any]:
        from google.genai import types

        prompt = self._build_prompt(text)
        # The new google-genai SDK requires the full model path with 'models/' prefix
        model_id = self.model_name if self.model_name.startswith("models/") else f"models/{self.model_name}"
        response = self.client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        return self._parse_json_response(response.text)

    def _build_prompt(self, text: str) -> str:
        fields = "\n".join(f"- {field}" for field in self.FIELD_NAMES)
        return f"""
You are an information extraction system for a credit risk platform.
Extract factual fields from the loan application text and return JSON only.

Rules:
- Do not approve, reject, recommend, judge, or score the loan.
- Do not invent missing values. Use null when unavailable.
- Preserve strings for identifiers such as PAN, phone, email, grade, and sub_grade.
- Convert money, percentages, month counts, and numeric counts to numbers when clear.
- Return exactly one JSON object with these keys:
{fields}

Document text:
\"\"\"{text[:12000]}\"\"\"
""".strip()

    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        extracted: Dict[str, Any] = {}
        for field, label in self.LABEL_MAP.items():
            extracted[field] = self._find_value_after_label(text, field, label)
        return extracted

    def _find_value_after_label(self, text: str, field: str, label: str) -> Optional[str]:
        aliases = sorted(self.LABEL_ALIASES.get(field, [label]), key=len, reverse=True)
        labels = sorted(
            {alias for values in self.LABEL_ALIASES.values() for alias in values}
            | set(self.LABEL_MAP.values()),
            key=len,
            reverse=True,
        )
        current_label_pattern = "|".join(re.escape(item) for item in aliases)
        next_label_pattern = "|".join(re.escape(item) for item in labels if item not in aliases)

        # Handles both two-line labels and OCR output like "Loan Amount: $25.000".
        pattern = rf"(?im)^\s*(?:{current_label_pattern})\s*:?\s*(?P<value>.*?)(?=^\s*(?:{next_label_pattern})\s*:?\s*|\Z)"
        match = re.search(pattern, text, flags=re.DOTALL | re.MULTILINE)
        if not match:
            return None

        lines = [line.strip() for line in match.group("value").splitlines() if line.strip()]
        return lines[0] if lines else None

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)

    def _normalize_result(self, data: Dict[str, Any], source_text: str) -> Dict[str, Any]:
        normalized = {field: data.get(field) for field in self.FIELD_NAMES}

        for field in self.NUMERIC_FIELDS:
            normalized[field] = self._to_number(normalized.get(field))

        missing_fields = [field for field, value in normalized.items() if value in (None, "")]

        return {
            "document_type": "loan_application",
            "extraction_method": "gemini_flash" if self.use_gemini else "regex_fallback",
            "fields": normalized,
            "missing_fields": missing_fields,
            "raw_text_char_count": len(source_text),
            "guardrail": "Extraction only. No loan approval, rejection, or recommendation is produced.",
        }

    def _to_number(self, value: Any) -> Optional[float | int]:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return value

        value_str = str(value).strip()
        value_str = re.sub(r"(?i)(?<=\d)\s*[o]\s*(?=\d)", "0", value_str)
        value_str = value_str.replace("S", "$")
        value_str = value_str.replace("s", "$")
        value_str = re.sub(r"(?<=\d)[.,]\s*(?=\d{3}\b)", "", value_str)

        number_match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?", value_str)
        if not number_match:
            return None

        number_text = number_match.group(0).replace(",", "")
        number = float(number_text)

        # OCR often drops the decimal point in percentages/ratios like 17.4 -> 174.
        # Covers: "17.4%" -> "174%", bare DTI values, and revolving_utilization.
        if isinstance(value, str) and any(token in value.lower() for token in ["dti", "%", "util"]) and number > 100:
            number = number / 10

        return int(number) if number.is_integer() else number


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    parser = argparse.ArgumentParser(description="Extract structured fields from a loan application PDF.")
    parser.add_argument("pdf_path", help="Path to a loan application PDF")
    parser.add_argument("--no-gemini", action="store_true", help="Use deterministic regex fallback only")
    args = parser.parse_args()

    extractor = LoanExtractor(use_gemini=not args.no_gemini)
    result = extractor.extract_from_pdf(args.pdf_path)
    print(json.dumps(result, indent=2))
