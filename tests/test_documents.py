import unittest
from src.document_ai.loan_extractor import LoanExtractor

class TestDocumentAI(unittest.TestCase):
    def setUp(self):
        self.extractor = LoanExtractor(use_gemini=False)

    def test_extractor_initialization(self):
        self.assertFalse(self.extractor.use_gemini)
        self.assertIsNone(self.extractor.client)

    def test_to_number_normalization(self):
        # Standard numbers
        self.assertEqual(self.extractor._to_number(12500), 12500)
        self.assertEqual(self.extractor._to_number(12.5), 12.5)
        
        # Currencies & formatting
        self.assertEqual(self.extractor._to_number("$25,000"), 25000)
        self.assertEqual(self.extractor._to_number("$ 25,000.50"), 25000.50)
        self.assertEqual(self.extractor._to_number("S25000"), 25000)
        
        # Percentages
        self.assertEqual(self.extractor._to_number("13.74%"), 13.74)
        
        # OCR scaling (ratios and DTIs multiplied by 10)
        self.assertEqual(self.extractor._to_number("dti: 174%"), 17.4)
        self.assertEqual(self.extractor._to_number("utilization: 483%"), 48.3)

    def test_regex_extraction_fallback(self):
        mock_text = """
        Applicant Name: John Doe
        DOB: 01-Jan-1990
        PAN: ABCDE1234F
        Address: 123 Main St, New York
        Employment Length: 5 years
        Home Ownership: RENT
        Loan Amount: $15,000
        Term: 36 months
        Interest Rate: 12.5%
        Installment: $500
        Annual Income: $60,000
        DTI: 15.2%
        FICO Avg: 710
        """
        result = self.extractor.extract(mock_text)
        
        self.assertEqual(result["document_type"], "loan_application")
        self.assertEqual(result["extraction_method"], "regex_fallback")
        
        fields = result["fields"]
        self.assertEqual(fields["applicant_name"], "John Doe")
        self.assertEqual(fields["dob"], "01-Jan-1990")
        self.assertEqual(fields["pan"], "ABCDE1234F")
        self.assertEqual(fields["home_ownership"], "RENT")
        
        # Normalized numeric fields
        self.assertEqual(fields["loan_amount"], 15000)
        self.assertEqual(fields["term_months"], 36)
        self.assertEqual(fields["interest_rate"], 12.5)
        self.assertEqual(fields["installment"], 500)
        self.assertEqual(fields["annual_income"], 60000)
        self.assertEqual(fields["fico_avg"], 710)
        self.assertEqual(fields["dti"], 15.2)

if __name__ == "__main__":
    unittest.main()
