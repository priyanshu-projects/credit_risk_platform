import unittest
from src.fraud.fraud_engine import evaluate_fraud

class TestFraudEngine(unittest.TestCase):
    def test_fraud_engine_clean_application(self):
        clean_app = {
            "fields": {
                "loan_amount": 12345.0,
                "annual_income": 75643.0,
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
        report = evaluate_fraud(clean_app)
        self.assertEqual(report.fraud_risk_level, "CLEAR")
        self.assertFalse(report.has_flags())
        self.assertEqual(report.highest_severity(), "none")

    def test_fraud_engine_round_numbers(self):
        round_app = {
            "fields": {
                "loan_amount": 30000.0,
                "annual_income": 80000.0,
                "dti": 15.0,
                "installment": 320.0,
                "fico_avg": 720.0,
            }
        }
        report = evaluate_fraud(round_app)
        self.assertEqual(report.fraud_risk_level, "LOW")
        self.assertEqual(report.highest_severity(), "low")
        self.assertEqual(len(report.flags), 2)
        flag_names = [f.check_name for f in report.flags]
        self.assertIn("round_number_income", flag_names)
        self.assertIn("round_number_loan_amount", flag_names)

    def test_fraud_engine_fico_dti_inconsistency(self):
        inconsistent_app = {
            "fields": {
                "loan_amount": 10000.0,
                "annual_income": 60000.0,
                "dti": 45.0,
                "fico_avg": 760.0,
            }
        }
        report = evaluate_fraud(inconsistent_app)
        self.assertEqual(report.fraud_risk_level, "MEDIUM")
        self.assertEqual(report.highest_severity(), "medium")
        self.assertTrue(any(f.check_name == "fico_dti_inconsistency" for f in report.flags))

    def test_fraud_engine_high_severity_mismatch(self):
        mismatch_app = {
            "fields": {
                "loan_amount": 10000.0,
                "annual_income": 60000.0,
                "fico_avg": 730.0,
                "delinquencies_2yrs": 3,
            }
        }
        report = evaluate_fraud(mismatch_app)
        self.assertEqual(report.fraud_risk_level, "HIGH")
        self.assertEqual(report.highest_severity(), "high")
        self.assertTrue(any(f.check_name == "delinquency_despite_high_fico" for f in report.flags))

    def test_fraud_engine_excessive_leverage(self):
        leverage_app = {
            "fields": {
                "loan_amount": 50000.0,
                "annual_income": 30000.0,
            }
        }
        report = evaluate_fraud(leverage_app)
        self.assertEqual(report.fraud_risk_level, "HIGH")
        self.assertTrue(any(f.check_name == "excessive_loan_to_income_leverage" for f in report.flags))

    def test_fraud_report_serialization(self):
        app = {
            "fields": {
                "loan_amount": 30000.0,
                "annual_income": 80000.0,
            }
        }
        report = evaluate_fraud(app)
        report_dict = report.to_dict()
        self.assertIn("fraud_risk_level", report_dict)
        self.assertIn("flag_counts", report_dict)
        self.assertIn("flags", report_dict)
        self.assertIsInstance(report_dict["flags"], list)

if __name__ == "__main__":
    unittest.main()
