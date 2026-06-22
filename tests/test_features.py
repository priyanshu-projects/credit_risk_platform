import unittest
import pandas as pd
from src.features.feature_engineering import FeatureEngineer, MODEL_FEATURE_COLUMNS

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.engineer = FeatureEngineer()

    def test_feature_engineer_transform(self):
        mock_extractor_result = {
            "fields": {
                "loan_amount": 25000,
                "interest_rate": 13.74,
                "installment": 850,
                "annual_income": 85000,
                "dti": 17.4,
                "fico_avg": 702.0,
                "open_accounts": 11,
                "total_accounts": 24,
                "revolving_balance": 14000,
                "revolving_utilization": 48.3,
                "total_credit_limit": 29000,
                "term_months": 36,
                "grade": "C",
                "sub_grade": "C3",
                "home_ownership": "RENT",
                "verification_status": "Verified",
                "purpose": "debt_consolidation",
                "employment_length": "7 years",
            }
        }
        df = self.engineer.transform(mock_extractor_result)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (1, len(MODEL_FEATURE_COLUMNS)))
        
        self.assertAlmostEqual(df["fico_avg"].iloc[0], 702.0)
        self.assertAlmostEqual(df["loan_to_income"].iloc[0], 25000 / 85000)
        self.assertAlmostEqual(df["installment_to_income"].iloc[0], 850 / 85000)
        # credit utilization ratio (revol_bal / total_rev_hi_lim) -> 14000 / 29000
        self.assertAlmostEqual(df["credit_utilization_ratio"].iloc[0], 14000 / 29000)
        # recent credit ratio (num_tl_op_past_12m / total_acc) -> defaults to imputed median 0.0833
        self.assertAlmostEqual(df["recent_credit_ratio"].iloc[0], 0.0833)

    def test_feature_engineer_missing_fields(self):
        df = self.engineer.transform({"fields": {}})
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (1, len(MODEL_FEATURE_COLUMNS)))
        for col in MODEL_FEATURE_COLUMNS:
            self.assertFalse(pd.isna(df[col].iloc[0]))

if __name__ == "__main__":
    unittest.main()
