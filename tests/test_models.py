import unittest
import pandas as pd
import numpy as np
from src.risk_models.predict import RiskModelPredictor

class TestRiskModels(unittest.TestCase):
    def setUp(self):
        self.predictor = RiskModelPredictor()

    def test_predictor_initialization(self):
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.metadata)
        self.assertGreater(len(self.predictor.expected_features), 0)

    def test_prepare_features(self):
        dummy_payload = {
            "loan_amnt": 10000,
            "int_rate": 10.5,
            "annual_inc": 75000,
            "fico_range_low": 700,
            "fico_range_high": 704
        }
        df = self.predictor.prepare_features(dummy_payload)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 1)
        for col in self.predictor.expected_features:
            self.assertIn(col, df.columns)

    def test_predict_risk(self):
        dummy_payload = {
            "loan_amnt": 10000,
            "int_rate": 10.5,
            "annual_inc": 75000,
            "fico_range_low": 700,
            "fico_range_high": 704
        }
        result = self.predictor.predict_risk(dummy_payload)
        self.assertIn("probability_of_default", result)
        self.assertIn("risk_tier", result)
        self.assertIn("model_version", result)
        self.assertIsInstance(result["probability_of_default"], float)
        self.assertTrue(0.0 <= result["probability_of_default"] <= 1.0)
        self.assertIn(result["risk_tier"], ["Low", "Medium", "High", "Very High"])

    def test_robust_missing_inputs(self):
        df = self.predictor.prepare_features({})
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 1)
        for col in self.predictor.expected_features:
            self.assertIn(col, df.columns)

if __name__ == "__main__":
    unittest.main()
