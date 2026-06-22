import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

class RiskModelPredictor:
    """
    Production interface for the trained LendingClub XGBoost model.
    Handles loading the model, metadata, and making predictions on new data.
    """
    def __init__(self, model_path: str = "models/xgboost_baseline.joblib", metadata_path: str = "data/processed/features/lendingclub_baseline_metadata.json"):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model = None
        self.metadata = None
        self.expected_features = []
        self._load_assets()

    def _load_assets(self):
        """Loads the saved joblib model and the metadata JSON."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Did you run Phase 1 training?")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}.")

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # In notebook 3, we dropped leaked columns before training. We must mirror that here.
        leakage_to_drop = [
            'total_rec_prncp', 
            'total_rec_int', 
            'total_rec_late_fee', 
            'last_fico_range_high', 
            'last_fico_range_low'
        ]
        
        # Load XGBoost model
        self.model = joblib.load(self.model_path)
        
        # Safely determine the exact features the model was trained on
        # If it's a pipeline or standard XGBoost classifier, feature_names_in_ is standard in modern scikit-learn/xgboost
        if hasattr(self.model, 'feature_names_in_'):
            self.expected_features = list(self.model.feature_names_in_)
        elif hasattr(self.model, 'get_booster'):
            self.expected_features = self.model.get_booster().feature_names
        else:
            # Fallback to metadata if we can't extract it from the model
            raw_features = self.metadata.get('feature_columns', [])
            leakage_to_drop = [
                'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 
                'last_fico_range_high', 'last_fico_range_low'
            ]
            self.expected_features = [col for col in raw_features if col not in leakage_to_drop]

    @staticmethod
    def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
        """Return a numeric ratio when both inputs are usable and denominator is non-zero."""
        try:
            numerator_val = float(numerator)
            denominator_val = float(denominator)
        except (TypeError, ValueError):
            return None

        if denominator_val == 0:
            return None
        return numerator_val / denominator_val

    def _add_engineered_features(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recreates lightweight features added in notebook 03 when raw inputs are present.
        Missing values are still handled later by training-time imputes or zero fallback.
        """
        enriched = dict(input_data)

        if "loan_to_income" not in enriched:
            value = self._safe_ratio(enriched.get("loan_amnt"), enriched.get("annual_inc"))
            if value is not None:
                enriched["loan_to_income"] = value

        if "installment_to_income" not in enriched:
            value = self._safe_ratio(enriched.get("installment"), enriched.get("annual_inc"))
            if value is not None:
                enriched["installment_to_income"] = value

        if "credit_utilization_ratio" not in enriched:
            value = self._safe_ratio(enriched.get("revol_bal"), enriched.get("total_rev_hi_lim"))
            if value is not None:
                enriched["credit_utilization_ratio"] = value

        if "recent_credit_ratio" not in enriched:
            value = self._safe_ratio(enriched.get("num_tl_op_past_12m"), enriched.get("total_acc"))
            if value is not None:
                enriched["recent_credit_ratio"] = value

        if "fico_avg" not in enriched:
            low = enriched.get("fico_range_low")
            high = enriched.get("fico_range_high")
            try:
                if low is not None and high is not None:
                    enriched["fico_avg"] = (float(low) + float(high)) / 2
            except (TypeError, ValueError):
                pass

        return enriched

    def prepare_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Takes raw dictionary input, aligns it with expected model features, 
        and imputes missing values based on training defaults if necessary.
        """
        input_data = self._add_engineered_features(input_data)

        # Create a dictionary to hold all column data before DataFrame creation
        # This avoids the pandas DataFrame fragmentation PerformanceWarning
        processed_data = {}
        impute_vals = self.metadata.get('numeric_impute_values', {})
        
        # Ensure all expected columns exist
        for col in self.expected_features:
            if col in input_data:
                processed_data[col] = [input_data[col]]
            else:
                # If a column is missing, try to use the imputation median/mode from metadata
                processed_data[col] = [impute_vals.get(col, 0.0)]
                
        # Create DataFrame from the fully constructed dictionary
        df = pd.DataFrame(processed_data)

        
        # Ensure numeric types
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        
        return df

    def predict_risk(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts the probability of default (Charged Off).
        Returns a structured dictionary with probability, risk tier, and decision context.
        """
        features_df = self.prepare_features(input_data)
        
        # XGBoost predict_proba returns [[prob_class_0, prob_class_1]]
        # Class 1 is "Charged Off" (Default)
        prob_default = float(self.model.predict_proba(features_df)[0][1])
        
        # Basic risk tiering logic (can be adjusted later)
        if prob_default < 0.10:
            risk_tier = "Low"
        elif prob_default < 0.25:
            risk_tier = "Medium"
        elif prob_default < 0.50:
            risk_tier = "High"
        else:
            risk_tier = "Very High"

        return {
            "probability_of_default": round(prob_default, 4),
            "risk_tier": risk_tier,
            "model_version": "tuned_xgboost_v1"
        }

if __name__ == "__main__":
    # Quick sanity check
    try:
        predictor = RiskModelPredictor()
        print("Model and metadata loaded successfully.")
        print(f"Expecting {len(predictor.expected_features)} features.")
        
        # Create a dummy payload
        dummy_payload = {"loan_amnt": 10000, "int_rate": 10.5, "annual_inc": 75000}
        result = predictor.predict_risk(dummy_payload)
        print("\nDummy Prediction Result:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Initialization failed: {e}")
