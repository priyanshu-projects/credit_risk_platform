import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List

class SHAPExplainer:
    """
    Utility class to generate explainability metrics for the XGBoost risk model.
    Provides structured text data for LLMs and visual plots for the UI.
    """
    def __init__(self, model):
        self.model = model
        # TreeExplainer is highly optimized for XGBoost
        self.explainer = shap.TreeExplainer(self.model)

    def get_shap_values(self, features_df: pd.DataFrame):
        """Calculates SHAP values for the provided feature DataFrame."""
        return self.explainer(features_df)

    def get_top_risk_factors(self, features_df: pd.DataFrame, top_n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extracts the top contributing features for a single prediction.
        Returns factors pushing risk UP (risk_drivers) and DOWN (mitigators).
        """
        shap_values = self.get_shap_values(features_df)
        
        feature_names = features_df.columns.tolist()
        
        # XGBoost binary classification SHAP values are typically 2D arrays: (samples, features)
        # We take the first row [0] since we evaluate one application at a time
        instance_shap_values = shap_values.values[0]
        instance_feature_values = features_df.iloc[0].values
        
        factors = []
        for name, shap_val, feat_val in zip(feature_names, instance_shap_values, instance_feature_values):
            factors.append({
                "feature": name,
                "value": float(feat_val) if isinstance(feat_val, (np.integer, np.floating)) else feat_val,
                "shap_contribution": float(shap_val)
            })
            
        # Sort by absolute SHAP value to find the strongest influencers
        factors.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)
        
        # Positive SHAP = pushes probability towards 1 (Default/Charged Off)
        # Negative SHAP = pushes probability towards 0 (Fully Paid)
        risk_drivers = [f for f in factors if f["shap_contribution"] > 0][:top_n]
        mitigators = [f for f in factors if f["shap_contribution"] < 0][:top_n]
        
        return {
            "base_value": float(shap_values.base_values[0]),
            "risk_drivers": risk_drivers,
            "mitigators": mitigators
        }

    def save_waterfall_plot(self, features_df: pd.DataFrame, save_path: str):
        """
        Generates and saves a SHAP waterfall plot to disk.
        """
        shap_values = self.get_shap_values(features_df)
        
        # Create a new figure to avoid overlapping plots in loops
        fig = plt.figure(figsize=(10, 6))
        
        # Generate the waterfall plot for the first instance
        shap.plots.waterfall(shap_values[0], show=False)
        
        # Ensure the destination directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    import sys
    import os
    import json
    
    # Add project root to path to import the predict module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.risk_models.predict import RiskModelPredictor
    
    try:
        # Load the model using our Phase 1 predictor
        predictor = RiskModelPredictor()
        explainer = SHAPExplainer(predictor.model)
        
        # Dummy data matching our previous test
        dummy_payload = {"loan_amnt": 10000, "int_rate": 18.5, "annual_inc": 45000, "dti": 25.0}
        features_df = predictor.prepare_features(dummy_payload)
        
        # Test Text Factors
        factors = explainer.get_top_risk_factors(features_df, top_n=3)
        print("--- Top Risk Factors ---")
        print(json.dumps(factors, indent=2))
        
        # Test Plot Generation
        test_plot_path = "artifacts/shap_plots/test_waterfall.png"
        explainer.save_waterfall_plot(features_df, test_plot_path)
        print(f"\n--- SHAP Plot ---")
        print(f"Saved successfully to: {test_plot_path}")
        
    except Exception as e:
        print(f"SHAP utility test failed: {e}")
