"""
train.py
--------
Reproducible training script for the LendingClub XGBoost risk model.
Replicates the training pipeline from notebooks/03_model_training.ipynb.

Usage:
    python src/risk_models/train.py [--tune]
"""

import gc
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

def train_model(tune: bool = False):
    print("Starting Phase 1 Risk Model training pipeline...")

    # Define paths
    metadata_path = Path("data/processed/features/lendingclub_baseline_metadata.json")
    data_path = Path("data/processed/features/lendingclub_baseline_features.parquet")
    model_output_path = Path("models/xgboost_baseline.joblib")
    model_metadata_path = Path("models/metadata.json")
    test_output_path = Path("data/processed/features/X_test_baseline.parquet")

    # Load baseline feature metadata
    print(f"Loading feature metadata from {metadata_path}...")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Exclude post-origination leakage columns
    leakage_to_drop = [
        'total_rec_prncp', 
        'total_rec_int', 
        'total_rec_late_fee', 
        'last_fico_range_high', 
        'last_fico_range_low'
    ]
    raw_features = metadata.get('feature_columns', [])
    features = [col for col in raw_features if col not in leakage_to_drop]
    print(f"Filtered to {len(features)} initial features after removing post-origination leakage.")

    # Load data
    print(f"Loading data from {data_path}...")
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")
    
    cols_to_load = features + ['target']
    df = pd.read_parquet(data_path, columns=cols_to_load)

    # Process and clean features
    X = df[features].copy()
    y = df['target'].copy()

    # Drop redundant columns
    X = X.drop(columns=['funded_amnt', 'funded_amnt_inv'], errors='ignore')

    # Create FICO average and drop FICO low/high ranges
    if {'fico_range_low', 'fico_range_high'}.issubset(X.columns):
        X['fico_avg'] = (X['fico_range_low'] + X['fico_range_high']) / 2
        X = X.drop(columns=['fico_range_low', 'fico_range_high'], errors='ignore')

    X = X.drop(columns=['num_sats'], errors='ignore')

    # Split into Train (70%), Validation (15%), Test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Add ratio engineered features
    ratio_features = {
        'loan_to_income': ('loan_amnt', 'annual_inc'),
        'installment_to_income': ('installment', 'annual_inc'),
        'credit_utilization_ratio': ('revol_bal', 'total_rev_hi_lim'),
        'recent_credit_ratio': ('num_tl_op_past_12m', 'total_acc'),
    }

    for new_col, (numerator, denominator) in ratio_features.items():
        for split_df in [X_train, X_val, X_test]:
            split_df[new_col] = split_df[numerator] / split_df[denominator].replace(0, np.nan)

        # Impute missing values with training split median
        median_val = X_train[new_col].median()
        X_train[new_col] = X_train[new_col].fillna(median_val)
        X_val[new_col] = X_val[new_col].fillna(median_val)
        X_test[new_col] = X_test[new_col].fillna(median_val)

    print(f"Training set shape  : {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Testing set shape   : {X_test.shape}")

    del X_temp, y_temp, df
    gc.collect()

    # Handle class imbalance
    neg_class_count = (y_train == 0).sum()
    pos_class_count = (y_train == 1).sum()
    scale_pos_weight_val = neg_class_count / pos_class_count
    print(f"Calculated scale_pos_weight: {scale_pos_weight_val:.2f}")

    if tune:
        print("Running XGBoost parameter tuning via GridSearchCV...")
        xgb_base = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            scale_pos_weight=scale_pos_weight_val,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            n_jobs=2
        )
        param_grid = {
            'n_estimators': [300, 500, 700],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.03, 0.05, 0.08],
        }
        grid = GridSearchCV(
            estimator=xgb_base,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3,
            n_jobs=1,
            verbose=2,
            refit=True
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_auc = float(grid.best_score_)
        print(f"Optimal parameters found: {best_params}")
    else:
        # Train directly with pre-tuned parameters
        print("Training model with optimal baseline parameters...")
        best_params = {
            "learning_rate": 0.03,
            "max_depth": 5,
            "n_estimators": 700
        }
        best_cv_auc = 0.7439  # Known CV score from original baseline run
        best_model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            scale_pos_weight=scale_pos_weight_val,
            learning_rate=best_params["learning_rate"],
            max_depth=best_params["max_depth"],
            n_estimators=best_params["n_estimators"],
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            n_jobs=2
        )
        best_model.fit(X_train, y_train)

    # Evaluate predictions
    y_train_prob = best_model.predict_proba(X_train)[:, 1]
    y_val_prob = best_model.predict_proba(X_val)[:, 1]
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = best_model.predict(X_test)

    train_auc = float(roc_auc_score(y_train, y_train_prob))
    val_auc = float(roc_auc_score(y_val, y_val_prob))
    test_auc = float(roc_auc_score(y_test, y_test_prob))

    print(f"\nEvaluation Results:")
    print(f"  Train ROC-AUC: {train_auc:.4f}")
    print(f"  Validation ROC-AUC: {val_auc:.4f}")
    print(f"  Test ROC-AUC: {test_auc:.4f}")

    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Save outputs
    print(f"Saving model to {model_output_path}...")
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_output_path)

    # Prepare and save metadata
    model_metadata = {
        'model_name': 'xgboost_baseline',
        'model_type': 'XGBClassifier',
        'selection_metric': 'validation_roc_auc',
        'best_params': best_params,
        'best_cv_roc_auc': best_cv_auc,
        'train_roc_auc': train_auc,
        'validation_roc_auc': val_auc,
        'test_roc_auc': test_auc,
        'feature_count': int(X_train.shape[1]),
        'training_rows': int(X_train.shape[0]),
        'validation_rows': int(X_val.shape[0]),
        'test_rows': int(X_test.shape[0]),
        'positive_class': 'Charged Off',
        'negative_class': 'Fully Paid',
        'llm_guardrail': 'The LLM never approves or rejects loans; it only summarizes evidence for a human underwriter.',
    }

    print(f"Saving metadata to {model_metadata_path}...")
    with open(model_metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)

    print(f"Saving test features to {test_output_path}...")
    X_test.to_parquet(test_output_path)

    print("Model training pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LendingClub XGBoost Risk Model.")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter search via GridSearchCV")
    args = parser.parse_args()

    train_model(tune=args.tune)
