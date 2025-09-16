#!/usr/bin/env python3
"""
Test script for nested cross-validation implementation
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.hyperparameter_tuning import HyperparameterTuner

def test_nested_cv():
    """Test the nested cross-validation implementation."""
    print("="*60)
    print("NESTED CROSS-VALIDATION TEST")
    print("="*60)
    
    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Target distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Initialize tuner
    print("\n2. Initializing HyperparameterTuner...")
    tuner = HyperparameterTuner()
    
    # Test nested CV with smaller parameter grid for faster execution
    print("\n3. Running nested cross-validation...")
    
    # Smaller parameter grid for testing
    gb_param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 4],
        'random_state': [42]
    }
    
    xgb_param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 4],
        'random_state': [42]
    }
    
    # Test individual nested CV
    print("\n--- Testing Gradient Boosting Nested CV ---")
    gb_results = tuner.nested_cross_validation(
        X_train, y_train, 'Gradient Boosting', gb_param_grid,
        cv_outer=3, cv_inner=3, n_jobs=1  # Smaller CV for testing
    )
    
    print("\n--- Testing XGBoost Nested CV ---")
    xgb_results = tuner.nested_cross_validation(
        X_train, y_train, 'XGBoost', xgb_param_grid,
        cv_outer=3, cv_inner=3, n_jobs=1  # Smaller CV for testing
    )
    
    # Test full nested CV
    print("\n4. Running full nested CV for all models...")
    all_results = tuner.nested_cv_all_models(
        X_train, y_train,
        cv_outer=3, cv_inner=3, n_jobs=1  # Smaller CV for testing
    )
    
    print("\n5. Results Summary:")
    print("-" * 40)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        for metric, scores in results.items():
            print(f"  {metric.upper()}: {scores['test_mean']:.4f} (+/- {scores['test_std'] * 2:.4f})")
    
    print("\n" + "="*60)
    print("NESTED CROSS-VALIDATION TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return all_results

if __name__ == "__main__":
    try:
        results = test_nested_cv()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
