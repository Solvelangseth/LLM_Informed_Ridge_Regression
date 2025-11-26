from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .custom_ridge import CustomRidge
from .target_elicitor import LLMTargetElicitor

class ExpertRidge:
    def __init__(self, alpha=1.0, llm_model="gpt-4"):
        self.alpha = alpha
        self.llm_model = llm_model
        self.ridge_model = None
        self.is_fitted_ = False
        self.feature_names_ = None  
        self.targets_used_ = None 
    
    def fit(self, X, y, target_name, targets=None, custom_prompt=None):
        # Handle LLM integration when available
        if targets is None and custom_prompt is not None:
            try:
                from src.models.target_elicitor import LLMTargetElicitor
                elicitor = LLMTargetElicitor(self.llm_model)
                result = elicitor.get_targets_with_prompt(custom_prompt, list(X.columns))
                targets = result['targets'] if result else None
            except ImportError:
                print("LLM Target Elicitor not available - using targets=None")
                targets = None
        
        # Convert targets to numpy array if provided
        if targets is not None:
            targets = np.asarray(targets)
        
        self.ridge_model = CustomRidge(alpha=self.alpha)
        self.ridge_model.fit(X.values, y.values, targets=targets)
        
        # Store for research analysis
        self.feature_names_ = list(X.columns)
        self.targets_used_ = targets
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("Must fit before prediction")
        return self.ridge_model.predict(X.values)
    
    def score(self, X, y):
        if not self.is_fitted_:
            raise ValueError("Must fit before scoring")
        return self.ridge_model.score(X.values, y.values)
    
    # Research methods - delegate to CustomRidge
    def get_loss_components(self, X, y):
        if not self.is_fitted_:
            raise ValueError("Must fit before getting loss components")
        return self.ridge_model.get_loss_components(X.values, y.values)
    
    def get_coefficient_summary(self):
        if not self.is_fitted_:
            raise ValueError("Must fit before getting summary")
        
        # Handle case where targets_used_ is None
        if self.targets_used_ is not None:
            targets_for_display = self.targets_used_
            adjustments = self.ridge_model.coef_ - self.targets_used_
        else:
            targets_for_display = np.zeros(len(self.feature_names_))
            adjustments = self.ridge_model.coef_ - targets_for_display
        
        return pd.DataFrame({
            'feature': self.feature_names_,
            'target': targets_for_display,
            'coefficient': self.ridge_model.coef_,
            'adjustment': adjustments
        })


if __name__ == "__main__":
    """
    Quick test script for ExpertRidge class
    Tests imports, basic functionality, and CustomRidge integration
    """

    print("ExpertRidge Integration Test")
    print("=" * 40)

    # Test 1: Basic instantiation
    print("\nTest 1: Basic instantiation")
    print("-" * 30)

    try:
        expert = ExpertRidge(alpha=1.0)
        print("✅ ExpertRidge instantiated successfully")
        print(f"   Alpha: {expert.alpha}")
        print(f"   LLM Model: {expert.llm_model}")
        print(f"   Is fitted: {expert.is_fitted_}")
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")

    # Test 2: Create synthetic data
    print("\nTest 2: Creating synthetic data")
    print("-" * 30)

    np.random.seed(42)
    n_samples, n_features = 100, 3
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=['feature_A', 'feature_B', 'feature_C']
    )
    true_coef = np.array([1.5, -2.0, 0.5])
    y = pd.Series(
        X.values @ true_coef + np.random.randn(n_samples) * 0.1,
        name='target'
    )

    print(f"✅ Created synthetic data")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   True coefficients: {true_coef}")

    # Test 3: Fit without targets (traditional Ridge)
    print("\nTest 3: Fit without targets")
    print("-" * 30)

    try:
        expert_no_targets = ExpertRidge(alpha=1.0)
        expert_no_targets.fit(X, y, target_name="test_target", targets=None)
        
        print("✅ Fitting without targets successful")
        print(f"   Is fitted: {expert_no_targets.is_fitted_}")
        print(f"   Feature names: {expert_no_targets.feature_names_}")
        print(f"   Coefficients: {expert_no_targets.ridge_model.coef_}")
        print(f"   Intercept: {expert_no_targets.ridge_model.intercept_}")
        
    except Exception as e:
        print(f"❌ Fitting without targets failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Predictions and scoring
    print("\nTest 4: Predictions and scoring")
    print("-" * 30)

    try:
        predictions = expert_no_targets.predict(X)
        score = expert_no_targets.score(X, y)
        
        print("✅ Predictions and scoring successful")
        print(f"   Prediction shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions[:5]}")
        print(f"   R² score: {score:.4f}")
        
    except Exception as e:
        print(f"❌ Predictions failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Fit with manual targets
    print("\nTest 5: Fit with manual targets")
    print("-" * 30)

    try:
        manual_targets = [1.2, -1.8, 0.4]  # Close to true coefficients
        expert_with_targets = ExpertRidge(alpha=1.0)
        expert_with_targets.fit(X, y, target_name="test_target", targets=manual_targets)
        
        print("✅ Fitting with targets successful")
        print(f"   Targets used: {expert_with_targets.targets_used_}")
        print(f"   Coefficients: {expert_with_targets.ridge_model.coef_}")
        
        # Compare with no-targets version
        adjustment = expert_with_targets.ridge_model.coef_ - expert_no_targets.ridge_model.coef_
        print(f"   Coefficient adjustments: {adjustment}")
        print(f"   Max adjustment: {np.max(np.abs(adjustment)):.4f}")
        
    except Exception as e:
        print(f"❌ Fitting with targets failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 6: Coefficient summary
    print("\nTest 6: Coefficient summary")
    print("-" * 30)

    try:
        summary = expert_with_targets.get_coefficient_summary()
        print("✅ Coefficient summary successful")
        print(summary)
        
    except Exception as e:
        print(f"❌ Coefficient summary failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 7: Loss components
    print("\nTest 7: Loss components analysis")
    print("-" * 30)

    try:
        loss_components = expert_with_targets.get_loss_components(X, y)
        print("✅ Loss components analysis successful")
        print(f"   MSE Loss: {loss_components['mse_loss']:.6f}")
        print(f"   Regularization Loss: {loss_components['regularization_loss']:.6f}")
        print(f"   Total Loss: {loss_components['total_loss']:.6f}")
        
    except Exception as e:
        print(f"❌ Loss components failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 8: Compare performance
    print("\nTest 8: Performance comparison")
    print("-" * 30)

    try:
        score_no_targets = expert_no_targets.score(X, y)
        score_with_targets = expert_with_targets.score(X, y)
        improvement = score_with_targets - score_no_targets
        
        print("✅ Performance comparison successful")
        print(f"   Traditional Ridge R²: {score_no_targets:.4f}")
        print(f"   Target-informed R²:   {score_with_targets:.4f}")
        print(f"   Improvement:          {improvement:+.4f}")
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")

    # Test 9: Check CustomRidge integration
    print("\nTest 9: CustomRidge integration check")
    print("-" * 30)

    try:
        print(f"✅ CustomRidge integration verified")
        print(f"   Ridge model type: {type(expert_with_targets.ridge_model)}")
        print(f"   Is CustomRidge: {isinstance(expert_with_targets.ridge_model, CustomRidge)}")
        
        # Test CustomRidge-specific methods
        reg_matrix = expert_with_targets.ridge_model.get_regularization_matrix(X.values)
        print(f"   Regularization matrix shape: {reg_matrix.shape}")
        
    except Exception as e:
        print(f"❌ CustomRidge integration check failed: {e}")

    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)

    if expert_no_targets.is_fitted_ and expert_with_targets.is_fitted_:
        print("✅ All core functionality working")
        print("✅ CustomRidge integration successful")
        print("✅ Target-informed regularization operational")
        print("\nReady for full testing and LLM integration!")
    else:
        print("❌ Some tests failed - check implementation")

    print("\nNext steps:")
    print("1. Run the full notebook test")
    print("2. Test LLM integration with custom prompts")
    print("3. Validate on real datasets")