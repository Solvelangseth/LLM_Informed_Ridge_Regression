import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.expert_ridge import ExpertRidge
from llm.prior_elicitor import LLMPriorElicitor
import pandas as pd
import numpy as np

def test_expert_ridge():
    """Test ExpertRidge with both LLM and mock priors"""
    
    # Create test data
    X = pd.DataFrame({
        'size': [1.0, 2.0, 3.0, 1.5, 2.5],
        'rooms': [1, 2, 3, 2, 3]
    })
    y = pd.Series([100, 200, 300, 150, 250])

    print("=== TESTING EXPERTRIDGE WITH LLM INTEGRATION ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Test with LLM
    print("\n" + "="*50)
    print("TEST 1: WITH LLM")
    print("="*50)
    
    model = ExpertRidge(alpha=1.0, llm_model="gpt-4")
    model.fit(X, y, target_name="price", use_llm=True)
    
    predictions = model.predict(X)
    summary = model.get_coefficient_summary()
    
    print("\nCoefficient Summary:")
    print(summary)
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    test_expert_ridge()