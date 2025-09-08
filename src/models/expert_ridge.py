from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llm.prior_elicitor import LLMPriorElicitor

class ExpertRidge:
  """
  Ridge Regression that incorporates expert priors from llm experts
  """

  def __init__(self, alpha: float = 1.0, llm_model: str = "gpt-4", validate_priors: bool = True):
    """
    Initialize expert ridge 
    Parameters:
    alpha: float 
      base ridge regularisation parameter 
    llm_model : str 
      LLM model for prior eliccitation 
    validate priors : bool 
      Whether to validate the priors before using them 
    """
    self.alpha = alpha 
    self.validate_priors = validate_priors
    self.llm_model = llm_model

    # Initialise components 
    #self.llm_elicitor = LLMPriorElicitor(model_name=llm_model)
    #self.prior_validator = PriorValidator() if validate_priors else None 


    self.feature_names_ = None
    self.target_name_ = None
    self.coefficients_ = None
    self.priors_used_ = None
    self.llm_response_ = None
    self.validation_results_ = None
    self.is_fitted_ = False



  def _validate_input_data(self, X: pd.DataFrame, y: pd.Series):

    # Type checking
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series")
    
    # Dimension checking
    if X.shape[0] != len(y):
        raise ValueError(f"X has {X.shape[0]} samples but y has {len(y)} samples")
    
    if X.shape[0] == 0:
        raise ValueError("Empty dataset")
        
    if X.shape[1] == 0:
        raise ValueError("No features in X")
    
    return True
  

  def _convert_to_arrays(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
     X_array = X.values
     y_array = y.values

     return X_array, y_array
  
  def _get_mock_priors(self, feature_names: List[str]) -> List[float]:
      # Return different number based on actual features
      return [200.0] * len(feature_names) 
    
  def _fit_ridge_with_priors(self, X, y, mu, lambda_reg):
    # Transform y with the mu from the llm 
    y_tilde = y - X @ mu
    # Fit ridge regression
    ridge = Ridge(alpha=lambda_reg, fit_intercept=False)
    ridge.fit(X, y_tilde)

    beta = ridge.coef_ + mu

    return beta


  def _check_is_fitted(self):
     fitted_attributes = ['coefficients_', 'feature_names_', 'priors_used_']

     for attr in fitted_attributes:
        if getattr(self, attr, None) is None:
          raise ValueError(f"This ExpertRidge instance is not fitted yet. "
                           f"Call 'fit' before using this method.")
        
  def fit(self, X: pd.DataFrame, y: pd.Series, target_name: str, domain: Optional[str] = None, use_llm: bool = True, custom_priors=None) -> 'ExpertRidge':
    """
    Fit ridge regression with LLM-generated priors
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature data
    y : pd.Series
        Target data  
    target_name : str
        Name of target variable
    domain : str, optional
        Domain context for LLM
    use_llm : bool, default=True
        If True, use LLM for priors. If False, use mock priors.
    """
    print("=== FITTING EXPERT RIDGE ===")
    
    # Validate input
    self._validate_input_data(X, y)
    print("✅ Input validation passed")

    # Convert to arrays & store feature names
    X_array, y_array = self._convert_to_arrays(X, y)
    print(f"✅ Converted to arrays: X{X_array.shape}, y{y_array.shape}")

    if custom_priors is not None:
      print("Using provided custom priors")
      priors = custom_priors
      self.llm_response_ = {'priors': custom_priors, 'source': 'custom'}
      # Get coeficcients
    elif use_llm:
        print("Getting coefficients from LLM...")
        try:
            # Create LLM elicitor
            llm_elicitor = LLMPriorElicitor(model_name=self.llm_model)
            
            # Get priors from LLM
            llm_response = llm_elicitor.get_priors(X, y, target_name, domain)
            
            if llm_response:
                priors = llm_response['priors']
                self.llm_response_ = llm_response
                print(f"✅ Got LLM priors: {priors}")
                print(f"   Domain identified: {llm_response.get('domain', 'unknown')}")
            else:
                print("⚠️ LLM failed, using mock priors")
                priors = self._get_mock_priors(list(X.columns))
                self.llm_response_ = None
                
        except Exception as e:
            print(f"⚠️ LLM integration failed: {e}")
            print("   Falling back to mock priors")
            priors = self._get_mock_priors(list(X.columns))
            self.llm_response_ = None
    else:
        print("📝 Using mock priors (LLM disabled)")
        priors = self._get_mock_priors(list(X.columns))
        self.llm_response_ = None

    print(f"   Final priors used: {priors}")

    # Validate priors (implement later)
    # if self.validate_priors:
    #     validation = self.prior_validator.validate_all(...)

    # Adjust alpha based on confidence (implement later)
    # adjusted_alpha = self._adjust_alpha_by_confidence(self.alpha, "medium")

    # Step 6: Fit ridge with priors
    print("🔧 Fitting ridge regression with priors...")
    coeffs = self._fit_ridge_with_priors(X_array, y_array, np.array(priors), self.alpha)
    print(f"✅ Ridge fitting complete")

    # Store results
    self.coefficients_ = coeffs
    self.priors_used_ = np.array(priors) 
    self.feature_names_ = list(X.columns)
    self.target_name_ = target_name
    self.is_fitted_ = True

    print("=== FITTING COMPLETE ===")
    print(f"   Features: {self.feature_names_}")
    print(f"   Priors:   {self.priors_used_}")
    print(f"   Final coefficients: {self.coefficients_}")
    print(f"   Adjustments: {self.coefficients_ - self.priors_used_}")

    return self
  
  def predict(self, X: pd.DataFrame) -> np.ndarray:
     self._check_is_fitted()

     X_array = X[self.feature_names_].values 

     return X_array @ self.coefficients_
  
  def get_coefficient_summary(self) -> pd.DataFrame:
    # Return DataFrame with:
    # ['feature', 'llm_prior', 'final_coefficient', 'adjustment']
    self._check_is_fitted()
    
    return pd.DataFrame({
        'feature': self.feature_names_,
        'llm_prior': self.priors_used_,
        'final_coefficient': self.coefficients_,
        'adjustment': self.coefficients_ - self.priors_used_
    })


