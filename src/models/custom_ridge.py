import numpy as np
from typing import Optional, Union, Tuple
import warnings
import pandas as pd


def sigmoid(z):
    """Stable sigmoid function."""
    # Clip z to prevent overflow
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))


def log_loss(y_true, y_pred, eps=1e-15):
    """Binary log-loss with numerical stability."""
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class CustomRidge:
    def __init__(self, alpha=1.0, fit_intercept=True, alpha2=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.alpha2 = alpha2
        self.is_fitted_ = False

    def fit(self, X_array, y_array, targets=None):
        if self._validate_arrays(X_array, y_array, targets) is False:
            print("Arrays are invalid")
            return 0
        
        n_samples, n_features = X_array.shape
        if self.fit_intercept is True:
            X_centered, y_centered, X_mean, y_mean = self._center_data(X_array, y_array)
        else:
            X_centered = X_array
            y_centered = y_array 
            X_mean = np.zeros(n_features)
            y_mean = 0

        # Solve the ridge system on centered data
        coef = self._solve_ridge_system(X_centered, y_centered, targets)
        
        # Compute the intercept 
        if self.fit_intercept is True:
            intercept = self._compute_intercept(X_mean, y_mean, coef)
        else:
            intercept = 0
        
        # Store results 
        self.coef_ = coef 
        self.intercept_ = intercept
        
        # Fixed targets copying logic
        if targets is not None:
            self.targets_used_ = np.asarray(targets)  # Convert list to array
        else:
            self.targets_used_ = np.zeros(n_features)
            
        self.is_fitted_ = True
        return self
    
    def predict(self, X_array):
        if not self.is_fitted_:
            raise ValueError("Must call fit() before predict()")
        if X_array.shape[1] != len(self.coef_):
            raise ValueError("Wrong number of features")
        # Linear prediction y = X @ coef + intercept 
        return X_array @ self.coef_ + self.intercept_
    
    def _solve_ridge_system(self, X, y, targets):
        n_features = X.shape[1]
        XtX = np.transpose(X) @ X
        regularisation_matrix = XtX + self.alpha * np.identity(n_features)

        Xty = np.transpose(X) @ y

        if targets is not None:
            targets_array = np.asarray(targets)  # Ensure it's numpy array
            rhs = Xty + self.alpha * targets_array
        else:
            rhs = Xty  # standard ridge (shrink toward zero)

        try:
            coef = np.linalg.solve(regularisation_matrix, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(regularisation_matrix) @ rhs
        return coef
    
    def _validate_arrays(self, X_array, y_array, targets=None):
        # Convert pandas DataFrame/Series to numpy arrays
        if isinstance(X_array, pd.DataFrame) or isinstance(X_array, pd.Series):
            X_array = X_array.values
        if isinstance(y_array, pd.Series):
            y_array = y_array.values

        # Check type
        if not isinstance(X_array, np.ndarray) or not isinstance(y_array, np.ndarray):
            raise TypeError("X_array and y_array must be numpy arrays or pandas objects.")

        # Check shapes
        if X_array.ndim != 2:
            raise ValueError(f"X_array must be 2D, got shape {X_array.shape}")
        if y_array.ndim not in [1, 2]:
            raise ValueError(f"y_array must be 1D or 2D, got shape {y_array.shape}")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError(f"Number of samples in X ({X_array.shape[0]}) and y ({y_array.shape[0]}) do not match.")

        # Check for missing values
        if np.isnan(X_array).any() or np.isnan(y_array).any():
            raise ValueError("X_array and y_array must not contain NaNs.")
        
        # Handle targets validation
        if targets is not None:
            targets = np.asarray(targets)  # Convert to array for validation
            if np.isnan(targets).any() or targets.shape[0] != X_array.shape[1]:
                raise ValueError("targets must be 1D array of length equal to number of features, and contain no NaNs.")

        return True
    
    def _compute_intercept(self, X_mean, y_mean, coef):
        return y_mean - np.dot(X_mean, coef)
    
    def _center_data(self, X, y):
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean
        return X_centered, y_centered, X_mean, y_mean
    
    # Research and analysis methods
    def get_loss_components(self, X, y):
        """Return MSE loss, regularization loss, and total loss components."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before computing loss.")
        
        y_pred = self.predict(X)
        residuals = y - y_pred
        mse_loss = np.mean(residuals**2)  # Fixed: ** instead of ^
        
        # Compute regularization loss
        if self.targets_used_ is not None and np.any(self.targets_used_ != 0):
            # Target-informed: α||β - targets||²
            reg_loss = self.alpha * np.sum((self.coef_ - self.targets_used_)**2)  # Fixed: ** instead of ^
        else:
            # Traditional: α||β||²
            reg_loss = self.alpha * np.sum(self.coef_**2)  # Fixed: ** instead of ^
        
        total_loss = mse_loss + reg_loss
        
        return {
            'mse_loss': mse_loss,
            'regularization_loss': reg_loss, 
            'total_loss': total_loss
        }
    
    def get_regularization_matrix(self, X):
        """Return (X^T X + αI) matrix for mathematical inspection."""
        n_features = X.shape[1]
        XtX = np.transpose(X) @ X
        regularization_matrix = XtX + self.alpha * np.identity(n_features) 
        return regularization_matrix
    
    def _compute_objective_value(self, X, y, targets=None):
        """Compute current objective function value."""
        if not self.is_fitted_:
            raise ValueError("Must fit model before computing objective value")
        
        # MSE component
        y_pred = self.predict(X)
        mse_component = np.mean((y - y_pred)**2)  # Fixed: ** instead of ^
        
        # Regularization component 
        if targets is not None:
            targets_array = np.asarray(targets)
            reg_component = self.alpha * np.sum((self.coef_ - targets_array)**2)  # Fixed: ** instead of ^
        else:
            reg_component = self.alpha * np.sum(self.coef_**2)  # Fixed: ** instead of ^

        objective_value = mse_component + reg_component
        return objective_value
    
    def get_coefficient_path(self, X, y, targets, alphas):
        """Compute regularization path for different alpha values."""
        path_results = []

        for alpha in alphas:
            temp_model = CustomRidge(alpha=alpha, fit_intercept=self.fit_intercept)
            temp_model.fit(X, y, targets=targets)
            
            result = {
                'alpha': alpha,
                'coefficients': np.copy(temp_model.coef_),
                'intercept': temp_model.intercept_
            }
            
            # Add loss components
            loss_components = temp_model.get_loss_components(X, y)
            result.update(loss_components)
            
            path_results.append(result)
            
        return path_results
    
    def score(self, X, y):
        """Return R² score for model evaluation."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before scoring")

        # Get predictions
        y_pred = self.predict(X)
        
        # Compute R² = 1 - SS_res/SS_tot
        SS_res = np.sum((y - y_pred)**2)           # Fixed: ** instead of ^
        SS_tot = np.sum((y - np.mean(y))**2)       # Fixed: ** instead of ^ and correct formula

        if SS_tot == 0:                            # Edge case: constant y
            if SS_res == 0:
                return 1.0                         # Perfect prediction
            else:
                return 0.0                         # Undefined, return 0

        r_squared = 1 - (SS_res / SS_tot)
        return r_squared