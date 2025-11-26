# ============================================================================
# target_informed_model.py
# ============================================================================
# Target-informed regression model that shrinks coefficients toward targets.
# Supports both Ridge and Logistic regression with target-informed regularization.
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, mean_squared_error
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from itertools import groupby
from operator import itemgetter 
import joblib
import json 
import re
import openai
from dotenv import load_dotenv
import os
load_dotenv()


class CovarianceTargetInformedModel(BaseEstimator):
    """
    Target-informed regression model that shrinks coefficients toward targets t.
    
    Ridge objective:
        min_β  ||y - Xβ||² + α ||β - t||²           [intercept unpenalized]
    
    Logistic objective:
        min_β  -log L(β) + (α/2)||β - t||²          [intercept unpenalized]
    
    Parameters
    ----------
    alpha : float (default 1.0)
        Strength of shrinkage toward targets t.
    model_type : {'ridge', 'logistic'} (default 'ridge')
        Type of regression model.
    fit_intercept : bool (default True)
        Whether to fit intercept (never penalized).
    random_state : int or None
        Random seed for logistic initialization.
    targets : array-like or None
        Target values for coefficients. If None, defaults to zeros.
    """
    
    def __init__(self, alpha=1.0, model_type='ridge', fit_intercept=True,
                 random_state=None, targets=None):
        self.alpha = float(alpha)
        self.model_type = model_type
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.targets = targets
        
        # Fitted attributes
        self.coef_ = None
        self.intercept_ = None
        self.targets_used_ = None
        self.feature_names_ = None
    
    # ---------- Utility methods ----------
    def _prepare_data(self, X, y):
        """Prepare and validate input data."""
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if y.ndim != 1:
            raise ValueError("y must be 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        return X, y
    
    # ---------- Ridge regression ----------
    def _solve_target_ridge(self, X, y, targets):
        """Solve ridge regression with target-informed regularization."""
        n, p = X.shape
        
        if self.fit_intercept:
            Xw = np.column_stack([np.ones(n), X])
            t_ext = np.concatenate([[0.0], targets])  # No target for intercept
            I_ext = np.eye(p + 1)
            I_ext[0, 0] = 0.0  # No penalty on intercept
        else:
            Xw = X
            t_ext = targets
            I_ext = np.eye(p)
        
        XtX = Xw.T @ Xw
        Xty = Xw.T @ y
        
        lhs = XtX + self.alpha * I_ext
        rhs = Xty + self.alpha * t_ext
        
        try:
            beta_ext = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta_ext = np.linalg.pinv(lhs) @ rhs
        
        if self.fit_intercept:
            self.intercept_ = float(beta_ext[0])
            self.coef_ = beta_ext[1:].astype(float)
        else:
            self.intercept_ = 0.0
            self.coef_ = beta_ext.astype(float)
        
        return self.coef_
    
    # ---------- Logistic regression ----------
    def _solve_target_logistic(self, X, y, targets):
        """Solve logistic regression with target-informed regularization."""
        n, p = X.shape
        
        if self.fit_intercept:
            Xw = np.column_stack([np.ones(n), X])
            t_ext = np.concatenate([[0.0], targets])
            d = p + 1
            mask = np.ones(d, dtype=float)
            mask[0] = 0.0  # No penalty on intercept
        else:
            Xw = X
            t_ext = targets
            d = p
            mask = np.ones(d, dtype=float)
        
        def objective(beta):
            z = np.clip(Xw @ beta, -500, 500)
            loglik = np.sum(y * z - np.logaddexp(0, z))
            diff = (beta - t_ext) * mask
            pen_t = 0.5 * self.alpha * np.dot(diff, diff)
            return -loglik + pen_t
        
        def gradient(beta):
            z = np.clip(Xw @ beta, -500, 500)
            p = 1.0 / (1.0 + np.exp(-z))
            grad_nll = -Xw.T @ (y - p)
            diff = (beta - t_ext) * mask
            grad = grad_nll + self.alpha * diff
            return grad
        
        # Initialize
        if self.random_state is not None:
            np.random.seed(self.random_state)
            beta0 = 0.01 * np.random.randn(d)
        else:
            beta0 = np.zeros(d)
        
        # Optimize
        res = minimize(
            objective, beta0, jac=gradient, method="L-BFGS-B",
            options={'maxiter': 1000}
        )
        
        if not res.success:
            print(f"Warning: Optimization did not converge. Message: {res.message}")
        
        beta = res.x
        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:].astype(float)
        else:
            self.intercept_ = 0.0
            self.coef_ = beta.astype(float)
        
        return self.coef_
    
    # ---------- Public API ----------
    def fit(self, X, y, feature_names=None, targets=None):
        """
        Fit the target-informed model.
        
        Args:
            X : array-like (n_samples, n_features)
                Training features.
            y : array-like (n_samples,)
                Target values.
            feature_names : list[str] or None
                Names of features.
            targets : array-like or None
                Target values for coefficients.
        
        Returns:
            self : object
                Fitted model.
        """
        X, y = self._prepare_data(X, y)
        n, p = X.shape
        
        # Set feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(p)]
        elif len(feature_names) != p:
            raise ValueError(f"feature_names must have {p} elements")
        self.feature_names_ = feature_names
        
        # Set targets
        if targets is None:
            targets = np.zeros(p) if self.targets is None else self.targets
        targets = np.asarray(targets, dtype=float)
        if len(targets) != p:
            raise ValueError(f"targets must have {p} elements")
        self.targets_used_ = targets
        
        # Fit model
        if self.model_type == "ridge":
            self._solve_target_ridge(X, y, targets)
        elif self.model_type == "logistic":
            # Ensure binary classification
            unique = np.unique(y)
            if len(unique) != 2:
                raise ValueError("For logistic regression, y must be binary")
            if not np.all(np.isin(unique, [0, 1])):
                y = (y == unique[1]).astype(float)
            self._solve_target_logistic(X, y, targets)
        else:
            raise ValueError("model_type must be 'ridge' or 'logistic'")
        
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X : array-like
                Input features.
        
        Returns:
            y_pred : array
                Predictions (continuous for ridge, probabilities for logistic).
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        linear_pred = X @ self.coef_ + self.intercept_
        
        if self.model_type == "ridge":
            return linear_pred
        elif self.model_type == "logistic":
            z = np.clip(linear_pred, -500, 500)
            return 1.0 / (1.0 + np.exp(-z))
    
    def predict_proba(self, X):
        """
        Predict probability for binary classification.
        
        Args:
            X : array-like
                Input features.
        
        Returns:
            proba : array (n_samples, 2)
                Class probabilities.
        """
        if self.model_type != "logistic":
            raise ValueError("predict_proba only available for logistic regression")
        
        p1 = self.predict(X)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
    
    def score(self, X, y):
        """
        Calculate model score.
        
        Returns:
            score : float
                R² for ridge, accuracy for logistic.
        """
        if self.model_type == "ridge":
            return r2_score(y, self.predict(X))
        elif self.model_type == "logistic":
            y_pred = (self.predict(X) > 0.5).astype(int)
            return accuracy_score(y, y_pred)
    
    def get_coefficient_summary(self):
        """
        Get summary of coefficients and their targets.
        
        Returns:
            summary : pd.DataFrame
                DataFrame with feature names, targets, coefficients, and adjustments.
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted first")
        
        return pd.DataFrame({
            "feature": self.feature_names_,
            "target": self.targets_used_,
            "coefficient": self.coef_,
            "adjustment": self.coef_ - self.targets_used_,
        })
    
    def report(self):
        """Print model summary."""
        summary = self.get_coefficient_summary()
        print("\n--- Model summary ---")
        print(f"Model type: {self.model_type}")
        print(f"Alpha: {self.alpha}")
        print(f"Intercept: {self.intercept_:.4f}")
        print("\nCoefficients:")
        print(summary.to_string(index=False))
    
    def save_model(self, path):
        """Save model to disk."""
        joblib.dump(self, path)
        print(f"Saved model → {path}")
    
    @staticmethod
    def load_model(path):
        """Load model from disk."""
        return joblib.load(path)


# ---------------------------------------------------------------------
# Cross-validation utilities
# ---------------------------------------------------------------------
def cross_validate_alpha(X, y, targets, alphas, model_type="ridge", 
                         n_splits=5, random_state=42):
    """
    Cross-validate to select best alpha value.
    
    Args:
        X : array-like
            Features.
        y : array-like
            Target values.
        targets : array-like
            Target coefficients.
        alphas : array-like
            Alpha values to test.
        model_type : str
            'ridge' or 'logistic'.
        n_splits : int
            Number of CV folds.
        random_state : int
            Random seed.
    
    Returns:
        best_alpha : float
            Best alpha value.
        scores : dict
            Scores for each alpha.
    """
    scores = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for alpha in alphas:
        fold_scores = []
        
        for train_idx, val_idx in kf.split(X):
            model = CovarianceTargetInformedModel(
                alpha=alpha, 
                model_type=model_type, 
                targets=targets,
                random_state=random_state
            )
            
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            
            if model_type == "ridge":
                metric = -mean_squared_error(y[val_idx], preds)
            else:  # logistic
                metric = roc_auc_score(y[val_idx], preds)
            
            fold_scores.append(metric)
        
        scores[alpha] = np.mean(fold_scores)
    
    best_alpha = max(scores, key=scores.get)
    return best_alpha, scores


def fit_and_evaluate(X_train, y_train, X_test, y_test, targets, 
                     alpha=1.0, model_type="ridge", feature_names=None):
    """
    Fit model and evaluate on test set.
    
    Args:
        X_train, y_train : arrays
            Training data.
        X_test, y_test : arrays
            Test data.
        targets : array-like
            Target coefficients.
        alpha : float
            Regularization strength.
        model_type : str
            'ridge' or 'logistic'.
        feature_names : list[str] or None
            Feature names.
    
    Returns:
        model : TargetInformedModel
            Fitted model.
        metrics : dict
            Evaluation metrics.
    """
    model = CovarianceTargetInformedModel(
        alpha=alpha, 
        model_type=model_type,
        targets=targets
    )
    
    model.fit(X_train, y_train, feature_names=feature_names)
    
    if model_type == "ridge":
        y_pred = model.predict(X_test)
        metrics = {
            "r2": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
        }
    else:  # logistic
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba)
        }
    
    return model, metrics

import os
import openai

def call_llm_api(prompt: str, api_key: str, api_provider: str, model_name):
    try:
        if api_provider == 'openai':
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content

        elif api_provider == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model_name,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

    except Exception as e:
        print(f"API call failed: {e}")
        return None


def rephrase_prompt(prompt: str):
    api_key = os.getenv("API_KEY")
    client = openai.OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": "Rewrite the prompt clearly while keeping all constraints exactly the same."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Rephrase failed: {e}")
        return prompt

import re
import json

def extract_targets(text: str):
    # 1. Prefer fenced blocks with json
    print("\n--- RAW LLM RESPONSE ---")
    print(text)
    print("--- END RESPONSE ---\n")

    fenced_json = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_json:
        candidate = fenced_json.group(1).strip()
        try:
            data = json.loads(candidate)
            if "targets" in data:
                return data["targets"]
        except json.JSONDecodeError:
            pass  # Fallback below
    
    # 2. Fallback: any fenced block ``` ... ```
    fenced = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            data = json.loads(candidate)
            if "targets" in data:
                return data["targets"]
        except json.JSONDecodeError:
            pass  # Next fallback
    
    # 3. Fallback: search for inline JSON object containing "targets"
    inline = re.search(r"\{[^{}]*\"targets\"[^{}]*\}", text, flags=re.DOTALL)
    if inline:
        # Try to load the entire object
        candidate = inline.group(0)
        try:
            data = json.loads(candidate)
            if "targets" in data:
                return data["targets"]
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON with a 'targets' object found.")




def sampling_llm(num_calls: int, base_prompt: str, model_name: str, api_provider: str, api_key: str):
    targets = []
    for _ in range(num_calls):
        #rephrased_prompt = rephrase_prompt(base_prompt)
        response = call_llm_api(
            prompt=base_prompt,
            api_key=api_key,
            api_provider=api_provider,
            model_name=model_name
        )
        targets.append(extract_targets(response))
    return targets


# Example
sample_prompt = """
You are an API, not a conversational assistant.
Your only job is to produce pure JSON and nothing else.
Do NOT include explanations, greetings, summaries, bullet points, or any text outside JSON.

Given these feature names:
- bmi
- height
- weight

Return a JSON object with numeric coefficient targets for a regularised linear model.
Use floating-point numbers without units.

Output must be EXACTLY and ONLY JSON format like this:

{
  "targets": {
    "bmi": <number>,
    "height": <number>,
    "weight": <number>
  }
}

Replace <number> with actual numeric values. Do not change keys. Do not add text before or after.
"""



sampled_response = sampling_llm(
    num_calls=3,
    base_prompt=sample_prompt,
    model_name="gpt-4o",
    api_provider="openai",
    api_key=os.getenv("API_KEY")
)

print(sampled_response)



result_sum = {}


for d in sampled_response:
    for key, value in d.items():
        result_sum[key] = result_sum.get(key, 0) + value

n = len(sampled_response)
result_mean = {key: total / n for key, total in result_sum.items()}

print("sum:", result_sum)
print("mean:", result_mean)