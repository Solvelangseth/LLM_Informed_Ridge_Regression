# ============================================================================
# llm_ensemble_utils.py
# ============================================================================
# Reusable utilities for ensemble learning with LLM predictions.
# Provides core functions for prediction, weight optimization, and evaluation.
# ============================================================================
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


# ---------------------------------------------------------------------
# 1. LLM-style logistic prediction
# ---------------------------------------------------------------------
def llm_predict_proba(X, beta, beta0=0.0, beta_scale=1.0):
    """
    Logistic regression predictor with optional coefficient scaling.
    
    Args:
        X : array-like (N, D)
            Input features.
        beta : array-like (D,)
            Coefficients.
        beta0 : float
            Intercept term.
        beta_scale : float or array-like
            Scaling factor for coefficients (default 1.0).
    
    Returns:
        proba : array (N,)
            Predicted probabilities.
    """
    X_arr = np.asarray(X)
    scale = np.asarray(beta_scale)
    z = beta0 + X_arr @ (np.asarray(beta) * scale)
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------
# 2. Ensemble probability prediction
# ---------------------------------------------------------------------
def ensemble_predict_proba(X, models, weights, llm_beta=None, llm_beta0=0.0, beta_scale=1.0):
    """
    Combine probabilities from base models and optional LLM model.
    
    Args:
        X : array-like
            Input features.
        models : list
            List of sklearn classifiers with predict_proba method.
        weights : array-like
            Ensemble weights (will be normalized to sum to 1).
        llm_beta : array-like or None
            LLM coefficients (if None, LLM not included).
        llm_beta0 : float
            LLM intercept.
        beta_scale : float or array-like
            Scaling factor for LLM coefficients.
    
    Returns:
        proba : array
            Ensemble predicted probabilities.
    """
    probs = [m.predict_proba(X)[:, 1] for m in models]
    
    if llm_beta is not None:
        probs.append(llm_predict_proba(X, llm_beta, llm_beta0, beta_scale))
    
    P = np.column_stack(probs)
    w = np.asarray(weights)
    w = np.maximum(w, 0)
    w = w / w.sum()
    
    p = np.clip(P @ w, 1e-12, 1 - 1e-12)
    return p


# ---------------------------------------------------------------------
# 3. Weight optimization
# ---------------------------------------------------------------------
def optimize_weights(probs_list, y_val, fixed_index=None, fixed_value=None, l2=0.0):
    """
    Optimize convex combination weights to minimize log loss.
    
    Args:
        probs_list : list of arrays
            Predicted probabilities from each model.
        y_val : array
            True labels.
        fixed_index : int or None
            Index of weight to fix (e.g., for LLM).
        fixed_value : float or None
            Value to fix the weight at.
        l2 : float
            L2 regularization strength.
    
    Returns:
        weights : array
            Optimal weights (sum to 1).
    """
    P = np.column_stack(probs_list)
    M = P.shape[1]
    w0 = np.ones(M) / M
    
    def objective(w):
        p = np.clip(P @ w, 1e-12, 1 - 1e-12)
        loss = log_loss(y_val, p)
        if l2 > 0:
            loss += l2 * np.sum(w**2)
        return loss
    
    bounds = [(0, 1)] * M
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    
    if fixed_index is not None and fixed_value is not None:
        cons.append({
            "type": "eq", 
            "fun": lambda w, i=fixed_index, v=fixed_value: w[i] - v
        })
    
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)
    
    if not res.success:
        print(f"Warning: optimization did not converge: {res.message}")
        w = np.maximum(w0, 0)
        return w / w.sum()
    
    return res.x


# ---------------------------------------------------------------------
# 4. Model evaluation metrics
# ---------------------------------------------------------------------
def evaluate_predictions(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate classification metrics.
    
    Args:
        y_true : array
            True labels.
        y_pred_proba : array
            Predicted probabilities.
        threshold : float
            Classification threshold.
    
    Returns:
        metrics : dict
            Dictionary with accuracy, AUC, and log loss.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred_proba),
        "logloss": log_loss(y_true, y_pred_proba),
    }
    
    return metrics


# ---------------------------------------------------------------------
# 5. Train-validation split helper
# ---------------------------------------------------------------------
def split_for_validation(X_train, y_train, val_size=0.25, random_state=42):
    """
    Split training data into train and validation sets.
    
    Args:
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
        val_size : float
            Proportion for validation set.
        random_state : int
            Random seed.
    
    Returns:
        X_sub, X_val, y_sub, y_val : arrays
            Split datasets.
    """
    return train_test_split(
        X_train, y_train, 
        test_size=val_size, 
        random_state=random_state, 
        stratify=y_train
    )


# ---------------------------------------------------------------------
# 6. Fit ensemble models
# ---------------------------------------------------------------------
def fit_ensemble_models(models, X_train, y_train):
    """
    Fit all models in the ensemble.
    
    Args:
        models : list
            List of sklearn models.
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
    
    Returns:
        fitted_models : list
            List of fitted models.
    """
    for model in models:
        model.fit(X_train, y_train)
    return models


# ---------------------------------------------------------------------
# 7. Get predictions from all models
# ---------------------------------------------------------------------
def get_model_predictions(models, X, include_llm=False, llm_beta=None, 
                         llm_beta0=0.0, beta_scale=1.0):
    """
    Get predictions from all models including optional LLM.
    
    Args:
        models : list
            List of fitted sklearn models.
        X : array-like
            Input features.
        include_llm : bool
            Whether to include LLM predictions.
        llm_beta : array-like
            LLM coefficients.
        llm_beta0 : float
            LLM intercept.
        beta_scale : float
            LLM coefficient scaling.
    
    Returns:
        probs_list : list
            List of prediction arrays.
    """
    probs_list = [m.predict_proba(X)[:, 1] for m in models]
    
    if include_llm and llm_beta is not None:
        probs_list.append(llm_predict_proba(X, llm_beta, llm_beta0, beta_scale))
    
    return probs_list


# ---------------------------------------------------------------------
# 8. Run ensemble experiment
# ---------------------------------------------------------------------
def run_ensemble_experiment(X_train, y_train, X_test, y_test, models,
                           llm_beta=None, llm_beta0=0.0,
                           use_optimizer=True, manual_weights=None,
                           fixed_weight=None, l2=0.0, beta_scale=1.0,
                           val_size=0.25, random_state=42):
    """
    Complete ensemble experiment: train, optimize weights, evaluate.
    
    Args:
        X_train, y_train : arrays
            Training data.
        X_test, y_test : arrays
            Test data.
        models : list
            List of sklearn models.
        llm_beta : array-like or None
            LLM coefficients.
        llm_beta0 : float
            LLM intercept.
        use_optimizer : bool
            Whether to optimize weights.
        manual_weights : array-like or None
            Manual weights if not optimizing.
        fixed_weight : tuple or None
            (index, value) to fix a weight.
        l2 : float
            Regularization strength.
        beta_scale : float
            LLM coefficient scaling.
        val_size : float
            Validation set size.
        random_state : int
            Random seed.
    
    Returns:
        weights : array
            Final ensemble weights.
        metrics : dict
            Evaluation metrics.
    """
    # Split for validation
    X_sub, X_val, y_sub, y_val = split_for_validation(
        X_train, y_train, val_size, random_state
    )
    
    # Fit models
    models = fit_ensemble_models(models, X_sub, y_sub)
    
    # Get validation predictions
    probs_val = get_model_predictions(
        models, X_val, 
        include_llm=(llm_beta is not None),
        llm_beta=llm_beta, 
        llm_beta0=llm_beta0, 
        beta_scale=beta_scale
    )
    
    # Determine weights
    if use_optimizer:
        fixed_index = fixed_weight[0] if fixed_weight else None
        fixed_value = fixed_weight[1] if fixed_weight else None
        weights = optimize_weights(probs_val, y_val, fixed_index, fixed_value, l2=l2)
    else:
        weights = np.asarray(manual_weights)
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
    
    # Test predictions
    p_test = ensemble_predict_proba(
        X_test, models, weights, 
        llm_beta, llm_beta0, beta_scale
    )
    
    # Evaluate
    metrics = evaluate_predictions(y_test, p_test)
    
    return weights, metrics


# ---------------------------------------------------------------------
# 9. Create default models
# ---------------------------------------------------------------------
def make_default_models(random_state=42):
    """
    Create a default set of ensemble models.
    
    Args:
        random_state : int
            Random seed.
    
    Returns:
        models : list
            List of sklearn models.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    
    logreg = make_pipeline(
        StandardScaler(), 
        LogisticRegression(max_iter=1000, random_state=random_state)
    )
    rf = RandomForestClassifier(
        n_estimators=200, 
        random_state=random_state
    )
    gb = GradientBoostingClassifier(
        random_state=random_state
    )
    
    return [logreg, rf, gb]


# ---------------------------------------------------------------------
# 10. Format results for display
# ---------------------------------------------------------------------
def format_ensemble_results(weights, metrics, model_names=None):
    """
    Format ensemble results for display.
    
    Args:
        weights : array
            Ensemble weights.
        metrics : dict
            Evaluation metrics.
        model_names : list or None
            Names of models.
    
    Returns:
        summary : pd.DataFrame
            Formatted results.
    """
    if model_names is None:
        model_names = [f"Model_{i}" for i in range(len(weights))]
    
    # Weights summary
    weights_df = pd.DataFrame({
        "Model": model_names,
        "Weight": weights
    })
    
    # Metrics summary
    metrics_df = pd.DataFrame([metrics])
    
    return weights_df, metrics_df


# ---------------------------------------------------------------------
# 11. Alpha sweep for LLM weight analysis
# ---------------------------------------------------------------------
def alpha_sweep(X_train, y_train, X_test, y_test, models,
                llm_beta, llm_beta0, alpha_values,
                l2=0.0, beta_scale=1.0):
    """
    Sweep over different fixed LLM weights.
    
    Args:
        X_train, y_train : arrays
            Training data.
        X_test, y_test : arrays
            Test data.
        models : list
            List of sklearn models.
        llm_beta : array
            LLM coefficients.
        llm_beta0 : float
            LLM intercept.
        alpha_values : array-like
            LLM weight values to test.
        l2 : float
            Regularization strength.
        beta_scale : float
            LLM coefficient scaling.
    
    Returns:
        results : pd.DataFrame
            Results for each alpha value.
    """
    results = []
    
    for alpha in alpha_values:
        weights, metrics = run_ensemble_experiment(
            X_train, y_train, X_test, y_test,
            models,
            llm_beta=llm_beta,
            llm_beta0=llm_beta0,
            use_optimizer=True,
            fixed_weight=(len(models), alpha),  # Fix LLM weight
            l2=l2,
            beta_scale=beta_scale
        )
        
        results.append({
            "alpha": alpha,
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "logloss": metrics["logloss"],
            "weights": weights
        })
    
    return pd.DataFrame(results)