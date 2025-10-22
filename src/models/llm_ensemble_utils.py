# ensemble_utils.py
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# 1. Fixed-coefficient logistic (LLM-style) with tunable scaling
# ---------------------------------------------------------------------
def llm_predict_proba(X, beta, beta0=0.0, beta_scale=1.0):
    """
    Logistic regression predictor with optional coefficient scaling.
    X: (N, D)
    beta: array-like (D,)
    beta0: float, intercept
    beta_scale: float or array-like (same length as beta)
                 Used to tune or perturb coefficients.
                 Example: 0.7 to 1.3 range for light tuning.
    Returns: (N,) probabilities
    """
    X_arr = np.asarray(X)
    scale = np.asarray(beta_scale)
    z = beta0 + X_arr @ (np.asarray(beta) * scale)
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------
# 2. Ensemble probability prediction (stack + weighted average)
# ---------------------------------------------------------------------
def ensemble_predict_proba(X, models, weights, llm_beta=None, llm_beta0=0.0, beta_scale=1.0):
    """
    Combine probabilities from base models and optional LLM model.
    - models: list of sklearn classifiers with predict_proba
    - weights: list/array of same length (sum to 1)
    - llm_beta, llm_beta0: if provided, add LLM as extra model
    - beta_scale: optional scaling factor for LLM coefficients
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
# 3. Weight optimizer (SLSQP convex combination minimizing log loss)
# ---------------------------------------------------------------------
def optimize_weights(
    probs_list,
    y_val,
    fixed_index=None,
    fixed_value=None,
    l2=0.0
):
    """
    Optimize convex weights that minimize log loss.
    probs_list: list of arrays (each N,)
    y_val: array (N,)
    fixed_index, fixed_value: optional (e.g., fix LLM weight)
    l2: optional regularization
    Returns: optimal weights (sum to 1)
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
        cons.append({"type": "eq", "fun": lambda w, i=fixed_index, v=fixed_value: w[i] - v})

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        print("Warning: optimization did not converge:", res.message)
        w = np.maximum(w0, 0)
        return w / w.sum()
    return res.x


# ---------------------------------------------------------------------
# 4. Run full ensemble experiment (train/val split + optimize + test)
# ---------------------------------------------------------------------
def run_ensemble_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    llm_beta=None,
    llm_beta0=0.0,
    use_optimizer=True,
    manual_weights=None,
    fixed_weight=None,   # (index, value)
    l2=0.0,
    beta_scale=1.0
):
    """
    Fit models on training data, optionally include LLM,
    optimize or use manual weights, and evaluate on test data.
    """
    # Split for validation
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    # Fit base models
    for m in models:
        m.fit(X_sub, y_sub)

    # Validation probabilities
    probs_val = [m.predict_proba(X_val)[:, 1] for m in models]
    if llm_beta is not None:
        probs_val.append(llm_predict_proba(X_val, llm_beta, llm_beta0, beta_scale))

    # Determine weights
    if use_optimizer:
        if fixed_weight is not None:
            fixed_index, fixed_value = fixed_weight
        else:
            fixed_index = fixed_value = None
        weights = optimize_weights(probs_val, y_val, fixed_index, fixed_value, l2=l2)
    else:
        weights = np.asarray(manual_weights)
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()

    # Test predictions
    p_test = ensemble_predict_proba(X_test, models, weights, llm_beta, llm_beta0, beta_scale)
    y_pred = (p_test >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, p_test),
        "logloss": log_loss(y_test, p_test),
    }

    return weights, metrics


# ---------------------------------------------------------------------
# 5. Optional alpha sweep (e.g., fix LLM weight across range)
# ---------------------------------------------------------------------
def alpha_sweep(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    llm_beta,
    llm_beta0,
    alpha_values,
    l2=0.0,
    beta_scale=1.0
):
    """
    Sweep over fixed α values (for LLM weight) while re-optimizing the rest.
    Returns a list of (alpha, metrics_dict, weights).
    """
    results = []
    for alpha in alpha_values:
        weights, metrics = run_ensemble_experiment(
            X_train, y_train, X_test, y_test,
            models,
            llm_beta=llm_beta,
            llm_beta0=llm_beta0,
            use_optimizer=True,
            fixed_weight=(len(models), alpha),  # fix LLM weight (last index)
            l2=l2,
            beta_scale=beta_scale
        )
        results.append({"alpha": alpha, "metrics": metrics, "weights": weights})
    return results


# ---------------------------------------------------------------------
# 6. Helper: make default models (LogReg, RF, GB)
# ---------------------------------------------------------------------
def make_default_models(random_state=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    gb = GradientBoostingClassifier(random_state=random_state)
    return [logreg, rf, gb]
