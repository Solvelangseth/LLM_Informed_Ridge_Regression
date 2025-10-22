import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score
from scipy.optimize import minimize

class TargetInformedModel(BaseEstimator):
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
    model_type : {'ridge','logistic'} (default 'ridge')
    fit_intercept : bool (default True)
        Intercept is never penalized.
    random_state : int or None
        Only used for logistic init.
    targets : array-like or None
        If None, defaults to zeros (standard ridge/logistic).
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

    # ---------- utils ----------
    def _prepare_data(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if y.ndim != 1:
            raise ValueError("y must be 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        return X, y

    # ---------- ridge ----------
    def _solve_target_ridge(self, X, y, targets):
        n, p = X.shape

        if self.fit_intercept:
            Xw = np.column_stack([np.ones(n), X])
            t_ext = np.concatenate([[0.0], targets])
            I_ext = np.eye(p + 1)
            I_ext[0, 0] = 0.0  # no penalty on intercept
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

    # ---------- logistic ----------
    def _solve_target_logistic(self, X, y, targets):
        n, p = X.shape

        if self.fit_intercept:
            Xw = np.column_stack([np.ones(n), X])
            t_ext = np.concatenate([[0.0], targets])
            d = p + 1
            mask = np.ones(d, dtype=float); mask[0] = 0.0
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

        if self.random_state is not None:
            np.random.seed(self.random_state)
            beta0 = 0.01 * np.random.randn(d)
        else:
            beta0 = np.zeros(d)

        res = minimize(objective, beta0, jac=gradient, method="L-BFGS-B",
                       options={'maxiter': 1000})
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

    # ---------- public API ----------
    def fit(self, X, y, feature_names=None, targets=None):
        X, y = self._prepare_data(X, y)
        n, p = X.shape

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(p)]
        elif len(feature_names) != p:
            raise ValueError(f"feature_names must have {p} elements")
        self.feature_names_ = feature_names

        if targets is None:
            targets = np.zeros(p) if self.targets is None else self.targets
        targets = np.asarray(targets, dtype=float)
        if len(targets) != p:
            raise ValueError(f"targets must have {p} elements")
        self.targets_used_ = targets

        if self.model_type == "ridge":
            self._solve_target_ridge(X, y, targets)
        elif self.model_type == "logistic":
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
        if self.coef_ is None:
            raise ValueError("Model must be fitted before prediction")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        lp = X @ self.coef_ + self.intercept_
        if self.model_type == "ridge":
            return lp
        elif self.model_type == "logistic":
            z = np.clip(lp, -500, 500)
            return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X):
        if self.model_type != "logistic":
            raise ValueError("predict_proba only available for logistic regression")
        p1 = self.predict(X)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def score(self, X, y):
        if self.model_type == "ridge":
            return r2_score(y, self.predict(X))
        elif self.model_type == "logistic":
            y_pred = (self.predict(X) > 0.5).astype(int)
            return accuracy_score(y, y_pred)

    def get_coefficient_summary(self):
        if self.coef_ is None:
            raise ValueError("Model must be fitted first")
        return pd.DataFrame({
            "feature": self.feature_names_,
            "target": self.targets_used_,
            "coefficient": self.coef_,
            "adjustment": self.coef_ - self.targets_used_,
        })
