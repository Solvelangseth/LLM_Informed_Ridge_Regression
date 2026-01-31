# ============================================================================
# covariance_target_informed_model.py (UPDATED)
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, mean_squared_error
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import joblib
import os


class CovarianceTargetInformedModel(BaseEstimator):
    """
    MAP regression model with multivariate Gaussian prior:

        β ~ N(μ, Σ)
        P = Σ^{-1}

    MAP ridge:
        minimize ||y - Xβ||² + (β - μ)^T P (β - μ)

    MAP logistic:
        minimize -log L(β) + (β - μ)^T P (β - μ)

    Parameters
    ----------
    mu : array-like, shape (p,)
        Prior mean vector from LLM sampling.

    P : array-like, shape (p,p)
        Precision matrix (inverse covariance) from LLM sampling.

    model_type : {'ridge', 'logistic'}
        Which likelihood to use.

    fit_intercept : bool
        If True, an intercept is included but NOT penalized.

    random_state : int
        For logistic initialization.
    """

    def __init__(self, mu=None, P=None, model_type="ridge",
                 fit_intercept=True, random_state=None, alpha=1.0):

        self.mu = None if mu is None else np.asarray(mu, float)
        self.P = None if P is None else np.asarray(P, float)

        self.model_type = model_type
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.alpha = alpha 

        # Fitted attributes
        self.coef_ = None
        self.intercept_ = None
        self.feature_names_ = None

    # ---------- Utility ----------
    def _prepare_data(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if len(y) != X.shape[0]:
            raise ValueError("X and y must align")

        return X, y

    # ---------- Ridge MAP solver ----------
    def _solve_ridge(self, X, y):
        n, p = X.shape

        if self.mu is None or self.P is None:
            raise ValueError("mu and P must be supplied for covariance-MAP regression")
        
        P_scaled = self.alpha * self.P

        # Expand matrices if intercept is used
        if self.fit_intercept:
            Xw = np.column_stack([np.ones(n), X])

            # expand precision
            P_ext = np.zeros((p + 1, p + 1))
            P_ext[1:, 1:] = P_scaled

            # expand mean
            mu_ext = np.concatenate([[0.0], self.mu])

        else:
            Xw = X
            P_ext = P_scaled
            mu_ext = self.mu

        XtX = Xw.T @ Xw
        Xty = Xw.T @ y

        lhs = XtX + P_ext
        rhs = Xty + P_ext @ mu_ext

        beta = np.linalg.solve(lhs, rhs)

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:].astype(float)
        else:
            self.intercept_ = 0.0
            self.coef_ = beta.astype(float)

    # ---------- Logistic MAP solver ----------
    def _solve_logistic(self, X, y):
        n, p = X.shape

        if self.mu is None or self.P is None:
            raise ValueError("mu and P must be supplied for covariance-MAP regression")
        
        P_scaled = self.alpha * self.P 
        if self.fit_intercept:
            Xw = np.column_stack([np.ones(n), X])
            d = p + 1

            P_ext = np.zeros((d, d))
            P_ext[1:, 1:] = P_scaled

            mu_ext = np.concatenate([[0.0], self.mu])
        else:
            Xw = X
            d = p
            P_ext = P_scaled
            mu_ext = self.mu

        def objective(beta):
            z = np.clip(Xw @ beta, -30, 30)
            loglik = np.sum(y * z - np.logaddexp(0, z))

            diff = (beta - mu_ext)
            prior = diff.T @ P_ext @ diff

            return -loglik + prior

        def gradient(beta):
            z = np.clip(Xw @ beta, -30, 30)
            p_hat = 1 / (1 + np.exp(-z))

            grad_ll = -Xw.T @ (y - p_hat)
            grad_prior = 2 * (P_ext @ (beta - mu_ext))

            return grad_ll + grad_prior

        if self.random_state is not None:
            np.random.seed(self.random_state)
            beta0 = 0.01 * np.random.randn(d)
        else:
            beta0 = np.zeros(d)

        res = minimize(objective, beta0, jac=gradient, method="L-BFGS-B")

        beta = res.x

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

    # ---------- Public API ----------
    def fit(self, X, y, feature_names=None):
        X, y = self._prepare_data(X, y)
        n, p = X.shape

        # feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(p)]
        self.feature_names_ = feature_names

        # run solver
        if self.model_type == "ridge":
            self._solve_ridge(X, y)
        elif self.model_type == "logistic":
            # clean labels
            unique = np.unique(y)
            if len(unique) != 2:
                raise ValueError("y must be binary for logistic")
            if not np.all(np.isin(unique, [0, 1])):
                y = (y == unique[1]).astype(float)

            self._solve_logistic(X, y)
        else:
            raise ValueError("model_type must be 'ridge' or 'logistic'")

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Fit the model first")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        z = X @ self.coef_ + self.intercept_

        if self.model_type == "ridge":
            return z
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        if self.model_type != "logistic":
            raise ValueError("predict_proba only for logistic")

        p1 = self.predict(X)
        return np.column_stack([1 - p1, p1])

    def score(self, X, y):
        if self.model_type == "ridge":
            return r2_score(y, self.predict(X))

        y_pred = (self.predict(X) > 0.5).astype(int)
        return accuracy_score(y, y_pred)

    def save_model(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load_model(path):
        return joblib.load(path)
