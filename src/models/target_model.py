import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, log_loss, roc_auc_score

class SklearnTargetModel(BaseEstimator):
    """
    Wrapper around sklearn Ridge and LogisticRegression to simulate
    target-informed shrinkage. When targets=0, behaves exactly like ridge/logistic ridge.
    """
    def __init__(self, alpha=1.0, model_type='linear', fit_intercept=True, targets=None):
        self.alpha = alpha
        self.model_type = model_type
        self.fit_intercept = fit_intercept
        self.targets = targets

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        if self.targets is None:
            self.targets = np.zeros(n_features)
        self.feature_names_ = feature_names or [f"feature_{i}" for i in range(n_features)]

        # Shift response by target offset
        offset = X @ self.targets

        if self.model_type == 'linear':
            # Equivalent to (y - X*targets), standard Ridge on theta
            y_shifted = y - offset
            self.model_ = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
            self.model_.fit(X, y_shifted)
            self.coef_ = self.model_.coef_ + self.targets
            self.intercept_ = self.model_.intercept_
        elif self.model_type == 'logistic':
            # For logistic regression, we fake an "offset" by adding as an extra column
            offset = offset.reshape(-1, 1)
            X_aug = np.hstack([X, offset])
            self.model_ = LogisticRegression(
                penalty="l2", C=1.0/self.alpha, solver="lbfgs", max_iter=1000,
                fit_intercept=self.fit_intercept
            )
            self.model_.fit(X_aug, y)
            # Separate coefficients
            self.coef_ = self.model_.coef_[0, :-1] + self.targets
            self.intercept_ = float(self.model_.intercept_ + self.model_.coef_[0, -1])
        else:
            raise ValueError("model_type must be 'linear' or 'logistic'")
        return self

    def predict(self, X):
        X = np.asarray(X)
        if self.model_type == 'linear':
            return X @ self.coef_ + self.intercept_
        elif self.model_type == 'logistic':
            # logistic probas
            logits = X @ self.coef_ + self.intercept_
            return 1 / (1 + np.exp(-logits))

    def score(self, X, y):
        if self.model_type == 'linear':
            return r2_score(y, self.predict(X))
        elif self.model_type == 'logistic':
            preds = self.predict(X)
            return accuracy_score(y, (preds > 0.5).astype(int))

    def get_coefficient_summary(self):
        # Convert everything to flat 1D arrays with consistent dtype
        features = np.asarray(self.feature_names_).ravel()
        targets = np.asarray(self.targets, dtype=float).ravel()
        coefs = np.asarray(self.coef_, dtype=float).ravel()
        adjustments = coefs - targets

        # Build DataFrame safely
        df = pd.DataFrame({
            'feature': features,
            'target': targets,
            'coefficient': coefs,
            'adjustment': adjustments
        })
        return df




# Example usage and testing
if __name__ == "__main__":
    np.random.seed(42)

    # === Ridge regression test ===
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([1.5, -2.0, 0.5, 0.0, -1.0])
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    manual_targets = np.array([0, 0, 0, 0, 0])

    ridge_sklearn = Ridge(alpha=1.0).fit(X, y)
    ridge_custom = SklearnTargetModel(alpha=1.0, model_type="linear", targets=np.zeros(n_features)).fit(X, y)
    ridge_informed = SklearnTargetModel(alpha=1.0, model_type="linear", targets=manual_targets).fit(X, y)

    print("\n=== Ridge Comparison ===")
    print("Sklearn Ridge R2:", r2_score(y, ridge_sklearn.predict(X)))
    print("Custom Ridge (zero target) R2:", ridge_custom.score(X, y))
    print("Target-Informed Ridge R2:", ridge_informed.score(X, y))
    print(ridge_informed.get_coefficient_summary())

    # === Logistic regression test ===
    n_samples, n_features = 200, 4
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([1.0, -1.5, 0.7, -0.3])
    logits = X @ true_coef
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)

    logistic_sklearn = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000).fit(X, y)
    logistic_custom = SklearnTargetModel(alpha=1.0, model_type="logistic", targets=np.zeros(n_features)).fit(X, y)
    logistic_informed = SklearnTargetModel(alpha=1.0, model_type="logistic", targets=np.zeros(n_features)).fit(X, y)

    print("\n=== Logistic Comparison ===")
    print("Sklearn Logistic accuracy:", accuracy_score(y, logistic_sklearn.predict(X)))
    print("Custom Logistic (zero target) accuracy:", logistic_custom.score(X, y))
    print("Target-Informed Logistic accuracy:", logistic_informed.score(X, y))
    print(logistic_informed.get_coefficient_summary())
