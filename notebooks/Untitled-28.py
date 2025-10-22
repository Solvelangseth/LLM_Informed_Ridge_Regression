# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score,
    roc_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve

# Import custom models
sys.path.append(os.path.dirname(os.getcwd()))
from src.models.target_informed_model import TargetInformedModel
from src.llm.target_elicitor import LLMTargetElicitor

np.random.seed(42)


# %%
# loading the data 
def load_heart_dataset(path, features, outcome="num"):
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    df = pd.read_csv(path, header=None, names=columns, na_values="?")
    df[outcome] = (df[outcome] > 0).astype(int)
    df = df[features + [outcome]].dropna()
    return df[features], df[outcome]

features = ["age", "sex", "trestbps", "chol", "thalach", "oldpeak", "cp", "exang", "fbs", "restecg"]

X, y = load_heart_dataset("data/heart+disease/processed.hungarian.data", features)
X_cleveland, y_cleveland = load_heart_dataset("data/heart+disease/processed.cleveland.data", features)



X = pd.get_dummies(X, columns=["cp", "restecg"], drop_first=True).astype(float)
X_cleveland = pd.get_dummies(X_cleveland, columns=["cp", "restecg"], drop_first=True).astype(float)

# Clean up dummy column names once, right after pd.get_dummies
X = X.rename(columns=lambda c: c.replace(".0", ""))
X_cleveland = X_cleveland.rename(columns=lambda c: c.replace(".0", ""))


features = list(X.columns)

print("Hungarian:", X.shape, "Cleveland:", X_cleveland.shape)


# %%
# %%
import numpy as np
import pandas as pd

CONT_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BIN_DUMMY_COLS = [c for c in X.columns if c not in CONT_COLS]  # sex, exang, fbs, cp_*, restecg_*

# Fit scaling statistics on in-domain (Cleveland)
scaler_stats = {
    col: {"mean": X_cleveland[col].mean(), "std": X_cleveland[col].std(ddof=0)}
    for col in CONT_COLS
}

def apply_standardization(df, stats):
    df = df.copy()
    for col in CONT_COLS:
        m = stats[col]["mean"]
        s = stats[col]["std"] if stats[col]["std"] > 0 else 1.0
        df[col] = (df[col] - m) / s
    return df

# Apply to Cleveland (train domain) and Hungarian (OOD)
X_clev_std = apply_standardization(X_cleveland, scaler_stats)
X_hu_std   = apply_standardization(X, scaler_stats)

# --- Ensure both datasets have identical feature sets ---
ALL_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak",  # continuous (z-scored)
    "sex", "exang", "fbs",                            # binary
    "cp_2", "cp_3", "cp_4",                           # chest pain dummies
    "restecg_1.0", "restecg_2.0"                      # restecg dummies
]

def align_features(df, all_features):
    """Ensure df has all features in all_features, in the same order."""
    for col in all_features:
        if col not in df.columns:
            df[col] = 0  # add missing dummy column if dataset doesn’t have it
    return df[all_features]  # reorder to consistent order

# Apply alignment
X_clev_std = align_features(X_clev_std, ALL_FEATURES)
X_hu_std   = align_features(X_hu_std, ALL_FEATURES)

# Store final feature order
features_std = ALL_FEATURES



# %%
import numpy as np

def llm_predict_proba(X, beta, beta0=0.0):
    """
    Logistic regression predictor with fixed coefficients (LLM-specified).
    X: pd.DataFrame or np.array with shape (n_samples, n_features)
    beta: array-like of length n_features
    beta0: intercept
    Returns: np.array of probabilities (n_samples,)
    """
    X_arr = np.asarray(X)
    beta = np.asarray(beta)
    z = beta0 + np.dot(X_arr, beta)
    return 1 / (1 + np.exp(-z))


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Train on Cleveland (in-domain, standardized)
logreg = LogisticRegression(max_iter=2000, random_state=42).fit(X_clev_std, y_cleveland)
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_clev_std, y_cleveland)
gb = GradientBoostingClassifier(random_state=42).fit(X_clev_std, y_cleveland)



# %%
def ensemble_predict_proba(X, models, weights, beta_llm=None, beta0_llm=0.0):
    """
    Weighted ensemble of scikit-learn models + optional LLM predictor.
    """
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    probs = [m.predict_proba(X)[:, 1] for m in models]

    if beta_llm is not None:  # Only include LLM if coefficients are given
        p_llm = llm_predict_proba(X, beta_llm, beta0_llm)
        probs.append(p_llm)

    p_ens = np.zeros(len(X))
    for w, p in zip(weights, probs):
        p_ens += w * p
    return p_ens


# %%
# LLM-suggested coefficients (standardized features)
llm_betas_dict = {
    "intercept": -0.85,
    "age": 0.60,
    "sex": 0.70,
    "trestbps": 0.20,
    "chol": 0.12,
    "thalach": -0.50,
    "oldpeak": 0.55,
    "exang": 0.60,
    "fbs": 0.35,
    "cp_2": -0.40,
    "cp_3": -0.80,
    "cp_4": 0.60,
    "restecg_1.0": 0.25,
    "restecg_2.0": 0.45,
}


beta0_llm = llm_betas_dict["intercept"]
beta_llm = np.array([llm_betas_dict[feat] for feat in features_std], dtype=float)

# %%


# %%
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_pred_proba),
        "auc": roc_auc_score(y_true, y_pred_proba)
    }


# %%
# Ensemble WITHOUT LLM
models = [logreg, rf, gb]
weights_no_llm = [0.33, 0.33, 0.34]  # equal weights
p_ens_no_llm = ensemble_predict_proba(X_clev_std, models, weights_no_llm, 0, 0)

# Ensemble WITH LLM
weights_with_llm = [0.3, 0.3, 0.2, 0.2]
p_ens_with_llm = ensemble_predict_proba(X_clev_std, models, weights_with_llm, beta_llm, beta0_llm)

# Evaluate
ens_no_llm_metrics = evaluate_model(y_cleveland, p_ens_no_llm)
ens_with_llm_metrics = evaluate_model(y_cleveland, p_ens_with_llm)

print("Ensemble (no LLM):", ens_no_llm_metrics)
print("Ensemble (with LLM):", ens_with_llm_metrics)


# %%
from sklearn.model_selection import train_test_split

# Cleveland train/val split
X_clev_train, X_clev_val, y_clev_train, y_clev_val = train_test_split(
    X_clev_std, y_cleveland, test_size=0.3, random_state=42, stratify=y_cleveland
)

# Re-train your base models on Cleveland train
logreg_clev = LogisticRegression(max_iter=2000).fit(X_clev_train, y_clev_train)
rf_clev = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_clev_train, y_clev_train)
gb_clev = GradientBoostingClassifier(random_state=42).fit(X_clev_train, y_clev_train)

models_clev = [logreg_clev, rf_clev, gb_clev]


# %%
def tune_llm_weight(X_val, y_val, models, beta_llm, beta0_llm, grid=np.linspace(0,0.5,11)):
    results = []
    for w_llm in grid:
        # distribute remaining weight equally among models
        w_rest = (1 - w_llm) / len(models)
        weights = [w_rest] * len(models) + [w_llm]
        
        p_ens = ensemble_predict_proba(X_val, models, weights, beta_llm, beta0_llm)
        metrics = evaluate_model(y_val, p_ens)
        results.append({
            "w_llm": w_llm,
            **metrics
        })
    return pd.DataFrame(results)

# Run tuning on Cleveland val
df_tuning = tune_llm_weight(X_clev_val, y_clev_val, models_clev, beta_llm, beta0_llm)
print(df_tuning)



# %%
best_row = df_tuning.loc[df_tuning["log_loss"].idxmin()]  # or auc/accuracy depending on goal
best_w_llm = best_row["w_llm"]
print("Best LLM weight on Cleveland val:", best_w_llm)


# %%
# Refit models on full Cleveland
logreg_clev = LogisticRegression(max_iter=2000).fit(X_clev_std, y_cleveland)
rf_clev = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_clev_std, y_cleveland)
gb_clev = GradientBoostingClassifier(random_state=42).fit(X_clev_std, y_cleveland)

models_clev = [logreg_clev, rf_clev, gb_clev]

# Ensemble with tuned LLM weight
w_rest = (1 - best_w_llm) / len(models_clev)
weights = [w_rest] * len(models_clev) + [best_w_llm]

# Evaluate on Hungarian (OOD)
p_ens_hu = ensemble_predict_proba(X_hu_std, models_clev, weights, beta_llm, beta0_llm)
ens_metrics_hu = evaluate_model(y, p_ens_hu)

print("Final Ensemble (with tuned LLM weight) on Hungarian:", ens_metrics_hu)


# %%
from sklearn.model_selection import train_test_split

def ood_weight_search(X, y, subgroup_mask, models, beta_llm, beta0_llm,
                      grid=np.linspace(0,0.5,11), min_size=15):
    """
    Train on complement, test on subgroup.
    Search LLM weights for the ensemble, pick best by log_loss.
    """
    X_train, y_train = X[~subgroup_mask], y[~subgroup_mask]
    X_test, y_test   = X[subgroup_mask], y[subgroup_mask]

    if len(y_test) < min_size or len(np.unique(y_test)) < 2:
        return None  # skip too small or single-class subgroup

    # Re-train models on training split
    logreg = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    rf     = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
    gb     = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    base_models = [logreg, rf, gb]

    results = []
    for w_llm in grid:
        w_rest = (1 - w_llm) / len(base_models)
        weights = [w_rest] * len(base_models) + [w_llm]

        p_ens = ensemble_predict_proba(X_test, base_models, weights, beta_llm, beta0_llm)
        metrics = evaluate_model(y_test, p_ens)
        results.append({"w_llm": w_llm, **metrics})

    df = pd.DataFrame(results)
    best_row = df.loc[df["log_loss"].idxmin()]
    return best_row["w_llm"], df


# %%
subgroups = {
    "age<45":       (X_hu_std["age"] < -0.4),   # since standardized, adjust threshold
    "male":         (X_hu_std["sex"] == 1),
    "female":       (X_hu_std["sex"] == 0),
    "high_BP":      (X_hu_std["trestbps"] > 1.0),
    "high_chol":    (X_hu_std["chol"] > 1.0),
    "exang=1":      (X_hu_std["exang"] == 1),
    "cp_2":         (X_hu_std["cp_2"] == 1),
    "cp_3":         (X_hu_std["cp_3"] == 1),
    "cp_4":         (X_hu_std["cp_4"] == 1),
    "oldpeak>=2":   (X_hu_std["oldpeak"] >= 2),
}

subgroup_results = {}
subgroup_sizes = {}   # <-- NEW

for name, mask in subgroups.items():
    try:
        best_w, df = ood_weight_search(X_hu_std, y, mask, models_clev, beta_llm, beta0_llm)
        subgroup_results[name] = best_w
        subgroup_sizes[name] = mask.sum()   # <-- NEW: size of subgroup
        print(f"✅ Subgroup {name}: best_w_llm={best_w} (n={mask.sum()})")
    except Exception as e:
        print(f"⚠️ Skipped subgroup {name}: {e}")

# --- Aggregation ---
subgroup_weights = pd.Series(subgroup_results, name="best_w_llm")
subgroup_sizes   = pd.Series(subgroup_sizes, name="n_test")

print("\nPer-subgroup LLM weights:")
print(subgroup_weights)

print("\nAggregated:")
print("Mean:", subgroup_weights.mean())
print("Median:", subgroup_weights.median())

weighted_mean = np.average(
    subgroup_weights.values,
    weights=subgroup_sizes.reindex(subgroup_weights.index).values
)
print("Weighted mean (by subgroup size):", weighted_mean)


# %%
subgroup_weights = pd.Series(subgroup_results)
print("\nPer-subgroup LLM weights:")
print(subgroup_weights)

print("\nAggregated:")
print("Mean:", subgroup_weights.mean())
print("Median:", subgroup_weights.median())
print("Weighted mean (by subgroup size): TBD if we track n_test)")


# %%
# Suppose in your loop you collected both best weights and subgroup sizes:
# subgroup_results[name] = best_w
# subgroup_sizes[name] = n_test

subgroup_weights = pd.Series(subgroup_results, name="best_w_llm")
subgroup_sizes   = pd.Series(subgroup_sizes, name="n_test")  # <-- make sure you collected this

print("\nPer-subgroup LLM weights:")
print(subgroup_weights)

print("\nAggregated:")
print("Mean:", subgroup_weights.mean())
print("Median:", subgroup_weights.median())

# Weighted mean using subgroup sizes
weighted_mean = np.average(
    subgroup_weights.values,
    weights=subgroup_sizes.reindex(subgroup_weights.index).values
)
print("Weighted mean (by subgroup size):", weighted_mean)


# %%
from sklearn.model_selection import StratifiedKFold

# --- Step 1: Global weight from subgroup tuning ---
w_llm_global = 0.38   # or try subgroup_weights.mean()/median
print("Using global LLM weight:", w_llm_global)

# Cleveland-trained models (already fit earlier)
models_clev = [logreg_clev, rf_clev, gb_clev]

# --- Step 2: Evaluate on Hungarian OOD test ---
# Baseline ensemble (no LLM)
p_base_hu = np.mean([m.predict_proba(X_hu_std)[:,1] for m in models_clev], axis=0)

metrics_base = {
    "accuracy": accuracy_score(y, (p_base_hu > 0.5).astype(int)),
    "log_loss": log_loss(y, p_base_hu),
    "auc": roc_auc_score(y, p_base_hu),
}

# LLM ensemble
p_llm_hu = ensemble_predict_proba(
    X_hu_std, models_clev, [0.3, 0.3, 0.2, w_llm_global], beta_llm, beta0_llm
)

metrics_llm = {
    "accuracy": accuracy_score(y, (p_llm_hu > 0.5).astype(int)),
    "log_loss": log_loss(y, p_llm_hu),
    "auc": roc_auc_score(y, p_llm_hu),
}

print("\n=== Hungarian Test (OOD) ===")
print("Baseline Ensemble:", metrics_base)
print("LLM Ensemble:", metrics_llm)

# --- Step 3: Cross-validation on Hungarian ---
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []
for train_idx, test_idx in kf.split(X_hu_std, y):
    X_tr, X_te = X_hu_std.iloc[train_idx], X_hu_std.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    # Train base models on fold
    logreg_cv = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    rf_cv = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
    gb_cv = GradientBoostingClassifier(random_state=42).fit(X_tr, y_tr)

    # Baseline ensemble (average probs)
    p_base = np.mean([m.predict_proba(X_te)[:,1] for m in [logreg_cv, rf_cv, gb_cv]], axis=0)

    # LLM ensemble (with tuned global weight)
    p_llm = ensemble_predict_proba(
        X_te, [logreg_cv, rf_cv, gb_cv], [0.3, 0.3, 0.2, w_llm_global], beta_llm, beta0_llm
    )

    cv_results.append({
        "acc_base": accuracy_score(y_te, (p_base > 0.5).astype(int)),
        "logloss_base": log_loss(y_te, p_base),
        "auc_base": roc_auc_score(y_te, p_base),
        "acc_llm": accuracy_score(y_te, (p_llm > 0.5).astype(int)),
        "logloss_llm": log_loss(y_te, p_llm),
        "auc_llm": roc_auc_score(y_te, p_llm),
    })

df_cv = pd.DataFrame(cv_results)
print("\n=== 5-fold CV on Hungarian ===")
print(df_cv.mean())


# %%



