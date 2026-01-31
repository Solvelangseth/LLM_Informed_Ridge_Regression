"""
Regression Experiment Template (MSE)

Copy this file into a new run folder under experiments/runs/<run_name>/
Edit ONLY the copied file.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

from llm_statistics import TargetInformedModel, CovarianceTargetInformedModel, LLMPriorElicitor

# -------------------------------
# CONFIG (edit in your run folder)
# -------------------------------
RUN_NAME = "YYYY-MM-DD_dataset_regression"
OUTPUT_DIR = os.path.join("results")

RANDOM_SEED = 42
ALPHA_GRID = np.logspace(-4, 6, 51)
GAMMA_GRID = np.linspace(0, 1, 101)

LLM_MODEL = "gpt-5.1"
NUM_LLM_SAMPLES = 10

# File paths (example placeholders)
X_PATH = "data/your_dataset_X.csv"
Y_PATH = "data/your_dataset_y.csv"

# Specify target column (if y is embedded in X)
TARGET_COL = None

# Provide your domain split logic
DOMAIN_COLUMN = None
DOMAIN_SOURCE_VALUE = None
DOMAIN_OOD_VALUE = None

# Features
NUMERIC_FEATURES = []
CATEGORICAL_FEATURES = []

# -------------------------------
# UTILS
# -------------------------------

def eval_mse(y_true, y_pred, name):
    return {"Model": name, "MSE": mean_squared_error(y_true, y_pred)}


def clean_categorical(df, cat_cols):
    out = df.copy()
    for col in cat_cols:
        out[col] = out[col].replace("?", np.nan)
        out[col] = out[col].fillna("Unknown")
        out[col] = out[col].astype(str)
    return out


def coerce_numeric(df, num_cols):
    out = df.copy()
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isnull().any():
            out[col] = out[col].fillna(out[col].median())
    return out


def build_feature_matrices(X_source, X_ood, num_cols, cat_cols):
    X_source = clean_categorical(coerce_numeric(X_source, num_cols), cat_cols)
    X_ood = clean_categorical(coerce_numeric(X_ood, num_cols), cat_cols)

    X_source_cat = pd.get_dummies(X_source[cat_cols], prefix=cat_cols, drop_first=False)
    X_ood_cat = pd.get_dummies(X_ood[cat_cols], prefix=cat_cols, drop_first=False)
    X_ood_cat = X_ood_cat.reindex(columns=X_source_cat.columns, fill_value=0)

    X_source_final = pd.concat([X_source[num_cols].reset_index(drop=True), X_source_cat.reset_index(drop=True)], axis=1)
    X_ood_final = pd.concat([X_ood[num_cols].reset_index(drop=True), X_ood_cat.reset_index(drop=True)], axis=1)

    feature_names = X_source_final.columns.tolist()
    return X_source_final.values, X_ood_final.values, feature_names


def build_llm_prompt(feature_names):
    feature_list = ", ".join(feature_names)
    return f"""
You are an expert applied statistician.

We are fitting a linear regression model with y in [0,1] (binary target treated as regression).
All features are standardized (z-scored) using training data mean/std.

Features (exact keys required):
[{feature_list}]

Return ONLY valid JSON in this exact format:
{{
  "targets": {{
    "feature_name": <number>,
    "another_feature": <number>
  }}
}}

Constraints:
- Use conservative magnitudes (roughly between -0.5 and +0.5 unless strongly justified)
- Do not include any text outside JSON.
"""


def elicit_single_prior(feature_names, out_path):
    elicitor = LLMPriorElicitor(LLM_MODEL)
    prompt = build_llm_prompt(feature_names)
    response = elicitor.call(prompt=prompt)
    targets = elicitor.extract_targets(response)
    with open(out_path, "w") as f:
        json.dump(targets, f, indent=2)
    return targets


def elicit_prior_samples(feature_names, out_path, num_samples):
    elicitor = LLMPriorElicitor(LLM_MODEL)
    prompt = build_llm_prompt(feature_names)
    samples = []
    attempts = 0
    max_attempts = num_samples * 5

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        response = elicitor.call(prompt=prompt)
        try:
            targets = elicitor.extract_targets(response)
            samples.append(targets)
        except Exception:
            continue

    if len(samples) < 2:
        raise ValueError("Not enough valid samples")

    with open(out_path, "w") as f:
        json.dump(samples, f, indent=2)
    return samples


def compute_priors_from_samples(samples, feature_names):
    elicitor = LLMPriorElicitor(LLM_MODEL)
    keys, means, cov, precision = elicitor._compute_prior_statistics(samples)

    key_idx = {k: i for i, k in enumerate(keys)}
    order = [key_idx[f] for f in feature_names]
    mu = means[order]
    P = precision[np.ix_(order, order)]
    return mu, P


def mix_preds(p_a, p_b, gamma):
    return (1 - gamma) * p_a + gamma * p_b


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH)
    y_raw = y.iloc[:, 0].astype(str)
    y_binary = (y_raw == ">50K").astype(float).values

    # Domain split
    source_mask = X[DOMAIN_COLUMN] == DOMAIN_SOURCE_VALUE
    ood_mask = X[DOMAIN_COLUMN] == DOMAIN_OOD_VALUE

    X_source = X[source_mask].drop(columns=[DOMAIN_COLUMN])
    X_ood = X[ood_mask].drop(columns=[DOMAIN_COLUMN])
    y_source = y_binary[source_mask.values]
    y_ood = y_binary[ood_mask.values]

    X_source_proc, X_ood_proc, feature_names = build_feature_matrices(
        X_source, X_ood, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )

    # Split source
    X_train_full, X_id, y_train_full, y_id = train_test_split(
        X_source_proc, y_source, test_size=0.20, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_SEED
    )

    # Standardize
    scaler_tune = StandardScaler()
    X_train_s = scaler_tune.fit_transform(X_train)
    X_val_s = scaler_tune.transform(X_val)

    scaler_final = StandardScaler()
    X_train_full_s = scaler_final.fit_transform(X_train_full)
    X_id_s = scaler_final.transform(X_id)
    X_ood_s = scaler_final.transform(X_ood_proc)

    # OLS
    ols = LinearRegression(fit_intercept=True)
    ols.fit(X_train_full_s, y_train_full)
    pred_ols_id = ols.predict(X_id_s)
    pred_ols_ood = ols.predict(X_ood_s)

    # Ridge (tune alpha)
    ridge_rows = []
    for a in ALPHA_GRID:
        m = Ridge(alpha=float(a), fit_intercept=True)
        m.fit(X_train_s, y_train)
        p_val = m.predict(X_val_s)
        ridge_rows.append({"alpha": float(a), "val_mse": mean_squared_error(y_val, p_val)})

    ridge_cv = pd.DataFrame(ridge_rows).sort_values("val_mse").reset_index(drop=True)
    best_ridge_alpha = float(ridge_cv.loc[0, "alpha"])

    ridge = Ridge(alpha=best_ridge_alpha, fit_intercept=True)
    ridge.fit(X_train_full_s, y_train_full)
    pred_ridge_id = ridge.predict(X_id_s)
    pred_ridge_ood = ridge.predict(X_ood_s)

    # LLM priors
    single_prior = elicit_single_prior(feature_names, os.path.join(OUTPUT_DIR, "single_prior.json"))
    samples = elicit_prior_samples(feature_names, os.path.join(OUTPUT_DIR, "prior_samples.json"), NUM_LLM_SAMPLES)
    mu, P = compute_priors_from_samples(samples, feature_names)
    t_target = np.array([single_prior[f] for f in feature_names], dtype=float)

    # Target-Informed Ridge
    tir_rows = []
    for a in ALPHA_GRID:
        tir = TargetInformedModel(alpha=float(a), model_type="ridge", targets=t_target, fit_intercept=True)
        tir.fit(X_train_s, y_train)
        p_val = tir.predict(X_val_s)
        tir_rows.append({"alpha": float(a), "val_mse": mean_squared_error(y_val, p_val)})

    tir_cv = pd.DataFrame(tir_rows).sort_values("val_mse").reset_index(drop=True)
    best_tir_alpha = float(tir_cv.loc[0, "alpha"])

    tir = TargetInformedModel(alpha=best_tir_alpha, model_type="ridge", targets=t_target, fit_intercept=True)
    tir.fit(X_train_full_s, y_train_full)
    pred_tir_id = tir.predict(X_id_s)
    pred_tir_ood = tir.predict(X_ood_s)

    # Covariance TI Ridge
    cov_rows = []
    for a in ALPHA_GRID:
        cov = CovarianceTargetInformedModel(mu=mu, P=P, model_type="ridge", fit_intercept=True, alpha=float(a))
        cov.fit(X_train_s, y_train)
        p_val = cov.predict(X_val_s)
        cov_rows.append({"alpha": float(a), "val_mse": mean_squared_error(y_val, p_val)})

    cov_cv = pd.DataFrame(cov_rows).sort_values("val_mse").reset_index(drop=True)
    best_cov_alpha = float(cov_cv.loc[0, "alpha"])

    cov = CovarianceTargetInformedModel(mu=mu, P=P, model_type="ridge", fit_intercept=True, alpha=best_cov_alpha)
    cov.fit(X_train_full_s, y_train_full)
    pred_cov_id = cov.predict(X_id_s)
    pred_cov_ood = cov.predict(X_ood_s)

    # Mixed model
    ridge_for_gamma = Ridge(alpha=best_ridge_alpha, fit_intercept=True)
    ridge_for_gamma.fit(X_train_s, y_train)
    p_ridge_val = ridge_for_gamma.predict(X_val_s)

    tir_for_gamma = TargetInformedModel(alpha=best_tir_alpha, model_type="ridge", targets=t_target, fit_intercept=True)
    tir_for_gamma.fit(X_train_s, y_train)
    p_tir_val = tir_for_gamma.predict(X_val_s)

    gamma_rows = []
    for g in GAMMA_GRID:
        p_mix = mix_preds(p_ridge_val, p_tir_val, float(g))
        gamma_rows.append({"gamma": float(g), "val_mse": mean_squared_error(y_val, p_mix)})

    gamma_df = pd.DataFrame(gamma_rows).sort_values("val_mse").reset_index(drop=True)
    best_gamma = float(gamma_df.loc[0, "gamma"])

    pred_mix_id = mix_preds(pred_ridge_id, pred_tir_id, best_gamma)
    pred_mix_ood = mix_preds(pred_ridge_ood, pred_tir_ood, best_gamma)

    # Results
    id_rows = [
        eval_mse(y_id, pred_ols_id, "OLS"),
        eval_mse(y_id, pred_ridge_id, "Ridge"),
        eval_mse(y_id, pred_tir_id, "Target-Informed Ridge"),
        eval_mse(y_id, pred_cov_id, "Covariance TI Ridge"),
        eval_mse(y_id, pred_mix_id, f"Mixed (gamma={best_gamma:.2f})"),
    ]

    ood_rows = [
        eval_mse(y_ood, pred_ols_ood, "OLS"),
        eval_mse(y_ood, pred_ridge_ood, "Ridge"),
        eval_mse(y_ood, pred_tir_ood, "Target-Informed Ridge"),
        eval_mse(y_ood, pred_cov_ood, "Covariance TI Ridge"),
        eval_mse(y_ood, pred_mix_ood, f"Mixed (gamma={best_gamma:.2f})"),
    ]

    pd.DataFrame(id_rows).to_csv(os.path.join(OUTPUT_DIR, "id_results.csv"), index=False)
    pd.DataFrame(ood_rows).to_csv(os.path.join(OUTPUT_DIR, "ood_results.csv"), index=False)
    ridge_cv.to_csv(os.path.join(OUTPUT_DIR, "ridge_alpha_tuning.csv"), index=False)
    tir_cv.to_csv(os.path.join(OUTPUT_DIR, "ti_ridge_alpha_tuning.csv"), index=False)
    cov_cv.to_csv(os.path.join(OUTPUT_DIR, "cov_ti_ridge_alpha_tuning.csv"), index=False)
    gamma_df.to_csv(os.path.join(OUTPUT_DIR, "mixed_gamma_tuning.csv"), index=False)
