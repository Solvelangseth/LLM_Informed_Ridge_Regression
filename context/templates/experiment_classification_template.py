"""
Classification Experiment Template (AUC or LogLoss)

Copy this file into a new run folder under experiments/runs/<run_name>/
Edit ONLY the copied file.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression

from llm_statistics import TargetInformedModel, CovarianceTargetInformedModel, LLMPriorElicitor

# -------------------------------
# CONFIG (edit in your run folder)
# -------------------------------
RUN_NAME = "YYYY-MM-DD_dataset_classification"
OUTPUT_DIR = os.path.join("results")

RANDOM_SEED = 42
C_GRID = np.logspace(-4, 6, 51)
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

# Choose metric: "auc" or "logloss"
TUNE_METRIC = "auc"

# -------------------------------
# UTILS
# -------------------------------

def ensure_p1(p):
    p = np.asarray(p)
    if p.ndim == 2:
        return p[:, 1]
    return p


def metric_value(y_true, p1, metric):
    p1 = np.clip(p1, 1e-12, 1 - 1e-12)
    if metric == "auc":
        return roc_auc_score(y_true, p1)
    if metric == "logloss":
        return -log_loss(y_true, p1)  # higher is better when maximizing
    raise ValueError("Unknown metric")


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

We are fitting a logistic regression model predicting y in {{0,1}}.
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
- Use conservative magnitudes (roughly between -1.0 and +1.0 unless strongly justified)
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
    y_binary = (y_raw == ">50K").astype(int).values

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

    # L2 Logistic (tune C)
    l2_rows = []
    for C in C_GRID:
        m = LogisticRegression(C=float(C), penalty="l2", solver="lbfgs", max_iter=5000)
        m.fit(X_train_s, y_train)
        p_val = m.predict_proba(X_val_s)[:, 1]
        l2_rows.append({"C": float(C), "val_metric": metric_value(y_val, p_val, TUNE_METRIC)})

    l2_cv = pd.DataFrame(l2_rows).sort_values("val_metric", ascending=False).reset_index(drop=True)
    best_C = float(l2_cv.loc[0, "C"])

    l2 = LogisticRegression(C=best_C, penalty="l2", solver="lbfgs", max_iter=5000)
    l2.fit(X_train_full_s, y_train_full)
    p_l2_id = l2.predict_proba(X_id_s)[:, 1]
    p_l2_ood = l2.predict_proba(X_ood_s)[:, 1]

    # LLM priors
    single_prior = elicit_single_prior(feature_names, os.path.join(OUTPUT_DIR, "single_prior.json"))
    samples = elicit_prior_samples(feature_names, os.path.join(OUTPUT_DIR, "prior_samples.json"), NUM_LLM_SAMPLES)
    mu, P = compute_priors_from_samples(samples, feature_names)
    t_target = np.array([single_prior[f] for f in feature_names], dtype=float)

    # Target-Informed Logistic
    tir_rows = []
    for a in ALPHA_GRID:
        tir = TargetInformedModel(alpha=float(a), model_type="logistic", targets=t_target, fit_intercept=True)
        tir.fit(X_train_s, y_train)
        p_val = ensure_p1(tir.predict_proba(X_val_s))
        tir_rows.append({"alpha": float(a), "val_metric": metric_value(y_val, p_val, TUNE_METRIC)})

    tir_cv = pd.DataFrame(tir_rows).sort_values("val_metric", ascending=False).reset_index(drop=True)
    best_tir_alpha = float(tir_cv.loc[0, "alpha"])

    tir = TargetInformedModel(alpha=best_tir_alpha, model_type="logistic", targets=t_target, fit_intercept=True)
    tir.fit(X_train_full_s, y_train_full)
    p_tir_id = ensure_p1(tir.predict_proba(X_id_s))
    p_tir_ood = ensure_p1(tir.predict_proba(X_ood_s))

    # Covariance TI Logistic
    cov_rows = []
    for a in ALPHA_GRID:
        cov = CovarianceTargetInformedModel(mu=mu, P=P, model_type="logistic", fit_intercept=True, alpha=float(a))
        cov.fit(X_train_s, y_train)
        p_val = ensure_p1(cov.predict_proba(X_val_s))
        cov_rows.append({"alpha": float(a), "val_metric": metric_value(y_val, p_val, TUNE_METRIC)})

    cov_cv = pd.DataFrame(cov_rows).sort_values("val_metric", ascending=False).reset_index(drop=True)
    best_cov_alpha = float(cov_cv.loc[0, "alpha"])

    cov = CovarianceTargetInformedModel(mu=mu, P=P, model_type="logistic", fit_intercept=True, alpha=best_cov_alpha)
    cov.fit(X_train_full_s, y_train_full)
    p_cov_id = ensure_p1(cov.predict_proba(X_id_s))
    p_cov_ood = ensure_p1(cov.predict_proba(X_ood_s))

    # Mixed model
    l2_for_gamma = LogisticRegression(C=best_C, penalty="l2", solver="lbfgs", max_iter=5000)
    l2_for_gamma.fit(X_train_s, y_train)
    p_l2_val = l2_for_gamma.predict_proba(X_val_s)[:, 1]

    tir_for_gamma = TargetInformedModel(alpha=best_tir_alpha, model_type="logistic", targets=t_target, fit_intercept=True)
    tir_for_gamma.fit(X_train_s, y_train)
    p_tir_val = ensure_p1(tir_for_gamma.predict_proba(X_val_s))

    gamma_rows = []
    for g in GAMMA_GRID:
        p_mix = mix_preds(p_l2_val, p_tir_val, float(g))
        gamma_rows.append({"gamma": float(g), "val_metric": metric_value(y_val, p_mix, TUNE_METRIC)})

    gamma_df = pd.DataFrame(gamma_rows).sort_values("val_metric", ascending=False).reset_index(drop=True)
    best_gamma = float(gamma_df.loc[0, "gamma"])

    p_mix_id = mix_preds(p_l2_id, p_tir_id, best_gamma)
    p_mix_ood = mix_preds(p_l2_ood, p_tir_ood, best_gamma)

    # Save results
    pd.DataFrame({"Model": [
        "L2 Logistic",
        "Target-Informed Logistic",
        "Covariance TI Logistic",
        f"Mixed (gamma={best_gamma:.2f})",
    ], "ID": [
        metric_value(y_id, p_l2_id, TUNE_METRIC),
        metric_value(y_id, p_tir_id, TUNE_METRIC),
        metric_value(y_id, p_cov_id, TUNE_METRIC),
        metric_value(y_id, p_mix_id, TUNE_METRIC),
    ], "OOD": [
        metric_value(y_ood, p_l2_ood, TUNE_METRIC),
        metric_value(y_ood, p_tir_ood, TUNE_METRIC),
        metric_value(y_ood, p_cov_ood, TUNE_METRIC),
        metric_value(y_ood, p_mix_ood, TUNE_METRIC),
    ]}).to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)

    l2_cv.to_csv(os.path.join(OUTPUT_DIR, "l2_c_tuning.csv"), index=False)
    tir_cv.to_csv(os.path.join(OUTPUT_DIR, "ti_logistic_alpha_tuning.csv"), index=False)
    cov_cv.to_csv(os.path.join(OUTPUT_DIR, "cov_ti_logistic_alpha_tuning.csv"), index=False)
    gamma_df.to_csv(os.path.join(OUTPUT_DIR, "mixed_gamma_tuning.csv"), index=False)
