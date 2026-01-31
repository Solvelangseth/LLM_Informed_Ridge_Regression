import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, f1_score
from dotenv import load_dotenv

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_statistics import LLMPriorElicitor, CovarianceTargetInformedModel
from llm_statistics.utils.preprocessing import fit_imputer_scaler, transform_data
from llm_statistics.utils.evaluation import evaluate_binary_classifier

# Fixed Alpha as requested
ALPHA_GRID = [1.0]

def find_best_threshold(y_true, y_probs):
    """Find the threshold that maximizes F1 score."""
    best_thresh = 0.5
    best_f1 = 0.0
    thresholds = np.linspace(0.01, 0.99, 99)
    
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
            
    return best_thresh, best_f1

def main():
    print("="*80)
    print("TEST: Covariance Target Informed Model (Alpha + Threshold Tuned)")
    print("="*80)

    # 1. Load Data
    print("\n1. Loading Data...")
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    data_dir = os.path.join(os.path.dirname(__file__), "../data/heart+disease")
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found at {data_dir}")
        return

    df_cleveland = pd.read_csv(f"{data_dir}/processed.cleveland.data", names=columns, na_values="?")
    df_hungary = pd.read_csv(f"{data_dir}/processed.hungarian.data", names=columns, na_values="?")
    df_switzerland = pd.read_csv(f"{data_dir}/processed.switzerland.data", names=columns, na_values="?")

    source_df = df_cleveland.copy()
    ood_df = pd.concat([df_hungary, df_switzerland], axis=0).copy()
    
    target_col = "target"
    features = [c for c in columns if c != target_col]

    def make_xy(df):
        X = df[features].copy()
        y = (df[target_col] > 0).astype(int).values
        return X, y

    X_source_raw, y_source = make_xy(source_df)
    X_ood_raw, y_ood = make_xy(ood_df)

    # Split (Same seed as analysis)
    print("2. Splitting and Preprocessing...")
    X_train_full_raw, X_id_raw, y_train_full, y_id = train_test_split(
        X_source_raw, y_source, test_size=0.20, random_state=42, stratify=y_source
    )

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_full_raw, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )

    # Preprocess
    imp, sc = fit_imputer_scaler(X_train_raw)
    
    X_train_s = transform_data(imp, sc, X_train_raw)
    X_val_s = transform_data(imp, sc, X_val_raw)
    X_train_full_s = transform_data(imp, sc, X_train_full_raw)
    X_id_s = transform_data(imp, sc, X_id_raw)
    X_ood_s = transform_data(imp, sc, X_ood_raw)

    # 2. Elicit Priors
    priors_file = "elicited_priors_gpt5.1_15samples.json"
    
    if os.path.exists(priors_file):
        print(f"\n3. Loading Priors from {priors_file}...")
        with open(priors_file, "r") as f:
            targets = json.load(f)
    else:
        print("Priors file not found! Falling back to generation (skipped here).")
        return

    elicitor = LLMPriorElicitor("gpt-5.1")
    keys, start_means, cov, precision = elicitor._compute_prior_statistics(targets)

    print("\nElicited Statistics:")
    print(f"Features: {keys}")
    print(f"Precision Matrix (P) trace: {np.trace(precision):.3f}")

    # 3. Fit & Tune Models
    print("\n4. Tuning Covariance Models (Alpha Grid Search + Threshold Optimization)...")
    
    final_results = []
    
    for m_type in ["ridge", "logistic"]:
        print(f"\n--- Tuning {m_type.upper()} ---")
        best_val_f1 = -1.0
        best_alpha = None
        best_thresh = None
        
        # Grid Search Alpha
        for alpha in ALPHA_GRID:
            # Fit on Train
            try:
                model = CovarianceTargetInformedModel(
                    mu=start_means,
                    P=precision,
                    model_type=m_type,
                    fit_intercept=True,
                    alpha=alpha
                )
                model.fit(X_train_s, y_train, feature_names=features)
                
                # Predict Val
                if m_type == "ridge":
                     z = model.predict(X_val_s)
                     p_val = 1 / (1 + np.exp(-z))
                else:
                     p_val = model.predict_proba(X_val_s)[:, 1]
                
                # Find best threshold for this alpha
                t, f1 = find_best_threshold(y_val, p_val)
                
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_alpha = alpha
                    best_thresh = t
            except Exception as e:
                print(f"Error fitting alpha={alpha}: {e}")
                continue

        print(f"Winner: Alpha={best_alpha}, Thresh={best_thresh:.2f}, Val F1={best_val_f1:.4f}")
        
        # 4. Refit on Full Train using Best Alpha
        print("    Refitting on full train and evaluating OOD...")
        final_model = CovarianceTargetInformedModel(
            mu=start_means,
            P=precision,
            model_type=m_type,
            fit_intercept=True,
            alpha=best_alpha
        )
        final_model.fit(X_train_full_s, y_train_full, feature_names=features)
        
        if m_type == "ridge":
            z_ood = final_model.predict(X_ood_s)
            p_ood = 1 / (1 + np.exp(-z_ood))
        else:
            p_ood = final_model.predict_proba(X_ood_s)[:, 1]
            
        preds_ood = (p_ood >= best_thresh).astype(int)
        
        ood_f1 = f1_score(y_ood, preds_ood)
        ood_auc = roc_auc_score(y_ood, p_ood)
        ood_logloss = log_loss(y_ood, np.clip(p_ood, 1e-15, 1-1e-15))
        
        final_results.append({
            "model": f"Covariance {m_type.capitalize()}",
            "alpha": best_alpha,
            "thresh": best_thresh,
            "ood_f1": ood_f1,
            "ood_auc": ood_auc,
            "ood_logloss": ood_logloss
        })

    print("\n" + "="*80)
    print("FINAL RESULTS (Covariance Model: Optimized Alpha & Threshold)")
    print("="*80)
    print(pd.DataFrame(final_results).round(4))

if __name__ == "__main__":
    main()
