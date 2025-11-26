# ============================================================================
# causal_selector_utils.py
# ============================================================================
# General-purpose feature selector for causal feature subsets.
# Provides utilities for selecting and filtering features based on causality.
# ============================================================================
import numpy as np
import pandas as pd


def select_causal_features(X, feature_names, causal_features, verbose=True):
    """
    Select only user-defined causal features from the dataset.
    
    Works with either numpy arrays or pandas DataFrames.
    
    Args:
        X : pd.DataFrame or np.ndarray
            Input feature matrix.
        feature_names : list[str]
            List of all feature names (must align with columns in X).
        causal_features : list[str]
            Names of causal features to keep.
        verbose : bool
            Whether to print summary information.
    
    Returns:
        X_causal : np.ndarray or pd.DataFrame
            Filtered data with only causal features.
        causal_indices : list[int]
            Indices of kept causal features.
        causal_feature_names : list[str]
            Names of kept causal features.
    """
    if len(feature_names) == 0:
        raise ValueError("feature_names cannot be empty.")
    
    if len(causal_features) == 0:
        raise ValueError("causal_features cannot be empty – nothing to keep.")
    
    # Validate that all causal features exist in feature_names
    missing = [f for f in causal_features if f not in feature_names]
    if missing:
        raise ValueError(f"These causal features are not in feature_names: {missing}")
    
    # Identify indices of causal features
    causal_indices = [feature_names.index(f) for f in causal_features]
    
    # Select data
    if isinstance(X, pd.DataFrame):
        X_causal = X[causal_features].copy()
    else:
        X_causal = np.asarray(X)[:, causal_indices]
    
    if verbose:
        dropped = [f for f in feature_names if f not in causal_features]
        print("\nCausal feature selection summary:")
        print(f"  Kept {len(causal_features)} / {len(feature_names)} features.")
        print(f"  Kept: {causal_features}")
        if dropped:
            print(f"  Dropped: {dropped}")
    
    return X_causal, causal_indices, causal_features


def filter_features_by_indices(X, feature_indices):
    """
    Filter features by their indices.
    
    Args:
        X : pd.DataFrame or np.ndarray
            Input feature matrix.
        feature_indices : list[int]
            Indices of features to keep.
    
    Returns:
        X_filtered : np.ndarray or pd.DataFrame
            Filtered data with only selected features.
    """
    if isinstance(X, pd.DataFrame):
        X_filtered = X.iloc[:, feature_indices].copy()
    else:
        X_filtered = np.asarray(X)[:, feature_indices]
    
    return X_filtered


def get_causal_coefficients(coefficients, feature_names, causal_features):
    """
    Extract coefficients corresponding to causal features.
    
    Args:
        coefficients : np.ndarray
            Full coefficient array.
        feature_names : list[str]
            List of all feature names.
        causal_features : list[str]
            Names of causal features.
    
    Returns:
        causal_coef : np.ndarray
            Coefficients for causal features only.
        causal_indices : list[int]
            Indices of causal features.
    """
    causal_indices = [feature_names.index(f) for f in causal_features 
                      if f in feature_names]
    
    if len(causal_indices) == 0:
        raise ValueError("No causal features found in feature_names")
    
    causal_coef = coefficients[causal_indices]
    
    return causal_coef, causal_indices


def drop_noncausal_features(df, causal_features, verbose=True):
    """
    Drop all columns from a pandas DataFrame that are not causal.
    
    Args:
        df : pd.DataFrame
            Input dataframe.
        causal_features : list[str]
            Names of causal features to keep.
        verbose : bool
            Whether to print summary information.
    
    Returns:
        df_causal : pd.DataFrame
            DataFrame with only causal features.
    """
    all_cols = list(df.columns)
    missing = [f for f in causal_features if f not in all_cols]
    if missing:
        raise ValueError(f"These causal features are missing in df: {missing}")
    
    dropped = [c for c in all_cols if c not in causal_features]
    df_causal = df[causal_features].copy()
    
    if verbose:
        print("\nDropped non-causal features:")
        print(f"  Kept ({len(causal_features)}): {causal_features}")
        if dropped:
            print(f"  Removed ({len(dropped)}): {dropped}")
    
    return df_causal


def get_feature_mask(feature_names, selected_features):
    """
    Create a boolean mask for feature selection.
    
    Args:
        feature_names : list[str]
            List of all feature names.
        selected_features : list[str]
            Names of features to select.
    
    Returns:
        mask : np.ndarray
            Boolean mask (True for selected features).
    """
    mask = np.array([f in selected_features for f in feature_names])
    return mask


def apply_feature_mask(X, mask):
    """
    Apply a boolean mask to select features.
    
    Args:
        X : pd.DataFrame or np.ndarray
            Input feature matrix.
        mask : np.ndarray
            Boolean mask for feature selection.
    
    Returns:
        X_masked : np.ndarray or pd.DataFrame
            Data with masked features only.
    """
    if isinstance(X, pd.DataFrame):
        X_masked = X.loc[:, mask].copy()
    else:
        X_masked = np.asarray(X)[:, mask]
    
    return X_masked