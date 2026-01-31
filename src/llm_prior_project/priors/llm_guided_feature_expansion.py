import numpy as np
import pandas as pd
import json
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer

class LLMGuidedFeatureExpansion(BaseEstimator, TransformerMixin):
    """
    Asks an LLM which non-linear features (interactions, squares, logs) 
    are scientifically relevant, then generates them.
    """
    
    def __init__(self, elicitor, max_interactions=5, include_logs=True):
        """
        Args:
            elicitor: Instance of your LLMPriorElicitor class.
            max_interactions: Hard limit on how many new terms to accept (prevents explosion).
            include_logs: Whether to allow logarithmic transformations (good for biological data).
        """
        self.elicitor = elicitor
        self.max_interactions = max_interactions
        self.include_logs = include_logs
        
        # State
        self.original_features_ = None
        self.new_features_map_ = {} # "feature_name": transformation_logic
        self.feature_names_out_ = None

    def fit(self, X, y=None, feature_names=None):
        # 1. Store original names
        if feature_names is None:
            if hasattr(X, "columns"):
                feature_names = list(X.columns)
            else:
                feature_names = [f"x{i}" for i in range(X.shape[1])]
        
        self.original_features_ = feature_names
        
        # 2. Construct the prompt
        prompt = self._build_prompt(feature_names)
        
        # 3. Call the LLM
        print(f"--- Asking LLM for feature interactions on: {feature_names} ---")
        response = self.elicitor.call(prompt)
        
        # 4. Parse response
        try:
            suggested_features = self._extract_suggestions(response)
            print(f"--- LLM Suggested {len(suggested_features)} new features ---")
            for f in suggested_features:
                print(f"  > {f}")
        except Exception as e:
            print(f"Warning: Failed to parse LLM feature suggestions. Using original features only. Error: {e}")
            suggested_features = []

        # 5. Filter and Map suggestions to actual math
        self.new_features_map_ = self._validate_and_map(suggested_features, feature_names)
        
        # 6. Set final feature names
        self.feature_names_out_ = self.original_features_ + list(self.new_features_map_.keys())
        
        return self

    def transform(self, X):
        # Ensure X is numpy or dataframe
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Validate shape
        if n_features != len(self.original_features_):
            raise ValueError(f"X has {n_features} features, but fit expecting {len(self.original_features_)}")

        # Create a DataFrame for easier indexing by name
        df = pd.DataFrame(X, columns=self.original_features_)
        
        # Calculate new columns
        new_cols = []
        for new_name, func in self.new_features_map_.items():
            try:
                col_data = func(df)
                new_cols.append(col_data)
            except Exception as e:
                print(f"Warning: Could not compute {new_name}. Skipping. Error: {e}")
        
        if not new_cols:
            return X
            
        # Stack original X with new features
        X_new = np.column_stack([X] + new_cols)
        return X_new

    def get_feature_names_out(self):
        return self.feature_names_out_

    # ---------------------------------------------------------
    # Internal Logic
    # ---------------------------------------------------------
    def _build_prompt(self, feature_names):
        return f"""
        You are an expert data scientist and domain expert. 
        I am training a linear model on a dataset with these variables: {', '.join(feature_names)}.
        
        The underlying phenomenon might be non-linear. 
        Identify the {self.max_interactions} most likely meaningful interactions or transformations.
        
        Valid operations:
        - Interaction: "var1 * var2"
        - Square: "var1^2"
        - Log: "log(var1)" (Only if variable is strictly positive like weight/age)
        
        Return ONLY valid JSON with this format:
        {{
            "features": [
                "bmi * age",
                "height^2",
                "log(income)"
            ]
        }}
        Do not explain. Just JSON.
        """

    def _extract_suggestions(self, text):
        # Use your existing regex logic or a simplified version
        import re
        fenced = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        candidate = fenced.group(1) if fenced else text
        
        try:
            data = json.loads(candidate)
            return data.get("features", [])
        except:
            # Fallback regex for list items if JSON fails
            return re.findall(r'"(.*?)"', text)

    def _validate_and_map(self, suggestions, original_names):
        """
        Parses strings like "bmi * age" into lambda functions.
        Securely ensures only valid columns are used.
        """
        valid_map = {}
        original_set = set(original_names)
        
        for expr in suggestions:
            if len(valid_map) >= self.max_interactions:
                break
                
            expr = expr.lower().replace(" ", "")
            
            # 1. Interaction: var1 * var2
            if "*" in expr:
                parts = expr.split("*")
                if len(parts) == 2 and parts[0] in original_set and parts[1] in original_set:
                    # Capture v1, v2 in closure using default args
                    func = lambda df, v1=parts[0], v2=parts[1]: df[v1] * df[v2]
                    valid_map[f"{parts[0]}_x_{parts[1]}"] = func

            # 2. Square: var^2
            elif "^2" in expr:
                var = expr.replace("^2", "")
                if var in original_set:
                    func = lambda df, v=var: df[v] ** 2
                    valid_map[f"{var}_sq"] = func

            # 3. Log: log(var)
            elif "log(" in expr and self.include_logs:
                # extract "bmi" from "log(bmi)"
                match = re.search(r"log\((.*?)\)", expr)
                if match:
                    var = match.group(1)
                    if var in original_set:
                        # Safety: add small epsilon or clip to avoid log(0)
                        func = lambda df, v=var: np.log(np.maximum(df[v], 1e-6))
                        valid_map[f"log_{var}"] = func

        return valid_map