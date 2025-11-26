import os, sys 
import pandas as pd 
import numpy as np 
import inspect

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from llm_prior_project.priors.target_informed_model import TargetInformedModel
from target_elicitor import LLMTargetElicitor

np.random.seed(42)

# --------------------------
# Prepare small synthetic dataset
# --------------------------

X = np.random.randn(100, 3)
true_coef = np.array([3.0, -1.5, 2.0])
y = X @ true_coef + 0.5 * np.random.randn(100)

df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
df["y"] = y

X_split = df[["feature1", "feature2", "feature3"]].values
y_split = df["y"].values

# --------------------------
# Confirm correct file is loaded
# --------------------------

import target_elicitor
print("\nLoaded LLMTargetElicitor from:", target_elicitor.__file__)

print("\n--- Showing actual run_rephrased_batch being executed ---")
print(inspect.getsource(target_elicitor.LLMTargetElicitor.run_rephrased_batch))

# --------------------------
# Set up elicitor
# --------------------------

elicitor = LLMTargetElicitor()

features = ["feature1", "feature2", "feature3"]

prompt = (
    "Provide coefficient targets for the features feature1, feature2, feature3. "
    "Reply ONLY in plaintext format:\n"
    "feature1: <number>\n"
    "feature2: <number>\n"
    "feature3: <number>\n"
    "confidence: <number between 0 and 1>\n"
)

print("\n--- Running batch target elicitation (10 variants) ---\n")

batch_results = elicitor.run_rephrased_batch(
    prompt=prompt,
    feature_names=features,
    n_variants=10,
    out_path="elicitation_batch_test_10.json"
)

print("\nBatch results keys:", batch_results.keys())
print("Variants parsed:", len(batch_results["variants"]))

# Show each variant’s targets + confidence
for i, v in enumerate(batch_results["variants"]):
    print(f"\nVariant {i}:")
    print("Rephrased prompt:", v["prompt_rephrased"])
    print("Targets:", v["targets"])
    print("Confidence:", v["confidence"])

print("\nUnweighted average targets:", batch_results["avg_targets_unweighted"])
print("Weighted average targets:", batch_results["avg_targets_weighted"])
print("Variance:", batch_results["targets_var"])
print("STD:", batch_results["targets_std"])

# Pick final targets
final_targets = batch_results["avg_targets_weighted"]
if final_targets is None:
    final_targets = batch_results["avg_targets_unweighted"]

print("\nFinal targets used for model:", final_targets)

model = TargetInformedModel(
    alpha=1.0,
    model_type="ridge",
    targets=np.array(final_targets)
)

model.fit(X_split, y_split, feature_names=features)
r2 = model.score(X_split, y_split)

print(f"\nModel R² with averaged LLM-derived targets: {r2}")

