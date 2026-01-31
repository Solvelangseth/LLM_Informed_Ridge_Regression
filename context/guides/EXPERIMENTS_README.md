# Experiments Guide (Codex-Friendly)

Purpose
- Provide a locked-in, repeatable template for new datasets.
- Keep new work isolated in a new folder so existing experiments are untouched.
- Standardize preprocessing, splits, tuning, and reporting for classification and regression.

How to use (for a new Codex agent)
1) Create a new work folder under `experiments/runs/` (e.g., `experiments/runs/2026-01-31_adult_v2/`).
2) Copy the template script(s) from `experiments/templates/` into the new folder.
3) Edit only the new folder files. Do not edit core code or existing experiments.
4) Run the script(s) inside the new folder.
5) Save outputs to a subfolder `results/` inside the run folder.

Project context (from README)
- Thesis goal: test whether LLM-elicited priors improve generalization (OOD) vs standard baselines.
- Models to compare (regression): OLS, Ridge, Target-Informed Ridge, Covariance TI Ridge, Mixed model.
- Models to compare (classification): L2 Logistic, Target-Informed Logistic, Covariance TI Logistic, Mixed model.
- Metrics:
  - Regression: MSE
  - Classification: AUC or LogLoss (choose one and stay consistent)

Standard experiment recipe (classification)
- Domain split (OOD): define a natural split (e.g., group, demographic, source/target dataset).
- Preprocess using source train only (fit transforms on train; apply to val/id/ood).
- Tune hyperparameters using validation set from source.
- Evaluate and report metrics on ID test (source) and OOD test (target).
- Save predictions and tuning curves.

Standard experiment recipe (regression)
- Same split and preprocessing rules.
- Tune alphas by validation MSE.
- Use LLM prior for target-informed model and sample-based priors for covariance model.
- Evaluate MSE on ID and OOD.
- Save tuning curves and results tables.

What to report
- Data split sizes (train/val/id/ood).
- Chosen hyperparameters (alpha, gamma, C, etc.).
- ID and OOD metrics in tables.
- Short interpretation (e.g., “LLM priors helped OOD MSE by X%”).

Guardrails
- Never edit heart disease scripts.
- Never edit core library modules unless requested.
- Always write outputs to the new run folder.

See also
- `experiments/templates/experiment_classification_template.py`
- `experiments/templates/experiment_regression_template.py`
