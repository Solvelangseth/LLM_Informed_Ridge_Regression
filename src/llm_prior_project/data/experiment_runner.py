from llm_prior_project.data.causal_selector_utils import select_causal_features, load_causal_features
import sys 
import os
sys.path.append(os.path.dirname(os.getcwd()))
from llm_prior_project.priors.target_informed_model import TargetInformedModel
from llm_prior_project.data.llm_ensemble_utils import run_full_ensemble_pipeline, make_default_models

def run_target_informed_experiment(X_train, y_train, X_test, y_test,
                                   targets, alphas, model_type="ridge"):
    best_alpha, scores = cross_validate_alpha(X_train, y_train, targets, alphas, model_type)
    model = TargetInformedModel(alpha=best_alpha, model_type=model_type, targets=targets)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    if model_type == "ridge":
        metric = r2_score(y_test, preds)
    else:
        from sklearn.metrics import roc_auc_score, log_loss
        metric = {
            "auc": roc_auc_score(y_test, preds),
            "logloss": log_loss(y_test, preds),
        }
    return model, best_alpha, metric, scores
