"""SHAP explainability. Handles binary/multiclass and version differences."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def _prep_background(prep, X: pd.DataFrame, n: int = 200, seed: int = 42) -> np.ndarray:
    Xs = X.sample(min(n, len(X)), random_state=seed)
    return prep.transform(Xs)


def shap_global_local(pipeline, X_test: pd.DataFrame, feature_names=None,
                      background_n: int = 200, explain_n: int = 500,
                      seed: int = 42) -> Dict:
    """Return shap values + expected_value. Caller plots/saves."""
    import shap
    from src.preprocessing import clean_frame
    X_test = clean_frame(X_test)
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    Xt = X_test.sample(min(explain_n, len(X_test)), random_state=seed)
    Xb = _prep_background(prep, X_test, background_n, seed)
    Xe = prep.transform(Xt)
    try:
        out_names = list(prep.get_feature_names_out())
    except Exception:
        out_names = [f"f{i}" for i in range(Xe.shape[1])]
    # Tree models: TreeExplainer; linear: LinearExplainer fallback to Kernel if needed
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xe)
    except Exception:
        try:
            explainer = shap.Explainer(clf, Xb)
            exp = explainer(Xe)
            sv = exp.values
        except Exception as e:
            return {"error": str(e), "feature_names": out_names}
    sv = np.asarray(sv) if not isinstance(sv, list) else sv
    return {"shap_values": sv, "feature_names": out_names,
            "X_sample": Xt, "Xe_shape": list(Xe.shape)}


def top_features(shap_values, feature_names, topk: int = 15) -> list:
    sv = np.asarray(shap_values)
    # binary: shape (n, f); multiclass: (n, f, c) or list
    if sv.ndim == 3:
        imp = np.abs(sv).mean(axis=(0, 2))
    else:
        imp = np.abs(sv).mean(axis=0)
    idx = np.argsort(imp)[::-1][:topk]
    return [(feature_names[i], float(imp[i])) for i in idx]
