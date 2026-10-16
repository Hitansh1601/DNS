"""Inference: load fitted pipeline, validate schema, predict CSV/single row."""
from __future__ import annotations
from typing import Dict, List
import joblib
import pandas as pd
from src.validation import validate_inference_df
from src.preprocessing import clean_frame


def load_artifact(path: str) -> Dict:
    obj = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj
    # legacy: raw pipeline
    return {"pipeline": obj, "feature_names": getattr(obj, "feature_names_in_", None)}


def predict_df(artifact: Dict, df: pd.DataFrame) -> pd.DataFrame:
    feats: List[str] = list(artifact["feature_names"])
    val = validate_inference_df(df, feats)
    if not val["ok"]:
        raise ValueError(f"Missing required columns: {val['missing']}. Extra ignored: {val['extra']}")
    X = clean_frame(df[feats].copy())
    pipe = artifact["pipeline"]
    pred = pipe.predict(X)
    out = df.copy()
    classes = artifact.get("label_classes", sorted(pd.Series(pred).unique().tolist()))
    # map int -> name if classes provided
    if isinstance(classes, list) and len(classes) > max(map(int, pd.Series(pred).unique())):
        out["prediction"] = [classes[int(p)] for p in pred]
    else:
        out["prediction"] = pred
    try:
        proba = pipe.predict_proba(X)
        import numpy as np
        proba = np.asarray(proba)
        for i, c in enumerate(classes if len(classes) == proba.shape[1] else range(proba.shape[1])):
            out[f"score_{c}"] = proba[:, i]
        out["score_max"] = proba.max(axis=1)
    except Exception:
        pass
    # disclaimer column
    out["note"] = "Estimate from training patterns, not proof of maliciousness."
    return out
