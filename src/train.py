"""Training: stratified splits, 3 models, timing, ablations, persistence."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Tuple
import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.data_loader import normalize_labels
from src.preprocessing import build_preprocessor, clean_frame, split_num_cat
from src.validation import select_features
from src.evaluate import evaluate_classifier


def load_config(path: str = "configs/model_config.yaml") -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(sss.split(X, y))
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]


def make_estimators(cfg: Dict, n_classes: int):
    m = cfg.get("models", {})
    lr = LogisticRegression(max_iter=m.get("logreg", {}).get("max_iter", 1000),
                            class_weight=m.get("logreg", {}).get("class_weight", "balanced"),
                            C=m.get("logreg", {}).get("C", 1.0))
    rf = RandomForestClassifier(
        n_estimators=m.get("rf", {}).get("n_estimators", 200),
        max_depth=m.get("rf", {}).get("max_depth"),
        min_samples_split=m.get("rf", {}).get("min_samples_split", 2),
        class_weight=m.get("rf", {}).get("class_weight", "balanced_subsample"),
        n_jobs=-1, random_state=cfg.get("random_state", 42))
    xp = m.get("xgb", {})
    xgb_clf = xgb.XGBClassifier(
        n_estimators=xp.get("n_estimators", 300), max_depth=xp.get("max_depth", 6),
        learning_rate=xp.get("learning_rate", 0.05), subsample=xp.get("subsample", 0.8),
        colsample_bytree=xp.get("colsample_bytree", 0.8),
        reg_alpha=xp.get("reg_alpha", 0.0), reg_lambda=xp.get("reg_lambda", 1.0),
        n_jobs=-1, tree_method=xp.get("tree_method", "hist"),
        random_state=cfg.get("random_state", 42),
        objective="binary:logistic" if n_classes == 2 else "multi:softprob",
        eval_metric="logloss")
    return {"logreg": lr, "rf": rf, "xgb": xgb_clf}


def train_task(df: pd.DataFrame, target_col: str, task: str,
               config_path: str = "configs/model_config.yaml") -> Dict:
    cfg = load_config(config_path)
    seed = cfg.get("random_state", 42)
    df = clean_frame(df)
    feats = select_features(df, target_col, config_path)
    y_all, label_info = normalize_labels(df[target_col], task=task, config_path=config_path)
    X_all = df.loc[y_all.index, feats].copy()
    # keep order, store dtypes for split
    num, cat = split_num_cat(X_all, feats)
    Xtr, Xte, ytr, yte = split_data(X_all, y_all, cfg.get("test_size", 0.2), seed)
    n_classes = int(y_all.nunique())
    estimators = make_estimators(cfg, n_classes)
    results: Dict = {"task": task, "target_col": target_col, "features": feats,
                     "num_features": num, "cat_features": cat,
                     "label_info": label_info, "n_classes": n_classes,
                     "models": {}}
    for name, clf in estimators.items():
        scale = (name == "logreg")  # scale only linear model
        prep = build_preprocessor(num, cat, scale_numeric=scale)
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        train_t = time.time() - t0
        t1 = time.time()
        pred = pipe.predict(Xte)
        infer_t = (time.time() - t1) / max(1, len(Xte))
        try:
            proba = pipe.predict_proba(Xte)
        except Exception:
            proba = None
        ev = evaluate_classifier(yte, pred, proba, task="binary" if n_classes == 2 else "multiclass")
        ev["train_seconds"] = train_t
        ev["infer_seconds_per_row"] = infer_t
        results["models"][name] = {"eval": ev, "pipeline": pipe}
    return results


def save_artifact(pipeline, feature_names, label_classes, task, metrics, config_snapshot, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_names": feature_names,
                 "label_classes": label_classes, "task": task,
                 "metrics": metrics, "config_snapshot": config_snapshot}, path)
    with open(Path(path).with_suffix(".meta.json"), "w") as f:
        json.dump({"feature_names": feature_names, "label_classes": label_classes,
                   "task": task, "metrics": metrics}, f, indent=2, default=str)
