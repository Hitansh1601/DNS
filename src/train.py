"""Training: stratified splits, CV, tuning, baselines, threshold analysis, persistence."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.data_loader import normalize_labels
from src.preprocessing import build_preprocessor, clean_frame, split_num_cat
from src.validation import select_features
from src.evaluate import evaluate_classifier, best_f1_threshold


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
    # NOTE: n_jobs=-1 hangs on some Windows/OpenMP builds; use 1 for reliability.
    # Override via config only with a positive int.
    xgb_jobs = xp.get("n_jobs", 1)
    if not isinstance(xgb_jobs, int) or xgb_jobs < 1:
        xgb_jobs = 1
    xgb_clf = xgb.XGBClassifier(
        n_estimators=xp.get("n_estimators", 300), max_depth=xp.get("max_depth", 6),
        learning_rate=xp.get("learning_rate", 0.05), subsample=xp.get("subsample", 0.8),
        colsample_bytree=xp.get("colsample_bytree", 0.8),
        reg_alpha=xp.get("reg_alpha", 0.0), reg_lambda=xp.get("reg_lambda", 1.0),
        n_jobs=xgb_jobs, tree_method=xp.get("tree_method", "hist"),
        random_state=cfg.get("random_state", 42),
        objective="binary:logistic" if n_classes == 2 else "multi:softprob",
        eval_metric="logloss")
    dummy = DummyClassifier(strategy="stratified", random_state=cfg.get("random_state", 42))
    return {"logreg": lr, "rf": rf, "xgb": xgb_clf, "dummy": dummy}


def _param_grid(cfg: Dict, name: str):
    t = cfg.get("tuning", {})
    if name == "rf":
        g = t.get("rf_grid", {})
        return {"clf__n_estimators": g.get("n_estimators", [100, 200]),
                "clf__max_depth": g.get("max_depth", [None, 12, 20]),
                "clf__min_samples_split": g.get("min_samples_split", [2, 5])}
    if name == "xgb":
        g = t.get("xgb_grid", {})
        return {"clf__n_estimators": g.get("n_estimators", [200, 300]),
                "clf__max_depth": g.get("max_depth", [4, 6]),
                "clf__learning_rate": g.get("learning_rate", [0.03, 0.05, 0.1]),
                "clf__subsample": g.get("subsample", [0.8, 1.0])}
    return {}


def _cross_val_scores(pipe, Xtr, ytr, task: str, folds: int, seed: int) -> Dict:
    """Stratified K-fold on TRAIN only. Test set is never touched here."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {"acc": "accuracy", "bal_acc": "balanced_accuracy",
               "f1": "f1" if task == "binary" else "f1_macro"}
    try:
        sc = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=1)
        return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in sc.items() if k.startswith("test_")}
    except Exception as e:
        return {"error": str(e)}


def train_task(df: pd.DataFrame, target_col: str, task: str,
               config_path: str = "configs/model_config.yaml",
               tune: bool | None = None) -> Dict:
    cfg = load_config(config_path)
    seed = cfg.get("random_state", 42)
    df = clean_frame(df)
    feats = select_features(df, target_col, config_path)
    y_all, label_info = normalize_labels(df[target_col], task=task, config_path=config_path)
    X_all = df.loc[y_all.index, feats].copy()
    num, cat = split_num_cat(X_all, feats)
    Xtr, Xte, ytr, yte = split_data(X_all, y_all, cfg.get("test_size", 0.2), seed)
    n_classes = int(y_all.nunique())
    estimators = make_estimators(cfg, n_classes)
    tune_enabled = cfg.get("tuning", {}).get("enabled", False) if tune is None else tune
    n_iter = int(cfg.get("tuning", {}).get("n_iter", 8))
    cv_folds = int(cfg.get("cv", {}).get("folds", 3))
    do_threshold = bool(cfg.get("threshold", {}).get("tune_binary_threshold", True))
    task_tag = "binary" if n_classes == 2 else "multiclass"
    results: Dict = {"task": task, "target_col": target_col, "features": feats,
                     "num_features": num, "cat_features": cat,
                     "label_info": label_info, "n_classes": n_classes,
                     "models": {}}
    for name, clf in estimators.items():
        scale = (name == "logreg")
        prep = build_preprocessor(num, cat, scale_numeric=scale)
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        cv_scores = _cross_val_scores(pipe, Xtr, ytr, task_tag, cv_folds, seed)
        best_params = None
        t0 = time.time()
        if tune_enabled and name in ("rf", "xgb"):
            try:
                search = RandomizedSearchCV(
                    pipe, _param_grid(cfg, name), n_iter=min(n_iter, 6),
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
                    scoring="f1" if task_tag == "binary" else "f1_macro",
                    random_state=seed, n_jobs=1)
                search.fit(Xtr, ytr)
                pipe = search.best_estimator_
                best_params = {k: v for k, v in search.best_params_.items()}
                cv_scores["tuning_best_score"] = float(search.best_score_)
            except Exception as e:
                cv_scores["tuning_error"] = str(e)
                pipe.fit(Xtr, ytr)
        else:
            pipe.fit(Xtr, ytr)
        train_t = time.time() - t0
        t1 = time.time()
        pred = pipe.predict(Xte)
        infer_t = (time.time() - t1) / max(1, len(Xte))
        try:
            proba = pipe.predict_proba(Xte)
        except Exception:
            proba = None
        ev = evaluate_classifier(yte, pred, proba, task=task_tag)
        ev["train_seconds"] = train_t
        ev["infer_seconds_per_row"] = infer_t
        ev["cv"] = cv_scores
        if best_params is not None:
            ev["best_params"] = best_params
        if do_threshold and task_tag == "binary" and proba is not None and name != "dummy":
            try:
                ev["threshold_analysis"] = best_f1_threshold(yte, np.asarray(proba)[:, 1])
            except Exception as e:
                ev["threshold_analysis"] = {"error": str(e)}
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
