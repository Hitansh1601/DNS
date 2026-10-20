"""Evaluation metrics for binary and multiclass. No fabricated results."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    balanced_accuracy_score, classification_report, precision_recall_curve,
)


def best_f1_threshold(y_true, y_score) -> Dict:
    """Security tradeoff: scan thresholds, report F1-optimal vs default 0.5.

    Why this matters: 0.5 is arbitrary. Lower threshold catches more attacks
    (higher recall, fewer FN) at cost of more false alarms (higher FPR).
    SOC teams tune this based on alert budget.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    # precision_recall_curve returns len(thr) == len(prec)-1
    best_i = int(np.argmax(f1s))
    best_thr = float(thr[best_i]) if best_i < len(thr) else 0.5
    # metrics at default 0.5
    pred_05 = (y_score >= 0.5).astype(int)
    pred_best = (y_score >= best_thr).astype(int)
    def _prf(y, p):
        return (float(precision_score(y, p, zero_division=0)),
                float(recall_score(y, p, zero_division=0)),
                float(f1_score(y, p, zero_division=0)))
    p05, r05, f05 = _prf(y_true, pred_05)
    pb, rb, fb = _prf(y_true, pred_best)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_best).ravel().tolist()
    return {"best_threshold": round(best_thr, 4),
            "at_0.5": {"precision": p05, "recall": r05, "f1": f05},
            "at_best": {"precision": pb, "recall": rb, "f1": fb,
                        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                        "fpr": float(fp / max(1, fp + tn)),
                        "fnr": float(fn / max(1, fn + tp))}}


def evaluate_classifier(y_true, y_pred, y_proba=None, task: str = "binary") -> Dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: Dict = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    out["confusion_matrix"] = cm.tolist()
    out["classification_report"] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    if task == "binary":
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        # FPR/FNR from 2x2 cm
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel().tolist()
            out.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                        "fpr": float(fp / max(1, fp + tn)),
                        "fnr": float(fn / max(1, fn + tp))})
        if y_proba is not None:
            try:
                prob1 = np.asarray(y_proba)[:, 1] if np.asarray(y_proba).ndim == 2 else np.asarray(y_proba)
                out["roc_auc"] = float(roc_auc_score(y_true, prob1))
                out["pr_auc"] = float(average_precision_score(y_true, prob1))
            except Exception as e:
                out["roc_auc_error"] = str(e)
    else:
        out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        out["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        out["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        out["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        if y_proba is not None:
            try:
                out["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
            except Exception as e:
                out["roc_auc_error"] = str(e)
    return out


def comparison_table(results: Dict) -> "object":
    import pandas as pd
    rows = []
    for name, m in results.get("models", {}).items():
        ev = m["eval"]
        cv = ev.get("cv", {})
        cv_f1 = cv.get("test_f1", {})
        rows.append({
            "model": name,
            "accuracy": ev.get("accuracy"),
            "balanced_acc": ev.get("balanced_accuracy"),
            "precision/recall/f1 or macro_f1": ev.get("f1", ev.get("macro_f1")),
            "roc_auc": ev.get("roc_auc", ev.get("roc_auc_ovr")),
            "cv_f1_mean": round(cv_f1.get("mean", 0), 4) if isinstance(cv_f1, dict) else None,
            "cv_f1_std": round(cv_f1.get("std", 0), 4) if isinstance(cv_f1, dict) else None,
            "train_s": round(ev.get("train_seconds", 0), 2),
        })
    # Sort: real models first by F1 desc, dummy last (baseline reference)
    df = pd.DataFrame(rows)
    if not df.empty:
        order = {"xgb": 0, "rf": 1, "logreg": 2, "dummy": 99}
        df["_o"] = df["model"].map(order).fillna(50)
        df = df.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    return df
