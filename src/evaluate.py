"""Evaluation metrics for binary and multiclass. No fabricated results."""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    balanced_accuracy_score, classification_report,
)


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
        rows.append({
            "model": name,
            "accuracy": ev.get("accuracy"),
            "balanced_acc": ev.get("balanced_accuracy"),
            "precision/recall/f1 or macro_f1": ev.get("f1", ev.get("macro_f1")),
            "roc_auc": ev.get("roc_auc", ev.get("roc_auc_ovr")),
            "train_s": round(ev.get("train_seconds", 0), 2),
        })
    return pd.DataFrame(rows)
