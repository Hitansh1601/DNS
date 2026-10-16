"""Generate SHAP summary + bar plots for the default binary XGB/RF pipeline."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, pandas as pd, numpy as np
from pathlib import Path
from src.explain import shap_global_local, top_features

Path("reports/figures").mkdir(parents=True, exist_ok=True)
for task in ("binary", "multiclass"):
    p = Path(f"models/pipeline_{task}.joblib")
    if not p.exists():
        print(f"skip {task}: no artifact"); continue
    art = joblib.load(p)
    # find a csv with matching features: prefer real, else synthetic
    import glob
    df = pd.read_csv("data/synthetic_dns_demo.csv")
    feats = art["feature_names"]
    if not set(feats) <= set(df.columns):
        print(f"skip {task}: feature mismatch with synthetic demo"); continue
    r = shap_global_local(art["pipeline"], df[feats], explain_n=400, background_n=150)
    if "error" in r:
        print(task, "SHAP error:", r["error"]); continue
    sv, fn = np.asarray(r["shap_values"]), r["feature_names"]
    print(task, "sv shape", sv.shape, "SYNTHETIC DEMO ONLY")
    tops = top_features(sv, fn, 15)
    # bar plot (mean |SHAP|)
    names, vals = zip(*tops)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(list(names)[::-1], list(vals)[::-1])
    ax.set_title(f"{task} mean|SHAP| top15 (SYNTHETIC DEMO, not CIC)")
    fig.tight_layout()
    fig.savefig(f"reports/figures/shap_bar_{task}.png", dpi=150)
    plt.close(fig)
    print(f"WROTE reports/figures/shap_bar_{task}.png")
