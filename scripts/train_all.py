"""End-to-end training. Uses real CSV if present, else clearly-synthetic demo data."""
from __future__ import annotations
import argparse, json, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import yaml

from src.data_loader import load_csv, detect_target_column
from src.synthetic import make_synthetic
from src.train import train_task, save_artifact
from src.evaluate import comparison_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CSV path. Default: first data/*.csv or synthetic.")
    ap.add_argument("--target", default=None)
    ap.add_argument("--synthetic-n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    with open("configs/model_config.yaml") as f:
        cfg = yaml.safe_load(f)

    csvs = sorted(glob.glob("data/*.csv"))
    # ignore synthetic unless explicitly chosen or no other csv
    real = [c for c in csvs if "synthetic" not in c]
    src_label = "synthetic (DEMO ONLY - not CIC data)"
    if a.data:
        df = load_csv(a.data)
        src_label = f"user-provided {a.data}"
    elif real:
        df = load_csv(real[0])
        src_label = f"real file {real[0]}"
    else:
        df = make_synthetic(a.synthetic_n, a.seed)
        df.to_csv("data/synthetic_dns_demo.csv", index=False)
    print(f"DATA SOURCE: {src_label} shape={df.shape}")

    target = a.target or detect_target_column(df)
    if target is None:
        raise SystemExit("Could not detect target column. Pass --target <name>. Columns: " + str(list(df.columns)))
    print("TARGET:", target, df[target].value_counts().to_dict())

    # Binary always; multiclass only if >=3 distinct labels that map to 3 buckets
    from src.data_loader import normalize_labels
    _, binfo = normalize_labels(df[target], task="binary")
    # check multiclass support: at least 2 of light/heavy keywords or >=3 uniques
    uniq = df[target].astype(str).str.lower().unique().tolist()
    multi_ok = (len(df[target].unique()) >= 3)
    all_metrics = {"data_source": src_label, "target_col": target, "shape": list(df.shape)}
    for task in (["binary", "multiclass"] if multi_ok else ["binary"]):
        print(f"\n=== TASK {task} ===")
        res = train_task(df, target, task)
        table = comparison_table(res)
        print(table.to_string())
        all_metrics[task] = {}
        for name, m in res["models"].items():
            ev = m["eval"]
            all_metrics[task][name] = {k: v for k, v in ev.items() if k not in ("confusion_matrix", "classification_report")}
            # save artifact for best? save all; app loads xgb by default, fallback rf/logreg
            tag = f"{task}"
            save_artifact(m["pipeline"], res["features"], res["label_info"]["classes"],
                          "binary" if res["n_classes"] == 2 else "multiclass",
                          ev, cfg, f"models/pipeline_{tag}_{name}.joblib")
            # confusion matrix figure
            try:
                import numpy as np
                from sklearn.metrics import ConfusionMatrixDisplay
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                disp = ConfusionMatrixDisplay(confusion_matrix=np.array(ev["confusion_matrix"]),
                                             display_labels=res["label_info"]["classes"])
                disp.plot(ax=ax2, colorbar=False)
                ax2.set_title(f"{task}-{name} confusion (source: {src_label[:40]})")
                fig2.tight_layout()
                fig2.savefig(f"reports/figures/cm_{task}_{name}.png", dpi=150)
                plt.close(fig2)
            except Exception as e:
                print("cm plot failed:", e)
        # default artifact: prefer xgb, else rf
        pref = "xgb" if "xgb" in res["models"] else list(res["models"])[0]
        save_artifact(res["models"][pref]["pipeline"], res["features"],
                      res["label_info"]["classes"],
                      "binary" if res["n_classes"] == 2 else "multiclass",
                      res["models"][pref]["eval"], cfg, f"models/pipeline_{task}.joblib")
    with open("reports/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print("\nWROTE reports/metrics.json + models/pipeline_*.joblib + reports/figures/cm_*.png")


if __name__ == "__main__":
    main()
