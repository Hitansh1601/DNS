"""DNS ThreatLens — Explainable DNS Traffic Detection (offline feature-CSV tool)."""
from __future__ import annotations
import json, sqlite3
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DNS ThreatLens", layout="wide")

DISCLAIMER = ("Predictions are estimates from training patterns, not proof a domain/host/event is malicious. "
              "This tool does not detect every DNS attack. It never resolves domains, fetches URLs, or blocks traffic.")

@st.cache_resource
def load_artifact(path: str):
    try:
        return joblib.load(path)
    except Exception:
        return None

def load_metrics():
    p = Path("reports/metrics.json")
    if p.exists():
        return json.loads(p.read_text())
    return {}

def artifact_for(task: str):
    for cand in [f"models/pipeline_{task}.joblib", f"models/pipeline_{task}_xgb.joblib",
                 f"models/pipeline_binary.joblib"]:
        a = load_artifact(cand)
        if a is not None:
            return a, cand
    return None, None

# --- history (optional sqlite) ---
def log_history(rows: pd.DataFrame, task: str):
    try:
        con = sqlite3.connect("app/history.db")
        rows.assign(task=task).to_sql("predictions", con, if_exists="append", index=False)
        con.close()
    except Exception as e:
        st.warning(f"History disabled: {e}")

# --- UI ---
st.sidebar.title("DNS ThreatLens")
page = st.sidebar.radio("Page", ["Overview", "Analyze DNS records", "Explain a prediction",
                                 "Model evaluation", "Prediction history"])
st.sidebar.caption(DISCLAIMER)

metrics = load_metrics()

if page == "Overview":
    st.title("DNS ThreatLens — Explainable DNS Traffic Detection")
    st.info(DISCLAIMER)
    st.markdown("""
**What this does:** classifies *prepared DNS feature records* (CSV rows) as benign / attack
(binary) or benign / light / heavy (multiclass, if labels support it).
**What it does NOT do:** live capture, PCAP parsing, DNS resolution, blocking, or SIEM integration.
""")
    st.subheader("Artifacts")
    st.json({k: (v if k in ("data_source", "target_col", "shape") else list(v.keys()) if isinstance(v, dict) else v)
             for k, v in metrics.items()})
    for t in ("binary", "multiclass"):
        art, path = artifact_for(t)
        if art:
            st.success(f"{t}: loaded `{path}` features={len(art['feature_names'])} classes={art['label_classes']}")
            st.write("Top features (order):", art["feature_names"][:12])
        else:
            st.warning(f"{t}: no pipeline found. Run `python scripts/train_all.py`.")

elif page == "Analyze DNS records":
    st.title("Analyze DNS records (CSV of prepared features)")
    st.caption("Upload a CSV with the same feature columns the model was trained on. Raw domain strings alone are not enough.")
    task = st.selectbox("Task", ["binary", "multiclass"])
    art, path = artifact_for(task)
    if not art:
        st.error("No model artifact. Train first.")
        st.stop()
    st.write(f"Model: `{path}` expects {len(art['feature_names'])} features.")
    with st.expander("Expected schema"):
        st.code("\n".join(map(str, art["feature_names"])))
    f = st.file_uploader("Upload feature CSV", type=["csv"])
    demo = st.checkbox("Use synthetic demo file (data/synthetic_dns_demo.csv, clearly synthetic)")
    df = None
    if demo and Path("data/synthetic_dns_demo.csv").exists():
        df = pd.read_csv("data/synthetic_dns_demo.csv", nrows=500)
        st.info("Demo records — SYNTHETIC ONLY, not real detections.")
    elif f:
        df = pd.read_csv(f)
    if df is not None:
        from src.predict import predict_df
        exp, missing, extra = set(art["feature_names"]), set(art["feature_names"]) - set(df.columns), set(df.columns) - set(art["feature_names"])
        if missing:
            st.error(f"Missing columns: {sorted(missing)}")
        else:
            if extra:
                st.warning(f"Extra columns ignored: {sorted(extra)}")
            try:
                res = predict_df(art, df)
                st.dataframe(res.head(100))
                st.bar_chart(res["prediction"].value_counts())
                csv = res.to_csv(index=False).encode()
                st.download_button("Download results CSV", csv, "dns_threatlens_results.csv", "text/csv")
                if st.checkbox("Save to history (SQLite, optional)"):
                    log_history(res[["prediction"] + [c for c in res.columns if c.startswith("score_")]].head(500), task)
                    st.success("Logged.")
            except Exception as e:
                st.error(str(e))

elif page == "Explain a prediction":
    st.title("Explain a prediction")
    st.caption("SHAP values explain the model, not causation. Positive pushes toward the predicted class.")
    task = st.selectbox("Task", ["binary", "multiclass"], key="ex_task")
    art, _ = artifact_for(task)
    if not art:
        st.error("No model artifact."); st.stop()
    st.write("Provide one record as JSON or upload single-row CSV matching expected features.")
    txt = st.text_area("JSON record (optional)", "{}")
    up = st.file_uploader("or single-row CSV", type=["csv"], key="ex_csv")
    row = None
    if up:
        row = pd.read_csv(up).head(1)
    elif txt.strip() not in ("", "{}"):
        try:
            row = pd.DataFrame([json.loads(txt)])
        except Exception as e:
            st.error(f"Bad JSON: {e}")
    if row is not None:
        from src.predict import predict_df
        try:
            res = predict_df(art, row)
            st.write(res.T)
            # SHAP local bar for this row
            try:
                from src.explain import shap_global_local
                import matplotlib.pyplot as plt
                import numpy as np
                pipe = art["pipeline"]
                prep = pipe.named_steps["prep"]
                Xe = prep.transform(row[art["feature_names"]])
                try:
                    fn = list(prep.get_feature_names_out())
                except Exception:
                    fn = art["feature_names"]
                import shap
                ex = shap.TreeExplainer(pipe.named_steps["clf"])
                sv = ex.shap_values(Xe)
                sv = np.asarray(sv)
                v = sv[0] if sv.ndim == 2 else sv[0, :, 0]
                idx = np.argsort(np.abs(v))[::-1][:12]
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh([fn[i] for i in idx][::-1], v[idx][::-1])
                ax.set_title("Top SHAP contributions (this record)")
                st.pyplot(fig)
                st.caption("Bars right = pushes toward predicted class; left = pushes away. Model-specific, not proof.")
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")
        except Exception as e:
            st.error(str(e))

elif page == "Model evaluation":
    st.title("Model evaluation (offline test set)")
    st.caption("Metrics below are from the saved offline test split, not from your uploads.")
    st.json(metrics)
    import glob
    for img in sorted(glob.glob("reports/figures/cm_*.png")):
        st.image(img, caption=img)

else:
    st.title("Prediction history")
    if st.button("Clear history"):
        try:
            Path("app/history.db").unlink(missing_ok=True)
            st.success("Cleared.")
        except Exception as e:
            st.error(str(e))
    try:
        con = sqlite3.connect("app/history.db")
        df = pd.read_sql("SELECT * FROM predictions ORDER BY rowid DESC LIMIT 500", con)
        con.close()
        st.dataframe(df)
    except Exception:
        st.info("No history yet (SQLite optional).")
