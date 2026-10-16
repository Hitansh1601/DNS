# Academic Report — Explainable ML-Based Malicious DNS Traffic Detection

## Abstract (180 words)
DNS is essential yet abused for data exfiltration via covert tunnels. This mini-project builds an offline, explainable classifier for prepared DNS feature records, using CIC-Bell-DNS-EXF-2021 when available. We compare Logistic Regression, Random Forest, and XGBoost under stratified splits with leakage controls, report binary (benign vs attack) and multiclass (benign/light/heavy) metrics, and explain predictions with SHAP. A Streamlit tool (DNS ThreatLens) validates schemas, predicts, explains, and logs history. Current measured results in this repo are **synthetic-demo only** (see reports/metrics.json): binary F1 ~0.998-0.999, multiclass macro-F1 ~0.96 on constructed separable data. These must NOT be presented as CIC results; re-run `scripts/train_all.py --data <REAL>` after gated download and replace this section. Limitations (representativeness, imbalance, drift, false positives/negatives) are discussed honestly.

## Ch 1 Introduction
DNS, queries/responses, resolvers, records; why firewalls allow DNS; exfiltration/tunneling/C2; domain vs record vs tunnel vs confirmed event. Objectives/scope as in README.

## Ch 2 Literature Review
Summarize CIC-Bell-DNS-EXF-2021 paper (Mahdavifar et al. ICCNS 2021), DoHBrw-2020 and DNS-2021 as alternatives with different tasks, plus ML-for-security, imbalance, SHAP. Verify every citation (title/authors/venue/year/DOI) before viva — do not invent references.

## Ch 3 Problem Definition
Threat model: offline batch scoring of feature rows. Intended users: students/SOC trainees. Exclusions: live capture, resolution, blocking, real-time claims.

## Ch 4 Dataset
Source, licence, citation (see data/README). Official stats: ~323k heavy, ~54k light, ~642k benign structured rows; 30 features Table 1; τ=100. **Replace with `python -m src.data_loader` output on your download.** State file names, shape, columns, label counts, missing/duplicates/inf, leakage columns.

## Ch 5 Methodology
Loading, audit, cleaning (inf→NaN, list-means e.g. unique_ttl, median/most-frequent impute, one-hot unknown-safe), scale only for LR, target normalization, stratified split (test untouched), group-aware limitation if no session IDs, imbalance via class_weight (no auto-SMOTE), tuning after baselines, metrics (Ch 8), SHAP sampled.

## Ch 6 System Design
Mermaid flowchart (dataset→validation→EDA→preprocess→leakage check→split→train→eval→SHAP→save→Streamlit→history). Component table + tech rationale.

## Ch 7 Implementation
Stack versions (requirements.txt), notebook sections, `scripts/train_all.py`, artifacts (`models/pipeline_*.joblib` + meta.json), app pages.

## Ch 8 Results (SYNTHETIC DEMO — replace with real)
Current `reports/metrics.json` (synthetic n=5000, seed 42):
- Binary: LR acc .998 F1 .998 ROC .9999; RF .998/.998/.9999; XGB .999/.999/.9999 (fp=1, fn≤1 on 1000 test).
- Multiclass: acc ~.969-.970, macro-F1 ~.960, ROC-OVR ~.996.
- Figures: `reports/figures/cm_*`, `shap_bar_*`. Top SHAP (binary XGB): rr_name_length, numeric, upper, labels_max, len.
Interpretation: synthetic classes were constructed separable, so near-perfect scores are expected and prove pipeline function only. Real CIC data will differ; investigate leakage/duplicates if real scores look suspiciously high.

## Ch 9 Demo
`streamlit run app/app.py`: Overview→Analyze (upload CSV/demo)→Explain→Evaluation→History. Screenshots + steps. Mark demo rows synthetic.

## Ch 10 Limitations
Gated data, synthetic demo, no group/temporal split without IDs, drift, FP/FN costs, feature availability, no live/PCAP/SIEM, SHAP≠causation, uncalibrated scores.

## Ch 11 Conclusion/Future
Implemented reproducible offline pipeline + app + tests (11 passing). Future: PCAP→feature parity check, streaming, DoH dataset variant, calibration, SIEM export — all distinct from done.

## Appendices
Abbreviations, hardware/software reqs, flowchart, equations (entropy, precision/recall/F1, ROC/PR), bibliography, 2-min + 5-min demo scripts (see docs/VIVA.md).
