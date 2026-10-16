# Viva Guide (simple language + technical depth)

**1. Problem?** Score prepared DNS feature rows for exfiltration. Not proof of maliciousness.
**2. Why DNS matters?** Required for normal browsing; firewalls allow it, so attackers piggyback encoded data.
**3. Tunneling/exfiltration?** Data encoded in queries (long/high-entropy labels, TXT/NULL types) to attacker NS; slow/low variants spread over time.
**4. Why this dataset?** CIC-Bell-DNS-EXF-2021 matches exfiltration with 30 documented features; benign/light/heavy labels. Alternatives change task (DoH tunneling vs domain reputation).
**5. Classes?** benign=normal; light=small files 15KB-924KB; heavy=4.5-26.9MB. Binary merges light+heavy→attack.
**6. Features?** Stateless (single query: lengths, entropy, counts) + stateful (window τ=100: distinct IPs/NS/ASN/countries, TTL stats). Dropped timestamps/IDs to avoid leakage; ablation with/without.
**7. Supervised ML?** Learn features→label from labeled rows; train/val/test separation.
**8. Why LR/RF/XGB?** LR=linear baseline (needs scaling); RF=bagged trees (non-linear, robust); XGB=boosted trees (often best tabular, needs tuning).
**9. Overfitting?** Memorizes train artifacts; check train vs test gap, limit depth, regularize.
**10. Leakage?** Test info into train (IDs, timestamps, fitted scaler on test). Fix: fit prep on train only, drop env IDs.
**11. Imbalance?** Benign majority; accuracy hides misses → use recall/F1, class_weight, keep representative test.
**12. Confusion matrix?** TN/FP/FN/TP table; security cares FN (missed attacks) and FP (alert fatigue).
**13. Recall?** Fraction of attacks caught; low FN critical.
**14. FP/FN?** FP=benign flagged; FN=attack missed. Report FPR/FNR.
**15. Macro-F1?** Average F1 across classes equally; reveals minority-class failures weighted acc hides.
**16. SHAP?** Per-feature contribution to a prediction; global=mean|SHAP|, local=per-row bars.
**17. SHAP≠proof?** Explains model, not causation or ground truth.
**18-20. Inference?** Load fitted pipeline joblib, validate columns, same order, `predict/predict_proba`; training fits, inference only transforms.
**21. Limitations?** Synthetic demo now; gated real data; no temporal/group split; drift; uncalibrated scores.
**22. vs blacklist?** Learns patterns for unseen variants; blacklist exact-match only, needs intel updates.
**23-25. All attacks? Live? SIEM?** No — offline exfiltration rows only; no capture/resolution/blocking; future: PCAP parity-checked ingestion, streaming, SIEM export.

## 2-min script
"DNS carries data for attackers. We classify feature rows benign vs exfiltration with RF/XGB, explain with SHAP, demo offline in Streamlit. Synthetic demo scores high by construction; real CIC evaluation after download."

## 5-min demo
1. Overview (metrics source) 2. Analyze upload (schema error demo + valid predict + download) 3. Explain one FP/FN with SHAP 4. Evaluation figures 5. History clear. Emphasize disclaimers.
