# DNS ThreatLens — Professor Explainer Script (ML-focused)

Use this file to explain the project in a viva / demo. It answers:
1. Is this a good ML + cybersecurity project?
2. Does it actually use ML? What exactly does ML do here?
3. What was improved and why?

---

## 1. One-line verdict (say this first)

> Yes — this is a legitimate **ML-first, cybersecurity-grounded** project: it does
> **supervised classification of DNS exfiltration feature records** (benign vs attack,
> and benign / light / heavy), with honest train/test separation, 4-model comparison
> including a dummy baseline, cross-validation, threshold analysis, and SHAP explanations.

Cybersecurity is secondary in the right way: the security value comes from
*which features* (DNS tunnel traits) and *how errors are measured* (missed attacks
vs false alarms), while the core work — preprocessing, training, evaluation,
explainability — is pure ML.

If the professor asks "where is the ML?": point to `src/train.py:train_task`,
`src/evaluate.py`, `configs/model_config.yaml`, and `reports/metrics.json`.

---

## 2. What exactly does ML do? (the 60-second version)

No rules, no blacklists, no `if entropy > X then attack`.

1. **Input:** one CSV row = one DNS traffic record with ~27 numeric/categorical
   features (`rr_name_length`, `entropy`, `upper`, `numeric`, `labels_max`,
   `ttl_mean`, `rr_type`, `reverse_dns`, ...). See `src/synthetic.py` /
   `data/README.md` for the CIC-Bell-DNS-EXF-2021 feature list.
2. **Learn:** from labeled rows (`benign`, `light_attack`, `heavy_attack`),
   learn a mapping `features -> label`. Binary task merges light+heavy to `attack`.
   Code: `src/data_loader.py:normalize_labels`, `src/train.py:split_data`.
3. **Models:** Logistic Regression (linear baseline, needs scaling),
   Random Forest (bagged trees), XGBoost (boosted trees), + Dummy stratified
   baseline. Code: `src/train.py:make_estimators`.
4. **Decide:** output `prediction` + `score_*` probabilities per row.
   Code: `src/predict.py:predict_df`. A threshold turns scores into decisions —
   0.5 by default, plus a reported F1-optimal threshold (`threshold_analysis`).
5. **Prove it works:** stratified 80/20 holdout (test never touched during fit),
   3-fold CV on train only, accuracy / balanced-acc / precision / recall / F1 /
   ROC-AUC / PR-AUC / confusion matrix / FPR / FNR. Multiclass adds macro-F1 and
   ROC-OVR. Code: `src/evaluate.py:evaluate_classifier`, `best_f1_threshold`.
6. **Explain:** SHAP global bar (mean |SHAP|) + per-row bars, with permutation
   importance fallback. Code: `src/explain.py`.

What ML does **not** do: resolve domains, capture packets, block traffic, or prove
maliciousness. It scores patterns. The app banner and `predict_df` note column say
this explicitly.

---

## 3. Why this is a *good* ML project (strengths to cite)

- Real tabular ML workflow: `ColumnTransformer` (median/most-frequent impute,
  one-hot unknown-safe, StandardScaler only for LogReg), fitted on train only.
  `src/preprocessing.py:build_preprocessor`.
- Leakage hygiene: identifier patterns flagged/reported/dropped, high-cardinality
  and constant-column checks, inference schema validation rejects silent mispredictions.
  `src/validation.py`.
- Honest evaluation: stratified split, CV means/std, dummy baseline lift,
  per-class report, FPR/FNR (security costs), threshold tradeoff.
- Model comparison, not single-model cherry-picking.
- Reproducibility: `configs/model_config.yaml` (seed, test_size, grids),
  saved `pipeline_*.joblib` + `.meta.json`, `reports/metrics.json`, 14 pytest tests.
- Scope honesty: synthetic demo labeled synthetic, real CIC gated download documented,
  no fake live-detection claims.

---

## 4. What was fixed / improved in this pass (cite if asked "what did you improve?")

1. **Leakage-pattern bug fixed** (`configs/model_config.yaml`).
   Old `.*ip.*`, `.*domain.*`, `.*fqdn.*` silently dropped legit features
   `distinct_ip`, `distinct_domains`, `FQDN_count`. New patterns match only true
   identifiers (`^src_ip$`, `^dst_ip$`, `session_id`, `timestamp`, ...).
   Test: `test_leakage_patterns_keep_aggregated_dns_features`.
2. **Dead tuning code activated** (`src/train.py`).
   `RandomizedSearchCV` was imported but never run despite `tuning.enabled:true`.
   Now RF/XGB actually search small grids on train only; best params saved in
   `eval.best_params`. Run fast with `scripts/train_all.py --no-tune`.
3. **Cross-validation added** (`train_task._cross_val_scores`).
   3-fold stratified CV on train only, reported as `eval.cv.test_f1.mean/std`.
   Test set remains untouched. Shown in `comparison_table` as `cv_f1_mean/std`.
4. **Dummy baseline added.** `DummyClassifier(strategy=stratified)` trains alongside
   real models. `train_all.py` prints `lift = acc - dummy_acc`. Proves learning beyond chance.
5. **Threshold analysis for security** (`evaluate.best_f1_threshold`).
   Reports F1-optimal threshold vs 0.5, with precision/recall/F1 and FPR/FNR at both.
   Lets you say: "lower threshold catches more exfiltration but raises alert load."
6. **Windows/XGB hang fixed.** `n_jobs=-1` deadlocked OpenMP on this box; XGB now
   defaults to `n_jobs=1` (configurable), search/CV use `n_jobs=1`.
7. **Explainability fallback** (`explain.permutation_fallback`).
   When SHAP fails (e.g., LogReg/Dummy), permutation importance still gives a ranking.
8. **Leakage ablation logging** (`scripts/train_all.py`).
   Prints which raw columns hit leakage patterns so the viva can discuss the
   with/without-drop delta instead of hiding it.
9. **Tests expanded 11 → 14.** New `tests/test_improvements.py` covers dummy-beats-chance,
   CV presence, threshold range, and the leakage fix.

---

## 5. Demo script — 2 minutes (use for quick walkthrough)

> "DNS is allowed through firewalls, so attackers hide stolen data in queries —
> long, high-entropy names, odd types like TXT. This project classifies prepared
> DNS feature rows, not raw domains. We train LogReg, Random Forest, and XGBoost
> plus a dummy baseline, split 80/20 stratified, report accuracy, F1, ROC-AUC, and
> false-positive / false-negative rates — because a missed exfiltration and a false
> alarm cost different things. SHAP tells us which features pushed each decision.
> Current numbers are synthetic-demo only; the pipeline is built for the real
> CIC-Bell-DNS-EXF-2021 CSV after gated download. The app validates schemas, predicts,
> explains one row, and logs history — fully offline."

Commands:
```
PYTHONPATH=. python scripts/train_all.py --no-tune
PYTHONPATH=. python -m pytest -q
streamlit run app/app.py
```

## 6. Demo script — 5 minutes (pages)

1. **Overview:** show `reports/metrics.json` source tag `synthetic (DEMO ONLY)`,
   artifact classes, feature count. Emphasize offline test metrics vs user uploads.
2. **Analyze:** upload a feature CSV → trigger a missing-column error on purpose
   (shows validation), then valid upload → predictions + `score_max` + download.
3. **Explain:** pick one FP and one FN from the confusion matrix, show SHAP bars,
   read the caption: bars push toward/away, model-specific, not proof.
4. **Evaluation:** open `reports/figures/cm_binary_xgb.png` + `shap_bar_binary.png`,
   quote dummy lift + CV std + threshold table. Say: "XGB beats dummy by ~0.49;
   CV std is small so the split is stable; threshold X gives F1 Y."
5. **History:** log a few rows, show SQLite, clear. Restate limitations.

## 7. Tough questions + one-line answers

- **"Why 99% accuracy? Overfit?"** Synthetic classes were constructed separable
  (longer names/higher entropy for attacks). It proves the pipeline runs, not that
  real CIC scores this high. Real run will be lower; investigate duplicates/leakage
  if real scores look suspiciously high. CV + dummy gap support this story.
- **"Why not deep learning?"** Tabular 27-feature problem, 5k–1M rows: boosted trees
  beat MLPs on accuracy/cost/interpretability. DL adds nothing here.
- **"Why three models?"** LogReg = interpretable linear baseline; RF = robust non-linear;
  XGB = best tabular accuracy. Agreement across all three = signal is real.
- **"What about imbalance?"** Stratified split preserves ratios, `class_weight=balanced`,
  report balanced-acc/macro-F1 not just accuracy, keep test representative.
- **"False positives vs false negatives?"** FN = missed exfiltration (breach);
  FP = analyst alert fatigue. Report FPR/FNR + threshold curve; let SOC pick the point.
- **"Does SHAP prove attack?"** No. It explains the model, not causation or ground truth.
- **"Live detection?"** No. Offline batch scoring of feature rows. No PCAP, no resolution,
  no blocking — v1 scope. Future: PCAP→feature parity, streaming, calibration, SIEM export.
- **"What would you do with real CIC data?"** `python -m src.data_loader` audit,
  `scripts/train_all.py --data data/<REAL>.csv`, replace Ch.4/Ch.8 of `reports/REPORT.md`,
  re-check leakage/duplicates, tune thresholds on validation, calibrate scores.

---

## 8. Files to open when the professor points

| Question | Open |
|---|---|
| Where is ML training? | `src/train.py:train_task`, `make_estimators` |
| Preprocessing / no leakage? | `src/preprocessing.py:build_preprocessor`, `src/validation.py` |
| Metrics? | `src/evaluate.py:evaluate_classifier`, `best_f1_threshold` |
| Config / reproducibility? | `configs/model_config.yaml`, `models/*.meta.json` |
| Results? | `reports/metrics.json`, `reports/figures/` |
| Inference? | `src/predict.py:predict_df` |
| Explainability? | `src/explain.py:shap_global_local`, `permutation_fallback` |
| Tests? | `tests/test_improvements.py`, `tests/test_prediction.py` |
| Report / viva? | `reports/REPORT.md`, `docs/VIVA.md` |

Rule: never present synthetic metrics as CIC results. The `data_source` field and
app banners exist to prevent exactly that mistake.
