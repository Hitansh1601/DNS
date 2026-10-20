"""New ML-rigor tests: baseline, CV, threshold, leakage-pattern fix (single train for speed)."""
from src.synthetic import make_synthetic
from src.train import train_task
from src.validation import find_leakage_columns, select_features
import yaml


def test_ml_rigor_bundle():
    # One training run covers dummy/CV/threshold asserts (keeps suite fast)
    df = make_synthetic(400, seed=11)
    res = train_task(df, "label", "binary", tune=False)
    # 1. dummy baseline present and beaten
    assert "dummy" in res["models"]
    dummy_acc = res["models"]["dummy"]["eval"]["accuracy"]
    assert res["models"]["xgb"]["eval"]["accuracy"] > dummy_acc
    # 2. CV scores present (train-only, test untouched)
    for name in ("logreg", "rf", "xgb"):
        cv = res["models"][name]["eval"].get("cv", {})
        assert "test_f1" in cv and "mean" in cv["test_f1"]
    # 3. threshold analysis present
    ta = res["models"]["xgb"]["eval"].get("threshold_analysis")
    assert ta is not None and 0.0 <= ta["best_threshold"] <= 1.0


def test_leakage_patterns_keep_aggregated_dns_features():
    with open("configs/model_config.yaml") as f:
        cfg = yaml.safe_load(f)
    cols = ["distinct_ip", "distinct_domains", "FQDN_count", "ttl_mean",
            "src_ip", "dst_ip", "session_id", "timestamp"]
    hits = find_leakage_columns(cols, cfg.get("leakage_patterns", []))
    assert "distinct_ip" not in hits
    assert "distinct_domains" not in hits
    assert "FQDN_count" not in hits
    assert "src_ip" in hits or "session_id" in hits


def test_feature_selection_keeps_dns_counts():
    df = make_synthetic(200, seed=4)
    feats = select_features(df, "label")
    for c in ("distinct_ip", "distinct_domains", "FQDN_count"):
        assert c in feats, f"{c} wrongly dropped as leakage"
