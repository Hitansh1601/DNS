import pandas as pd
from src.synthetic import make_synthetic
from src.data_loader import detect_target_column, normalize_labels
from src.validation import data_quality_report, select_features

def test_target_detection():
    df = make_synthetic(300, seed=0)
    assert detect_target_column(df) == "label"

def test_normalize_binary():
    df = make_synthetic(300, seed=1)
    y, info = normalize_labels(df["label"], task="binary")
    assert set(y.unique()) <= {0, 1}
    assert info["classes"] == ["benign", "attack"]

def test_normalize_multiclass():
    df = make_synthetic(300, seed=2)
    y, info = normalize_labels(df["label"], task="multiclass")
    assert set(y.unique()) <= {0, 1, 2}

def test_quality_report():
    df = make_synthetic(200, seed=3)
    rep = data_quality_report(df)
    assert rep["shape"][0] == 200
    assert "inf_cells" in rep

def test_feature_selection_drops_nothing_critical():
    df = make_synthetic(200, seed=4)
    feats = select_features(df, "label")
    assert "label" not in feats
    assert len(feats) > 10
