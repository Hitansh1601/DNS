import pandas as pd
from src.synthetic import make_synthetic
from src.train import train_task, save_artifact
from src.predict import predict_df
from src.validation import validate_inference_df

def _small():
    return make_synthetic(600, seed=11)

def test_train_binary_runs():
    df = _small()
    res = train_task(df, "label", "binary")
    assert "logreg" in res["models"] and "rf" in res["models"] and "xgb" in res["models"]
    for m in res["models"].values():
        assert 0.0 <= m["eval"]["accuracy"] <= 1.0

def test_predict_output_format(tmp_path):
    df = _small()
    res = train_task(df, "label", "binary")
    feats = res["features"]
    art = {"pipeline": res["models"]["rf"]["pipeline"], "feature_names": feats,
           "label_classes": res["label_info"]["classes"]}
    out = predict_df(art, df[feats].head(5))
    assert "prediction" in out.columns and "score_max" in out.columns
    assert ((out["score_max"] >= 0) & (out["score_max"] <= 1)).all()
    # missing column raises
    try:
        predict_df(art, df[feats[1:]].head(2))
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "Missing" in str(e)
    # empty upload handled by caller; validate reports ok flag
    v = validate_inference_df(df[feats].head(1), feats)
    assert v["ok"]

def test_probability_shape():
    df = _small()
    res = train_task(df, "label", "binary")
    import numpy as np
    pipe = res["models"]["xgb"]["pipeline"]
    from src.preprocessing import clean_frame
    X = clean_frame(df[res["features"]].head(10))
    p = pipe.predict_proba(X)
    assert p.shape == (10, 2)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
