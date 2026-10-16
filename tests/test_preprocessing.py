import numpy as np, pandas as pd
from src.synthetic import make_synthetic
from src.preprocessing import clean_frame, build_preprocessor, split_num_cat
from src.validation import select_features

def test_clean_handles_inf_nan():
    df = make_synthetic(200, seed=5)
    feats = select_features(df, "label")
    X = clean_frame(df[feats])
    num = X.select_dtypes(include=[np.number])
    assert not np.isinf(num.to_numpy()).any()

def test_preprocessor_fit_transform():
    df = make_synthetic(300, seed=6)
    feats = select_features(df, "label")
    df_c = clean_frame(df)
    num, cat = split_num_cat(df_c[feats], feats)
    prep = build_preprocessor(num, cat, scale_numeric=True)
    Xt = prep.fit_transform(df_c[feats].iloc[:100])
    assert Xt.shape[0] == 100
    # unseen category safe
    row = df_c[feats].iloc[:1].copy()
    for c in cat:
        row[c] = "__UNSEEN__"
    assert prep.transform(row).shape[1] == Xt.shape[1]

def test_feature_order_preserved():
    df = make_synthetic(100, seed=7)
    feats = select_features(df, "label")
    assert feats == sorted(feats, key=feats.index)  # order stable, no shuffle
