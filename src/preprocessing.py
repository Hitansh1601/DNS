"""Reproducible preprocessing. Test set never fits transformations."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def coerce_lists_to_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Paper notes unique_ttl lists -> average. Handle any string like '[1, 2]' or '1;2;3'."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str)
            looks_list = s.str.contains(r"[\[\],;]").mean() > 0.3
            if looks_list:
                def _mean(v):
                    try:
                        t = str(v).replace("[", "").replace("]", "").replace(";", ",")
                        nums = [float(x) for x in t.split(",") if x.strip() not in ("", "nan", "None")]
                        return float(np.mean(nums)) if nums else np.nan
                    except Exception:
                        return np.nan
                parsed = s.map(_mean)
                if parsed.notna().mean() > 0.5:
                    df[c] = parsed
    return df


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = coerce_lists_to_mean(df)
    # coerce numeric-like objects
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.9:
                df[c] = conv
    # inf -> nan (imputed later)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    return df


def split_num_cat(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in features if c not in num]
    return num, cat


def build_preprocessor(num_features: List[str], cat_features: List[str],
                       scale_numeric: bool = False) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if num_features:
        transformers.append(("num", Pipeline(num_steps), num_features))
    if cat_features:
        transformers.append(("cat", cat_pipe, cat_features))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
