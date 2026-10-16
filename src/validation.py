"""Schema validation and data-quality checks."""
from __future__ import annotations
import re
from typing import Dict, List
import numpy as np
import pandas as pd
import yaml


def load_leakage_patterns(config_path: str = "configs/model_config.yaml") -> List[str]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("leakage_patterns", [])


def find_leakage_columns(columns: List[str], patterns: List[str]) -> Dict[str, str]:
    hits: Dict[str, str] = {}
    for c in columns:
        for pat in patterns:
            try:
                if re.match(pat, c, flags=re.IGNORECASE):
                    hits[c] = pat
                    break
            except re.error:
                if pat.lower() in c.lower():
                    hits[c] = pat
                    break
    return hits


def data_quality_report(df: pd.DataFrame, config_path: str = "configs/model_config.yaml") -> Dict:
    from src.data_loader import basic_summary
    rep = basic_summary(df)
    patterns = load_leakage_patterns(config_path)
    rep["leakage_hits"] = find_leakage_columns(list(df.columns), patterns)
    # identifier check: high-cardinality object columns
    high_card = {}
    for c in df.select_dtypes(include=["object"]).columns:
        try:
            nu = int(df[c].nunique())
            if nu > max(1000, 0.9 * len(df)):
                high_card[c] = nu
        except Exception:
            pass
    rep["high_cardinality_object"] = high_card
    # constant columns
    rep["constant_columns"] = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return rep


def validate_inference_df(df: pd.DataFrame, expected_features: List[str]) -> Dict:
    cols = set(df.columns)
    exp = set(expected_features)
    return {
        "missing": sorted(exp - cols),
        "extra": sorted(cols - exp),
        "ok": (exp - cols) == set(),
        "n_rows": len(df),
    }


def select_features(df: pd.DataFrame, target_col: str,
                    config_path: str = "configs/model_config.yaml") -> List[str]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    drop_exact = {d.lower() for d in cfg.get("drop_exact", [])}
    feats = [c for c in df.columns if c != target_col and c.lower() not in drop_exact]
    if cfg.get("drop_leakage_columns"):
        hits = find_leakage_columns(feats, cfg.get("leakage_patterns", []))
        feats = [c for c in feats if c not in hits]
    return feats
