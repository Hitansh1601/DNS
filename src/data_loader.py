"""Data loading and schema inspection. No assumed label/column names."""
from __future__ import annotations
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


def find_csvs(data_dir: str | Path = "data") -> List[str]:
    pats = ["*.csv", "*.CSV", "*.csv.gz", "*.zip"]
    out: List[str] = []
    for p in pats:
        out.extend(glob.glob(str(Path(data_dir) / p)))
    return sorted(set(out))


def load_csv(path: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def basic_summary(df: pd.DataFrame) -> Dict:
    num = df.select_dtypes(include=[np.number])
    inf_cells = int(np.isinf(num.to_numpy()).sum()) if num.shape[1] else 0
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_top20": df.isna().sum().sort_values(ascending=False).head(20).to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
        "inf_cells": inf_cells,
    }


def candidate_label_info(df: pd.DataFrame, max_unique: int = 20) -> Dict[str, Dict]:
    info: Dict[str, Dict] = {}
    for c in df.columns:
        try:
            nu = int(df[c].nunique(dropna=False))
        except Exception:
            continue
        if nu <= max_unique or df[c].dtype == object:
            vc = df[c].value_counts(dropna=False).head(20)
            info[c] = {"nunique": nu, "dtype": str(df[c].dtype),
                       "values": {str(k): int(v) for k, v in vc.items()}}
    return info


def detect_target_column(df: pd.DataFrame, candidates: Optional[List[str]] = None,
                         config_path: str = "configs/model_config.yaml") -> Optional[str]:
    if candidates is None:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            if cfg.get("target"):
                t = cfg["target"]
                if t in df.columns:
                    return t
            candidates = cfg.get("target_candidates", [])
        except FileNotFoundError:
            candidates = []
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in (candidates or []):
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # heuristic: object column with 2-4 uniques containing benign/attack keywords
    keywords = ("benign", "attack", "light", "heavy", "malicious", "normal")
    for c in df.columns:
        try:
            nu = int(df[c].nunique(dropna=False))
        except Exception:
            continue
        if 2 <= nu <= 6:
            vals = " ".join(str(v).lower() for v in df[c].dropna().unique()[:10])
            if any(k in vals for k in keywords):
                return c
    return None


def normalize_labels(s: pd.Series, task: str = "binary",
                     config_path: str = "configs/model_config.yaml") -> Tuple[pd.Series, Dict]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    mapping = cfg.get("binary_map" if task == "binary" else "multiclass_map", {})

    def norm(v) -> str:
        v0 = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        # common variants: "benign", "light exfiltration", "heavy exfiltration", "dns exfiltration"
        if "light" in v0:
            return "light_attack" if task == "multiclass" else mapping and "attack" or "attack"
        if "heavy" in v0:
            return "heavy_attack" if task == "multiclass" else "attack"
        if "benign" in v0 or v0 == "normal":
            return "benign"
        return v0

    normed = s.map(norm)
    if task == "binary":
        # merge everything non-benign to attack
        normed = normed.map(lambda x: "benign" if x == "benign" else "attack")
        label_map = {"benign": 0, "attack": 1}
    else:
        # keep benign/light_attack/heavy_attack; fold unknown attacks to attack->light bucket
        normed = normed.map(lambda x: x if x in ("benign", "light_attack", "heavy_attack") else ("benign" if x == "benign" else "light_attack"))
        label_map = {"benign": 0, "light_attack": 1, "heavy_attack": 2}
    y = normed.map(label_map)
    # drop unmapped (should be none after folding)
    mask = y.notna()
    return y[mask].astype(int), {"classes": sorted(label_map, key=label_map.get), "label_map": label_map,
                                "dropped_unmapped": int((~mask).sum())}


if __name__ == "__main__":
    import json
    files = find_csvs("data")
    print("csv files:", files)
    for p in files:
        print("\n===", p)
        try:
            df = load_csv(p, nrows=5)
            print("columns:", list(df.columns))
            print(df.head(2).to_string())
        except Exception as e:
            print("ERROR:", e)
    # full audit on first file if present
    if files and not any(f.endswith((".zip", ".gz")) for f in files):
        df = load_csv(files[0])
        print("\nSUMMARY:", json.dumps(basic_summary(df), indent=2)[:3000])
        print("\nTARGET GUESS:", detect_target_column(df))
        print("\nCANDIDATES:", json.dumps(candidate_label_info(df), indent=2)[:4000])
