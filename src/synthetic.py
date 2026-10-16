"""Synthetic demo data. CLEARLY SYNTHETIC - never present as CIC data."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

# Subset of documented CIC-Bell-DNS-EXF-2021 feature names (verified from official page).
# Values below are RANDOM and carry no real traffic meaning.
NUMERIC_FEATURES = [
    "rr_count", "rr_name_length", "rr_name_entropy",
    "distinct_ns", "a_records", "unique_country", "unique_asn",
    "distinct_ip", "distinct_domains", "ttl_mean", "ttl_variance",
    "FQDN_count", "subdomain_length", "upper", "lower", "numeric",
    "entropy", "special", "labels", "labels_max", "labels_average", "len",
]
CATEGORICAL_FEATURES = ["rr_type", "sld", "longest_word", "subdomain", "reverse_dns"]
RR_TYPES = ["A", "AAAA", "TXT", "MX", "CNAME", "NS", "SOA", "PTR"]


def make_synthetic(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_benign = n // 2
    n_light = n // 4
    n_heavy = n - n_benign - n_light
    frames = []
    for kind, cnt in [("benign", n_benign), ("light_attack", n_light), ("heavy_attack", n_heavy)]:
        d: dict = {}
        # attack rows: longer names, higher entropy, more numeric/upper/special
        loc_len = {"benign": 25, "light_attack": 60, "heavy_attack": 90}[kind]
        loc_ent = {"benign": 3.2, "light_attack": 4.3, "heavy_attack": 4.8}[kind]
        d["rr_count"] = rng.integers(1, 5, cnt)
        d["rr_name_length"] = rng.normal(loc_len, 10, cnt).clip(5, 253).round().astype(int)
        d["rr_name_entropy"] = rng.normal(loc_ent, 0.5, cnt).clip(0, 8)
        d["distinct_ns"] = rng.integers(1, 5, cnt)
        d["a_records"] = rng.integers(1, 6, cnt)
        d["unique_country"] = rng.integers(1, 4, cnt)
        d["unique_asn"] = rng.integers(1, 4, cnt)
        d["distinct_ip"] = rng.integers(1, 6, cnt)
        d["distinct_domains"] = rng.integers(1, 6, cnt)
        d["ttl_mean"] = rng.normal(3600, 1500, cnt).clip(60, 86400)
        d["ttl_variance"] = np.abs(rng.normal(500, 800, cnt))
        d["FQDN_count"] = d["rr_name_length"] + rng.integers(0, 10, cnt)
        d["subdomain_length"] = (d["rr_name_length"] * rng.uniform(0.5, 0.9, cnt)).astype(int)
        d["upper"] = rng.integers(0, 3 if kind == "benign" else 12, cnt)
        d["lower"] = rng.integers(5, 30, cnt)
        d["numeric"] = rng.integers(0, 3 if kind == "benign" else 15, cnt)
        d["entropy"] = d["rr_name_entropy"] + rng.normal(0, 0.2, cnt)
        d["special"] = rng.integers(0, 2 if kind == "benign" else 6, cnt)
        d["labels"] = rng.integers(2, 5, cnt)
        d["labels_max"] = rng.integers(5, 20 if kind == "benign" else 63, cnt)
        d["labels_average"] = rng.normal(8, 3, cnt).clip(2, 63)
        d["len"] = d["rr_name_length"]
        probs = [0.5, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05] if kind == "benign" else [0.1, 0.1, 0.5, 0.05, 0.05, 0.05, 0.05, 0.1]
        d["rr_type"] = rng.choice(RR_TYPES, cnt, p=np.array(probs) / sum(probs))
        d["sld"] = rng.choice(["example", "testsite", "mydomain", "sample"], cnt)
        d["longest_word"] = rng.choice(["mail", "login", "video", "data"], cnt)
        d["subdomain"] = rng.choice(["yes", "no"], cnt, p=[0.6, 0.4])
        d["reverse_dns"] = rng.choice(["found", "not_found"], cnt, p=[0.8, 0.2] if kind == "benign" else [0.3, 0.7])
        df = pd.DataFrame(d)
        df["label"] = kind
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # shuffle
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # inject 1% missing + a couple inf to exercise cleaning
    for c in ["ttl_mean", "entropy"]:
        idx = rng.choice(len(out), size=max(1, len(out) // 100), replace=False)
        out.loc[idx, c] = np.nan
    out.loc[0, "ttl_variance"] = np.inf
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate SYNTHETIC DNS demo CSV (not real CIC data).")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="data/synthetic_dns_demo.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = make_synthetic(a.n, a.seed)
    df.to_csv(a.out, index=False)
    print(f"WROTE synthetic demo csv: {a.out} shape={df.shape} (SYNTHETIC ONLY)")


if __name__ == "__main__":
    main()
