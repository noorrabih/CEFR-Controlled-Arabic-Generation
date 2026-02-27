"""Compute correlation between CEFR levels and BAREC (Taha-19) readability.

This script expects one or more CSV files with at least:
  - CEFR column (e.g., A1..C2)
  - avg_level column (numeric readability)

Example:
  python utils/barec_correlation.py --inputs data1.csv data2.csv --out_csv correlation_results.csv
"""

from __future__ import annotations

import argparse
from typing import List, Dict

import pandas as pd
from scipy.stats import spearmanr, pearsonr


CEFR_MAPPING: Dict[str, int] = {
    "A1": 1,
    "A2": 2,
    "B1": 3,
    "B2": 4,
    "C1": 5,
    "C2": 6,
}


def compute(df: pd.DataFrame, cefr_col: str = "CEFR", level_col: str = "avg_level") -> pd.DataFrame:
    if cefr_col not in df.columns or level_col not in df.columns:
        raise ValueError(f"Input must contain columns '{cefr_col}' and '{level_col}'. Found: {list(df.columns)}")

    tmp = df.copy()
    tmp["CEFR_numeric"] = tmp[cefr_col].map(CEFR_MAPPING)
    tmp[level_col] = pd.to_numeric(tmp[level_col], errors="coerce")
    tmp = tmp.dropna(subset=["CEFR_numeric", level_col])

    if len(tmp) < 3:
        raise ValueError("Not enough valid rows to compute correlation (need >= 3).")

    spearman_corr, spearman_p = spearmanr(tmp["CEFR_numeric"], tmp[level_col])
    pearson_corr, pearson_p = pearsonr(tmp["CEFR_numeric"], tmp[level_col])

    return pd.DataFrame([
        {"metric": "spearman_rho", "value": float(spearman_corr), "p_value": float(spearman_p)},
        {"metric": "pearson_r", "value": float(pearson_corr), "p_value": float(pearson_p)},
        {"metric": "n_rows", "value": int(len(tmp)), "p_value": ""},
    ])


def main(args):
    dfs: List[pd.DataFrame] = [pd.read_csv(p) for p in args.inputs]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    res = compute(df, cefr_col=args.cefr_col, level_col=args.level_col)

    print(res.to_string(index=False))

    if args.out_csv:
        res.to_csv(args.out_csv, index=False)
        print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute correlation between CEFR and BAREC avg_level.")
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more CSV files to concatenate.")
    ap.add_argument("--cefr_col", default="CEFR")
    ap.add_argument("--level_col", default="avg_level")
    ap.add_argument("--out_csv", default=None)
    main(ap.parse_args())
