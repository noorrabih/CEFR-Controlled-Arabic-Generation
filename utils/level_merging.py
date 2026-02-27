"""Merge CEFR levels (e.g., 6-level -> 5-level or 3-level buckets).

This is used when you want to evaluate / profile at coarser granularity.

Example:
  python utils/level_merging.py --input_csv level_summary_6levels.csv --scheme 5 --out_csv level_summary_5levels.csv

Schemes:
- 5:  A1+A2 -> A, keep B1,B2,C1,C2
- 3:  A1+A2 -> A, B1+B2 -> B, C1+C2 -> C
"""

from __future__ import annotations

import argparse
import pandas as pd


def merge_levels(df: pd.DataFrame, scheme: int, level_col: str = "CEFR") -> pd.DataFrame:
    if level_col not in df.columns:
        raise ValueError(f"Missing level_col '{level_col}'. Found: {list(df.columns)}")

    mapping = {}
    if scheme == 5:
        mapping = {"A1": "A", "A2": "A", "B1": "B1", "B2": "B2", "C1": "C1", "C2": "C2", "A": "A"}
    elif scheme == 3:
        mapping = {"A1": "A", "A2": "A", "B1": "B", "B2": "B", "C1": "C", "C2": "C", "A": "A", "B": "B", "C": "C"}
    else:
        raise ValueError("scheme must be 3 or 5")

    out = df.copy()
    out[level_col] = out[level_col].map(mapping).fillna(out[level_col])

    # Average numeric columns after merging
    numeric_cols = out.select_dtypes(include='number').columns.tolist()
    other_cols = [c for c in out.columns if c not in numeric_cols and c != level_col]

    agg = {c: 'first' for c in other_cols}
    for c in numeric_cols:
        agg[c] = 'mean'

    out = out.groupby(level_col, as_index=False).agg(agg)

    # Sort
    order = ["A", "B1", "B2", "C1", "C2"] if scheme == 5 else ["A", "B", "C"]
    out[level_col] = pd.Categorical(out[level_col], categories=order, ordered=True)
    out = out.sort_values(level_col).reset_index(drop=True)

    return out


def main(args):
    df = pd.read_csv(args.input_csv)
    out = merge_levels(df, scheme=args.scheme, level_col=args.level_col)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge CEFR levels into coarser buckets.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--scheme", type=int, choices=[3,5], required=True)
    ap.add_argument("--level_col", default="CEFR")
    ap.add_argument("--out_csv", required=True)
    main(ap.parse_args())
