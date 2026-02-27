"""Visualize alignment between CEFR and predicted readability levels.

Creates heatmaps for:
  - rounded avg_level vs CEFR
  - rounded max_level vs CEFR

Example:
  python utils/readability_visualization.py \
    --input_csv generated_essays_readability.csv \
    --out_dir outputs/ \
    --avg_col avg_level --max_col max_level --cefr_col CEFR
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(ct: pd.DataFrame, out_path: str, xlabel: str = "CEFR", ylabel: str = "readability"):
    plt.figure(figsize=(10, 7))
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Count'}, linewidths=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    if args.cefr_col not in df.columns:
        raise ValueError(f"Missing CEFR column '{args.cefr_col}'. Found: {list(df.columns)}")

    # Optional filtering
    if args.drop_unassessable:
        df = df[df[args.cefr_col] != "Unassessable"]

    # Avg heatmap
    if args.avg_col in df.columns:
        tmp = df.dropna(subset=[args.avg_col, args.cefr_col]).copy()
        tmp["avg_rounded"] = pd.to_numeric(tmp[args.avg_col], errors="coerce").round().astype('Int64')
        tmp = tmp.dropna(subset=["avg_rounded"])
        ct = pd.crosstab(tmp["avg_rounded"], tmp[args.cefr_col]).fillna(0).astype(int)
        plot_heatmap(ct, os.path.join(args.out_dir, args.out_prefix + "_avg_heatmap.png"), ylabel=args.avg_col)

    # Max heatmap
    if args.max_col in df.columns:
        tmp = df.dropna(subset=[args.max_col, args.cefr_col]).copy()
        tmp["max_rounded"] = pd.to_numeric(tmp[args.max_col], errors="coerce").round().astype('Int64')
        tmp = tmp.dropna(subset=["max_rounded"])
        ct = pd.crosstab(tmp["max_rounded"], tmp[args.cefr_col]).fillna(0).astype(int)
        plot_heatmap(ct, os.path.join(args.out_dir, args.out_prefix + "_max_heatmap.png"), ylabel=args.max_col)

    print(f"Saved heatmaps to: {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot heatmaps of readability (avg/max) vs CEFR.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_prefix", default="readability_vs_cefr")
    ap.add_argument("--cefr_col", default="CEFR")
    ap.add_argument("--avg_col", default="avg_level")
    ap.add_argument("--max_col", default="max_level")
    ap.add_argument("--drop_unassessable", action="store_true")
    main(ap.parse_args())
