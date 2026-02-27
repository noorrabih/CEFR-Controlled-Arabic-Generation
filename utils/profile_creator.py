"""Select monotonic features across CEFR levels.

Given a CSV with a level column ("level" or "CEFR") and feature columns,
this script finds columns whose per-level means are monotonic increasing or decreasing.

Example:
  python utils/profile_creator.py \
    --input_csv profiling_data/ZAEBUC+ARWI/6levels/level_summary.csv \
    --trend increasing \
    --out_csv profiling_data/ZAEBUC+ARWI/6levels/monotonic_increasing.csv
"""

from __future__ import annotations

import argparse
from typing import List

import pandas as pd


def get_monotonic_columns(
    df: pd.DataFrame,
    trend: str = "increasing",
    cefr_order: List[str] | None = None,
    exclude_pattern: str = r"ratio|---|_median|std|count",
    level_col: str | None = None,
) -> List[str]:
    if level_col is None:
        if "level" in df.columns:
            level_col = "level"
        elif "CEFR" in df.columns:
            level_col = "CEFR"
        else:
            raise ValueError("CSV must contain a 'level' or 'CEFR' column")

    if cefr_order is None:
        cefr_order = ["A1", "A2", "B1", "B2", "C1", "C2"]

    keep_cols = df.columns[~df.columns.str.contains(exclude_pattern, case=False, regex=True)].tolist()
    if level_col not in keep_cols:
        keep_cols = [level_col] + keep_cols

    level_df = df.loc[:, keep_cols].copy()
    level_df[level_col] = pd.Categorical(level_df[level_col], categories=cefr_order, ordered=True)

    feat_cols = [c for c in level_df.columns if c != level_col]
    level_means = (
        level_df.groupby(level_col, observed=True)[feat_cols]
        .mean()
        .reindex(cefr_order)
    )

    if trend == "increasing":
        cols = [c for c in level_means.columns if level_means[c].is_monotonic_increasing]
    elif trend == "decreasing":
        cols = [c for c in level_means.columns if level_means[c].is_monotonic_decreasing]
    else:
        raise ValueError("trend must be 'increasing' or 'decreasing'")

    return cols


def main(args):
    df = pd.read_csv(args.input_csv)
    cefr_order = args.cefr_order.split(",") if args.cefr_order else None

    cols = get_monotonic_columns(
        df,
        trend=args.trend,
        cefr_order=cefr_order,
        exclude_pattern=args.exclude_pattern,
        level_col=args.level_col,
    )

    out_df = pd.DataFrame({"feature": cols})

    if args.out_csv:
        out_df.to_csv(args.out_csv, index=False)
        print(f"Saved: {args.out_csv}")
    else:
        print("\n".join(cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Find monotonic features across CEFR levels.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--trend", choices=["increasing", "decreasing"], default="increasing")
    ap.add_argument("--cefr_order", default=None, help="Comma-separated CEFR order, e.g. A1,A2,B1,B2,C1,C2")
    ap.add_argument("--exclude_pattern", default=r"ratio|---|_median|std|count")
    ap.add_argument("--level_col", default=None, help="Override level column name (default: auto-detect level/CEFR).")
    ap.add_argument("--out_csv", default=None)
    main(ap.parse_args())
