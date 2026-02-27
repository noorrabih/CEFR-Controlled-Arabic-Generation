"""Filter a feature summary table to a selected feature set.

Typical use:
- You have a CSV listing the selected (monotonic) features as columns.
- You have separate level-wise summaries from surface/camel/syntax.
- You want a single CSV containing only those selected features.

Example:
  python utils/features_filter.py \
    --selected_features_csv selected_readability_monotonic_features.csv \
    --surface_csv surface_feats/level_summary.csv \
    --camel_csv surface_feats/level_features_camel.csv \
    --syntax_csv syntax/level_syntax_stats.csv \
    --out_csv filtered_monotonic_features.csv
"""

from __future__ import annotations

import argparse
import pandas as pd


def main(args):
    selected = pd.read_csv(args.selected_features_csv)
    selected_cols = list(selected.columns)

    surface_df = pd.read_csv(args.surface_csv)
    camel_df = pd.read_csv(args.camel_csv)
    syntax_df = pd.read_csv(args.syntax_csv)

    # Normalize join key
    join_key = args.join_key
    for d in (surface_df, camel_df, syntax_df):
        if join_key not in d.columns:
            raise ValueError(f"Missing join key '{join_key}' in one of the inputs. Columns: {list(d.columns)}")

    merged = surface_df.merge(camel_df, on=join_key, how="inner").merge(syntax_df, on=join_key, how="inner")

    # Keep columns that exist; create missing ones as NaN for schema compatibility
    keep = [c for c in selected_cols if c in merged.columns]
    filtered = merged.loc[:, keep].copy()

    missing = [c for c in selected_cols if c not in merged.columns]
    for c in missing:
        filtered[c] = float('nan')

    # Reorder to match selected schema
    filtered = filtered[selected_cols]

    filtered.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")
    if missing:
        print(f"Warning: {len(missing)} selected columns were missing in merged inputs and were filled with NaN.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Filter merged feature summaries to a selected feature set.")
    ap.add_argument("--selected_features_csv", required=True)
    ap.add_argument("--surface_csv", required=True)
    ap.add_argument("--camel_csv", required=True)
    ap.add_argument("--syntax_csv", required=True)
    ap.add_argument("--join_key", default="level")
    ap.add_argument("--out_csv", required=True)
    main(ap.parse_args())
