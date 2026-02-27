"""Expand prompt rows by mapping coarse levels to CEFR buckets.

Example:
  python generation_data/topic_prompts/prompt_process.py \
    --input_csv generation_data/topic_prompts/prompts.csv \
    --out_csv generation_data/topic_prompts/prompts_with_3cefr_levels.csv \
    --level_col Level
"""

from __future__ import annotations

import argparse
import pandas as pd


DEFAULT_MAPPING = {
    "Beginner": ["A"],
    "Intermediate": ["B"],
    "Advanced": ["C"],
}


def main(args):
    df = pd.read_csv(args.input_csv)
    if args.level_col not in df.columns:
        raise ValueError(f"Missing level column '{args.level_col}'. Found: {list(df.columns)}")

    expanded_rows = []
    for _, row in df.iterrows():
        lvl = row[args.level_col]
        targets = DEFAULT_MAPPING.get(str(lvl), [None])
        for cefr in targets:
            new_row = row.copy()
            new_row[args.out_cefr_col] = cefr
            expanded_rows.append(new_row)

    out = pd.DataFrame(expanded_rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Expand prompts by mapping coarse level to CEFR.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--level_col", default="Level")
    ap.add_argument("--out_cefr_col", default="CEFR_level")
    main(ap.parse_args())
