"""Compute BAREC readability–CEFR correlation for Arabic essay datasets.

Reads one or more readability CSV files (containing CEFR and avg_level
columns), concatenates them, and reports Spearman and Pearson correlations
between CEFR level (encoded as an integer) and the BAREC readability score.

Expected CSV columns:
  - CEFR      : CEFR label (A1, A2, B1, B2, C1, C2)
  - avg_level : BAREC readability score (numeric)

Example:
  python utils/barec_corelation.py \\
    --csvs path/to/bea.csv path/to/zaebuc.csv
"""

from __future__ import annotations

import argparse
from typing import List

import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# CEFR encoding
# ---------------------------------------------------------------------------

CEFR_MAPPING = {
    "A1": 1,
    "A2": 2,
    "B1": 3,
    "B2": 4,
    "C1": 5,
    "C2": 6,
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_correlations(
    csv_paths: List[str],
    cefr_col: str = "CEFR",
    score_col: str = "avg_level",
) -> None:
    """Load CSVs, encode CEFR, and print Spearman/Pearson correlations.

    Args:
        csv_paths:  One or more paths to readability CSV files.
        cefr_col:   Name of the CEFR label column.
        score_col:  Name of the numeric readability score column.

    Raises:
        ValueError: If required columns are missing from any CSV.
    """
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        missing = {cefr_col, score_col} - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV '{path}' is missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["CEFR_numeric"] = df[cefr_col].map(CEFR_MAPPING)
    df = df.dropna(subset=["CEFR_numeric", score_col])

    spearman_corr, spearman_p = spearmanr(df["CEFR_numeric"], df[score_col])
    pearson_corr, pearson_p   = pearsonr(df["CEFR_numeric"], df[score_col])

    print(f"Rows analysed : {len(df)}")
    print(f"\nSpearman correlation (recommended):")
    print(f"  ρ = {spearman_corr:.4f},  p = {spearman_p:.6f}")
    print(f"\nPearson correlation:")
    print(f"  r = {pearson_corr:.4f},  p = {pearson_p:.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute BAREC readability–CEFR correlation."
    )
    p.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="One or more readability CSV files to concatenate.",
    )
    p.add_argument(
        "--cefr_col",
        default="CEFR",
        help="Name of the CEFR label column (default: CEFR).",
    )
    p.add_argument(
        "--score_col",
        default="avg_level",
        help="Name of the numeric readability score column (default: avg_level).",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    compute_correlations(
        csv_paths=args.csvs,
        cefr_col=args.cefr_col,
        score_col=args.score_col,
    )
