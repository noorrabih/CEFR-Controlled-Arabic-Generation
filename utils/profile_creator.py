"""Select monotonically increasing features across CEFR levels.

Merges surface, CAMeL, and syntactic level-summary CSVs, applies
source-specific post-filters, and writes the merged selected features
CSV plus 5 diagnostic plots.

Each input CSV must contain a 'level' or 'CEFR' column.

Output (all written to --out_dir):
  selected_features.csv
  camel_increasing_features_across_cefr.png
  words_increasing_features_across_cefr.png
  camelandwords_increasing_features_across_cefr.png
  pos_increasing_features_across_cefr.png
  dep_increasing_features_across_cefr.png
  depPOS_increasing_features_across_cefr.png

Example:
  python utils/profile_creator.py \
    --surface_csv <surface_csv> \
    --camel_csv <camel_csv> \
    --syntax_csv <syntax_csv> \
    --out_csv <out_csv> \
    --out_dir <out_dir>
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CEFR_ORDER      = ["A1", "A2", "B1", "B2", "C1", "C2"]
EXCLUDE_PATTERN = r"ratio|---|_median|_std|_count|_per_token"

# Syntax post-filter: always drop these columns even if monotonic
SYNTAX_EXCLUDE_COLS = {
    "depPOS_SBJ(ADV)_mean",
    "depPOS_SBJ(CCONJ)_mean",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_level_col(df: pd.DataFrame, path: str) -> str:
    """Detect the level column ('level' or 'CEFR') in a DataFrame.

    Args:
        df:   Input DataFrame.
        path: File path used in error message.

    Returns:
        Name of the level column.

    Raises:
        ValueError: If neither 'level' nor 'CEFR' is present.
    """
    if "level" in df.columns:
        return "level"
    if "CEFR" in df.columns:
        return "CEFR"
    raise ValueError(
        f"CSV must contain a 'level' or 'CEFR' column. "
        f"Found: {list(df.columns)} in: {path}"
    )


def _load(path: str) -> Tuple[pd.DataFrame, str]:
    """Load a level-summary CSV and return it with its level column name.

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of (DataFrame, level_col_name).
    """
    df        = pd.read_csv(path, dtype=str)
    level_col = _detect_level_col(df, path)

    for col in df.columns:
        if col != level_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, level_col


def get_monotonic_increasing(
    df: pd.DataFrame,
    level_col: str,
    exclude_pattern: str = EXCLUDE_PATTERN,
) -> List[str]:
    """Return feature columns that are monotonically increasing across CEFR levels.

    Args:
        df:              Level-summary DataFrame.
        level_col:       Name of the level column.
        exclude_pattern: Regex for columns to skip before checking.

    Returns:
        List of monotonically increasing column names.
    """
    keep_cols = df.columns[
        ~df.columns.str.contains(exclude_pattern, case=False, regex=True)
    ].tolist()
    keep_cols = [c for c in keep_cols if c != level_col]

    level_df            = df[[level_col] + keep_cols].copy()
    level_df[level_col] = pd.Categorical(
        level_df[level_col], categories=CEFR_ORDER, ordered=True
    )

    level_means = (
        level_df.groupby(level_col, observed=True)[keep_cols]
        .mean()
        .reindex(CEFR_ORDER)
    )

    return [c for c in level_means.columns if level_means[c].is_monotonic_increasing]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    surface_csv: str,
    camel_csv: str,
    syntax_csv: str,
    out_csv: str,
    out_dir: str,
) -> None:
    """Select monotonic features and save merged CSV + plots.

    Args:
        surface_csv: Path to surface-level summary CSV.
        camel_csv:   Path to CAMeL-level summary CSV.
        syntax_csv:  Path to syntax-level summary CSV.
        out_csv:     Output path for the merged selected features CSV.
        out_dir:     Directory for plot PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Load ---
    surface_df, surface_lvl = _load(surface_csv)
    camel_df,   camel_lvl   = _load(camel_csv)
    syntax_df,  syntax_lvl  = _load(syntax_csv)

    # --- Select monotonically increasing features ---
    surface_cols = get_monotonic_increasing(surface_df, surface_lvl)
    camel_cols   = get_monotonic_increasing(camel_df,   camel_lvl)
    syntax_cols  = get_monotonic_increasing(syntax_df,  syntax_lvl)

    # --- Post-filters ---
    # Syntax: drop (X), specific named cols, and _sum columns
    syntax_cols = [
        c for c in syntax_cols
        if "(X)"               not in c
        and c                  not in SYNTAX_EXCLUDE_COLS
        and "sum"              not in c
    ]

    print(f"[profile_creator] Surface features selected : {len(surface_cols)}")
    print(f"[profile_creator] CAMeL features selected   : {len(camel_cols)}")
    print(f"[profile_creator] Syntax features selected  : {len(syntax_cols)}")

    # --- Normalise level column to 'CEFR' for merging ---
    surface_df = surface_df.rename(columns={surface_lvl: "CEFR"})[["CEFR"] + surface_cols]
    camel_df   = camel_df.rename(columns={camel_lvl:   "CEFR"})[["CEFR"] + camel_cols]
    syntax_df  = syntax_df.rename(columns={syntax_lvl: "CEFR"})[["CEFR"] + syntax_cols]

    # --- Merge ---
    merged = (
        surface_df
        .merge(camel_df,  on="CEFR", how="inner")
        .merge(syntax_df, on="CEFR", how="inner")
    )

    merged["CEFR"] = pd.Categorical(merged["CEFR"], categories=CEFR_ORDER, ordered=True)
    merged = merged.sort_values("CEFR").reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[profile_creator] Saved {merged.shape[1] - 1} features to: {out_csv}")

    # --- Plots ---
    _save_plots(merged, surface_cols, camel_cols, syntax_cols, out_dir)


def _save_plots(
    merged: pd.DataFrame,
    surface_cols: List[str],
    camel_cols: List[str],
    syntax_cols: List[str],
    out_dir: str,
) -> None:
    """Save diagnostic line plots of selected features vs CEFR.

    Imports matplotlib and seaborn lazily since they are only needed
    for plotting.

    Args:
        merged:       Merged features DataFrame with a CEFR column.
        surface_cols: Selected surface feature column names.
        camel_cols:   Selected CAMeL feature column names.
        syntax_cols:  Selected syntax feature column names.
        out_dir:      Directory to write PNG files.
    """
    import matplotlib.pyplot as plt  # optional dep — imported here intentionally
    import seaborn as sns             # optional dep — imported here intentionally

    def _plot(feature_cols: List[str], title: str, filename: str) -> None:
        if not feature_cols:
            print(f"[profile_creator] Skipping '{filename}' — no features.")
            return
        melt_df = merged.melt(
            id_vars=["CEFR"],
            value_vars=feature_cols,
            var_name="feature",
            value_name="value",
        )
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=melt_df, x="CEFR", y="value", hue="feature", marker="o")
        plt.xlabel("CEFR Level")
        plt.ylabel("Feature Value")
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        out_path = os.path.join(out_dir, filename)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[profile_creator] Saved plot: {out_path}")

    pos_cols    = [c for c in syntax_cols if "pos_"    in c]
    dep_cols    = [c for c in syntax_cols if "dep_"    in c and "depPOS_" not in c]
    deppos_cols = [c for c in syntax_cols if "depPOS_" in c]

    _plot(camel_cols,                "CAMeL Features vs CEFR",           "camel_increasing_features_across_cefr.png")
    _plot(surface_cols,              "Surface Features vs CEFR",          "words_increasing_features_across_cefr.png")
    surface_no_totals = [c for c in surface_cols if "total" not in c]
    _plot(surface_no_totals + camel_cols, "Surface + CAMeL Features vs CEFR", "camelandwords_increasing_features_across_cefr.png")
    _plot(pos_cols,                  "POS Features vs CEFR",              "pos_increasing_features_across_cefr.png")
    _plot(dep_cols,                  "Dependency Features vs CEFR",       "dep_increasing_features_across_cefr.png")
    _plot(deppos_cols,               "Dep(POS) Features vs CEFR",        "depPOS_increasing_features_across_cefr.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Select monotonically increasing features across CEFR levels."
    )
    p.add_argument("--surface_csv", required=True, help="Surface-level summary CSV.")
    p.add_argument("--camel_csv",   required=True, help="CAMeL-level summary CSV.")
    p.add_argument("--syntax_csv",  required=True, help="Syntax-level summary CSV.")
    p.add_argument("--out_csv",     required=True, help="Output path for merged selected features CSV.")
    p.add_argument("--out_dir",     required=True, help="Output directory for plot PNGs.")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(
        surface_csv=args.surface_csv,
        camel_csv=args.camel_csv,
        syntax_csv=args.syntax_csv,
        out_csv=args.out_csv,
        out_dir=args.out_dir,
    )