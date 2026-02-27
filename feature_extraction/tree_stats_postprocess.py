"""Aggregate tree statistics per essay and optionally visualize vs CEFR.

Inputs:
  --stats_dir       Directory containing per-file .xlsx stats produced by
                    tree_stats.py (one xlsx per .conllx file).
  --readability_csv CSV containing a Document_ID column and a CEFR column.

Outputs:
  --out_csv         Aggregated CSV per essay (with CEFR joined).
  --out_dir         Directory for optional heatmap PNGs.

Example:
  python feature_extraction/tree_stats_postprocess.py \
    --stats_dir <stats_dir> \
    --readability_csv <readability_csv> \
    --out_csv <out_csv> \
    --out_dir <out_dir> \
    --make_heatmaps
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_tree_stats(
    stats_dir: str,
    include_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aggregate per-sentence tree stats to per-essay level.

    Reads all .xlsx files in ``stats_dir`` (as produced by tree_stats.py),
    derives an essay ID from the sentence ID (everything before the last dash),
    and aggregates numeric stat columns with mean, max, and list.

    Args:
        stats_dir:    Directory containing .xlsx stat files.
        include_cols: Stat columns to aggregate. Defaults to depth, breadth,
                      total_nodes, max_branching_factor.

    Returns:
        DataFrame with one row per essay and aggregated stat columns.

    Raises:
        FileNotFoundError: If no .xlsx files are found in ``stats_dir``.
        ValueError:        If required columns are missing.
    """
    include_cols = include_cols or [
        "depth", "breadth", "total_nodes", "max_branching_factor"
    ]

    all_files = sorted(glob(os.path.join(stats_dir, "*.xlsx")))
    if not all_files:
        raise FileNotFoundError(f"No .xlsx files found in: {stats_dir}")

    dfs = [pd.read_excel(f) for f in all_files]
    df  = pd.concat(dfs, ignore_index=True)

    if "sentence_id" not in df.columns:
        raise ValueError(
            f"Expected 'sentence_id' column in tree stats. "
            f"Found: {list(df.columns)}"
        )

    # Essay ID = everything before the last dash  (e.g. "AR-030-268469-3" → "AR-030-268469")
    df["essay_id"] = df["sentence_id"].astype(str).str.rsplit("-", n=1).str[0]

    stat_cols = [
        c for c in include_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not stat_cols:
        raise ValueError(
            f"No numeric stat columns found among: {include_cols}. "
            f"Available: {list(df.columns)}"
        )

    agg_funcs = {col: ["mean", "max", list] for col in stat_cols}
    grouped   = df.groupby("essay_id").agg(agg_funcs)

    grouped.columns = [f"{c[0]}_{c[1]}" for c in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()

    # Serialize list columns to comma-separated strings for CSV compatibility
    for col in grouped.columns:
        if col.endswith("_list"):
            grouped[col] = grouped[col].apply(
                lambda x: ",".join(str(v) for v in x)
            )

    return grouped


# ---------------------------------------------------------------------------
# CEFR joining
# ---------------------------------------------------------------------------

def merge_cefr(
    grouped: pd.DataFrame,
    readability_csv: str,
    id_col: str = "Document_ID",
    cefr_col: Optional[str] = "CEFR",
) -> pd.DataFrame:
    """Join CEFR labels from a readability CSV onto the aggregated stats.

    Attempts a direct join on essay_id == id_col. If no matches are found,
    retries by stripping everything up to the last dash from essay_id
    (e.g. "AR-030-268469" → "268469") to handle ZAEBUC-style IDs.

    Args:
        grouped:         Aggregated stats DataFrame from :func:`aggregate_tree_stats`.
        readability_csv: Path to CSV containing ``id_col`` and ``cefr_col``.
        id_col:          ID column name in the readability CSV.
        cefr_col:        CEFR column name in the readability CSV.

    Returns:
        Merged DataFrame with a ``CEFR`` column added.

    Raises:
        ValueError: If required columns are missing from the readability CSV.
    """
    read_df = pd.read_csv(readability_csv, dtype=str)

    if id_col not in read_df.columns:
        raise ValueError(
            f"readability_csv missing '{id_col}' column. "
            f"Found: {list(read_df.columns)}"
        )

    if cefr_col and cefr_col in read_df.columns:
        read_df["CEFR"] = read_df[cefr_col].astype(str)
    else:
        # Fallback: infer CEFR from ID patterns like "1_A1_1"
        print(
            f"[WARNING] CEFR column '{cefr_col}' not found in readability CSV. "
            "Attempting to infer CEFR from ID field (expects pattern like '1_A1_1'). "
            "Results may be incorrect if IDs do not follow this pattern."
        )
        read_df["CEFR"] = read_df[id_col].astype(str).str.split("_").str[1]

    read_df["Document_ID"] = read_df[id_col].astype(str)
    grouped = grouped.copy()
    grouped["essay_id"] = grouped["essay_id"].astype(str)

    merged = grouped.merge(
        read_df[["Document_ID", "CEFR"]],
        left_on="essay_id",
        right_on="Document_ID",
        how="left",
    )

    # Retry with suffix-stripped IDs if nothing matched
    if merged["CEFR"].isna().all():
        print(
            "[WARNING] No CEFR matches found on full essay_id. "
            "Retrying with last segment of essay_id (e.g. 'AR-030-268469' → '268469')."
        )
        grouped["essay_id"] = grouped["essay_id"].str.split("-").str[-1]
        merged = grouped.merge(
            read_df[["Document_ID", "CEFR"]],
            left_on="essay_id",
            right_on="Document_ID",
            how="left",
        )

    if merged["CEFR"].isna().all():
        print(
            "[WARNING] CEFR column is entirely empty after merge. "
            "Check that Document_ID values in the readability CSV match essay IDs."
        )

    merged = merged.drop(columns=["Document_ID"])
    return merged


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_heatmaps(df: pd.DataFrame, out_dir: str, metrics: List[str]) -> None:
    """Save heatmap PNGs showing the distribution of each metric vs CEFR level.

    Imports matplotlib and seaborn lazily since they are optional dependencies
    only needed when --make_heatmaps is passed.

    Args:
        df:      DataFrame with metric columns and a ``CEFR`` column.
        out_dir: Directory to write heatmap PNG files.
        metrics: List of column names to plot.
    """
    import matplotlib.pyplot as plt  # optional dep — imported here intentionally
    import seaborn as sns             # optional dep — imported here intentionally

    os.makedirs(out_dir, exist_ok=True)
    print(f"[make_heatmaps] Metrics: {metrics}")
    print(f"[make_heatmaps] CEFR levels: {sorted(df['CEFR'].dropna().unique())}")

    for metric in metrics:
        if metric not in df.columns:
            print(f"[make_heatmaps] Skipping '{metric}' — column not found.")
            continue

        tmp = df.dropna(subset=[metric, "CEFR"]).copy()
        if tmp.empty:
            continue

        if tmp[metric].dtype.kind in "fc":
            tmp["metric_rounded"] = tmp[metric].round(0).astype(int)
        else:
            tmp["metric_rounded"] = (
                pd.to_numeric(tmp[metric], errors="coerce").round(0).astype("Int64")
            )

        tmp = tmp.dropna(subset=["metric_rounded"])
        pivot = tmp.pivot_table(
            index="metric_rounded", columns="CEFR", aggfunc="size", fill_value=0
        )

        plt.figure(figsize=(10, 7))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")
        plt.title(f"{metric} vs CEFR")
        plt.xlabel("CEFR")
        plt.ylabel(f"{metric} (rounded)")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{metric}_vs_CEFR_heatmap.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[make_heatmaps] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    grouped = aggregate_tree_stats(args.stats_dir)

    merged = merge_cefr(
        grouped,
        readability_csv=args.readability_csv,
        id_col=args.readability_id_col,
        cefr_col=args.readability_cefr_col,
    )

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"[tree_stats_postprocess] Saved: {args.out_csv}")

    if args.make_heatmaps:
        metrics = [m.strip() for m in args.heatmap_metrics.split(",")]
        make_heatmaps(merged, args.out_dir, metrics)
        print(f"[tree_stats_postprocess] Saved heatmaps to: {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Aggregate tree stats per essay and optionally visualize vs CEFR."
    )
    ap.add_argument(
        "--stats_dir",
        required=True,
        help="Directory containing .xlsx stat files from tree_stats.py.",
    )
    ap.add_argument(
        "--readability_csv",
        required=True,
        help="CSV with Document_ID and CEFR columns.",
    )
    ap.add_argument(
        "--readability_id_col",
        default="Document_ID",
        help="ID column name in readability CSV (default: Document_ID).",
    )
    ap.add_argument(
        "--readability_cefr_col",
        default="CEFR",
        help="CEFR column name in readability CSV (default: CEFR).",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV path for aggregated stats.",
    )
    ap.add_argument(
        "--out_dir",
        default=".",
        help="Output directory for heatmap PNGs (default: current directory).",
    )
    ap.add_argument(
        "--make_heatmaps",
        action="store_true",
        help="If set, save heatmap PNGs to --out_dir.",
    )
    ap.add_argument(
        "--heatmap_metrics",
        default="depth_mean,depth_max,breadth_mean,breadth_max,total_nodes_mean,total_nodes_max",
        help="Comma-separated list of metrics to plot (default: depth/breadth/total_nodes mean+max).",
    )
    main(ap.parse_args())