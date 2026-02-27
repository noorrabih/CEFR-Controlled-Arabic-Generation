"""Extract surface-level features from a sentence-level CSV and aggregate by CEFR.

Reads a sentence-level CSV, computes per-sentence surface metrics, aggregates
to essay level and CEFR level, and writes multiple output CSVs and plots.

Input CSV must contain columns: Document_ID, Sentence, CEFR
  - If CEFR is missing, it will be inferred from Document_ID (pattern: 1_A1_1)
  - If inference fails, provide --levels_csv as a fallback mapping

Outputs (all written to --out_dir):
  sentence_level_metrics.csv   per-sentence metrics
  essay_level_summary.csv      per-essay aggregated metrics
  surface_level_summary.csv    per-CEFR-level aggregated metrics
  cefr_words_metrics_plot.png  total words & unique words vs CEFR
  cefr_sentence_word_metrics_plot.png  sentence/word metrics vs CEFR

Example:
  python feature_extraction/cefr_surface_feats.py \
    --input_csv <input_csv> \
    --out_dir <out_dir>

  # With fallback CEFR mapping:
  python feature_extraction/cefr_surface_feats.py \
    --input_csv <input_csv> \
    --out_dir <out_dir> \
    --levels_csv <levels_csv>
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import pandas as pd
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.charsets import UNICODE_PUNCT_CHARSET


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CEFR_ORDERS = {
    3: ["A", "B", "C"],
    5: ["A", "B1", "B2", "C1", "C2"],
    6: ["A1", "A2", "B1", "B2", "C1", "C2"],
}


# ---------------------------------------------------------------------------
# Sentence-level metrics
# ---------------------------------------------------------------------------

def sentence_metrics(sentence: str) -> Tuple[int, int, float, int, int]:
    """Compute surface metrics for a single sentence.

    Tokenizes using CAMeL Tools simple word tokenizer and removes
    punctuation tokens before computing metrics.

    Args:
        sentence: Raw sentence string.

    Returns:
        Tuple of (word_count, unique_word_count, avg_word_len,
                  min_word_len, max_word_len).
        Returns (0, 0, 0.0, 0, 0) for empty or non-string input.
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return 0, 0, 0.0, 0, 0

    tokens = simple_word_tokenize(sentence)
    tokens = [t for t in tokens if all(c not in UNICODE_PUNCT_CHARSET for c in t)]

    word_count = len(tokens)
    if word_count == 0:
        return 0, 0, 0.0, 0, 0

    lengths = [len(w) for w in tokens]
    return (
        word_count,
        len(set(tokens)),
        sum(lengths) / word_count,
        min(lengths),
        max(lengths),
    )


# ---------------------------------------------------------------------------
# CEFR resolution
# ---------------------------------------------------------------------------

def resolve_cefr(
    df: pd.DataFrame,
    levels_csv: Optional[str],
) -> pd.DataFrame:
    """Ensure the DataFrame has a valid CEFR column.

    Resolution order:
      1. Use existing CEFR column if present and has 2+ unique values.
      2. Infer from Document_ID string (pattern: '1_A1_1').
      3. Join from levels_csv if provided.

    Args:
        df:         Essay-level DataFrame with a ``Document_ID`` column.
        levels_csv: Optional path to a CSV with Document_ID and CEFR columns.

    Returns:
        DataFrame with a resolved CEFR column, Unassessable rows removed.
    """
    df = df.copy()

    # Attempt to infer CEFR from Document_ID if column is missing or empty
    if "CEFR" not in df.columns or df["CEFR"].nunique() < 2:
        df["CEFR"] = df["Document_ID"].astype(str).str.split("_").str[1]

    if df["CEFR"].nunique() < 2:
        if levels_csv:
            print(
                "[WARNING] Could not extract CEFR from Document_ID. "
                f"Falling back to levels_csv: {levels_csv}"
            )
            levels = pd.read_csv(levels_csv, dtype=str)
            if "Document_ID" not in levels.columns or "CEFR" not in levels.columns:
                raise ValueError(
                    f"levels_csv must contain columns: Document_ID, CEFR. "
                    f"Found: {list(levels.columns)}"
                )
            cefr_map = dict(zip(levels["Document_ID"].astype(str), levels["CEFR"]))
            df["CEFR"] = df["Document_ID"].astype(str).map(cefr_map).fillna("NA")
        else:
            print(
                "[WARNING] Could not extract CEFR from Document_ID and no "
                "--levels_csv provided. CEFR column will be empty."
            )

    # Drop unassessable rows
    df = df[~df["CEFR"].isin(["Unassessable", "NA"])].copy()

    # Apply categorical ordering if level count is known
    n_levels = df["CEFR"].nunique()
    cefr_order = CEFR_ORDERS.get(n_levels)
    if cefr_order:
        df["CEFR"] = pd.Categorical(df["CEFR"], categories=cefr_order, ordered=True)
    else:
        print(
            f"[WARNING] Unexpected number of CEFR levels ({n_levels}). "
            "Sorting alphabetically."
        )

    print(f"[cefr_surface_feats] CEFR distribution:\n{df['CEFR'].value_counts().to_string()}")
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_essay_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentence-level metrics to essay level.

    Args:
        df: Sentence-level DataFrame with Document_ID and metric columns.

    Returns:
        Essay-level DataFrame with one row per Document_ID.
    """
    return (
        df.groupby("Document_ID")
        .agg(
            sentences             =("Document_ID",      "count"),
            total_words           =("word_count",        "sum"),
            avg_words_per_sentence=("word_count",        "mean"),
            total_unique_words    =("unique_word_count", "sum"),
            avg_unique_words      =("unique_word_count", "mean"),
            overall_avg_word_len  =("avg_word_len",      "mean"),
            overall_max_word_len  =("max_word_len",      "max"),
        )
        .reset_index()
    )


def compute_level_summary(essay_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate essay-level metrics to CEFR-level summary.

    Args:
        essay_df: Essay-level DataFrame with a CEFR column.

    Returns:
        CEFR-level summary DataFrame with column ``level`` instead of ``CEFR``.
    """
    return (
        essay_df.groupby("CEFR")
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("CEFR")
        .rename(columns={"CEFR": "level"})
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(summary: pd.DataFrame, out_dir: str) -> None:
    """Save surface feature plots vs CEFR level.

    Imports matplotlib lazily since it is only needed for plotting.

    Args:
        summary:  CEFR-level summary DataFrame with a ``level`` column.
        out_dir:  Directory to write PNG files.
    """
    import matplotlib.pyplot as plt  # optional dep — imported here intentionally

    # Plot 1: total words & total unique words
    plt.figure(figsize=(8, 5))
    plt.plot(summary["level"], summary["total_words"],        marker="o", label="total_words")
    plt.plot(summary["level"], summary["total_unique_words"], marker="o", label="total_unique_words")
    plt.xlabel("CEFR Level")
    plt.ylabel("Count")
    plt.title("Total Words & Total Unique Words vs CEFR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cefr_words_metrics_plot.png"))
    plt.close()

    # Plot 2: sentence and word-level metrics
    metrics_to_plot = [
        "sentences",
        "avg_words_per_sentence",
        "avg_unique_words",
        "overall_avg_word_len",
        "overall_max_word_len",
    ]
    plt.figure(figsize=(10, 5))
    for metric in metrics_to_plot:
        if metric in summary.columns:
            plt.plot(summary["level"], summary[metric], marker="o", label=metric)
    plt.xlabel("CEFR Level")
    plt.ylabel("Average Value")
    plt.title("Sentence & Word-Level Metrics vs CEFR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cefr_sentence_word_metrics_plot.png"))
    plt.close()

    print(f"[cefr_surface_feats] Saved plots to: {out_dir}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    input_csv: str,
    out_dir: str,
    levels_csv: Optional[str] = None,
) -> None:
    """Run the full surface feature extraction pipeline.

    Args:
        input_csv:  Path to input CSV (Document_ID, Sentence, CEFR).
        out_dir:    Directory to write all output files.
        levels_csv: Optional fallback CSV with Document_ID and CEFR columns.

    Raises:
        ValueError: If required columns are missing from the input CSV.
    """
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv, dtype=str)

    required = {"Document_ID", "Sentence"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # --- Sentence-level metrics ---
    metrics = df["Sentence"].apply(sentence_metrics)
    df[["word_count", "unique_word_count", "avg_word_len", "min_word_len", "max_word_len"]] = (
        pd.DataFrame(metrics.tolist(), index=df.index)
    )

    out_sent = os.path.join(out_dir, "sentence_level_metrics.csv")
    df.to_csv(out_sent, index=False, encoding="utf-8")
    print(f"[cefr_surface_feats] Saved sentence-level metrics to: {out_sent}")

    # --- Essay-level aggregation ---
    essay_df = compute_essay_level(df)

    # Carry CEFR from sentence-level df into essay-level df
    if "CEFR" in df.columns:
        cefr_map = df.drop_duplicates("Document_ID").set_index("Document_ID")["CEFR"]
        essay_df["CEFR"] = essay_df["Document_ID"].map(cefr_map)

    essay_df = resolve_cefr(essay_df, levels_csv)

    out_essay = os.path.join(out_dir, "essay_level_summary.csv")
    essay_df.to_csv(out_essay, index=False, encoding="utf-8")
    print(f"[cefr_surface_feats] Saved essay-level summary to: {out_essay}")

    # --- CEFR-level summary ---
    summary = compute_level_summary(essay_df)

    out_level = os.path.join(out_dir, "surface_level_summary.csv")
    summary.to_csv(out_level, index=False, encoding="utf-8")
    print(f"[cefr_surface_feats] Saved surface level summary to: {out_level}")

    # --- Plots ---
    save_plots(summary, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract surface-level features and aggregate by CEFR level."
    )
    p.add_argument(
        "--input_csv",
        required=True,
        help="Input CSV with columns: Document_ID, Sentence, CEFR.",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for all CSVs and plots.",
    )
    p.add_argument(
        "--levels_csv",
        default=None,
        help=(
            "Optional fallback CSV with columns: Document_ID, CEFR. "
            "Used when CEFR cannot be inferred from Document_ID."
        ),
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        levels_csv=args.levels_csv,
    )