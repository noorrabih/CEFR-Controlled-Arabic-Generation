"""Identify prompts with too few SAMER-filtered vocabulary words.

Reads the filtered vocab CSV produced by gpt_tosamer.py, finds prompts
with fewer than --min_words matched words, merges with the original prompts
CSV, and saves them for the relevance expansion step.

Inputs:
  --filtered_vocab_csv  Output of gpt_tosamer.py (prompt_id, topic_id,
                        CEFR_level, level, words)
  --prompts_csv         Original prompts CSV containing prompt_id

Output:
  --out_csv             Prompts with low word counts for relevance expansion

Example:
  python vocabulary_construction/low_vocab_prompts.py \
    --filtered_vocab_csv <filtered_vocab_csv> \
    --prompts_csv <prompts_csv> \
    --out_csv <out_csv> \
    --min_words 3
"""

from __future__ import annotations

import argparse
import ast
import os
import re
from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex to replace bare `nan` tokens in string-serialised lists
_NAN_TOKEN = re.compile(r"(?<![\w])nan(?![\w])", flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_words_cell(s: str) -> list:
    """Parse a string-serialised list of (word, pos, gloss) tuples.

    Handles bare `nan` tokens that pandas writes when a value is NaN,
    replacing them with None before eval so ast.literal_eval succeeds.

    Args:
        s: String representation of a Python list.

    Returns:
        Parsed list, or empty list on parse failure.
    """
    s = _NAN_TOKEN.sub("None", str(s).strip())
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        print(f"[WARNING] Could not parse words cell: {s[:80]}...")
        return []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    filtered_vocab_csv: str,
    prompts_csv: str,
    out_csv: str,
    min_words: int = 3,
) -> None:
    """Find low-vocab prompts and merge with original prompts CSV.

    Args:
        filtered_vocab_csv: Path to filtered vocab CSV from gpt_tosamer.py.
        prompts_csv:        Path to original prompts CSV.
        out_csv:            Output path for low-vocab prompts.
        min_words:          Prompts with fewer than this many words are flagged.

    Raises:
        ValueError: If required columns are missing from either input.
    """
    # --- Load filtered vocab ---
    vocab_df = pd.read_csv(filtered_vocab_csv, dtype=str)
    required = {"prompt_id", "words"}
    missing  = required - set(vocab_df.columns)
    if missing:
        raise ValueError(
            f"Filtered vocab CSV missing required columns: {missing}. "
            f"Found: {list(vocab_df.columns)}"
        )

    vocab_df["words"]      = vocab_df["words"].apply(parse_words_cell)
    vocab_df["word_count"] = vocab_df["words"].apply(len)

    # --- Distribution stats ---
    dist = vocab_df["word_count"].value_counts().sort_index()
    print(f"[low_vocab_prompts] Word count distribution:")
    print(dist.to_string())

    # --- Filter low-vocab prompts ---
    low_vocab = vocab_df[vocab_df["word_count"] < min_words].copy()
    print(
        f"[low_vocab_prompts] {len(low_vocab)} prompts have fewer than "
        f"{min_words} words (out of {len(vocab_df)} total)."
    )

    # --- Merge with original prompts ---
    prompts_df = pd.read_csv(prompts_csv, dtype=str)
    if "prompt_id" not in prompts_df.columns:
        raise ValueError(
            f"Prompts CSV missing 'prompt_id' column. "
            f"Found: {list(prompts_df.columns)}"
        )

    merged = low_vocab.merge(prompts_df, on="prompt_id", how="left")

    # --- Save ---
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[low_vocab_prompts] Saved {len(merged)} low-vocab prompts to: {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Find prompts with too few SAMER-filtered words for relevance expansion."
    )
    p.add_argument(
        "--filtered_vocab_csv",
        required=True,
        help="Filtered vocab CSV produced by gpt_tosamer.py.",
    )
    p.add_argument(
        "--prompts_csv",
        required=True,
        help="Original prompts CSV containing prompt_id.",
    )
    p.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV path for low-vocab prompts.",
    )
    p.add_argument(
        "--min_words",
        type=int,
        default=3,
        help="Prompts with fewer than this many matched words are flagged (default: 3).",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(
        filtered_vocab_csv=args.filtered_vocab_csv,
        prompts_csv=args.prompts_csv,
        out_csv=args.out_csv,
        min_words=args.min_words,
    )