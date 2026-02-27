"""Compute tree statistics (depth/breadth/branching) from CoNLL-X files.

Input:
  --src_dir   directory containing *.conllx files (with '# id = ...' sentence ids)
  --out_dir   directory to write per-file Excel stats and all_files_stats.xlsx

Example:
  python feature_extraction/tree_stats.py \
    --src_dir <src_dir> \
    --out_dir <out_dir>
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd
import tqdm

# conllx_df is a local dependency cloned alongside this repo.
# We add the repo root to sys.path so it can be imported regardless of
# the working directory the script is called from.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from conllx_df.src.conllx_df import ConllxDf
from conllx_df.src.conll_utils import get_children_ids_of, get_token_count


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONLLX_EXT          = ".conllx"



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_sentence_ids(conll_file: str) -> List[str]:
    """Extract ordered sentence IDs from '# id = ...' comment lines.

    Args:
        conll_file: Path to a CoNLL-X file.

    Returns:
        List of sentence ID strings in file order.
    """
    sent_ids: List[str] = []
    with open(conll_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# id ="):
                sent_ids.append(line.split("=", 1)[1].strip())
    return sent_ids


def calculate_tree_stats(sent_df: pd.DataFrame) -> dict:
    """Compute depth, breadth, and branching statistics for a dependency tree.

    Args:
        sent_df: DataFrame for a single sentence from ConllxDf.

    Returns:
        Dict with keys: depth, breadth, depth_normalized, breadth_normalized,
        max_branching_factor, avg_branching_factor, min_branching_factor,
        total_nodes.
    """
    tree = [[0]]
    while True:
        new_level = []
        for parent_id in tree[-1]:
            new_level.extend(get_children_ids_of(sent_df, parent_id))
        if not new_level:
            break
        tree.append(new_level)

    depth    = len(tree) - 1
    breadth  = max(len(level) for level in tree)

    branching_factors = []
    for level in tree[1:]:
        for node_id in level:
            c = len(get_children_ids_of(sent_df, node_id))
            if c > 0:
                branching_factors.append(c)

    total_nodes = get_token_count(sent_df)

    return {
        "depth":                depth,
        "breadth":              breadth,
        "depth_normalized":     depth  / total_nodes if total_nodes > 0 else 0,
        "breadth_normalized":   breadth / total_nodes if total_nodes > 0 else 0,
        "max_branching_factor": max(branching_factors) if branching_factors else 0,
        "avg_branching_factor": (
            sum(branching_factors) / len(branching_factors)
            if branching_factors else 0
        ),
        "min_branching_factor": min(branching_factors) if branching_factors else 0,
        "total_nodes":          total_nodes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(args.src_dir) if f.endswith(CONLLX_EXT)])
    if not files:
        raise FileNotFoundError(f"No {CONLLX_EXT} files found in: {args.src_dir}")

    for file in tqdm.tqdm(files):
        conll_file   = os.path.join(args.src_dir, file)
        sentence_ids = extract_sentence_ids(conll_file)
        conll_df     = ConllxDf(conll_file)

        # Validate that sentence ID count matches parsed sentence count
        n_parsed = conll_df.get_sentence_count()
        if len(sentence_ids) != n_parsed:
            raise ValueError(
                f"[{file}] Sentence ID count ({len(sentence_ids)}) does not match "
                f"parsed sentence count ({n_parsed}). "
                "Check that every sentence has a '# id = ...' comment."
            )

        sent_rows = []
        for i, sent_id in enumerate(sentence_ids):
            sent_df = conll_df.get_df_by_id(i)
            stats   = calculate_tree_stats(sent_df)
            sent_rows.append({"sentence_id": sent_id, **stats})

        stats_df = pd.DataFrame(sent_rows)
        out_xlsx = os.path.join(args.out_dir, file.replace(CONLLX_EXT, ".xlsx"))
        stats_df.to_excel(out_xlsx, index=False)

    print(f"[tree_stats] Wrote stats to: {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Compute tree statistics (depth/breadth/branching) from CoNLL-X files."
    )
    ap.add_argument(
        "--src_dir",
        required=True,
        help=f"Directory containing {CONLLX_EXT} files.",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for per-file .xlsx stats and all_files_stats.xlsx.",
    )
    main(ap.parse_args())