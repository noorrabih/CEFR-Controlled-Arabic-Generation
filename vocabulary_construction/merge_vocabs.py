"""Merge two vocabulary sources (e.g., GPT vocab + relevance-expanded vocab).

Both inputs should have at least:
  - prompt_id
  - CEFR_level
  - words (JSON-list string or comma-separated)

Output:
  - merged word list per prompt_id (union)

Example:
  python vocabulary_construction/merge_vocabs.py \
    --gpt_csv vocab/full_vocab_filtered_by_samer.csv \
    --relevant_csv vocab/relevant_words.csv \
    --out_csv vocab/merged_vocabs.csv
"""

from __future__ import annotations

import argparse
import ast
import json
from typing import List, Set

import pandas as pd


def _parse_words(x) -> List[str]:
    if x is None:
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(w).strip() for w in v if str(w).strip()]
    except Exception:
        pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(w).strip() for w in v if str(w).strip()]
    except Exception:
        pass
    return [w.strip() for w in s.split(",") if w.strip()]


def main(args):
    gpt = pd.read_csv(args.gpt_csv)
    rel = pd.read_csv(args.relevant_csv)

    for df, name in [(gpt, "gpt"), (rel, "relevant")]:
        for col in [args.prompt_id_col, args.cefr_col, args.words_col]:
            if col not in df.columns:
                raise ValueError(f"{name} CSV missing '{col}'. Found: {list(df.columns)}")

    merged_rows = []
    all_ids = sorted(set(gpt[args.prompt_id_col].astype(str)) | set(rel[args.prompt_id_col].astype(str)))

    gpt_map = {str(r[args.prompt_id_col]): r for _, r in gpt.iterrows()}
    rel_map = {str(r[args.prompt_id_col]): r for _, r in rel.iterrows()}

    for pid in all_ids:
        g = gpt_map.get(pid)
        r = rel_map.get(pid)

        cefr = None
        if g is not None:
            cefr = g[args.cefr_col]
        elif r is not None:
            cefr = r[args.cefr_col]

        words: Set[str] = set()
        if g is not None:
            words.update(_parse_words(g[args.words_col]))
        if r is not None:
            words.update(_parse_words(r[args.words_col]))

        merged_rows.append({
            "prompt_id": pid,
            "CEFR_level": str(cefr) if cefr is not None else "",
            "words": json.dumps(sorted(words), ensure_ascii=False),
            "n_words": len(words),
        })

    out = pd.DataFrame(merged_rows)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge two vocab sources into a union per prompt.")
    ap.add_argument("--gpt_csv", required=True)
    ap.add_argument("--relevant_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--prompt_id_col", default="prompt_id")
    ap.add_argument("--cefr_col", default="CEFR_level")
    ap.add_argument("--words_col", default="words")
    main(ap.parse_args())
