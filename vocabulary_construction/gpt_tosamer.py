"""Map GPT-generated vocabulary to SAMER readability levels and filter by target CEFR.

Input CSV should have columns:
  - prompt_id
  - CEFR_level (A1..C2 or coarse A/B/C)
  - words (a JSON list string or Python list string)

SAMER lexicon TSV should include:
  - lex_s31 (word form)
  - readability (rounded average)  (integer)

This script:
1) Looks up each word in the SAMER lexicon
2) Keeps words whose readability level matches the target CEFR bucket
3) Writes an output CSV with filtered words and optional POS/gloss (if available)

Example:
  python vocabulary_construction/gpt_tosamer.py \
    --input_csv vocab/full_vocab_prompts.csv \
    --samer_tsv vocabulary_construction/SAMER_Readability_Levels_camelmorph.tsv \
    --out_csv vocab/full_vocab_filtered_by_samer.csv
"""

from __future__ import annotations

import argparse
import ast
import json
from typing import Dict, List, Tuple

import pandas as pd


def _parse_words_cell(x) -> List[str]:
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


def build_samer_lookup(df: pd.DataFrame, word_col: str, level_col: str, pos_col: str | None, gloss_col: str | None) -> Dict[str, Tuple[int, str, str]]:
    out: Dict[str, Tuple[int, str, str]] = {}
    for _, r in df.iterrows():
        w = str(r[word_col]).strip()
        if not w:
            continue
        try:
            lvl = int(float(r[level_col]))
        except Exception:
            continue
        pos = str(r[pos_col]).strip() if pos_col and pos_col in df.columns else ""
        gloss = str(r[gloss_col]).strip() if gloss_col and gloss_col in df.columns else ""
        # keep minimum level if duplicates
        if w not in out or lvl < out[w][0]:
            out[w] = (lvl, pos, gloss)
    return out


def cefr_to_samer_targets(cefr: str) -> List[int]:
    cefr = str(cefr).strip().upper()
    # Heuristic buckets (adjust as needed for your setup)
    mapping = {
        "A1": [1],
        "A2": [2],
        "B1": [3],
        "B2": [4],
        "C1": [5],
        "C2": [5],
        "A": [1, 2],
        "B": [3, 4],
        "C": [5],
    }
    return mapping.get(cefr, [])


def main(args):
    vocab_df = pd.read_csv(args.input_csv)
    samer_df = pd.read_csv(args.samer_tsv, sep="\t")

    for col in [args.prompt_id_col, args.cefr_col, args.words_col]:
        if col not in vocab_df.columns:
            raise ValueError(f"Missing column '{col}' in input_csv. Found: {list(vocab_df.columns)}")

    # Detect columns in SAMER
    word_col = args.samer_word_col
    level_col = args.samer_level_col
    if word_col not in samer_df.columns or level_col not in samer_df.columns:
        raise ValueError(f"SAMER TSV must include '{word_col}' and '{level_col}'. Found: {list(samer_df.columns)}")

    lookup = build_samer_lookup(samer_df, word_col, level_col, args.samer_pos_col, args.samer_gloss_col)

    out_rows = []
    for _, r in vocab_df.iterrows():
        cefr = r[args.cefr_col]
        targets = set(cefr_to_samer_targets(cefr))
        words = _parse_words_cell(r[args.words_col])

        kept = []
        meta = []
        for w in words:
            if w in lookup and (not targets or lookup[w][0] in targets):
                kept.append(w)
                lvl, pos, gloss = lookup[w]
                meta.append({"word": w, "samer_level": lvl, "pos": pos, "gloss": gloss})

        out_rows.append({
            "prompt_id": r[args.prompt_id_col],
            "CEFR_level": str(cefr),
            "words": json.dumps(kept, ensure_ascii=False),
            "n_words": len(kept),
            "words_meta": json.dumps(meta, ensure_ascii=False),
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Filter GPT vocab lists using SAMER readability levels.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--samer_tsv", required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--prompt_id_col", default="prompt_id")
    ap.add_argument("--cefr_col", default="CEFR_level")
    ap.add_argument("--words_col", default="words")

    ap.add_argument("--samer_word_col", default="lex_s31")
    ap.add_argument("--samer_level_col", default="readability (rounded average)")
    ap.add_argument("--samer_pos_col", default="pos_s31")
    ap.add_argument("--samer_gloss_col", default="stemgloss_s31")

    main(ap.parse_args())
