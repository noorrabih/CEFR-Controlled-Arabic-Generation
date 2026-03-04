"""Extract CAMeL-based lexical/surface features at sentence and essay level.

Features:
- camel_tokens
- avg_syllables_per_word
- max_syllables
- unique_lemmas
- unique_lemma_pos

Input CSV must include:
- ID: sentence id
- Document_ID: essay id
- Sentence: sentence text

Optional levels CSV can be provided to map essay IDs to CEFR when Document_ID does not encode it.

Example:
  export CAMEL_MORPH_DB=/path/to/camel_morph_msa_v1.0.db
  python feature_extraction/cefr_sentence_camel_features.py \
    --input_csv generated_essays_sentences.csv \
    --out_dir surface_feats/ \
    --levels_csv generated_essays_readability.csv \
    --levels_id_col ID \
    --levels_cefr_col CEFR
"""

from __future__ import annotations

import argparse
import os
import re
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from camel_tools.utils.charsets import AR_DIAC_CHARSET, UNICODE_PUNCT_CHARSET
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB


VOWELS = {'a','e','i','o','u','aa','ee','ii','oo','uu'}
ARABIC_VOWELS = {'ي','و','ا','ى','ؤ','ئ','إ','أ','آ','ة'}
TANWEEN = {'ً','ٍ','ٌ'}


def init_camel(db_path: str):
    db = MorphologyDB(db_path, "a")
    analyzer = Analyzer(db, 'NONE', cache_size=100000)

    bert = BERTUnfactoredDisambiguator.pretrained(model_name='msa', pretrained_cache=False)
    bert._analyzer = analyzer  # attach analyzer
    return bert


def count_syllables(caphi: Optional[str], diac: str, prc0: Optional[str], prc2: Optional[str]) -> int:
    diac = diac or ""
    syllable_count = 0

    if caphi:
        parts = caphi.split('_')
        for i, part in enumerate(parts):
            if part in VOWELS:
                if i == len(parts) - 1 and diac and diac[-1] in AR_DIAC_CHARSET:
                    continue
                syllable_count += 1

        if len(parts) > 2 and parts[-1] == 'n' and parts[-2] in VOWELS and diac and diac[-1] in TANWEEN:
            syllable_count -= 1

        if prc0 == 'Al_det':
            syllable_count -= 1
        if prc2 == 'wa_conj':
            syllable_count -= 1
    else:
        for ch in diac:
            if ch in ARABIC_VOWELS:
                syllable_count += 1

    return max(syllable_count, 0)


def make_disambiguator(bert):
    @lru_cache(maxsize=4096)
    def disambiguate_sentence(sentence_text: str):
        toks = simple_word_tokenize(sentence_text)
        return bert.tag_sentence(toks)

    return disambiguate_sentence


def sentence_camel_features(sentence: str, disambiguate_fn) -> Dict[str, float]:
    disambig = disambiguate_fn(sentence)

    lemmas: List[str] = []
    lemma_pos_pairs: List[Tuple[str, str]] = []
    syllables: List[int] = []
    token_count = 0

    for item in disambig:
        token_text = item.get('diac', '') or item.get('tok', '') or ''

        # punctuation-only token
        if token_text and all(c in UNICODE_PUNCT_CHARSET for c in token_text):
            token_count += 1
            continue

        lemma = item.get('lex', '') or ''
        pos = item.get('pos', '') or ''

        caphi = item.get('caphi', '')
        diac = item.get('diac', '')
        prc0 = item.get('prc0', '')
        prc2 = item.get('prc2', '')

        syllables.append(count_syllables(caphi, diac, prc0, prc2))
        token_count += 1

        if lemma:
            lemmas.append(lemma)
            lemma_pos_pairs.append((lemma, pos))

    return {
        "camel_tokens": float(token_count),
        "avg_syllables_per_word": float(np.mean(syllables)) if syllables else 0.0,
        "max_syllables": float(np.max(syllables)) if syllables else 0.0,
        "unique_lemmas": float(len(set(lemmas))),
        "unique_lemma_pos": float(len(set(lemma_pos_pairs))),
    }


def infer_cefr_order(levels: List[str]) -> List[str]:
    levels = [str(x) for x in levels if pd.notna(x)]
    if set(levels) <= {"A", "B", "C"}:
        return ["A", "B", "C"]
    if set(levels) <= {"A", "B1", "B2", "C1", "C2"}:
        return ["A", "B1", "B2", "C1", "C2"]
    return ["A1", "A2", "B1", "B2", "C1", "C2"]


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    out_sentence_csv = os.path.join(args.out_dir, args.output_sentence_csv)

    if args.sentence_csv:
        print(f"Loading existing sentence features from {args.sentence_csv}")
        sent_df = pd.read_csv(args.sentence_csv, encoding="utf-8-sig")
    else:
        if not args.db_path:
            raise ValueError("--db_path is required (or set CAMEL_MORPH_DB env var).")

        df = pd.read_csv(args.input_csv)
        for col in ["ID", "Document_ID", "Sentence"]:
            if col not in df.columns:
                raise ValueError(f"Input CSV must include columns: ID, Document_ID, Sentence. Found: {list(df.columns)}")

        bert = init_camel(args.db_path)
        disambiguate_fn = make_disambiguator(bert)

        feats = df["Sentence"].astype(str).apply(lambda s: sentence_camel_features(s, disambiguate_fn))
        feats_df = pd.DataFrame(list(feats), index=df.index)
        sent_df = pd.concat([df, feats_df], axis=1)
        sent_df.to_csv(out_sentence_csv, index=False, encoding="utf-8-sig")

    # Essay-level aggregation
    essay_df = (
        sent_df.groupby("Document_ID")
        .agg(
            sentences=("ID", "count"),
            avg_syllables_per_word=("avg_syllables_per_word", "mean"),
            max_syllables=("max_syllables", "max"),
            avg_unique_lemmas=("unique_lemmas", "mean"),
            avg_unique_lemma_pos=("unique_lemma_pos", "mean"),
            avg_camel_tokens=("camel_tokens", "mean"),
        )
        .reset_index()
    )

    # Map CEFR from levels_csv using direct Document_ID match
    if args.levels_csv:
        levels = pd.read_csv(args.levels_csv, dtype=str)
        if args.levels_id_col not in levels.columns or args.levels_cefr_col not in levels.columns:
            raise ValueError(
                f"levels_csv must include '{args.levels_id_col}' and '{args.levels_cefr_col}'. Found: {list(levels.columns)}"
            )
        mapping = dict(zip(levels[args.levels_id_col].astype(str), levels[args.levels_cefr_col].astype(str)))
        essay_df["CEFR"] = essay_df["Document_ID"].astype(str).map(mapping)

    if args.drop_unassessable:
        essay_df = essay_df[essay_df["CEFR"] != "Unassessable"]

    order = infer_cefr_order(essay_df["CEFR"].dropna().unique().tolist())
    essay_df["CEFR"] = pd.Categorical(essay_df["CEFR"], categories=order, ordered=True)

    out_group_csv = os.path.join(args.out_dir, args.output_group_csv)
    essay_df.to_csv(out_group_csv, index=False, encoding="utf-8-sig")

    # Per-level summary
    level_summary = (
        essay_df.groupby("CEFR")
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"CEFR": "level"})
        .sort_values("level")
    )

    out_level_csv = os.path.join(args.out_dir, args.output_level_csv)
    level_summary.to_csv(out_level_csv, index=False, encoding="utf-8-sig")

    # Plot
    plt.figure(figsize=(10, 5))
    for col in [
        "avg_syllables_per_word",
        "max_syllables",
        "avg_unique_lemmas",
        "avg_unique_lemma_pos",
        "avg_camel_tokens",
    ]:
        if col in level_summary.columns:
            plt.plot(level_summary["level"], level_summary[col], marker="o", label=col)

    plt.xlabel("CEFR Level")
    plt.ylabel("Average Value")
    plt.title("CAMeL-based Features vs CEFR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "camel_features_vs_cefr.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_sentence_csv}")
    print(f"Saved: {out_group_csv}")
    print(f"Saved: {out_level_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract CAMeL-based sentence/essay features.")
    ap.add_argument("--input_csv", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--db_path", default=os.environ.get("CAMEL_MORPH_DB"))

    ap.add_argument("--levels_csv", default=None)
    ap.add_argument("--levels_id_col", default="Document_ID")
    ap.add_argument("--levels_cefr_col", default="CEFR")
    ap.add_argument("--drop_unassessable", action="store_true")

    ap.add_argument("--sentence_csv", default=None, help="Skip CAMeL processing and load existing sentence features CSV.")
    ap.add_argument("--output_sentence_csv", default="sentence_features_camel.csv")
    ap.add_argument("--output_group_csv", default="group_features_camel.csv")
    ap.add_argument("--output_level_csv", default="level_features_camel.csv")

    main(ap.parse_args())
