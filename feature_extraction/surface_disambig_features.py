# cefr_sentence_camel_features.py

import re
import string
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from camel_tools.utils.charsets import AR_DIAC_CHARSET
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.utils.charsets import UNICODE_PUNCT_CHARSET


# -----------------------------
# Your syllable resources
# -----------------------------
vowels = ['a', 'e', 'i', 'o', 'u', 'aa', 'ee', 'ii', 'oo', 'uu']
arabic_vowels = ['ي', 'و', 'ا', 'ى', 'ؤ', 'ئ', 'إ', 'أ', 'آ', 'ة']
tanween = ['ً', 'ٍ', 'ٌ']


# -----------------------------
# Init CAMeL (BERT + Analyzer)
# -----------------------------
def init_camel(db_path: str):
    """
    Initialize MorphologyDB, Analyzer, and BERT disambiguator.
    """
    db = MorphologyDB(db_path, "a")
    analyzer = Analyzer(db, 'NONE', cache_size=100000)

    bert = BERTUnfactoredDisambiguator.pretrained(
        model_name='msa',
        pretrained_cache=False
    )
    # attach analyzer (as you did)
    bert._analyzer = analyzer
    return bert


# -----------------------------
# Your syllable counting (wrapped safely)
# -----------------------------
def count_syllables(caphi: Optional[str], diac: str, prc0: Optional[str], prc2: Optional[str]) -> int:
    syllable_count = 0

    diac = diac or ""

    if caphi:
        caphi_parts = caphi.split('_')

        for i, part in enumerate(caphi_parts):
            if part in vowels:
                # if last is vowel and diac ends with a diacritic char, skip it
                if i == len(caphi_parts) - 1:
                    if diac and diac[-1] in AR_DIAC_CHARSET:
                        continue
                syllable_count += 1

        # tanween adjustment
        if len(caphi_parts) > 2:
            if caphi_parts[-1] == 'n' and caphi_parts[-2] in vowels and diac and diac[-1] in tanween:
                syllable_count -= 1

        # exclude "Al" and "wa" prefixes if present
        if prc0 == 'Al_det':
            syllable_count -= 1
        if prc2 == 'wa_conj':
            syllable_count -= 1

    else:
        # fallback heuristic on diac string
        for ch in diac:
            if ch in arabic_vowels:
                syllable_count += 1

    return max(syllable_count, 0)


# -----------------------------
# Disambiguation with caching
# -----------------------------
def make_disambiguator(bert):
    @lru_cache(maxsize=2048)
    def disambiguate_sentence(sentence_text: str):
        toks = simple_word_tokenize(sentence_text)
        return bert.tag_sentence(toks)  # list of disambig results

    return disambiguate_sentence



# -----------------------------
# Sentence-level feature extraction
# -----------------------------
def sentence_camel_features(sentence: str, disambiguate_fn) -> Dict[str, float]:
    """
    Returns:
      - avg_syllables_per_word
      - unique_lemmas
      - unique_lemma_pos
      - tokens_count (CAMeL tokenizer)
    """
    disambig = disambiguate_fn(sentence)
    lemmas: List[str] = []
    lemma_pos_pairs: List[Tuple[str, str]] = []
    syllables = []
    token_count = 0

    for item in disambig:
        # if it is in UNICODE_PUNCT_CHARSET continue but add one to token count
        token_text = item.get('diac', '')
        if all(c in UNICODE_PUNCT_CHARSET for c in token_text):
            token_count += 1
            continue

        lemma = item.get('lex', '')
        pos = item.get('pos', '')

        # syllables from analysis fields
        caphi = item.get('caphi','')
        diac = item.get('diac', '')
        prc0 = item.get('prc0', '')
        prc2 = item.get('prc2', '')
        syllables_count = count_syllables(caphi, diac, prc0, prc2)
        syllables.append(syllables_count)
        # import pdb; pdb.set_trace()
        token_count += 1
        if lemma:
            lemmas.append(lemma)
            if pos:
                lemma_pos_pairs.append((lemma, pos))
            else:
                print('no pos')
                lemma_pos_pairs.append((lemma, ""))

    unique_lemmas = len(set(lemmas))
    unique_lemma_pos = len(set(lemma_pos_pairs))



    avg_syllables = np.mean(syllables) if syllables else 0.0
    max_syllables = np.max(syllables) if syllables else 0

    return {
        "camel_tokens": token_count,               # number of analyzed word tokens
        "avg_syllables_per_word": avg_syllables,   # mean syllables per word
        "max_syllables": max_syllables,             # max syllables in a word
        "unique_lemmas": unique_lemmas,             # type count (lemmas)
        "unique_lemma_pos": unique_lemma_pos,       # (lemma, POS) diversity
    }



# -----------------------------
# Main pipeline
# -----------------------------
def main(
    input_csv: str,
    levels_csv:str,
    output_sentence_csv: str = "sentence_features_camel.csv",
    output_group_csv: str = "group_features_camel.csv",
    db_path: str | None = None,
):
    df = pd.read_csv(input_csv)
    levels = pd.read_csv(levels_csv)
    # Expect at least: ID, Sentence
    if "ID" not in df.columns or "Sentence" not in df.columns:
        raise ValueError(f"CSV must include columns: ID, Sentence. Found: {list(df.columns)}")

    # init CAMeL
    bert = init_camel(db_path)
    disambiguate_fn = make_disambiguator(bert)

    # compute features
    feats = df["Sentence"].astype(str).apply(lambda s: sentence_camel_features(s, disambiguate_fn))

    feats_df = pd.DataFrame(list(feats), index=df.index)
    df = pd.concat([df, feats_df], axis=1)

    # save sentence-level
    df.to_csv(output_sentence_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_sentence_csv}")

    # group-level aggregation
    group_df = (
        df.groupby("anon_id")
          .agg(
              sentences=("ID", "count"),
              avg_syllables_per_word=("avg_syllables_per_word", "mean"),
              max_max_syllables=("max_syllables", "max"),
              avg_unique_lemmas=("unique_lemmas", "mean"),
              avg_unique_lemma_pos=("unique_lemma_pos", "mean"),
              avg_camel_tokens=("camel_tokens", "mean"),
          )
          .reset_index()
    )

    group_df.to_csv(output_group_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_group_csv}")

    df = group_df
    # Extract CEFR
    if "CEFR" not in df.columns:
        df["CEFR"] = df["anon_id"].str.split("_").str[1]
    # cefr_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
    if df["CEFR"].nunique() < 6:
        cefr_order = ["A", "B1", "B2", "C1", "C2"]
    else:
        cefr_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
    df["CEFR"] = pd.Categorical(df["CEFR"], categories=cefr_order, ordered=True)
        # Extract CEFR
    print(df["CEFR"].value_counts())
    if df["CEFR"].nunique() < 2:
        print("Warning: Some CEFR levels could not be extracted.")
        # cefr level mapping file
        levels["essay_id"] = levels["ID"].astype(str)
        cefr_dict = dict(zip(levels["ID"], levels["CEFR"]))
        def get_cefr_from_dict(group_id):
            # group_id = group_id.split("-")[2]
            return cefr_dict.get(group_id, "NA")
        df["CEFR"] = df["anon_id"].apply(get_cefr_from_dict)
        df = df[df["CEFR"] != "Unassessable"]

    else:
        print("All CEFR levels extracted successfully.")
        # cefr_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
        df["CEFR"] = pd.Categorical(df["CEFR"], categories=cefr_order, ordered=True)

    # Aggregate per CEFR
    cefr_summary = (
        df.groupby("CEFR")
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("CEFR")
    )
    cefr_summary = cefr_summary.rename(columns={"CEFR": "level"})

    cefr_summary.to_csv(f"{directory}/level_features_camel.csv", index=False, encoding="utf-8-sig")

    # Single combined graph
    plt.figure(figsize=(10, 5))

    plt.plot(cefr_summary["level"], cefr_summary["avg_syllables_per_word"], marker="o", label="avg_syllables_per_word")
    plt.plot(cefr_summary["level"], cefr_summary["max_max_syllables"], marker="o", label="max_syllables")
    plt.plot(cefr_summary["level"], cefr_summary["avg_unique_lemmas"], marker="o", label="avg_unique_lemmas")
    plt.plot(cefr_summary["level"], cefr_summary["avg_unique_lemma_pos"], marker="o", label="avg_unique_lemma_pos")
    plt.plot(cefr_summary["level"], cefr_summary["avg_camel_tokens"], marker="o", label="avg_camel_tokens")
    plt.xlabel("CEFR Level")
    plt.ylabel("Average Value")
    plt.title("CAMeL-based Linguistic Features vs CEFR Level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_group_csv.replace('.csv', '')}_plot.png")

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="Extract CAMeL-based sentence/group features (syllables, lemmas, etc.).")
    ap.add_argument("--input_csv", required=True, help="CSV with at least columns: ID, Sentence, anon_id")
    ap.add_argument("--levels_csv", default=None, help="Optional CSV to map IDs to CEFR if CEFR cannot be extracted from anon_id")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--db_path", default=os.environ.get("CAMEL_MORPH_DB"), help="Path to camel morphology DB. Can also be set via CAMEL_MORPH_DB env var.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(
        input_csv=args.input_csv,
        levels_csv=args.levels_csv or args.input_csv,
        output_sentence_csv=os.path.join(args.out_dir, "sentence_features_camel.csv"),
        output_group_csv=os.path.join(args.out_dir, "group_features_camel.csv"),
        db_path=args.db_path,
    )
