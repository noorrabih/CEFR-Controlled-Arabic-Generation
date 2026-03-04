"""Disambiguate GPT vocabulary words with CAMeL and filter by SAMER readability.

Pipeline:
  1. Extract unique words from a GPT vocab JSONL.
  2. Disambiguate each unique word with CAMeL BERT, get lemma/POS/gloss.
  3. Merge with SAMER TSV to get readability levels.
  4. Filter words whose SAMER level matches the target CEFR level.

Input JSONL must contain fields: prompt_id, topic_id, level, words
Input SAMER TSV must contain columns: lex_CM, pos_CM, stemgloss_CM,
  readability (rounded average)

Outputs (all written to --out_dir):
  unique_vocab.csv       unique words extracted from JSONL
  disambig_samer.csv     unique words with lemma/POS/gloss + SAMER level
  filtered_vocab.csv     per-prompt filtered word lists matching CEFR level

Example:
  export CAMEL_MORPH_DB=/path/to/camel_morph_msa_v1.0.db

  python vocabulary_construction/disambig_samer.py \
    --input_jsonl <input_jsonl> \
    --samer_tsv <samer_tsv> \
    --out_dir <out_dir> \
    --levels 6

  # Explicit DB path:
  python vocabulary_construction/disambig_samer.py \
    --input_jsonl <input_jsonl> \
    --samer_tsv <samer_tsv> \
    --db_path <db_path> \
    --out_dir <out_dir> \
    --levels 3
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.tokenizers.word import simple_word_tokenize


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIACRITICS = {"َ", "ِ", "ُ", "ً", "ٍ", "ٌ", "ْ", "ّ"}

LEVEL_MAPPINGS: Dict[int, Dict[str, Union[int, List[int]]]] = {
    6: {"A1": 1, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5},
    5: {"A":  1, "B1": 2, "B2": 3, "C1": 4, "C2": 5},
    3: {"A":  1, "B":  [2, 3], "C": [4, 5]},
}

# Default SAMER level assigned to unmatched words
DEFAULT_SAMER_LEVEL = 5


# ---------------------------------------------------------------------------
# CAMeL helpers
# ---------------------------------------------------------------------------

def init_camel(db_path: str) -> Tuple:
    """Initialise CAMeL morphology analyzer and BERT disambiguator.

    Args:
        db_path: Path to the CAMeL morphology DB file.

    Returns:
        Tuple of (analyzer, bert).
    """
    db       = MorphologyDB(db_path, "a")
    analyzer = Analyzer(db, "NONE", cache_size=100000)
    bert     = BERTUnfactoredDisambiguator.pretrained(
        model_name="msa", pretrained_cache=False
    )
    bert._analyzer = analyzer
    return analyzer, bert


def clean_lemma(lemma: str) -> str:
    """Remove a trailing diacritic from a lemma string.

    Args:
        lemma: Lemma string from CAMeL disambiguation.

    Returns:
        Lemma with trailing diacritic removed if present.
    """
    return lemma[:-1] if lemma and lemma[-1] in _DIACRITICS else lemma


def make_analyzer_fn(analyzer, bert):
    """Return a cached token analysis function.

    Wraps CAMeL BERT disambiguation with an LRU cache so each unique
    token is analyzed only once. Re-analyzes noun_prop and pron_rel
    tokens with the morphological analyzer to get a better POS tag.

    Args:
        analyzer: CAMeL Analyzer instance.
        bert:     CAMeL BERTUnfactoredDisambiguator instance.

    Returns:
        Cached callable: token -> (lemma, pos, stemgloss).
    """
    @lru_cache(maxsize=None)
    def analyze_token(token: str) -> Tuple[str, str, str]:
        disambig = bert.tag_sentence(simple_word_tokenize(token))
        if not disambig:
            return ("N/A", "N/A", "N/A")

        best = disambig[0]
        pos  = best.get("pos", "N/A")

        # Re-analyze special POS tags with morphological analyzer
        if pos in ("noun_prop", "pron_rel"):
            analyses      = analyzer.analyze(token)
            noun_analyses = [a for a in analyses if a.get("pos") == "noun"]
            adj_analyses  = [a for a in analyses if a.get("pos") == "adj"]
            if noun_analyses:
                best = noun_analyses[0]
            elif pos == "noun_prop" and adj_analyses:
                best = adj_analyses[0]
            pos = best.get("pos", pos)

        lemma     = clean_lemma(best.get("lex",       "N/A") or "N/A")
        stemgloss = best.get("stemgloss", "N/A") or "N/A"
        return (lemma, pos or "N/A", stemgloss)

    return analyze_token


# ---------------------------------------------------------------------------
# Stage 1: extract unique words
# ---------------------------------------------------------------------------

def extract_unique_words(input_jsonl: str) -> Tuple[List[dict], List[str]]:
    """Load vocab JSONL and extract unique words.

    Args:
        input_jsonl: Path to GPT vocab JSONL file.

    Returns:
        Tuple of (all_entries, unique_words).
    """
    entries: List[dict] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    all_words: List[str] = [
        word
        for entry in entries
        for word in entry.get("words", [])
    ]
    unique_words = sorted(set(all_words))
    print(f"[disambig_samer] Loaded {len(entries)} entries, {len(unique_words)} unique words.")
    return entries, unique_words


# ---------------------------------------------------------------------------
# Stage 2: disambiguate + merge with SAMER
# ---------------------------------------------------------------------------

def disambiguate_words(
    unique_words: List[str],
    analyze_token,
) -> pd.DataFrame:
    """Disambiguate each unique word and return a word/subword-level DataFrame.

    Multi-word phrases are split into subwords; each subword is analyzed
    independently. Results are merged back to retain the original phrase.

    Args:
        unique_words:  List of unique word strings (may include phrases).
        analyze_token: Cached token analysis function from :func:`make_analyzer_fn`.

    Returns:
        DataFrame with columns: word, subword, lemma, pos, stemgloss.
    """
    rows = [{"word": w} for w in unique_words]
    df   = pd.DataFrame(rows)

    df["subwords"] = df["word"].astype(str).str.split()
    df = (
        df.explode("subwords", ignore_index=True)
        .rename(columns={"subwords": "subword"})
    )
    df["subword"] = df["subword"].str.strip()
    df = df[df["subword"].ne("")]

    uniq_tokens = df["subword"].unique()
    print(f"[disambig_samer] Analyzing {len(uniq_tokens)} unique tokens...")
    lookup = {tok: analyze_token(tok) for tok in uniq_tokens}

    df_lookup = (
        pd.DataFrame.from_dict(
            lookup, orient="index", columns=["lemma", "pos", "stemgloss"]
        )
        .reset_index()
        .rename(columns={"index": "subword"})
    )

    df = df.merge(df_lookup, on="subword", how="left")
    return df[["word", "subword", "lemma", "pos", "stemgloss"]]


def merge_with_samer(
    disambig_df: pd.DataFrame,
    samer_tsv: str,
) -> pd.DataFrame:
    """Merge disambiguated words with SAMER readability levels.

    Matches on lemma + POS + stemgloss. Unmatched words are assigned
    SAMER level 5 (hardest).

    Args:
        disambig_df: DataFrame from :func:`disambiguate_words`.
        samer_tsv:   Path to SAMER TSV file.

    Returns:
        DataFrame with an added ``samer_readability_level`` column.

    Raises:
        ValueError: If required columns are missing from the SAMER TSV.
    """
    samer_df = pd.read_csv(samer_tsv, sep="\t", dtype=str)

    required = {"lex_CM", "pos_CM", "stemgloss_CM", "readability (rounded average)"}
    missing  = required - set(samer_df.columns)
    if missing:
        raise ValueError(
            f"SAMER TSV missing required columns: {missing}. "
            f"Found: {list(samer_df.columns)}"
        )

    samer_df["lex_CM"] = samer_df["lex_CM"].apply(
        lambda x: clean_lemma(str(x)) if pd.notna(x) else x
    )

    merged = pd.merge(
        disambig_df,
        samer_df[["lex_CM", "pos_CM", "stemgloss_CM", "readability (rounded average)"]],
        left_on=["lemma", "pos", "stemgloss"],
        right_on=["lex_CM", "pos_CM", "stemgloss_CM"],
        how="left",
    )
    merged = merged.drop(columns=["lex_CM", "pos_CM", "stemgloss_CM"])
    merged = merged.rename(columns={"readability (rounded average)": "samer_readability_level"})
    merged["samer_readability_level"] = (
        pd.to_numeric(merged["samer_readability_level"], errors="coerce")
        .fillna(DEFAULT_SAMER_LEVEL)
    )

    matched = merged["samer_readability_level"].ne(DEFAULT_SAMER_LEVEL).sum()
    print(f"[disambig_samer] SAMER match rate: {matched}/{len(merged)} subwords.")
    return merged


# ---------------------------------------------------------------------------
# Stage 3: filter by CEFR level
# ---------------------------------------------------------------------------

def _level_matches(samer_level: float, target: Union[int, List[int]]) -> bool:
    if isinstance(target, list):
        return samer_level in target
    return samer_level == target


def filter_by_cefr(
    entries: List[dict],
    disambig_samer: pd.DataFrame,
    level_mapping: Dict[str, Union[int, List[int]]],
) -> pd.DataFrame:
    """Filter vocab words to those whose SAMER level matches the CEFR target.

    For multi-word phrases, the maximum SAMER level across subwords is used.

    Args:
        entries:        All entries from the input JSONL.
        disambig_samer: Merged disambig + SAMER DataFrame.
        level_mapping:  CEFR level string → SAMER level(s).

    Returns:
        DataFrame with columns: prompt_id, topic_id, CEFR_level, level, words.
        The words column contains lists of (word, pos, gloss) tuples.
    """
    rows: List[dict] = []

    for entry in entries:
        prompt_id = entry["prompt_id"]
        topic_id  = entry["topic_id"]
        cefr      = str(entry["level"])
        target    = level_mapping.get(cefr)

        if target is None:
            print(f"[WARNING] Unknown level '{cefr}' for prompt {prompt_id}. Skipping.")
            continue

        matched_words: List[tuple] = []

        for word in entry.get("words", []):
            subwords = word.split() if " " in word else [word]

            if len(subwords) > 1:
                # Multi-word phrase: use max SAMER level across subwords
                levels: List[float] = []
                last_match = None
                for subword in subwords:
                    match = disambig_samer[
                        (disambig_samer["subword"] == subword) &
                        (disambig_samer["word"]    == word)
                    ]
                    if not match.empty:
                        if len(match) > 1:
                            print(
                                f"[WARNING] Multiple matches for subword "
                                f"'{subword}' in '{word}'. Using first."
                            )
                        samer_level = float(match["samer_readability_level"].values[0])
                        levels.append(samer_level)
                        last_match = match

                if levels and last_match is not None:
                    if _level_matches(max(levels), target):
                        matched_words.append((
                            word,
                            last_match["pos"].values[0],
                            last_match["stemgloss"].values[0],
                        ))

            else:
                # Single word
                match = disambig_samer[disambig_samer["subword"] == word]
                if not match.empty:
                    samer_level = float(match["samer_readability_level"].values[0])
                    if _level_matches(samer_level, target):
                        matched_words.append((
                            word,
                            match["pos"].values[0],
                            match["stemgloss"].values[0],
                        ))

        rows.append({
            "prompt_id":  prompt_id,
            "topic_id":   topic_id,
            "CEFR_level": cefr,
            "level":      target,
            "words":      matched_words,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    input_jsonl: str,
    samer_tsv: str,
    out_dir: str,
    db_path: str,
    n_levels: int = 6,
) -> None:
    """Run the full disambiguate-and-filter pipeline.

    Args:
        input_jsonl: Path to GPT vocab JSONL.
        samer_tsv:   Path to SAMER TSV file.
        out_dir:     Directory to write all output CSVs.
        db_path:     Path to CAMeL morphology DB.
        n_levels:    CEFR level granularity: 3, 5, or 6.

    Raises:
        ValueError: If n_levels is not 3, 5, or 6, or db_path is not set.
    """
    if not db_path:
        raise ValueError(
            "--db_path is required. "
            "Alternatively set the CAMEL_MORPH_DB environment variable."
        )
    if n_levels not in LEVEL_MAPPINGS:
        raise ValueError(f"--levels must be 3, 5, or 6. Got: {n_levels}")

    os.makedirs(out_dir, exist_ok=True)
    level_mapping = LEVEL_MAPPINGS[n_levels]
    print(f"[disambig_samer] Using {n_levels}-level mapping: {level_mapping}")

    # --- Stage 1: unique words ---
    entries, unique_words = extract_unique_words(input_jsonl)

    unique_df = pd.DataFrame({"word": unique_words})
    out_unique = os.path.join(out_dir, "unique_vocab.csv")
    unique_df.to_csv(out_unique, index=False)
    print(f"[disambig_samer] Saved unique vocab to: {out_unique}")

    # --- Stage 2: disambiguate + SAMER merge ---
    analyzer, bert = init_camel(db_path)
    analyze_token  = make_analyzer_fn(analyzer, bert)

    disambig = disambiguate_words(unique_words, analyze_token)
    disambig_samer = merge_with_samer(disambig, samer_tsv)

    out_disambig = os.path.join(out_dir, "disambig_samer.csv")
    disambig_samer.to_csv(out_disambig, index=False)
    print(f"[disambig_samer] Saved disambig+SAMER to: {out_disambig}")

    # --- Stage 3: filter by CEFR level ---
    filtered = filter_by_cefr(entries, disambig_samer, level_mapping)

    out_filtered = os.path.join(out_dir, "filtered_vocab.csv")
    filtered.to_csv(out_filtered, index=False)
    print(f"[disambig_samer] Saved filtered vocab to: {out_filtered}")

    # --- Stats ---
    lengths = filtered["words"].apply(len)
    print(f"[disambig_samer] Word count distribution after filtering:")
    print(lengths.value_counts().sort_index().to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Disambiguate GPT vocabulary with CAMeL BERT and filter "
            "by SAMER readability level matching CEFR."
        )
    )
    p.add_argument(
        "--input_jsonl",
        required=True,
        help="GPT vocab JSONL with fields: prompt_id, topic_id, level, words.",
    )
    p.add_argument(
        "--samer_tsv",
        required=True,
        help="SAMER TSV with columns: lex_CM, pos_CM, stemgloss_CM, readability (rounded average).",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for all CSVs.",
    )
    p.add_argument(
        "--db_path",
        default=os.environ.get("CAMEL_MORPH_DB"),
        help="Path to CAMeL morphology DB. Defaults to CAMEL_MORPH_DB env var.",
    )
    p.add_argument(
        "--levels",
        type=int,
        choices=[3, 5, 6],
        default=5,
        help="CEFR level granularity: 3 (A/B/C), 5 (A/B1/B2/C1/C2), 6 (A1-C2). Default: 6.",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(
        input_jsonl=args.input_jsonl,
        samer_tsv=args.samer_tsv,
        out_dir=args.out_dir,
        db_path=args.db_path,
        n_levels=args.levels,
    )
