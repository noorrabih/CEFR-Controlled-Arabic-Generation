"""Find relevant vocabulary items for a topic using a readability lexicon.

This script ranks lexicon entries (lemmas) by semantic similarity to a given topic/sentence,
restricted to a given SAMER readability level.

Input lexicon file must contain:
- readability (rounded average)
- lemma#pos

Example:
  python vocabulary_construction/vocab-relavance.py \
    --topic "اكتب عن السفر" \
    --lexicon_tsv vocabulary_construction/SAMER_Readability_Lexicon.tsv \
    --level 3 \
    --top_k 50 \
    --out_csv vocab/relevant_words_level3.csv
"""

from __future__ import annotations

import argparse
from typing import Optional, Set

import pandas as pd


DEFAULT_UNWANTED_POS: Set[str] = {
    "PUNCT", "SYM", "X", "NUM", "PART", "DET", "ADP", "CCONJ", "SCONJ",
}


def read_any_delim(path: str) -> pd.DataFrame:
    # try TSV then CSV
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path)


def relevant_words_from_file(
    sentence: str,
    path: str,
    level: int,
    threshold: float = 0.35,
    top_k: Optional[int] = None,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    unwanted_pos: Optional[Set[str]] = None,
    device: str = "auto",
) -> pd.DataFrame:
    """Return lexicon entries relevant to `sentence` at a given readability `level`."""

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    if unwanted_pos is None:
        unwanted_pos = DEFAULT_UNWANTED_POS

    df = read_any_delim(path).fillna("")
    if "readability (rounded average)" not in df.columns:
        raise ValueError("Lexicon must include column 'readability (rounded average)'.")
    if "lemma#pos" not in df.columns:
        raise ValueError("Lexicon must include column 'lemma#pos'.")

    df["readability (rounded average)"] = pd.to_numeric(df["readability (rounded average)"], errors="coerce")
    df = df[df["readability (rounded average)"] == level]

    parts = df["lemma#pos"].astype(str).str.split("#", n=1, expand=True)
    df["lemma"] = parts[0]
    df["pos"] = parts[1] if parts.shape[1] > 1 else ""

    if unwanted_pos:
        df = df[~df["pos"].isin(unwanted_pos)]

    if df.empty:
        return df

    model = SentenceTransformer(model_name)
    if device != "auto":
        model = model.to(device)

    lemmas = df["lemma"].astype(str).tolist()
    sent_emb = model.encode([sentence], normalize_embeddings=True)
    lemma_emb = model.encode(lemmas, normalize_embeddings=True, batch_size=1024)

    sims = cosine_similarity(sent_emb, lemma_emb)[0]
    df = df.copy()
    df["similarity"] = sims
    df = df[df["similarity"] >= threshold].sort_values("similarity", ascending=False)

    if top_k is not None:
        df = df.head(int(top_k))

    return df[["lemma#pos", "lemma", "pos", "readability (rounded average)", "similarity"]]


def main(args):
    out = relevant_words_from_file(
        sentence=args.topic,
        path=args.lexicon_tsv,
        level=args.level,
        threshold=args.threshold,
        top_k=args.top_k,
        model_name=args.model_name,
        device=args.device,
    )

    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Saved: {args.out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rank relevant lexicon words for a topic at a readability level.")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--lexicon_tsv", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--model_name", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--out_csv", required=True)
    main(ap.parse_args())
