"""Extract semantically relevant words from SAMER lexicon for low-vocab prompts.

For each prompt in --prompts_csv, finds lemmas from the SAMER lexicon whose
embedding similarity to the prompt text exceeds --threshold, filtered to the
prompt's target SAMER readability level. Falls back to a lower threshold if
no lemmas are found at the primary threshold.

Input CSV (--prompts_csv) must contain columns:
  prompt_id, level, Arabic Text, Translated Text, Topic_ID
  (as produced by low_vocab_prompts.py)

Input lexicon (--lexicon_path) must be a TSV with columns:
  lex_CM, pos_CM, stemgloss_CM, readability (rounded average)

Output CSV (--out_csv) columns:
  prompt_id, Level, Arabic Text, Translated Text, Topic_ID, lemmas

Example:
  python vocabulary_construction/vocab_relevance.py \
    --prompts_csv <prompts_csv> \
    --lexicon_path <lexicon_path> \
    --out_csv <out_csv> \
    --threshold 0.50 \
    --fallback_threshold 0.35 \
    --top_k 40 \
    --device auto
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import List, Optional, Set

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_UNWANTED_POS: Set[str] = {
    "prep", "pron_dem", "pron_rel", "part_neg", "conj", "conj_sub", "part",
    "part_det", "part_focus", "part_fut", "part_interrog", "part_restrict",
    "part_verb", "part_voc", "pron", "punc", "adv_rel", "pron_interrog",
    "pron_exclam",
}

_AR_DIAC = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]"
)

READABILITY_COL = "readability (rounded average)"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_ar(s: str) -> str:
    """Lightly normalize Arabic text by removing diacritics and unifying forms."""
    if not isinstance(s, str):
        return s
    s = _AR_DIAC.sub("", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ة", "ه").replace("ى", "ي").replace("ـ", "")
    return s


def read_any_delim(path: str) -> pd.DataFrame:
    """Read CSV or TSV, trying comma then tab then sniffing delimiter."""
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE)
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", quoting=csv.QUOTE_NONE)


# ---------------------------------------------------------------------------
# Model (lazy singleton)
# ---------------------------------------------------------------------------

_ST_MODEL = None
_ST_DEVICE = "cpu"


def _get_device(preferred: Optional[str] = "auto") -> str:
    try:
        import torch
        if preferred in ("cuda", "auto") and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_model(
    name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device: Optional[str] = "auto",
):
    """Load the SentenceTransformer model as a lazy singleton.

    Args:
        name:   Model name or path.
        device: 'auto', 'cuda', or 'cpu'.

    Returns:
        Tuple of (model, actual_device).
    """
    global _ST_MODEL, _ST_DEVICE
    if _ST_MODEL is not None:
        return _ST_MODEL, _ST_DEVICE
    dev = _get_device(device)
    from sentence_transformers import SentenceTransformer
    _ST_MODEL  = SentenceTransformer(name, device=dev)
    _ST_DEVICE = dev
    print(f"[relevant_words] Loaded model '{name}' on device: {dev}")
    return _ST_MODEL, _ST_DEVICE


def _encode_in_batches(texts: List[str], batch_size: int, use_fp16: bool):
    """Encode texts in batches with optional FP16 autocast on CUDA."""
    import torch
    use_amp    = use_fp16 and (_ST_DEVICE == "cuda")
    embeddings = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        if use_amp:
            with torch.cuda.amp.autocast():
                emb = _ST_MODEL.encode(
                    chunk, batch_size=len(chunk), convert_to_tensor=True,
                    normalize_embeddings=True, show_progress_bar=False,
                )
        else:
            emb = _ST_MODEL.encode(
                chunk, batch_size=len(chunk), convert_to_tensor=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0) if len(embeddings) > 1 else embeddings[0]


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def relevant_words_from_lexicon(
    sentence: str,
    lexicon_df: pd.DataFrame,
    level: int,
    threshold: float = 0.35,
    top_k: Optional[int] = None,
    normalize_arabic: bool = True,
    unwanted_pos: Optional[Set[str]] = None,
    use_fp16: bool = True,
    batch_size: int = 1024,
) -> pd.DataFrame:
    """Find lexicon words semantically relevant to a sentence at a given level.

    Args:
        sentence:         Arabic sentence to match against.
        lexicon_df:       Pre-loaded SAMER lexicon DataFrame with columns:
                          lex_CM, pos_CM, stemgloss_CM,
                          readability (rounded average).
        level:            Target SAMER readability level (1-5).
        threshold:        Minimum cosine similarity to include a word.
        top_k:            Maximum number of results to return.
        normalize_arabic: Whether to normalize Arabic text before encoding.
        unwanted_pos:     POS tags to exclude.
        use_fp16:         Use FP16 autocast on CUDA.
        batch_size:       Encoding batch size.

    Returns:
        DataFrame with columns: lex_CM, pos_CM, stemgloss_CM,
        readability (rounded average), similarity.
    """
    if unwanted_pos is None:
        unwanted_pos = DEFAULT_UNWANTED_POS

    empty = pd.DataFrame(
        columns=["lex_CM", "pos_CM", "stemgloss_CM", READABILITY_COL, "similarity"]
    )

    tmp = lexicon_df[lexicon_df[READABILITY_COL] == level].copy()
    if tmp.empty:
        return empty

    # Filter unwanted POS
    tmp = tmp[~tmp["pos_CM"].isin(unwanted_pos)].copy()
    if tmp.empty:
        return empty

    # Deduplicate by lemma, preferring higher lex_score if available
    if "lex_score" in tmp.columns:
        tmp = tmp.sort_values(["lex_CM", "lex_score"], ascending=[True, False])
    tmp = tmp.drop_duplicates(subset=["lex_CM"], keep="first")

    # Normalize text for embedding
    sentence_proc = normalize_ar(sentence) if normalize_arabic else sentence
    lemmas_proc   = (
        [normalize_ar(x) for x in tmp["lex_CM"].tolist()]
        if normalize_arabic else tmp["lex_CM"].tolist()
    )

    # Encode and compute similarity
    import torch
    from sentence_transformers import util

    s_emb  = _ST_MODEL.encode(
        sentence_proc, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    w_embs = _encode_in_batches(lemmas_proc, batch_size=batch_size, use_fp16=use_fp16)
    sims   = util.cos_sim(s_emb, w_embs)[0].detach().cpu().numpy()

    tmp = tmp.assign(similarity=sims)

    out_cols = [c for c in ["lex_CM", "pos_CM", "stemgloss_CM", READABILITY_COL, "similarity"] if c in tmp.columns]

    result = tmp[tmp["similarity"] >= threshold].sort_values("similarity", ascending=False)
    if top_k is not None:
        result = result.head(top_k)

    return result[out_cols]


def get_relevant_lemmas(
    sentence: str,
    lexicon_df: pd.DataFrame,
    level: int,
    threshold: float = 0.50,
    fallback_threshold: float = 0.35,
    top_k: int = 40,
    fallback_top_k: int = 20,
    normalize_arabic: bool = True,
    unwanted_pos: Optional[Set[str]] = None,
    use_fp16: bool = True,
    batch_size: int = 1024,
) -> List[tuple]:
    """Get relevant lemmas for a sentence, with fallback threshold if needed.

    Args:
        sentence:           Arabic sentence text.
        lexicon_df:         Pre-loaded SAMER lexicon DataFrame.
        level:              Target SAMER readability level (1-5).
        threshold:          Primary cosine similarity threshold.
        fallback_threshold: Threshold if primary returns no results.
        top_k:              Max results at primary threshold.
        fallback_top_k:     Max results at fallback threshold.
        normalize_arabic:   Whether to normalize Arabic before encoding.
        unwanted_pos:       POS tags to exclude.
        use_fp16:           Use FP16 autocast on CUDA.
        batch_size:         Encoding batch size.

    Returns:
        List of row tuples from the result DataFrame.
    """
    df = relevant_words_from_lexicon(
        sentence=sentence, lexicon_df=lexicon_df, level=level,
        threshold=threshold, top_k=top_k,
        normalize_arabic=normalize_arabic, unwanted_pos=unwanted_pos,
        use_fp16=use_fp16, batch_size=batch_size,
    )

    if df.empty:
        print(
            f"[relevant_words] No results at threshold={threshold} "
            f"for level {level}. Trying fallback={fallback_threshold}."
        )
        df = relevant_words_from_lexicon(
            sentence=sentence, lexicon_df=lexicon_df, level=level,
            threshold=fallback_threshold, top_k=fallback_top_k,
            normalize_arabic=normalize_arabic, unwanted_pos=unwanted_pos,
            use_fp16=use_fp16, batch_size=batch_size,
        )

    return list(df.itertuples(index=False, name=None))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Run relevant word extraction for all prompts."""
    # --- Load prompts ---
    prompts_df = read_any_delim(args.prompts_csv).fillna("")
    required   = {"prompt_id", "level", "Arabic Text"}
    missing    = required - set(prompts_df.columns)
    if missing:
        raise ValueError(
            f"Prompts CSV missing required columns: {missing}. "
            f"Found: {list(prompts_df.columns)}"
        )

    # --- Load lexicon ---
    lexicon_df = read_any_delim(args.lexicon_path).fillna("")
    required_lex = {"lex_CM", "pos_CM", READABILITY_COL}
    missing_lex  = required_lex - set(lexicon_df.columns)
    if missing_lex:
        raise ValueError(
            f"Lexicon TSV missing required columns: {missing_lex}. "
            f"Found: {list(lexicon_df.columns)}"
        )

    lexicon_df[READABILITY_COL] = pd.to_numeric(
        lexicon_df[READABILITY_COL], errors="coerce"
    )

    # --- Load model once ---
    _get_model(args.model_name, device=args.device)

    # --- Process each prompt ---
    out_rows = []
    total    = len(prompts_df)

    for i, (_, r) in enumerate(prompts_df.iterrows(), 1):
        prompt_id = r.get("prompt_id", "")
        sentence  = str(r.get("Arabic Text", ""))
        level     = int(float(str(r.get("level", 1)).strip()))

        print(f"[relevant_words] [{i}/{total}] prompt_id={prompt_id} level={level}")

        lemmas = get_relevant_lemmas(
            sentence=sentence,
            lexicon_df=lexicon_df,
            level=level,
            threshold=args.threshold,
            fallback_threshold=args.fallback_threshold,
            top_k=args.top_k,
            fallback_top_k=args.fallback_top_k,
            normalize_arabic=True,
            use_fp16=args.use_fp16,
            batch_size=args.batch_size,
        )

        print(f"[relevant_words]   -> {len(lemmas)} lemmas found.")

        out_rows.append({
            "prompt_id":       prompt_id,
            "Level":           level,
            "Arabic Text":     sentence,
            "Translated Text": r.get("Translated Text", ""),
            "Topic_ID":        r.get("Topic_ID", ""),
            "lemmas":          lemmas,
        })

    # --- Save ---
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[relevant_words] Saved {len(out_rows)} rows to: {args.out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract semantically relevant SAMER lexicon words for low-vocab prompts."
    )
    p.add_argument("--prompts_csv",  required=True, help="Input CSV with columns: prompt_id, level, Arabic Text.")
    p.add_argument("--lexicon_path", required=True, help="SAMER lexicon TSV with columns: lex_CM, pos_CM, stemgloss_CM, readability (rounded average).")
    p.add_argument("--out_csv",      required=True, help="Output CSV path.")
    p.add_argument("--threshold",          type=float, default=0.50,  help="Primary cosine similarity threshold (default: 0.50).")
    p.add_argument("--fallback_threshold", type=float, default=0.35,  help="Fallback threshold if primary returns no results (default: 0.35).")
    p.add_argument("--top_k",              type=int,   default=40,    help="Max results at primary threshold (default: 40).")
    p.add_argument("--fallback_top_k",     type=int,   default=20,    help="Max results at fallback threshold (default: 20).")
    p.add_argument("--model_name",  default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformer model name.")
    p.add_argument("--device",      default="auto", choices=["auto", "cuda", "cpu"], help="Device for embedding model (default: auto).")
    p.add_argument("--use_fp16",    action="store_true", default=True, help="Use FP16 autocast on CUDA (default: True).")
    p.add_argument("--batch_size",  type=int, default=1024, help="Encoding batch size (default: 1024).")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)