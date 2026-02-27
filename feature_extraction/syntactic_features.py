"""Extract dependency/POS/dep(POS) counts from CoNLL-X files and aggregate by essay/CEFR.

Reads parsed .conllx files (recursively), extracts per-sentence syntactic counts,
joins CEFR labels from a readability CSV, and writes essay-level and level-level
aggregated statistics.

Input:
  --parsed_dir      Directory containing parsed .conllx files (searched recursively).
  --readability_csv CSV with columns: Document_ID, CEFR.
  --out_dir         Output directory for CSVs.

Output CSVs:
  sentences_with_dep_pos_counts.csv       per-sentence syntactic counts
  sentences_with_dep_pos_counts_cefr.csv  same + CEFR column
  essay_syntax_stats.csv                  per-essay aggregated counts + per-token ratios
  level_syntax_stats.csv                  per-CEFR-level aggregated statistics

Example:
  python feature_extraction/syntactic_features.py \
    --parsed_dir <parsed_dir> \
    --readability_csv <readability_csv> \
    --out_dir <out_dir>
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Matches "# id = <sentence_id>" comment lines in CoNLL-X files
ID_RE = re.compile(r"^#\s*id\s*=\s*(\S+)\s*$")

# Extracts UD POS from FEATS column values like "ud=NOUN|pos=noun|..."
UD_RE = re.compile(r"(?:^|[|;])ud=([^|;]+)")

# Matches CEFR level embedded in essay IDs like "1_A1_1"
CEFR_IN_ID_RE = re.compile(r"_(A1|A2|B1|B2|C1|C2)_")


# ---------------------------------------------------------------------------
# CoNLL-X parsing
# ---------------------------------------------------------------------------

def iter_sentences_with_ids(path: str) -> Iterator[Tuple[str, List[List[str]]]]:
    """Yield (essay_id, sent_rows) for each sentence block in a CoNLL-X file.

    Sentence blocks are separated by blank lines. Essay IDs are read from
    '# id = ...' comment lines. If no ID comment is found, the filename
    (without extension) is used as a fallback.

    Args:
        path: Path to a .conllx file.

    Yields:
        Tuples of (essay_id, list_of_token_rows) where each token row is a
        list of CoNLL-X field strings.
    """
    current_essay_id: Optional[str] = None
    sent: List[List[str]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            m = ID_RE.match(line.strip())
            if m:
                current_essay_id = m.group(1)
                continue

            if not line.strip():
                if sent:
                    if current_essay_id is None:
                        current_essay_id = os.path.splitext(os.path.basename(path))[0]
                    yield current_essay_id, sent
                    sent = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) >= 8:
                sent.append(parts)

        # Yield final sentence if file does not end with a blank line
        if sent:
            if current_essay_id is None:
                current_essay_id = os.path.splitext(os.path.basename(path))[0]
            yield current_essay_id, sent


def get_ud_pos(feats: str, fallback: Optional[str] = None) -> Optional[str]:
    """Extract UD POS tag from a CoNLL-X FEATS column value.

    Args:
        feats:    FEATS column string (e.g. 'ud=NOUN|pos=noun|...').
        fallback: Value to return if no UD POS is found.

    Returns:
        UD POS string, or ``fallback`` if not found.
    """
    if feats is None:
        return fallback
    m = UD_RE.search(str(feats))
    return m.group(1) if m else fallback


def extract_level_from_essay_id(essay_id: str) -> Optional[str]:
    """Attempt to extract a CEFR level embedded in an essay ID string.

    Handles patterns like '1_A1_1' or 'topic_B2_essay'.
    Used only as a last-resort fallback when CEFR cannot be joined from a CSV.

    Args:
        essay_id: Essay ID string.

    Returns:
        CEFR level string (e.g. 'A1'), or None if not found.
    """
    m = CEFR_IN_ID_RE.search(str(essay_id))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# CEFR joining
# ---------------------------------------------------------------------------

def add_cefr_column(sent_df: pd.DataFrame, readability_csv: str) -> pd.DataFrame:
    """Attach a CEFR column to a sentence-level DataFrame.

    Joins from a readability CSV on Document_ID. If no match is found on the
    full essay_id, falls back to matching on the last dash-separated segment
    (e.g. 'AR-030-268469' → '268469'). If still unmatched, attempts to parse
    the CEFR level directly from the essay_id string.

    Args:
        sent_df:         Sentence-level DataFrame with an ``essay_id`` column.
        readability_csv: Path to CSV with ``Document_ID`` and ``CEFR`` columns.

    Returns:
        Copy of ``sent_df`` with ``CEFR`` and ``base_id`` columns added.

    Raises:
        ValueError: If required columns are missing from the readability CSV.
    """
    cefr_df = pd.read_csv(readability_csv, dtype=str)

    required = {"Document_ID", "CEFR"}
    missing  = required - set(cefr_df.columns)
    if missing:
        raise ValueError(
            f"Readability CSV must contain columns {required}. "
            f"Missing: {missing}. Found: {list(cefr_df.columns)}"
        )

    cefr_map: Dict[str, str] = dict(
        zip(cefr_df["Document_ID"].astype(str), cefr_df["CEFR"].astype(str))
    )

    out = sent_df.copy()
    out["essay_id"] = out["essay_id"].astype(str)

    # Attempt 1: direct match on full essay_id
    out["CEFR"] = out["essay_id"].map(cefr_map)

    # Attempt 2: strip trailing sentence-number suffix
    # e.g. "AR-030-268469-1" → "AR-030-268469"
    still_missing = out["CEFR"].isna()
    if still_missing.any():
        stripped = out["essay_id"].str.rsplit("-", n=1).str[0]
        out.loc[still_missing, "CEFR"] = stripped[still_missing].map(cefr_map)

    out["base_id"] = out["essay_id"].str.rsplit("-", n=1).str[0].where(
        out["CEFR"].notna(), other=out["essay_id"]
    )

    # Attempt 3: match on last dash-separated segment
    # e.g. "AR-030-268469" → "268469"
    if out["CEFR"].isna().all():
        print(
            "[WARNING] No CEFR matches on full essay_id. "
            "Retrying with last segment (e.g. 'AR-030-268469' → '268469')."
        )
        out["base_id"] = out["essay_id"].str.rsplit("-", n=1).str[-1]
        out["CEFR"]    = out["base_id"].map(cefr_map)

    # Attempt 3: parse CEFR from essay_id string
    still_missing = out["CEFR"].isna()
    if still_missing.any():
        print(
            f"[WARNING] {still_missing.sum()} rows still missing CEFR. "
            "Attempting to parse from essay_id string."
        )
        out.loc[still_missing, "CEFR"] = (
            out.loc[still_missing, "essay_id"].apply(extract_level_from_essay_id)
        )

    if out["CEFR"].isna().all():
        print(
            "[WARNING] CEFR column is entirely empty after all fallbacks. "
            "Check that Document_ID values in the readability CSV match essay IDs."
        )

    return out


# ---------------------------------------------------------------------------
# Sentence-level feature extraction
# ---------------------------------------------------------------------------

def build_sentence_counts(parsed_dir: str) -> pd.DataFrame:
    """Extract per-sentence dependency, POS, and dep(POS) counts from CoNLL-X files.

    Walks ``parsed_dir`` recursively, parses all .conllx files, and returns
    one row per sentence with count columns for each dependency relation,
    POS tag, and dep(POS) combination seen across the corpus.

    Args:
        parsed_dir: Root directory to search for .conllx files.

    Returns:
        DataFrame with columns: essay_id, sentence, dep_*, pos_*, depPOS_*.

    Raises:
        RuntimeError: If no sentences could be parsed from ``parsed_dir``.
    """
    rows: List[dict] = []
    all_dep:     set = set()
    all_pos:     set = set()
    all_dep_pos: set = set()

    for root, _, files in os.walk(parsed_dir):
        for fn in sorted(files):
            if fn.startswith(".") or not fn.endswith(".conllx"):
                continue
            path = os.path.join(root, fn)

            try:
                for essay_id, sent in iter_sentences_with_ids(path):
                    tokens   = [tok[1] for tok in sent]   # FORM
                    deprels  = [tok[7] for tok in sent]   # DEPREL
                    feats    = [tok[5] for tok in sent]   # FEATS

                    # Prefer UD POS from FEATS; fall back to POSTAG column (index 3)
                    pos_ud = [
                        get_ud_pos(f, fallback=tok[3])
                        for f, tok in zip(feats, sent)
                    ]

                    dep_counts     = Counter(deprels)
                    pos_counts     = Counter(pos_ud)
                    dep_pos_labels = [f"{r}({p})" for r, p in zip(deprels, pos_ud)]
                    dep_pos_counts = Counter(dep_pos_labels)

                    all_dep.update(dep_counts.keys())
                    all_pos.update(pos_counts.keys())
                    all_dep_pos.update(dep_pos_counts.keys())

                    row: dict = {
                        "essay_id": essay_id,
                        "sentence": " ".join(tokens),
                    }
                    for rel, c in dep_counts.items():
                        row[f"dep_{rel}"] = c
                    for p, c in pos_counts.items():
                        row[f"pos_{p}"] = c
                    for lab, c in dep_pos_counts.items():
                        row[f"depPOS_{lab.replace(' ', '_')}"] = c

                    rows.append(row)

            except Exception as e:
                print(f"[WARNING] Skipping file due to error: {path}\n  -> {e}")

    if not rows:
        raise RuntimeError(
            f"No sentences were parsed. Check --parsed_dir: {parsed_dir}"
        )

    df = pd.DataFrame(rows)

    # Ensure all observed feature columns exist (fill missing with 0)
    dep_cols    = [f"dep_{lab}"    for lab in sorted(all_dep)]
    pos_cols    = [f"pos_{lab}"    for lab in sorted(all_pos)]
    depPOS_cols = [
        f"depPOS_{lab.replace(' ', '_')}" for lab in sorted(all_dep_pos)
    ]

    for col in dep_cols + pos_cols + depPOS_cols:
        if col not in df.columns:
            df[col] = 0

    base_cols = ["essay_id", "sentence"]
    df = df[base_cols + dep_cols + pos_cols + depPOS_cols].fillna(0)

    for col in dep_cols + pos_cols + depPOS_cols:
        df[col] = df[col].astype(int)

    return df


# ---------------------------------------------------------------------------
# Essay and level aggregation
# ---------------------------------------------------------------------------

def build_essay_and_level_stats(
    df: pd.DataFrame,
    out_dir: str,
) -> Tuple[str, str]:
    """Aggregate sentence-level counts to essay-level and CEFR-level statistics.

    Per-essay: sums all count columns, computes per-token ratios.
    Per-level: mean, median, std, count of per-token ratios; sum and mean of raw counts.

    Args:
        df:      Sentence-level DataFrame with CEFR and count columns.
        out_dir: Directory to write output CSVs.

    Returns:
        Tuple of (essay_csv_path, level_csv_path).
    """
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()

    # Use base_id (essay-level, suffix stripped) if available, else essay_id
    id_col = "base_id" if "base_id" in df.columns else "essay_id"
    count_cols = [
        c for c in df.columns if c.startswith(("dep_", "pos_", "depPOS_"))
    ]

    df["n_tokens"] = (
        df["sentence"].fillna("").astype(str).apply(lambda s: len(s.split()))
    )

    level_col = "CEFR" if "CEFR" in df.columns else None
    if level_col is None:
        print(
            "[WARNING] No CEFR column found. "
            "Attempting to extract level from essay_id."
        )
        df["CEFR"] = df["essay_id"].apply(extract_level_from_essay_id)
        level_col  = "CEFR"

    # --- Essay-level aggregation (sum counts across sentences) ---
    essay = df.groupby(
        [id_col, level_col], as_index=False
    )[count_cols + ["n_tokens"]].sum()
    essay = essay.rename(columns={id_col: "Document_ID"})

    # Per-token ratios
    for c in count_cols:
        essay[f"{c}_per_token"] = (
            essay[c] / essay["n_tokens"].where(essay["n_tokens"] > 0, 1)
        )

    out_essay = os.path.join(out_dir, "essay_syntax_stats.csv")
    essay.to_csv(out_essay, index=False, encoding="utf-8")

    # --- Level-level aggregation ---
    ratio_cols = [f"{c}_per_token" for c in count_cols]

    ratio_stats = (
        essay.groupby(level_col)[ratio_cols]
        .agg(["mean", "median", "std", "count"])
    )
    ratio_stats.columns = [f"{col}_{stat}" for col, stat in ratio_stats.columns]
    ratio_stats = ratio_stats.reset_index()

    sum_stats  = essay.groupby(level_col)[count_cols].sum().add_suffix("_sum")
    mean_stats = essay.groupby(level_col)[count_cols].mean().add_suffix("_mean")

    level_stats = ratio_stats.merge(
        pd.concat([sum_stats, mean_stats], axis=1).reset_index(),
        on=level_col,
        how="left",
    )

    out_level = os.path.join(out_dir, "level_syntax_stats.csv")
    level_stats.to_csv(out_level, index=False, encoding="utf-8")

    return out_essay, out_level


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extract dependency/POS/dep(POS) counts from .conllx files "
            "and aggregate by essay and CEFR level."
        )
    )
    ap.add_argument(
        "--parsed_dir",
        required=True,
        help="Directory containing parsed .conllx files (searched recursively).",
    )
    ap.add_argument(
        "--readability_csv",
        required=True,
        help="CSV with columns: Document_ID, CEFR.",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for CSVs.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: sentence-level counts
    sent_df  = build_sentence_counts(args.parsed_dir)
    out_sent = os.path.join(args.out_dir, "sentences_with_dep_pos_counts.csv")
    sent_df.to_csv(out_sent, index=False, encoding="utf-8")
    print(f"[OK] Saved {len(sent_df)} sentences to: {out_sent}")

    # Step 2: attach CEFR
    sent_cefr_df  = add_cefr_column(sent_df, args.readability_csv)
    out_sent_cefr = os.path.join(
        args.out_dir, "sentences_with_dep_pos_counts_cefr.csv"
    )
    sent_cefr_df.to_csv(out_sent_cefr, index=False, encoding="utf-8")
    print(f"[OK] Saved with CEFR to: {out_sent_cefr}")

    # Step 3: essay + level stats
    out_essay, out_level = build_essay_and_level_stats(sent_cefr_df, args.out_dir)
    print(f"[OK] Saved essay stats: {out_essay}")
    print(f"[OK] Saved level stats: {out_level}")
    print(
        f"[OK] CEFR levels found: "
        f"{sorted(sent_cefr_df['CEFR'].dropna().astype(str).unique())}"
    )


if __name__ == "__main__":
    main()