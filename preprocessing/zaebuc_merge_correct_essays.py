"""Merge ZAEBUC corrected essays into one row per essay.

ZAEBUC corrected essays are provided one word per row. This script:
1) Reads the alignment TSV containing columns: Document, Corrected
2) Joins Corrected tokens into a single essay per Document
3) Optionally extracts CEFR + word_count from an analyzed TSV that contains <doc ...> XML lines
4) Writes a CSV with columns: Document_ID, Essay, (optional) CEFR, word_count

Example:
  python preprocessing/zaebuc_merge_correct_essays.py \
    --alignment_tsv ZAEBUC/AR-all.alignment-FINAL.tsv \
    --analyzed_tsv ZAEBUC/AR-all.extracted.corrected.analyzed.corrected-FINAL.tsv \
    --out_csv ZAEBUC/corrected_full_texts.csv
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple, List

import pandas as pd


def _extract_doc_meta_from_xml_lines(xml_lines: List[str]) -> Tuple[List[str], List[str]]:
    """Extract CEFR and word_count from ZAEBUC <doc ...> XML lines.

    ZAEBUC analyzed TSV sometimes contains a column where some rows are XML strings like:
      <doc CEFR="A2" word_count="123"> ...

    Args:
        xml_lines: List of raw XML strings from the Document column.

    Returns:
        Tuple of (grades, word_counts), each a list of strings aligned to
        document order. Malformed or missing entries are skipped silently.
    """
    import xmltodict

    grades: List[str] = []
    word_counts: List[str] = []

    for xml in xml_lines:
        if not isinstance(xml, str):
            continue
        if xml.strip() == "</doc>":
            continue
        if not xml.strip().startswith("<"):
            continue

        try:
            doc = xmltodict.parse(xml)
            grades.append(doc["doc"].get("@CEFR", ""))
            word_counts.append(doc["doc"].get("@word_count", ""))
        except Exception:
            continue

    return grades, word_counts


def merge_corrected(
    alignment_tsv: str,
    out_csv: str,
    analyzed_tsv: Optional[str] = None,
    corrected_col: str = "Corrected",
    document_col: str = "Document",
    encoding: str = "utf-8",
) -> None:
    """Merge one-token-per-row ZAEBUC essays into one-essay-per-row CSV.

    Args:
        alignment_tsv:  Path to ZAEBUC alignment TSV (Document + Corrected columns).
        out_csv:        Output CSV path.
        analyzed_tsv:   Optional analyzed TSV containing <doc ...> XML lines
                        for CEFR / word_count metadata.
        corrected_col:  Name of the corrected-token column.
        document_col:   Name of the document ID column.
        encoding:       File encoding for reading/writing.

    Raises:
        ValueError: If required columns are missing from input files.
    """
    df = pd.read_csv(alignment_tsv, sep="\t", dtype=str, encoding=encoding)

    for col in (document_col, corrected_col):
        if col not in df.columns:
            raise ValueError(
                f"alignment_tsv must contain column '{col}'. "
                f"Found: {list(df.columns)}"
            )

    # --- Merge tokens into full essays (pandas 2.x safe) ---
    merged = (
        df.groupby(document_col, sort=False)[corrected_col]
        .apply(lambda x: " ".join(x.dropna()))
        .reset_index()
        .rename(columns={document_col: "Document_ID", corrected_col: "Essay"})
    )

    n_essays = len(merged)
    if n_essays == 0:
        raise ValueError(
            "Merge produced 0 essays — check that alignment_tsv is not empty "
            "and that document_col / corrected_col are correct."
        )
    print(f"[merge_corrected] Merged {n_essays} essays from '{alignment_tsv}'.")

    # --- Optional: attach CEFR / word_count metadata ---
    if analyzed_tsv:
        ana = pd.read_csv(analyzed_tsv, sep="\t", dtype=str, encoding=encoding)

        if document_col not in ana.columns:
            raise ValueError(
                f"analyzed_tsv must contain a '{document_col}' column. "
                f"Found: {list(ana.columns)}"
            )

        xml_lines = (
            ana[document_col]
            .apply(lambda x: x if isinstance(x, str) and x.startswith("<") else None)
            .dropna()
            .tolist()
        )

        grades, word_counts = _extract_doc_meta_from_xml_lines(xml_lines)

        if grades:
            if len(grades) != n_essays:
                print(
                    f"[WARNING] Extracted {len(grades)} CEFR labels but found "
                    f"{n_essays} essays — metadata may be misaligned. "
                    "Truncating to essay count."
                )
            merged["CEFR"] = grades[:n_essays]

        if word_counts:
            if len(word_counts) != n_essays:
                print(
                    f"[WARNING] Extracted {len(word_counts)} word_count values but found "
                    f"{n_essays} essays — metadata may be misaligned. "
                    "Truncating to essay count."
                )
            merged["word_count"] = word_counts[:n_essays]

    merged.to_csv(out_csv, sep=",", index=False, encoding=encoding)
    print(f"[merge_corrected] Saved: {out_csv}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge ZAEBUC corrected essays (one token per row) into one row per essay."
    )
    p.add_argument(
        "--alignment_tsv",
        required=True,
        help="Path to ZAEBUC alignment TSV (must include Document and Corrected columns).",
    )
    p.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV path.",
    )
    p.add_argument(
        "--analyzed_tsv",
        default=None,
        help="Optional analyzed TSV containing <doc ...> XML lines for CEFR/word_count.",
    )
    p.add_argument(
        "--corrected_col",
        default="Corrected",
        help="Name of corrected-token column in alignment TSV (default: Corrected).",
    )
    p.add_argument(
        "--document_col",
        default="Document",
        help="Name of document ID column (default: Document).",
    )
    p.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8).",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    merge_corrected(
        alignment_tsv=args.alignment_tsv,
        out_csv=args.out_csv,
        analyzed_tsv=args.analyzed_tsv,
        corrected_col=args.corrected_col,
        document_col=args.document_col,
        encoding=args.encoding,
    )