"""Split essays into sentences.

Reads a TSV/CSV file with document IDs and essay text, and produces a CSV where each row
is a sentence with columns: ID, Document_ID, Sentence.

Example:
  python utils/split_sentences.py \
    --input_path generated_essays.tsv --sep '\t' \
    --id_col Document_ID --text_col essay \
    --out_csv generated_essays_sentences.csv
"""

from __future__ import annotations

import argparse
import re
import pandas as pd


sent_end_pattern = re.compile(r"(?<=[\.\!\?؟؛…])\s+")


def split_sentences(text: str):
    if not isinstance(text, str):
        return []
    t = text.strip()
    if not t:
        return []
    t = re.sub(r"[ \t]+", " ", t)
    parts = sent_end_pattern.split(t)
    sents = [p.strip() for p in parts if p and p.strip()]
    return sents if sents else [t]


def main(args):
    df = pd.read_csv(args.input_path, sep=args.sep, dtype=str, keep_default_na=False)

    id_col = args.id_col
    text_col = args.text_col

    if id_col not in df.columns:
        raise ValueError(f"Missing id_col '{id_col}'. Found: {list(df.columns)}")
    if text_col not in df.columns:
        raise ValueError(f"Missing text_col '{text_col}'. Found: {list(df.columns)}")

    rows = []
    for _, r in df.iterrows():
        base = str(r[id_col]).strip()
        anon_id = base
        sents = split_sentences(r[text_col])
        for n, s in enumerate(sents, start=1):
            rows.append({"ID": f"{base}-{n}", "Document_ID": anon_id, "Sentence": s})

    out_df = pd.DataFrame(rows, columns=["ID", "Document_ID", "Sentence"])
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Split essays into sentences.")
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--sep", default="\t", help="Input separator (default: tab).")
    ap.add_argument("--id_col", default="Document_ID")
    ap.add_argument("--text_col", default="essay")
    ap.add_argument("--out_csv", required=True)
    main(ap.parse_args())
