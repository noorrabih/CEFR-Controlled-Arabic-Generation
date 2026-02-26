# Script to read a TSV file with documents and their corrected text,
# split the corrected text into sentences, and output a CSV file with each sentence

import pandas as pd
import re
from pathlib import Path

in_path = Path("/home/nour.rabih/arwi/readability_controlled_generation/generation/syntax_vocab_prompt/5levels/generated_essays_5levels.tsv")
out_path = Path("/home/nour.rabih/arwi/readability_controlled_generation/generation/syntax_vocab_prompt/5levels/generated_essays_sentences.csv")

df = pd.read_csv(in_path, sep="\t", dtype=str, keep_default_na=False)

id_col = "Document_ID"  # column with document IDs
essay_col = "essay"     # column with corrected essay text

if id_col not in df.columns:
    id_col = "ID"

if essay_col not in df.columns:
    essay_col = "text"

# Ensure columns
for col in [id_col, essay_col]:
    if col not in df.columns:
        raise RuntimeError(f"Missing required column: {col}")

# Sentence splitter
sent_end_pattern = re.compile(r"(?<=[\.\!\?؟؛…])\s+")
def split_sentences(t: str):
    if not isinstance(t, str):
        return []
    t = t.strip()
    if not t:
        return []
    t = re.sub(r"[ \t]+", " ", t)
    parts = sent_end_pattern.split(t)
    sents = [p.strip() for p in parts if p and p.strip()]
    if not sents:
        sents = [t]
    return sents

# Base extractor: choose the LONGEST numeric group; if tie, take the last occurrence
def extract_base(doc: str):
    nums = re.findall(r"\d+", str(doc))
    if not nums:
        return str(doc).strip()
    maxlen = max(len(x) for x in nums)
    candidates = [x for x in nums if len(x) == maxlen]
    return candidates[-1]

rows = []
for _, r in df.iterrows():
    base = r[id_col]  # e.g., 268469
    anon_id = base
    sents = split_sentences(r[essay_col])
    for n, s in enumerate(sents, start=1):
        rows.append({"ID": f"{base}-{n}", "anon_id": anon_id, "Sentence": s})

out_df = pd.DataFrame(rows, columns=["ID", "anon_id", "Sentence"])
out_df.to_csv(out_path, index=False)

print(f"Wrote: {out_path}")
