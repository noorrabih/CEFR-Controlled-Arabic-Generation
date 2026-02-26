# split the corrected text into sentences,
# returns a csv with essays and their corresponding d3tok, CEFR level and word count.
from __future__ import annotations
import re, sys, json, os, tempfile, subprocess
from pathlib import Path
from typing import List, Optional
import pandas as pd

# # open /mnt/data/users/nour.rabih/arwi/arabic-aes-bea25/Data/arwi_cefr_levels.csv 
# levels = pd.read_csv("/l/users/nour.rabih/arabic-aes-bea25/Data/arwi_cefr_levels.csv", dtype=str)
# print(levels.columns)
# # open /mnt/data/users/nour.rabih/arwi/arabic-aes-bea25/Data/arwi_original_essays.csv
# essays = pd.read_csv("/l/users/nour.rabih/arabic-aes-bea25/Data/arwi_original_essays.csv", dtype=str)
# print(essays.columns)
# # merge on Document_ID
# merged = essays.merge(levels, on="Document_ID", how="left")
# merged = merged.rename(columns={"CEFR":"Grade","Original_Essay":"Corrected"})


DOC_ID_RE = re.compile(r"^AR-\d{3}-(\d+)$")

def extract_anon_id(doc: str) -> str:
    """'AR-030-268469' -> '268469'; fallback to raw string if no match."""
    if not isinstance(doc, str):
        return ""
    m = DOC_ID_RE.match(doc.strip())
    return m.group(1) if m else doc.strip()

def word_count(text: str) -> int:
    return len(str(text).split()) if isinstance(text, str) else 0

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

def simple_sentence_split(text: str) -> List[str]:
    """
    Lightweight sentence splitter for Arabic/English punctuation.
    Produces one sentence per line for the external CLI (if used).
    """
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r"(?<=[\.\!\?؟…])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def add_d3tok_via_cli(
    df: pd.DataFrame,
    preprocess_script: str,
    db_path: Optional[str] = None,
    input_variant: str = "D3Tok",
) -> pd.DataFrame:
    """
    Runs your CLI once over all sentences, maps outputs back row-wise.
    If you don’t want this, just omit --preprocess_script and we’ll skip it.
    """
    perrow_sent_lists: List[List[str]] = []
    flat: List[str] = []

    for txt in df["text"].astype(str):
        sents = simple_sentence_split(txt)
        perrow_sent_lists.append(sents)
        flat.extend(sents)

    if not flat:
        out = df.copy()
        out["d3tok"] = [json.dumps([], ensure_ascii=False)] * len(out)
        return out

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path  = os.path.join(tmpdir, "all_input.txt")
        out_path = os.path.join(tmpdir, "all_output.txt")
        with open(in_path, "w", encoding="utf-8") as f:
            f.write("\n".join(flat) + "\n")

        cmd = [sys.executable, preprocess_script,
               "--input", in_path,
               "--input_var", input_variant,
               "--output", out_path]
        if db_path:
            cmd += ["--db", db_path]

        run_cwd = os.path.dirname(preprocess_script) or None
        res = subprocess.run(cmd, cwd=run_cwd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"preprocess.py failed (code {res.returncode}).\n"
                f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

        with open(out_path, "r", encoding="utf-8") as f:
            out_lines = [ln.rstrip("\n") for ln in f]

    # map back
    idx = 0
    d3tok_lists: List[str] = []
    for sents in perrow_sent_lists:
        n = len(sents)
        chunk = out_lines[idx: idx + n] if n > 0 else []
        idx += n
        d3tok_lists.append(json.dumps(chunk, ensure_ascii=False))

    out = df.copy()
    out["d3tok"] = d3tok_lists
    return out

def build_csv_from_tsv(
    tsv_path, # 
    output_csv: str,
    preprocess_script: Optional[str] = None,
    morph_db: Optional[str] = None,
):
    """
    Your TSV columns (as seen): 
      - Document  (e.g., AR-030-268469)
      - Corrected (essay text)
      - Grade     (CEFR label)
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    # rows = []
    # for _, r in df.iterrows():
    #     sents = split_sentences(r["essay"])
    #     ID = r["Document_ID"]
    #     grade = r["Grade"]
    #     for n, s in enumerate(sents, start=1):
    #         rows.append({"Document_ID": f"{ID}-{n}" , "Grade": grade, "Sentence": s})

    # df = pd.DataFrame(rows, columns=["Document_ID", "Grade", "Sentence"])

    out = pd.DataFrame({
        "ID": df["Document_ID"].apply(extract_anon_id),
        "CEFR": df["Grade"].fillna(""),
        "text": df["essay"].fillna(""), #Sentence
    })
    out["word_count"] = out["text"].apply(word_count)

    # Optional D3Tok via your CLI
    if preprocess_script:
        out = add_d3tok_via_cli(out,
                                preprocess_script=preprocess_script,
                                db_path=morph_db,
                                input_variant="D3Tok")
    else:
        out["d3tok"] = [json.dumps([], ensure_ascii=False)] * len(out)

    out = out[["ID", "CEFR", "word_count", "text", "d3tok"]]
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Wrote: {output_csv}")

if __name__ == "__main__":
    # Usage:
    # python clean_essay_process.py <full_texts.tsv> <output_csv> [<preprocess_script>] [<morph_db>]
    if len(sys.argv) < 3:
        print("Usage: python clean_essays_process.py <full_texts.tsv> <output_csv> [<preprocess_script>] [<morph_db>]")
        sys.exit(1)
    TSV = sys.argv[1] # full_texts.tsv merged
    OUT = sys.argv[2]
    PREP = sys.argv[3] if len(sys.argv) >= 4 else "/home/nour.rabih/arwi/barec_analyzer/scripts/preprocess.py"
    DB   = sys.argv[4] if len(sys.argv) >= 5 else "/home/nour.rabih/arwi/calima-msa-s31.db"
    build_csv_from_tsv(TSV, OUT, preprocess_script=PREP, morph_db=DB)

