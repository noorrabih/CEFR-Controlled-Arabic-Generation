"""Prepare essays CSV with optional D3 tokenization.

Reads a CSV of essays (one essay per row), runs D3 tokenization
via an external CLI script, and writes a CSV ready for BAREC readability
prediction.

Expected CSV columns:
  - Document_ID
  - Essay
  - CEFR

Output CSV columns:
  - Document_ID
  - CEFR
  - word_count
  - text
  - d3tok   (JSON list of tokenized sentences; empty list [] if skipped)

Example:
  python utils/essays_d3tok.py \
    --input_csv <input_csv> \
    --out_csv <out_csv> \
    --preprocess_script <path/to/d3tok_preprocess.py> \
    --morph_db <path/to/calima-msa-s31.db>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PREPROCESS_SCRIPT = "preprocessing/d3tok_preprocess.py"
DEFAULT_MORPH_DB           = "calima-msa-s31.db"


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENT_END_RE = re.compile(r"(?<=[\.\!\?؟؛…])\s+")


def split_sentences(text: str) -> List[str]:
    """Split Arabic/English text into sentences on punctuation boundaries.

    Args:
        text: Raw essay string.

    Returns:
        List of non-empty sentence strings. Returns the full text as a
        single-element list if no sentence boundaries are found.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    text = re.sub(r"[ \t]+", " ", text.strip())
    parts = _SENT_END_RE.split(text)
    sents = [p.strip() for p in parts if p.strip()]
    return sents if sents else [text.strip()]


# ---------------------------------------------------------------------------
# Word count
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    """Return whitespace-token count for a string.

    Args:
        text: Input string.

    Returns:
        Number of whitespace-separated tokens, or 0 for non-string input.
    """
    return len(str(text).split()) if isinstance(text, str) else 0


# ---------------------------------------------------------------------------
# D3 tokenization via external CLI
# ---------------------------------------------------------------------------

def add_d3tok_via_cli(
    df: pd.DataFrame,
    preprocess_script: str,
    db_path: Optional[str] = None,
    input_variant: str = "D3Tok",
) -> pd.DataFrame:
    """Run D3 tokenization over all essay sentences via an external CLI script.

    Flattens all sentences across all essays into a single input file,
    runs the CLI once, then maps outputs back to their source essays.

    Args:
        df:                 DataFrame with a ``text`` column.
        preprocess_script:  Absolute path to the D3Tok preprocess script.
        db_path:            Optional path to the CAMeL morphology DB.
        input_variant:      Input variant flag passed to the CLI (default: D3Tok).

    Returns:
        Copy of ``df`` with a ``d3tok`` column added. Each value is a
        JSON-encoded list of tokenized sentence strings.

    Raises:
        RuntimeError: If the external CLI exits with a non-zero return code.
    """
    perrow_sent_lists: List[List[str]] = []
    flat: List[str] = []

    for txt in df["text"].astype(str):
        sents = split_sentences(txt)
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
            # Sanitize: collapse any embedded newlines within a sentence into a
            # space so each sentence occupies exactly one line.
            sanitized = [s.replace("\r\n", " ").replace("\r", " ").replace("\n", " ") for s in flat]
            f.write("\n".join(sanitized) + "\n")

        cmd = [
            sys.executable, preprocess_script,
            "--input", in_path,
            "--input_var", input_variant,
            "--output", out_path,
        ]
        if db_path:
            cmd += ["--db", db_path]

        run_cwd = os.path.dirname(preprocess_script) or None
        res = subprocess.run(cmd, cwd=run_cwd, capture_output=True, text=True)

        if res.returncode != 0:
            raise RuntimeError(
                f"D3Tok preprocess script failed (exit code {res.returncode}).\n"
                f"Script: {preprocess_script}\n"
                f"STDOUT:\n{res.stdout}\n"
                f"STDERR:\n{res.stderr}"
            )

        with open(out_path, "r", encoding="utf-8") as f:
            out_lines = [ln.rstrip("\n") for ln in f]

    # Strip trailing blank lines emitted by some D3Tok versions
    while out_lines and out_lines[-1] == "":
        out_lines.pop()

    # Validate output line count matches input
    if len(out_lines) != len(flat):
        raise RuntimeError(
            f"D3Tok output line count ({len(out_lines)}) does not match "
            f"input sentence count ({len(flat)}). "
            "Check the preprocess script output."
        )

    # Map tokenized lines back to their source essays
    idx = 0
    d3tok_col: List[str] = []
    for sents in perrow_sent_lists:
        n = len(sents)
        chunk = out_lines[idx: idx + n] if n > 0 else []
        idx += n
        d3tok_col.append(json.dumps(chunk, ensure_ascii=False))

    out = df.copy()
    out["d3tok"] = d3tok_col
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_csv_from_csv(
    csv_path: str,
    output_csv: str,
    preprocess_script: Optional[str] = DEFAULT_PREPROCESS_SCRIPT,
    morph_db: Optional[str] = DEFAULT_MORPH_DB,
) -> None:
    """Build the output CSV from a CSV of essays with optional D3 tokenization.

    Args:
        csv_path:           Path to input CSV (Document_ID, Essay, CEFR).
        output_csv:         Path to write output CSV.
        preprocess_script:  Optional absolute path to D3Tok preprocess script.
                            If None or not found, d3tok column will be empty [].
        morph_db:           Optional path to CAMeL morphology DB.

    Raises:
        ValueError: If required columns are missing from the input CSV.
    """
    df = pd.read_csv(csv_path, sep=",", dtype=str)

    required_cols = {"Document_ID", "Essay", "CEFR"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "Document_ID": df["Document_ID"].fillna(""),
        "CEFR":        df["CEFR"].fillna(""),
        "text":        df["Essay"].fillna(""),
    })

    out["word_count"] = out["text"].apply(word_count)

    # --- D3 tokenization (optional) ---
    if preprocess_script is not None:
        script_path = Path(preprocess_script).resolve()
        if not script_path.exists():
            print(
                f"[WARNING] Preprocess script not found: {script_path}. "
                "Skipping D3Tok — d3tok column will be empty."
            )
            out["d3tok"] = [json.dumps([], ensure_ascii=False)] * len(out)
        else:
            db_path = str(Path(morph_db).resolve()) if morph_db else None
            out = add_d3tok_via_cli(
                out,
                preprocess_script=str(script_path),
                db_path=db_path,
                input_variant="D3Tok",
            )
    else:
        print("[INFO] No preprocess script provided. Skipping D3Tok.")
        out["d3tok"] = [json.dumps([], ensure_ascii=False)] * len(out)

    out = out[["Document_ID", "CEFR", "word_count", "text", "d3tok"]]
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[essays_d3tok] Wrote {len(out)} rows to: {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare essays CSV with optional D3 tokenization."
    )
    p.add_argument(
        "--input_csv",
        required=True,
        help="Path to input CSV (required columns: Document_ID, Essay, CEFR).",
    )
    p.add_argument(
        "--out_csv",
        required=True,
        help="Path to write output CSV.",
    )
    p.add_argument(
        "--preprocess_script",
        default=DEFAULT_PREPROCESS_SCRIPT,
        help="Path to D3Tok preprocess script (default: DEFAULT_PREPROCESS_SCRIPT).",
    )
    p.add_argument(
        "--morph_db",
        default=DEFAULT_MORPH_DB,
        help="Path to CAMeL morphology DB (default: DEFAULT_MORPH_DB).",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    build_csv_from_csv(
        csv_path=args.input_csv,
        output_csv=args.out_csv,
        preprocess_script=args.preprocess_script,
        morph_db=args.morph_db,
    )