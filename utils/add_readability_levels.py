"""Predict BAREC readability levels for essays using a Hugging Face pipeline.

Loads a CSV with a 'd3tok' JSON-list column (produced by essays_d3tok.py),
runs the CAMeL-Lab/readability-arabertv2-d3tok-CE model sentence-by-sentence,
and appends three columns to the output CSV:
  - readability_levels  JSON list of per-sentence integer levels
  - max_level           maximum level across sentences (int or null)
  - avg_level           mean level across sentences (float or null)

Compatible with both ZAEBUC and ARWI datasets (both use Document_ID as the
essay identifier column).

Example:
  python utils/add_readability_levels.py \
    --input_csv <input_csv> \
    --output_csv <output_csv>
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from typing import Iterator, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eprint(*args) -> None:
    print(*args, file=sys.stderr)


def label_to_level(lbl: str) -> Optional[int]:
    """Extract a numeric level from labels like 'LABEL_0' or 'CE_18', then +1."""
    if not isinstance(lbl, str):
        return None
    m = re.search(r'(\d+)', lbl)
    return (int(m.group(1)) + 1) if m else None


def batched(iterable: list, batch_size: int) -> Iterator[list]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i: i + batch_size]


def parse_d3tok_cell(cell) -> List[str]:
    """Robustly parse a d3tok cell into a list of sentence strings.

    Handles:
      ["[ '…', '…' ]"]   JSON outer list wrapping a Python list literal
      "['…','…']"         Python list literal as a plain string
      ["a","b"]           proper JSON list of strings
      plain text          fallback: splitlines
    """
    if cell is None:
        return []
    text = str(cell).strip()
    if not text:
        return []

    # 1) Try JSON first
    try:
        val = json.loads(text)
        if isinstance(val, list) and len(val) == 1 and isinstance(val[0], str) and val[0].strip().startswith('['):
            try:
                inner = ast.literal_eval(val[0])
                if isinstance(inner, list):
                    return [str(s).strip() for s in inner if str(s).strip()]
            except Exception:
                pass
        if isinstance(val, list) and all(isinstance(x, str) for x in val):
            return [x.strip() for x in val if x.strip()]
        if isinstance(val, str):
            text = val
    except Exception:
        pass

    # 2) Try as Python list literal
    try:
        inner = ast.literal_eval(text)
        if isinstance(inner, list):
            return [str(s).strip() for s in inner if str(s).strip()]
    except Exception:
        pass

    # 3) Fallback: splitlines
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    input_csv: str,
    output_csv: str,
    d3tok_col: str = "d3tok",
    model_name: str = "CAMeL-Lab/readability-arabertv2-d3tok-CE",
    batch_size: int = 32,
    device: Optional[int] = None,
    local_only: bool = False,
) -> None:
    """Run readability prediction and write enriched CSV.

    Args:
        input_csv:    Path to input CSV. Must contain 'Document_ID' and d3tok_col.
        output_csv:   Path to write output CSV.
        d3tok_col:    Name of the column containing D3-tokenised sentence lists.
        model_name:   HuggingFace model identifier.
        batch_size:   Inference batch size.
        device:       Torch device index (-1 = CPU, 0 = cuda:0). Auto-detected if None.
        local_only:   Pass local_files_only=True to the HF pipeline.
    """
    try:
        import torch
        from transformers import pipeline
    except Exception:
        _eprint("[ERROR] Please install: pip install transformers torch")
        raise

    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)

    required_cols = {"Document_ID", d3tok_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Gather all sentences and record per-row boundaries
    all_sents: List[str] = []
    boundaries: List[Tuple[int, int]] = []

    for cell in df[d3tok_col]:
        sents = parse_d3tok_cell(cell)
        start = len(all_sents)
        all_sents.extend(sents)
        boundaries.append((start, len(all_sents)))

    if not all_sents:
        df["readability_levels"] = json.dumps([], ensure_ascii=False)
        df["max_level"] = None
        df["avg_level"] = None
        df.to_csv(output_csv, index=False)
        print(f"[INFO] No sentences found in column '{d3tok_col}'. Wrote: {output_csv}")
        return

    # Auto-detect device
    if device is None:
        try:
            device = 0 if torch.cuda.is_available() else -1
        except Exception:
            device = -1

    pipe_kwargs = dict(task="text-classification", model=model_name, device=device)
    if local_only:
        pipe_kwargs["local_files_only"] = True
    readability = pipeline(**pipe_kwargs)

    # Run inference
    levels_all: List[Optional[int]] = []
    for batch in batched(all_sents, batch_size):
        preds = readability(batch, truncation=True)
        for p in preds:
            levels_all.append(label_to_level(p.get("label")))

    # Map predictions back to rows
    levels_per_row_json: List[str] = []
    max_levels: List[Optional[int]] = []
    avg_levels: List[Optional[float]] = []
    for start, end in boundaries:
        lvls = [lvl for lvl in levels_all[start:end] if lvl is not None]
        levels_per_row_json.append(json.dumps(lvls, ensure_ascii=False))
        max_levels.append(max(lvls) if lvls else None)
        avg_levels.append(sum(lvls) / len(lvls) if lvls else None)

    df["readability_levels"] = levels_per_row_json
    df["max_level"] = max_levels
    df["avg_level"] = avg_levels

    df.to_csv(output_csv, index=False)
    print(f"[OK] Wrote {len(df)} rows to: {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Predict readability levels using a Hugging Face model."
    )
    ap.add_argument("--input_csv",   required=True,  help="Path to input CSV (must have Document_ID and d3tok columns).")
    ap.add_argument("--output_csv",  required=True,  help="Path to write output CSV.")
    ap.add_argument("--d3tok_col",   default="d3tok", help="Column containing D3-tokenised sentences (default: d3tok).")
    ap.add_argument("--model_name",  default="CAMeL-Lab/readability-arabertv2-d3tok-CE")
    ap.add_argument("--batch_size",  type=int, default=32)
    ap.add_argument("--device",      type=int, default=None, help="-1 for CPU, 0 for cuda:0. Auto-detected if omitted.")
    ap.add_argument("--local_only",  action="store_true")
    args = ap.parse_args()
    run(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        d3tok_col=args.d3tok_col,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        local_only=args.local_only,
    )
