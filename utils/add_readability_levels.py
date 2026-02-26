# Create a ready-to-run script that:
# - loads your CSV with a 'd3tok' JSON-list column
# - runs the HF pipeline CAMeL-Lab/readability-arabertv2-d3tok-CE
# - adds three new columns: 'readability_levels' (JSON list) and 'max_level' (int) and avg_level (float)
# - writes an updated CSV

import os
import re
import json
import sys
import math
import ast
import pandas as pd
from typing import Optional

def _eprint(*a):
    print(*a, file=sys.stderr)

# def label_to_level(lbl: str) -> int | None:
#     """Extract a numeric level from labels like 'LABEL_0', 'CE_18', etc., then +1."""
#     if not isinstance(lbl, str):
#         return None
#     m = re.search(r'(\d+)', lbl)
#     return (int(m.group(1)) + 1) if m else None

def label_to_level(lbl):
    if not isinstance(lbl, str):
        return None
    m = re.search(r'(\d+)', lbl)
    return (int(m.group(1)) + 1) if m else None


def batched(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]

def parse_d3tok_cell(cell):
    """Robustly parse your d3tok cells:

    Handles forms like:
      ["[ '…', '…' ]"]          (JSON outer list with inner Python list literal)
      "['…','…']"                (Python list literal as a string)
      ["a","b"]                  (proper JSON list of strings)
      plain text / multi-lines   (fallback splitlines)
    Returns: list[str]
    """
    if cell is None:
        return []
    text = str(cell).strip()
    if not text:
        return []

    # 1) Try JSON first
    try:
        val = json.loads(text)
        # Case: ["[ '…', '…' ]"] -> inner Python list literal
        if isinstance(val, list) and len(val) == 1 and isinstance(val[0], str) and val[0].strip().startswith('['):
            try:
                inner = ast.literal_eval(val[0])
                if isinstance(inner, list):
                    return [str(s).strip() for s in inner if str(s).strip()]
            except Exception:
                pass
        # Case: already a list of strings
        if isinstance(val, list) and all(isinstance(x, str) for x in val):
            return [x.strip() for x in val if x.strip()]
        # Case: JSON returned a string; continue parsing that
        if isinstance(val, str):
            text = val
    except Exception:
        pass

    # 2) Try as Python list literal directly
    try:
        inner = ast.literal_eval(text)
        if isinstance(inner, list):
            return [str(s).strip() for s in inner if str(s).strip()]
    except Exception:
        pass

    # 3) Fallback: splitlines (covers raw text)
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

def run(
    input_csv: str,
    output_csv: str,
    d3tok_col: str = "d3tok",
    model_name: str = "CAMeL-Lab/readability-arabertv2-d3tok-CE",
    batch_size: int = 32,
    device: Optional[int] = None,
    local_only: bool = False,
):
    try:
        import torch
        from transformers import pipeline
    except Exception as e:
        _eprint("[ERROR] Please install: pip install transformers torch")
        raise

    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)

    # gather all sentences across rows and remember boundaries
    all_sents: list[str] = []
    boundaries: list[tuple[int,int]] = []  # (start, end) per row

    for s in df.get(d3tok_col, []):
        sents = parse_d3tok_cell(s)
        start = len(all_sents)
        all_sents.extend(sents)
        end = len(all_sents)
        boundaries.append((start, end))

    if not all_sents:
        df["readability_levels"] = json.dumps([], ensure_ascii=False)
        df["max_level"] = None
        df['avg_level'] = None
        df.to_csv(output_csv, index=False)
        print(f"[INFO] No sentences found in column '{d3tok_col}'. Wrote: {output_csv}")
        return

    # init pipeline
    if device is None:
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except Exception:
            device = -1

    pipe_kwargs = dict(task="text-classification", model=model_name, device=device)
    if local_only:
        pipe_kwargs["local_files_only"] = True
    readability = pipeline(**pipe_kwargs)

    # run predictions in batches
    levels_all: list[int | None] = []
    for batch in batched(all_sents, batch_size):
        preds = readability(batch, truncation=True)
        for p in preds:
            # lvl = p['CEFR'] if 'CEFR' in p else label_to_level(p.get("label"))
            # import pdb; pdb.set_trace()
            lvl = label_to_level(p.get("label"))
            levels_all.append(lvl)

    # map predictions back to rows
    levels_per_row_json: list[str] = []
    max_levels: list[int | None] = []
    avg_levels: list[float | None] = []
    for (start, end) in boundaries:
        lvls = [lvl for lvl in levels_all[start:end] if lvl is not None]
        levels_per_row_json.append(json.dumps(lvls, ensure_ascii=False))
        max_levels.append(max(lvls) if lvls else None)
        avg_levels.append(sum(lvls)/len(lvls) if lvls else None)

    df["readability_levels"] = levels_per_row_json
    df["max_level"] = max_levels
    df["avg_level"] = avg_levels

    df.to_csv(output_csv, index=False)
    print(f"[OK] Wrote: {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_readability_levels.py <input_csv> <output_csv> [d3tok_col] [batch_size] [device] [local_only]")
        print("  d3tok_col defaults to 'd3tok'")
        print("  device: -1=CPU, 0=CUDA:0, ...  (default: auto-detect)")
        print("  local_only: 0 or 1 (default: 0)")
        sys.exit(1)
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    d3tok_col = sys.argv[3] if len(sys.argv) >= 4 else "d3tok"
    batch_size = int(sys.argv[4]) if len(sys.argv) >= 5 else 32
    device = int(sys.argv[5]) if len(sys.argv) >= 6 else None
    local_only = bool(int(sys.argv[6])) if len(sys.argv) >= 7 else False
    run(input_csv, output_csv, d3tok_col=d3tok_col, batch_size=batch_size, device=device, local_only=local_only)


