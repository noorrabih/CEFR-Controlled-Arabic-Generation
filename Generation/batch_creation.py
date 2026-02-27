"""Create OpenAI Batch JSONL requests for CEFR-controlled essay generation.

This produces a JSONL file where each line is a request payload suitable for OpenAI Batch.

Input CSV expected columns (configurable):
- prompt_id (default: prompt_id)
- topic text (default: Arabic Text)
- CEFR level (default: CEFR_level)
- profile specifications (optional; default: profile_json)
- vocabulary list (optional; default: words)

Conditions:
- P1: topic only
- P2: topic + level
- P3: topic + level + profile
- P4: topic + level + vocab list
- P5: topic + level + profile + vocab list

Example:
  python Generation/batch_creation.py \
    --input_csv generation/prompts_with_6cefr_levels.csv \
    --out_jsonl generation/batch_input.jsonl \
    --condition P3 \
    --model gpt-4o
"""

from __future__ import annotations

import argparse
import ast
import json
from typing import Any, Dict, List

import pandas as pd


SYSTEM_MSG = "You are a helpful Arabic writing assistant."


def _parse_list_cell(x: str) -> List[str]:
    if x is None:
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        return list(ast.literal_eval(s))
    except Exception:
        # fallback: comma-separated
        return [w.strip() for w in s.split(",") if w.strip()]


def _parse_json_cell(x: str) -> Dict[str, Any]:
    if x is None:
        return {}
    s = str(x).strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def build_user_prompt(topic_text: str, level: str | None, profile: Dict[str, Any] | None, vocab: List[str] | None, condition: str) -> str:
    topic_text = str(topic_text).strip()
    level = str(level).strip() if level is not None else ""
    profile = profile or {}
    vocab = vocab or []

    base = [
        "Write a short Arabic essay.",
        f"Topic: {topic_text}",
    ]

    if condition in {"P2", "P3", "P4", "P5"} and level:
        base.append(f"Target CEFR level: {level}")

    if condition in {"P3", "P5"} and profile:
        base.append("Follow these level-specific profile constraints:")
        base.append(json.dumps(profile, ensure_ascii=False, indent=2))

    if condition in {"P4", "P5"} and vocab:
        base.append("Try to use the following vocabulary where appropriate:")
        base.append(", ".join(vocab))

    base.append("Return only the essay text. Do not add explanations.")
    return "\n".join(base)


def make_request(custom_id: str, model: str, user_prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }


def main(args):
    df = pd.read_csv(args.input_csv)

    for col in [args.prompt_id_col, args.topic_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Found: {list(df.columns)}")

    # Optional columns
    if args.level_col and args.level_col not in df.columns and args.condition != "P1":
        raise ValueError(f"Condition {args.condition} requires --level_col, but '{args.level_col}' not found.")

    out_rows = []
    for _, r in df.iterrows():
        pid = str(r[args.prompt_id_col])
        topic = r[args.topic_col]
        level = r[args.level_col] if args.level_col in df.columns else None
        profile = _parse_json_cell(r[args.profile_col]) if args.profile_col in df.columns else {}
        vocab = _parse_list_cell(r[args.vocab_col]) if args.vocab_col in df.columns else []

        user_prompt = build_user_prompt(topic, level, profile, vocab, args.condition)
        out_rows.append(make_request(pid, args.model, user_prompt, args.temperature, args.max_tokens))

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote: {args.out_jsonl} ({len(out_rows)} requests)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create OpenAI Batch JSONL input for essay generation.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--condition", choices=["P1", "P2", "P3", "P4", "P5"], required=True)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=800)

    ap.add_argument("--prompt_id_col", default="prompt_id")
    ap.add_argument("--topic_col", default="Arabic Text")
    ap.add_argument("--level_col", default="CEFR_level")
    ap.add_argument("--profile_col", default="profile_json")
    ap.add_argument("--vocab_col", default="words")

    main(ap.parse_args())
