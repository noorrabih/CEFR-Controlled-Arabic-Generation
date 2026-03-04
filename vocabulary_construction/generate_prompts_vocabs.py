#!/usr/bin/env python3
"""Generate vocabulary lists per prompt using OpenAI.

Input CSV should contain at least:
  - prompt_id
  - Topic_ID
  - Arabic Text
  - CEFR_level (or a column you pass via --cefr_col)

The script calls an OpenAI chat model and asks for a comma-separated list of Arabic words.

Example:
  export OPENAI_API_KEY=... 
  python vocabulary_construction/generate_prompts_vocabs.py \
    --input_csv generation_data/topic_prompts/prompts_with_6cefr_levels.csv \
    --out_jsonl generation_data/vocabulary/6levels/full_vocab.jsonl \
    --out_csv generation_data/vocabulary/6levels/full_vocab.csv \
    --model gpt-4o \
    --cefr_col CEFR_level \
    --arabic_col "Arabic Text"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

import pandas as pd
from openai import OpenAI


CEFR_VOCAB_SIZES: Dict[str, Tuple[int, int]] = {
    "A1": (20, 30),
    "A2": (25, 35),
    "B1": (30, 40),
    "B2": (35, 45),
    "C1": (40, 50),
    "C2": (40, 60),
    # coarse buckets
    "A": (20, 35),
    "B": (30, 45),
    "C": (40, 60),
}


def make_prompt(arabic_text: str, cefr: str) -> str:
    low, high = CEFR_VOCAB_SIZES.get(str(cefr), (30, 40))
    return (
        "You are an Arabic vocabulary instructor designing word lists aligned with CEFR levels.\n"
        f"Target CEFR level: {cefr}\n"
        f"Topic (Arabic): \"{arabic_text}\"\n\n"
        "Task:\n"
        f"- Suggest a list of vocabulary suitable for a learner at level {cefr}, related to the topic above, that the learner can use to write an Arabic essay.\n"
        f"- Ensure the words are appropriate in frequency and difficulty for {cefr}.\n"
        "- Output only the list of Arabic words, separated by commas.\n"
        f"- Number of words: between {low} and {high}.\n\n"
        "Example format (no numbering, no extra text):\n"
        "word1, word2, word3, word4, ...\n"
    )


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text).strip()
        text = re.sub(r"\n?```$", "", text).strip()
    return text


def parse_word_list(reply: str) -> List[str]:
    reply = strip_code_fences(reply)
    reply = reply.replace("\n", ",").replace("؛", ",").replace("،", ",")
    reply = reply.replace('"', "").replace("'", "")
    parts = [p.strip() for p in reply.split(",")]
    words = [w for w in parts if w]

    seen = set()
    uniq = []
    for w in words:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def main(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Export it before running.")

    df = pd.read_csv(args.input_csv)
    for col in [args.prompt_id_col, args.topic_id_col, args.arabic_col, args.cefr_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Found: {list(df.columns)}")

    client = OpenAI(api_key=api_key)

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    rows_out = []
    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for _, r in df.iterrows():
            prompt_id = r[args.prompt_id_col]
            topic_id = r[args.topic_id_col]
            cefr = str(r[args.cefr_col]).strip()
            topic_ar = str(r[args.arabic_col]).strip()

            gpt_prompt = make_prompt(topic_ar, cefr)

            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": gpt_prompt}],
                temperature=args.temperature,
            )

            text = resp.choices[0].message.content
            words = parse_word_list(text)

            record = {
                "prompt_id": prompt_id,
                "topic_id": topic_id,
                "CEFR_level": cefr,
                "words": words,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            rows_out.append({
                "prompt_id": prompt_id,
                "topic_id": topic_id,
                "CEFR_level": cefr,
                "words": json.dumps(words, ensure_ascii=False),
                "n_words": len(words),
            })

            if args.rate_delay > 0:
                time.sleep(args.rate_delay)

    pd.DataFrame(rows_out).to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Wrote: {args.out_jsonl}")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate vocabulary lists per prompt using OpenAI.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--rate_delay", type=float, default=0.5)

    ap.add_argument("--prompt_id_col", default="prompt_id")
    ap.add_argument("--topic_id_col", default="Topic_ID")
    ap.add_argument("--arabic_col", default="Arabic Text")
    ap.add_argument("--cefr_col", default="CEFR_level")

    main(ap.parse_args())
