#!/usr/bin/env python3
# generate_vocab_lists.py

import os, json, time, re
from typing import List, Dict, Tuple
import pandas as pd
from openai import OpenAI

# ---------- Config ----------
INPUT_CSV = "/home/nour.rabih/arwi/generation/prompts_with_3cefr_levels.csv"   # columns: Level,Arabic Text,Translated Text,Topic_ID,prompt_id
MODEL = "gpt-4o"                               # or "gpt-4o-mini"
TEMP = 0.6                                       
RATE_DELAY = 0.5                               # seconds between calls
OUT_JSONL = "/home/nour.rabih/arwi/vocab/3levels/full_vocab_prompts_3levels.jsonl"         # NDJSON output
OUT_CSV = "/home/nour.rabih/arwi/vocab/3levels/full_vocab_prompts_3levels.csv"            # summary CSV
ENGINE = "gpt-4o"

OpenAI.api_key = "key"



# CEFR mapping (each Level -> 2 CEFR levels)
# CEFR_LEVELS_MAPPING = {
#     "Beginner": ["A1", "A2"],
#     "Intermediate": ["B1", "B2"],
#     "Advanced": ["C1", "C2"],
# }

CEFR_LEVELS_MAPPING = {
    "Beginner": ["A"],
    "Intermediate": ["B"],
    "Advanced": ["C"],
}

# Target ranges per CEFR (you can tweak)
CEFR_VOCAB_SIZES: Dict[str, Tuple[int,int]] = {
    "A1": (20, 30),
    "A2": (25, 35),
    "B1": (30, 40),
    "B2": (35, 45),
    "C1": (40, 50),
    "C2": (40, 60),
}

# ---------- Prompt builder (English; words only as output) ----------
def make_gpt_prompt(arabic_text: str, cefr: str) -> str:
    low, high = CEFR_VOCAB_SIZES.get(cefr, (30, 40))
    return f"""You are an Arabic vocabulary instructor designing word lists aligned with CEFR levels.
Target CEFR level: {cefr}
Topic (Arabic): "{arabic_text}"

Task:
- Suggest a list of vocabulary suitable for a learner at level {cefr}, related to the topic above, that the learner can use to write an Arabic Essay.
- Ensure the words are appropriate in frequency and difficulty for {cefr}.
- Output only the list of Arabic words, separated by commas.
- Number of words: between {low} and {high}.


Example format (no numbering, no extra text):
word1, word2, word3, word4, ...
"""

# ---------- Utilities ----------
def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # remove opening fence (with optional language) and trailing fence
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text).strip()
        text = re.sub(r"\n?```$", "", text).strip()
    return text

def parse_word_list(reply: str) -> List[str]:
    """
    Accepts: 'كلمة1, كلمة2, كلمة3 ...'
    Returns: ['كلمة1', 'كلمة2', 'كلمة3', ...] (stripped, deduped, non-empty)
    """
    reply = strip_code_fences(reply)
    # Replace newlines/semicolons with commas, normalize arabic commas/spaces
    reply = reply.replace("\n", ",").replace("؛", ",").replace("،", ",")
    # Remove accidental English quotes
    reply = reply.replace('"', "").replace("'", "")
    parts = [p.strip() for p in reply.split(",")]
    words = [w for w in parts if w]
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for w in words:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq

def expand_to_cefr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CEFR_list"] = df["Level"].apply(lambda x: CEFR_LEVELS_MAPPING[x])
    long_df = df.explode("CEFR_list", ignore_index=True).rename(columns={"CEFR_list": "CEFR"})
    return long_df

# ---------- Main ----------
def main():
    # Load inputs
    df = pd.read_csv(INPUT_CSV)
    required = {"Level", "Arabic Text", "Translated Text", "Topic_ID", "prompt_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Expand to per-CEFR rows
    # long_df = expand_to_cefr(df)
    long_df = df
    # Build prompts
    long_df["gpt_prompt"] = long_df.apply(
        lambda r: make_gpt_prompt(
            arabic_text=r["Arabic Text"],
            cefr=r["CEFR_level"],
        ),
        axis=1
    )

    client = OpenAI(
    # This is the default and can be omitted
    api_key=OpenAI.api_key
    )

    # Prepare outputs
    results_jsonl = []
    csv_rows = []

    for i, row in long_df.iterrows():
        prompt_id = row["prompt_id"]
        topic_id = row["Topic_ID"]
        cefr = row["CEFR_level"]
        prompt_text = row["gpt_prompt"]

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=TEMP,
            )
            reply = resp.choices[0].message.content.strip()
            words = parse_word_list(reply)

            record = {
                "prompt_id": int(prompt_id),
                "topic_id": int(topic_id),
                "level": cefr,
                "words": words,
            }
            results_jsonl.append(record)
            csv_rows.append({
                "prompt_id": int(prompt_id),
                "topic_id": int(topic_id),
                "level": cefr,
                "num_words": len(words),
                "words_preview": ", ".join(words[:8])  # quick glance
            })

            print(f"[OK] prompt_id={prompt_id} level={cefr} -> {len(words)} words")
            time.sleep(RATE_DELAY)

        except Exception as e:
            print(f"[ERR] prompt_id={prompt_id} level={cefr}: {e}")
            # still record an error stub to keep place
            results_jsonl.append({
                "prompt_id": int(prompt_id),
                "topic_id": int(topic_id),
                "level": cefr,
                "error": str(e),
                "words": []
            })

    # Save NDJSON
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in results_jsonl:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Save CSV summary
    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nSaved:\n - {OUT_JSONL}\n - {OUT_CSV}")

if __name__ == "__main__":
    main()
