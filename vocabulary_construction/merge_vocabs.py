# merging the gpt and vocab relevance results

import re, ast
import pandas as pd

gpt_vocab_path = '/home/nour.rabih/arwi/vocab/lowvocab_gpt_vocab_responses_filtered_by_samer_5levels_withpos_gloss.csv'
relevant_vocab_path = '/home/nour.rabih/arwi/vocab/relevant_words_samer.csv'

prompt_vocabs = pd.read_csv(gpt_vocab_path)
relevant_vocabs = pd.read_csv(relevant_vocab_path)

_nan_token = re.compile(r'(?<![\w])nan(?![\w])', flags=re.IGNORECASE)

def parse_words_cell(s):
    s = str(s).strip()
    # Replace bare nan with None so it's a valid Python literal
    s = _nan_token.sub("None", s)
    return ast.literal_eval(s)

prompt_vocabs["words"] = prompt_vocabs["words"].apply(parse_words_cell)

# (<3 words)
short_prompts = prompt_vocabs[prompt_vocabs["words"].apply(len) < 3].copy()
print(f"Short prompts (<3 words): {len(short_prompts)}")
# merge relevant vocabs only for short prompts
merged_df = relevant_vocabs.merge(short_prompts, on=["prompt_id", "Level"], how="inner")
print(f"Merged short prompts with relevant vocabs: {len(merged_df)}")
# anti-join: keep rows from prompt_vocabs that are NOT in short_prompts by (prompt_id, Level)
keys = ["prompt_id", "Level"]
long_prompts = (
    prompt_vocabs.merge(short_prompts[keys].drop_duplicates(), on=keys, how="left", indicator=True)
               .query("_merge == 'left_only'")
               .drop(columns=["_merge"])
)
print(f"Long prompts (>=3 words): {len(long_prompts)}")
# align columns before concat (optional but recommended)
# This keeps all columns from both; adjust if you want a specific schema.
final_df = pd.concat([merged_df, long_prompts], ignore_index=True, sort=False)

final_df.to_csv("relevance_gpt_lowvocab.csv", index=False)


import ast
import pandas as pd

# --- paths (edit if needed) ---
base_path = "gpt_vocab_responses_filtered_by_samer_5levels_withpos_gloss.csv"   # 322 rows
other_path = "relevance_gpt_lowvocab.csv"
out_path = "merged_common_words_322rows.csv"

KEYS = ["prompt_id", "topic_id", "CEFR_level"]



def merge_preserve_order(a, b):
    """
    Merge two lists (often list of tuples), preserving order:
    keep items from a first, then add unseen items from b.
    Uniqueness is checked by the full element value.
    """
    out = []
    seen = set()
    for item in a:
        if item not in seen:
            out.append(item)
            seen.add(item)
    for item in b:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out

# --- load ---
base = pd.read_csv(base_path)
other = pd.read_csv(other_path)

# --- parse words columns (adjust column name if yours differs) ---
if "words" not in base.columns:
    raise ValueError("Base file must contain a 'words' column.")
if "words" not in other.columns:
    raise ValueError("Other file must contain a 'words' column.")

base["words"] = base["words"].apply(parse_words_cell)
other["words"] = other["words"].apply(parse_words_cell)

# --- keep only the keys + words from the other file for a clean merge ---
other_small = other[KEYS + ["words"]].rename(columns={"words": "words_lowvocab"})

# --- left merge so we keep all 322 rows from base ---
merged = base.merge(other_small, on=KEYS, how="left")

# --- optional: keep originals for auditing ---
merged["words_original"] = merged["words"]

# --- merge words only when there's a match ---
def merge_row(row):
    a = row["words_original"]
    b = row["words_lowvocab"]
    if isinstance(b, list) and len(b) > 0:
        return merge_preserve_order(a, b)
    return a

merged["words"] = merged.apply(merge_row, axis=1)

# --- make sure words columns are saved as strings (literal list) in CSV ---
for col in ["words", "words_original", "words_lowvocab"]:
    if col in merged.columns:
        merged[col] = merged[col].apply(lambda x: str(x) if isinstance(x, list) else ("" if pd.isna(x) else str(x)))

# --- save ---
merged.to_csv(out_path, index=False)

# --- quick sanity prints ---
print("Base rows:", len(base))
print("Output rows:", len(merged))
print("Matches on keys:", merged["words_lowvocab"].astype(str).ne("").sum())
print("Saved:", out_path)
