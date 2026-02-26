import os
import re
import pandas as pd
from collections import Counter, defaultdict

directory = "/home/nour.rabih/arwi/readability_controlled_generation/generation/vocabs_prompt/5levels/"
BASE_DIR = f"{directory}essays_parsed_Text_100"
OUT_CSV  = f"{directory}syntax/sentences_with_dep_pos_counts.csv"

ID_RE = re.compile(r"^#\s*id\s*=\s*(\S+)\s*$")

def iter_sentences_with_ids(path):
    """
    Yields (essay_id, sent_rows) for each sentence block.
    Sentence blocks separated by blank lines.
    """
    current_essay_id = None
    sent = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            m = ID_RE.match(line.strip())
            if m:
                current_essay_id = m.group(1)
                continue

            if not line.strip():
                if sent:
                    if current_essay_id is None:
                        current_essay_id = os.path.splitext(os.path.basename(path))[0]
                    yield current_essay_id, sent
                    sent = []
                continue

            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) >= 8:
                sent.append(parts)

        if sent:
            if current_essay_id is None:
                current_essay_id = os.path.splitext(os.path.basename(path))[0]
            yield current_essay_id, sent


UD_RE = re.compile(r"(?:^|[|;])ud=([^|;]+)")

def get_ud_pos(feats: str, fallback: str = None):
    """
    Extract UD POS from FEATS column (e.g., 'ud=NOUN|pos=noun|...').
    Returns fallback if not found.
    """
    if feats is None:
        return fallback
    m = UD_RE.search(str(feats))
    return m.group(1) if m else fallback


rows = []
all_dep = set()
all_pos = set()
all_dep_pos = set()
sent_counter = defaultdict(int)

for root, _, files in os.walk(BASE_DIR):
    for fn in files:
        if fn.startswith(".") or not fn.endswith(".conllx"):
            continue
        path = os.path.join(root, fn)

        try:
            for essay_id, sent in iter_sentences_with_ids(path):
                sent_counter[essay_id] += 1
                i = sent_counter[essay_id]

                tokens  = [tok[1] for tok in sent]        # FORM
                deprels = [tok[7] for tok in sent]        # DEPREL
                feats   = [tok[5] for tok in sent]        # FEATS

                # Use UD POS from FEATS; fallback to column 4 if missing
                pos_ud  = [get_ud_pos(f, fallback=tok[3]) for f, tok in zip(feats, sent)]
                bad = [p for p in pos_ud if not re.fullmatch(r"[A-Z_]+", str(p))]
                if bad:
                    print("Non-standard POS detected:", set(bad))

                dep_counts = Counter(deprels)
                pos_counts = Counter(pos_ud)

                # normalized: REL(POS_dep)
                dep_pos_labels = [f"{r}({p})" for r, p in zip(deprels, pos_ud)]
                dep_pos_counts = Counter(dep_pos_labels)

                all_dep.update(dep_counts.keys())
                all_pos.update(pos_counts.keys())
                all_dep_pos.update(dep_pos_counts.keys())

                row = {
                    "essay_id": essay_id,
                    # "sentence_id": f"{essay_id}_{i}",
                    "sentence": " ".join(tokens),
                }

                # deprel counts
                for rel, c in dep_counts.items():
                    row[f"dep_{rel}"] = c

                # pos counts
                for p, c in pos_counts.items():
                    row[f"pos_{p}"] = c

                # rel(pos_dep) counts
                for lab, c in dep_pos_counts.items():
                    # column names shouldn't contain weird chars; keep () but replace spaces just in case
                    safe = lab.replace(" ", "_")
                    row[f"depPOS_{safe}"] = c

                rows.append(row)

        except Exception as e:
            print(f"[WARN] Skipping file due to error: {path}\n  -> {e}")

if not rows:
    raise RuntimeError(f"No sentences were parsed. Check BASE_DIR={BASE_DIR}")

df = pd.DataFrame(rows)

# Ensure all columns exist
dep_cols = [f"dep_{lab}" for lab in sorted(all_dep)]
pos_cols = [f"pos_{lab}" for lab in sorted(all_pos)]
depPOS_cols = [f"depPOS_{lab.replace(' ', '_')}" for lab in sorted(all_dep_pos)]

for col in dep_cols + pos_cols + depPOS_cols:
    if col not in df.columns:
        df[col] = 0

base_cols = ["essay_id", "sentence"]
df = df[base_cols + dep_cols + pos_cols + depPOS_cols].fillna(0)

for col in dep_cols + pos_cols + depPOS_cols:
    df[col] = df[col].astype(int)

df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"✅ Saved {len(df)} sentences to: {OUT_CSV}")
print(f"✅ Dependency columns: {len(dep_cols)} | POS columns: {len(pos_cols)} | depPOS columns: {len(depPOS_cols)}")



# add cefr levels to corrected essays sentences file
IN_CSV =  f"{directory}syntax/sentences_with_dep_pos_counts.csv"
OUT_CSV = f"{directory}syntax/sentences_with_dep_pos_counts_cefr.csv"


# cefr from 
cefr = f"{directory}generated_essays_p3_readability.csv"
import pandas as pd

df = pd.read_csv(IN_CSV)
cefr_df = pd.read_csv(cefr) # , sep='\t'
# cefr_df['Document_ID'].split('-')[1]
# cefr_df['Document_ID'] = cefr_df['Document_ID'].apply(lambda x: x.split('-')[1])
cefr_df['Document_ID'] = cefr_df['ID'].astype(str)

cefr_dict = dict(zip(cefr_df['Document_ID'], cefr_df['CEFR']))

def get_cefr(essay_id):
    return essay_id.split('_')[1].split('-')[0]
    # base_id = essay_id.split('-')[1]
    # return cefr_dict.get(base_id, 'NA')
df['CEFR'] = df['essay_id'].apply(get_cefr)

df.to_csv(OUT_CSV, index=False)
print(f"✅ Saved with CEFR: {OUT_CSV}")


# import pandas as pd
# import re

# # IN_CSV = "/home/nour.rabih/arwi/readability_controlled_generation/arabic-aes-bea25/Data/sentences_with_dep_pos_counts_cefr.csv"
# # OUT_ESSAY = "/home/nour.rabih/arwi/readability_controlled_generation/arabic-aes-bea25/Data/essay_syntax_stats.csv"
# # OUT_LEVEL = "/home/nour.rabih/arwi/readability_controlled_generation/arabic-aes-bea25/Data/level_syntax_stats.csv"
# # df = pd.read_csv(IN_CSV)
# # the id is AR-030-268469-11 in df it is 268469
def get_base_id(essay_id):
    return essay_id.split('-')[0]


# # bea = pd.read_csv("/home/nour.rabih/arwi/readability_controlled_generation/arabic-aes-bea25/Data/sentences_with_dep_pos_counts_cefr.csv")
# # AR_gpt-4o_15_64_1276-5
# # bea['base_id'] = bea['essay_id'].apply(get_base_id)

# # def get_base_id(essay_id):
# #     return essay_id.split('-')[2]
# # zaebuc = pd.read_csv("/home/nour.rabih/arwi/readability_controlled_generation/ZAEBUC-v1.01/sentences_with_dep_pos_counts_cefr.csv")
# # # AR-130-90813-1
# # zaebuc['base_id'] = zaebuc['essay_id'].apply(get_base_id)


# # df = pd.concat([bea, zaebuc], ignore_index=True)

OUT_ESSAY = f"{directory}syntax/essay_syntax_stats.csv"
OUT_LEVEL = f"{directory}syntax/level_syntax_stats.csv"

df['base_id'] = df['essay_id'].apply(get_base_id)
print(df.columns)
# --- collect syntactic count columns ---
count_cols = [c for c in df.columns if c.startswith(("dep_", "pos_", "depPOS_"))]

# --- tokens per sentence ---
df["n_tokens"] = df["sentence"].fillna("").astype(str).apply(lambda s: len(s.split()))

def extract_level(essay_id: str):
    m = re.search(r"_((A|A1|A2|B1|B2|C1|C2))_", str(essay_id))
    return m.group(1) if m else None

if "CEFR" not in df.columns:
    df["level"] = df["essay_id"].apply(extract_level)
else:
    df["level"] = df["CEFR"]
    print("CEFR")


# --- essay aggregation (SUM) ---
essay = df.groupby(["base_id", "level"], as_index=False)[count_cols + ["n_tokens"]].sum()

# --- ratios per essay (per token) for ALL count columns ---
for c in count_cols:
    essay[c.replace("_", "_ratio_", 1)] = essay[c] / essay["n_tokens"].where(essay["n_tokens"] > 0, 1)
    # Explanation: dep_X -> dep_ratio_X ; pos_X -> pos_ratio_X ; depPOS_X -> depPOS_ratio_X


essay.to_csv(OUT_ESSAY, index=False, encoding="utf-8")

# --- level stats ---
ratio_cols = [c for c in essay.columns if c.startswith(("dep_ratio_", "pos_ratio_", "depPOS_ratio_"))]

ratio_stats = essay.groupby("level")[ratio_cols].agg(["mean", "median", "std", "count"])
ratio_stats.columns = [f"{col}_{stat}" for col, stat in ratio_stats.columns]
ratio_stats = ratio_stats.reset_index()

# raw counts sum + mean at level
sum_stats = essay.groupby("level")[count_cols].sum().add_suffix("_sum")
mean_stats = essay.groupby("level")[count_cols].mean().add_suffix("_mean")

level_stats = ratio_stats.merge(
    pd.concat([sum_stats, mean_stats], axis=1).reset_index(),
    on="level",
    how="left"
)

level_stats.to_csv(OUT_LEVEL, index=False, encoding="utf-8")

print("✅ Saved essay stats:", OUT_ESSAY)
print("✅ Saved level stats:", OUT_LEVEL)
print("Levels found:", sorted([x for x in level_stats["level"].dropna().unique()]))
