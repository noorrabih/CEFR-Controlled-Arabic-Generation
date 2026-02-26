# relevant_words_gpu_readability.py
# -*- coding: utf-8 -*-

import re, csv
from typing import Optional, Set, List
import pandas as pd
import numpy as np
# from torch.optim.lr_scheduler import LRScheduler

# ---------- predefined unwanted POS ----------
DEFAULT_UNWANTED_POS: Set[str] = {
    "prep","pron_dem","pron_rel","part_neg","conj","conj_sub","part",
    "part_det","part_focus","part_fut","part_interrog","part_restrict",
    "part_verb","part_voc","pron","punc","adv_rel","pron_interrog","pron_exclam"
}
# Add more if desired:
# DEFAULT_UNWANTED_POS.update({"interj","noun_num","adj_num","verb_pseudo"})

# ---------- light Arabic normalization ----------
_AR_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]")
def normalize_ar(s: str) -> str:
    if not isinstance(s, str): return s
    s = _AR_DIAC.sub("", s)
    s = s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    s = s.replace("ة","ه").replace("ى","ي").replace("ـ","")
    return s

def read_any_delim(path: str) -> pd.DataFrame:
    """Read CSV or TSV; try comma, then tab, then sniff."""
    try:
        return pd.read_csv(path)  # comma
    except Exception:
        try:
            return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE)  # tab
        except Exception:
            return pd.read_csv(path, sep=None, engine="python", quoting=csv.QUOTE_NONE)  # sniff

# -------- SentenceTransformer model (GPU-aware) --------
_ST_MODEL = None
_ST_DEVICE = "cpu"

def _get_device(preferred: Optional[str] = "auto") -> str:
    try:
        import torch
        if preferred == "cuda" or (preferred == "auto" and torch.cuda.is_available()):
            return "cuda"
    except Exception:
        pass
    return "cpu"

def _get_model(
    name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device: Optional[str] = "auto"
):
    global _ST_MODEL, _ST_DEVICE
    if _ST_MODEL is not None:
        return _ST_MODEL, _ST_DEVICE
    dev = _get_device(device)
    from sentence_transformers import SentenceTransformer
    _ST_MODEL = SentenceTransformer(name, device=dev)
    _ST_DEVICE = dev
    return _ST_MODEL, _ST_DEVICE

def _encode_in_batches(model, texts: List[str], batch_size: int, normalize_embeddings: bool, use_fp16: bool):
    """
    Encode long lists on the chosen device with batching and optional FP16 autocast.
    """
    try:
        import torch
        use_amp = use_fp16 and (model._target_device.type == "cuda")
        embeddings = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            if use_amp:
                with torch.cuda.amp.autocast():
                    emb = model.encode(
                        chunk,
                        batch_size=len(chunk),
                        convert_to_tensor=True,
                        normalize_embeddings=normalize_embeddings,
                        show_progress_bar=False,
                    )
            else:
                emb = model.encode(
                    chunk,
                    batch_size=len(chunk),
                    convert_to_tensor=True,
                    normalize_embeddings=normalize_embeddings,
                    show_progress_bar=False,
                )
            embeddings.append(emb)
        if len(embeddings) == 1:
            return embeddings[0]
        return torch.cat(embeddings, dim=0)
    except ImportError:
        # Fallback: single call (CPU)
        return model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )

# ---- small helper to find a column by name case-insensitively ----
def _find_col_casefold(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cand_cf = [c.casefold() for c in candidates]
    for col in df.columns:
        if col.casefold() in cand_cf:
            return col
    return None

# -------- main API --------
def relevant_words_from_file(
    sentence: str,
    path: str,
    level: int,
    threshold: float = 0.35,
    top_k: Optional[int] = None,
    normalize_arabic: bool = True,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    unwanted_pos: Optional[Set[str]] = None,
    device: Optional[str] = "auto",   # "auto" | "cuda" | "cpu"
    use_fp16: bool = True,            # use FP16 when on CUDA
    batch_size: int = 1024            # tune to your VRAM
) -> pd.DataFrame:
    """
    GPU-aware relevant word extraction for a sentence from a CSV/TSV with 'lemma#pos'.
    - Filters out POS in `unwanted_pos` (defaults to DEFAULT_UNWANTED_POS).
    - Uses lemma (before '#') for embeddings.
    - Runs on CUDA if available (or if device='cuda') with optional FP16.
    - Includes 'readability (rounded average)' column in output if present.
    """
    if unwanted_pos is None:
        unwanted_pos = DEFAULT_UNWANTED_POS

    df = read_any_delim(path).fillna("")
    df = df[df['readability (rounded average)'] == level]
    if "lemma#pos" not in df.columns:
        raise ValueError("Input file must include a 'lemma#pos' column.")

    tmp = df.copy()

    # Robust split "lemma#pos"
    parts = tmp["lemma#pos"].astype(str).str.split("#", n=1, expand=True)
    if parts.shape[1] == 1:
        parts[1] = ""
    parts.columns = ["lemma", "pos"]
    tmp["lemma"] = parts["lemma"]
    tmp["pos"]   = parts["pos"].fillna("")

    # Optional occurrences for tie-breaks
    if "Occurrences" in tmp.columns:
        def _to_int(x):
            try: return int(str(x).strip())
            except Exception: return 0
        tmp["Occurrences_int"] = tmp["Occurrences"].map(_to_int)
    else:
        tmp["Occurrences_int"] = 0

    # ---- locate readability column (case-insensitive) ----
    readability_col = _find_col_casefold(tmp, [
        "readability (rounded average)",
        "readability",
        "rounded average readability"
    ])
    if readability_col is not None:
        # Try to coerce to numeric (keeps strings if cannot)
        try:
            tmp[readability_col] = pd.to_numeric(tmp[readability_col], errors="ignore")
        except Exception:
            pass

    # Drop unwanted POS
    tmp = tmp[~tmp["pos"].isin(unwanted_pos)].copy()
    if tmp.empty:
        # include readability column in the empty frame if available
        base_cols = ["lemma","pos","similarity"]
        if readability_col is not None:
            base_cols.append(readability_col)
        return pd.DataFrame(columns=base_cols)

    # Deduplicate by lemma
    tmp = tmp.sort_values(["lemma","Occurrences_int"], ascending=[True, False])
    tmp = tmp.drop_duplicates(subset=["lemma"], keep="first")

    # Prep text
    if normalize_arabic:
        sentence_proc = normalize_ar(sentence)
        lemmas_proc: List[str] = [normalize_ar(x) for x in tmp["lemma"].tolist()]
    else:
        sentence_proc = sentence
        lemmas_proc = tmp["lemma"].tolist()

    # Model + device
    model, actual_device = _get_model(model_name, device=device)

    # Encode
    import torch
    use_amp_sentence = use_fp16 and (actual_device == "cuda")
    if use_amp_sentence:
        with torch.cuda.amp.autocast():
            s_emb = model.encode(sentence_proc, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    else:
        s_emb = model.encode(sentence_proc, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)

    w_embs = _encode_in_batches(model, lemmas_proc, batch_size=batch_size, normalize_embeddings=True, use_fp16=use_fp16)

    # Cosine similarity on device
    from sentence_transformers import util
    sims = util.cos_sim(s_emb, w_embs)[0].detach().cpu().numpy()

    # Attach + filter
    tmp = tmp.assign(similarity=sims)

    # Build output columns
    out_cols = ["lemma","pos","similarity"]
    if readability_col is not None:
        out_cols.append(readability_col)  # keep exact original name
    if "Gloss" in tmp.columns:
        out_cols.append("Gloss")
    if "Occurrences" in tmp.columns:
        out_cols.append("Occurrences")

    result = tmp[tmp["similarity"] >= threshold].sort_values(
        ["similarity","Occurrences_int","lemma"], ascending=[False, False, True]
    )
    if top_k is not None:
        result = result.head(top_k)

    return result[out_cols]

def relevant_lemmas_list(
    sentence: str,
    path: str,
    level: int,
    threshold: float = 0.35,
    top_k: Optional[int] = None,
    normalize_arabic: bool = True,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    unwanted_pos: Optional[Set[str]] = None,
    device: Optional[str] = "auto",
    use_fp16: bool = True,
    batch_size: int = 1024
) -> List[str]:
    df = relevant_words_from_file(
        sentence=sentence,
        path=path,
        level= level,
        threshold=threshold,
        top_k=top_k,
        normalize_arabic=normalize_arabic,
        model_name=model_name,
        unwanted_pos=unwanted_pos,
        device=device,
        use_fp16=use_fp16,
        batch_size=batch_size
    )
    df = df[df['Occurrences']>50]
    
    # list of (lemma, pos) pairs
    return list(df[["lemma", "pos", "similarity", 'readability (rounded average)', "Gloss"]].itertuples(index=False, name=None))


# sentence = "اوصف يومك المفضل,"
# path = "samer-readability-lexicon-v2/SAMER-Readability-Lexicon-v2.tsv"


# # Auto GPU if available (FP16 on), keep top 200:
# df = relevant_words_from_file(sentence, path, level = 1, threshold=0.4, top_k=40, device="auto", use_fp16=True, batch_size=1024)
# occ = df[df['Occurrences']>70]

# occ.to_csv('words2.csv')

# # Force GPU (error if CUDA not available):
# df_gpu = relevant_words_from_file(sentence, path, device="cuda")

# Just lemmas:
# lemmas = relevant_lemmas_list(sentence, path, threshold=0.4, device="auto")

# for each prompt in /Users/nour.rabih/Desktop/arwi/prompts.csv
# get the relavant words based on the level. 
# level Beginner -> 1 and 2
# Intermediate -> 3 and 4
# Advanced -> 5

# -------------------- add this at the very end --------------------
# Simple runner: read prompts.csv, get relevant words per prompt by level, save CSV.
# # -------------------- minimal CSV writer (uses relevant_lemmas_list) --------------------
import pandas as pd
from collections import OrderedDict

PROMPTS_CSV = "lowvocab_prompts.csv"
LEXICON_PATH = "/home/nour.rabih/Readability-interpretability/data/SAMER-Readability-Lexicon.tsv"  # <-- set this
OUTPUT_CSV = "relevant_words_samer.csv"

# Level -> allowed readability levels
# LEVEL2READ = {
#     "Beginner": [1],
#     "Intermediate": [2, 3],
#     "Advanced": [4, 5],
# }
LEVEL2READ = {
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
}

prompts_df = read_any_delim(PROMPTS_CSV).fillna("")
# prompts_df= prompts_df[prompts_df['prompt_id'] == 117]
out_rows = []

for _, r in prompts_df.iterrows():
    level_name = str(r.get("level", "")).strip() #Level
    # print(type(level_name))

    # levels = LEVEL2READ.get(level_name, [1, 2, 3, 4, 5])
    levels = [int(level_name)]
    # print("levels ", levels)
    sentence = str(r.get("Arabic Text", ""))

    # collect lemmas across allowed readability levels (dedup, preserve insertion order)
    ordered = OrderedDict()
    # print(levels)
    for lv in levels:
        print(lv)
        # print(f"Processing level {lv} for prompt ID {r.get('prompt_id', '')}...")
        lemmas = relevant_lemmas_list(
            sentence=sentence,
            path=LEXICON_PATH,
            level=lv,
            threshold=0.50,      # tweak if you want stricter/looser
            top_k=40,          # keep all above threshold
            normalize_arabic=True,
            device="auto",
            use_fp16=True,
            batch_size=1024
        )
        if len(lemmas) == 0:
            print(f"No lemmas found for level {lv} and prompt ID {r.get('prompt_id', '')}.")
            lemmas = relevant_lemmas_list(
                sentence=sentence,
                path=LEXICON_PATH,
                level=lv,
                threshold=0.35,      # tweak if you want stricter/looser
                top_k=20,          # keep all above threshold
                normalize_arabic=True,
                device="auto",
                use_fp16=True,
                batch_size=1024
            )
        out_rows.append({
        "prompt_id": r.get("prompt_id", ""),
        "Level": lv,
        "Arabic Text": sentence,
        "Translated Text": r.get("Translated Text", ""),
        "Topic_ID": r.get("Topic_ID", ""),
        "lemmas": lemmas  # <-- list serialized as a single column
        })
        print(f"appended {len(lemmas)} lemmas for level {lv}.")
        # for L in lemmas:
        #     ordered.setdefault(L, None)

    # lemmas_list = list(ordered.keys())
    # lemmas_joined = "; ".join(lemmas_list)

    # out_rows.append({
    #     "prompt_id": r.get("prompt_id", ""),
    #     "Level": level_name,
    #     "Arabic Text": sentence,
    #     "Translated Text": r.get("Translated Text", ""),
    #     "Topic_ID": r.get("Topic_ID", ""),
    #     "lemmas": lemmas  # <-- list serialized as a single column
    # })

pd.DataFrame(out_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"Saved: {OUTPUT_CSV}")

