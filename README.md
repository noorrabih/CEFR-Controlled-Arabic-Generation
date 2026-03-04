# CEFR-Controlled Arabic Generation

This repo contains the code for generating **CEFR-controlled Arabic essays** and evaluating them using:

- **Readability alignment** (predicting BAREC / Taha-19 levels)
- **Linguistic profiling** (surface, lexical, syntactic features)
- **Profile-based evaluation** (feature filtering + cosine similarity)
- **Vocabulary list construction** (GPT vocab generation + SAMER filtering + relevance expansion)

The pipeline is built around two learner corpora: **ZAEBUC** and **ARWI**.

---

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Data Sources](#data-sources)
4. [Readability Alignment (BAREC)](#readability-alignment-barec)
5. [Linguistic Profiling](#linguistic-profiling)
6. [Profile Filtering](#profile-filtering)
7. [Vocabulary List Construction](#vocabulary-list-construction)
8. [Essay Generation (OpenAI Batch)](#essay-generation-openai-batch)
9. [Evaluation Against Profiles](#evaluation-against-profiles)
10. [Notes](#notes)

---

## Installation

**Python**: 3.9+

```bash
pip install -r requirements.txt
```

**CAMeL Tools** (required for morphological analysis and parsing):

```bash
pip install camel-tools
camel_data -i morphology-db-msa-s31
```

> For CAMeL Parser, clone the repo separately and point `--model_dir` to its models directory:
> ```bash
> git clone https://github.com/CAMeL-Lab/camel_parser
> ```

**OpenAI** (required for generation and vocab construction):

```bash
export OPENAI_API_KEY=your_key_here
```

**CAMeL Morphology DB** (required for feature extraction):

```bash
export CAMEL_MORPH_DB=path/to/camel_morph_msa_v1.0.db
```

---

## Project Structure

```
.
├── preprocessing/              # Dataset formatting utilities
│   └── zaebuc_merge_correct_essays.py
├── utils/                      # Readability, filtering, evaluation helpers
│   ├── essays_d3tok.py
│   ├── add_readability_levels.py
│   ├── barec_correlation.py
│   ├── readability_visualization.py
│   ├── split_sentences.py
│   ├── profile_creator.py
│   ├── features_filter.py
│   └── cosine_similarity.py
├── feature_extraction/         # Syntactic and surface feature extraction
│   ├── tree_parsing.py
│   ├── tree_stats.py
│   ├── tree_stats_postprocess.py
│   ├── syntactic_features.py
│   ├── surface_feats.py
│   └── camel_features.py
├── vocabulary_construction/    # Vocab list generation and filtering
│   ├── generate_prompts_vocabs.py
│   ├── gpt_tosamer.py
│   ├── vocab_relevance.py
│   └── merge_vocabs.py
└── generation/                 # OpenAI batch generation
    ├── build_batch.py
    └── upload_run.py
```

---

## Data Sources

### ZAEBUC

ZAEBUC corrected essays are provided **one token per row**. We merge them into full essays before running experiments.

```bash
python preprocessing/zaebuc_merge_correct_essays.py \
    --alignment_tsv <alignment_tsv> \
    --analyzed_tsv <analyzed_tsv> \
    --out_csv <out_csv>
```

Output CSV columns: `Document_ID`, `Essay`, `CEFR`, `word_count`

### ARWI

ARWI essays are already formatted as full essays — no merging step needed.
Expected input columns: `Document_ID`, `Essay`, `CEFR`

---

## Readability Alignment (BAREC)

We predict BAREC / Taha-19 readability levels for essays, then compute Spearman correlation
with CEFR ordinal ranks to validate alignment.

### Step 1 — Sentence splitting + D3 tokenization

```bash
python utils/essays_d3tok.py \
  --input_csv <input_csv> \
  --out_csv <out_csv> \
  --preprocess_script <preprocess_script> \   # optional
  --morph_db <morph_db>                       # optional
```

Output CSV columns: `Document_ID`, `Essay`, `d3tok`, `CEFR`, `word_count`

### Step 2 — Predict readability with BAREC

```bash
python utils/add_readability_levels.py \
  --input_csv <input_csv> \
  --output_csv <output_csv>
```

### Step 3 — Spearman correlation (CEFR vs Taha-19)

```bash
python utils/barec_corelation.py \
  --csvs paths/to/cefr_barec.csv
```

### Step 4 — Visualization

```bash
python utils/readability_visualization.py \
  --input_csv <input_csv> \
  --out_dir <out_dir> \
  --drop_unassessable
```
 
---

## Linguistic Profiling

We build per-CEFR-level linguistic profiles using **syntactic**, **lexical**, and **surface** features.

### Syntactic Features

**1 — Split essays into sentences**

```bash
python utils/split_sentences.py \
  --input_path <input_path> \
  --sep "," \
  --id_col Document_ID \
  --text_col Essay \
  --out_csv <out_csv>
```

**2 — Parse with CAMeL Parser (outputs CoNLL-X)**

```bash
python feature_extraction/tree_parsing.py \
  --input_csv <input_csv> \
  --batch_dir <batch_dir> \
  --out_dir <out_dir>
  # --model_dir and --clitic_feats_csv default to the local camel_parser paths
```

**3 — Compute tree statistics (depth, breadth, branching)**

```bash
python feature_extraction/tree_stats.py \
  --src_dir <src_dir> \
  --out_dir <out_dir>
```
**4 — Aggregate per essay + optional heatmaps**

```bash
python feature_extraction/tree_stats_postprocess.py \
  --stats_dir <stats_dir> \
  --readability_csv <readability_csv> \
  --out_csv <out_csv> \
  --out_dir <out_dir> \
  --make_heatmaps
```

**5 — POS / Dependency / Dep(POS) extraction**

```bash
python feature_extraction/syntactic_features.py \
  --parsed_dir <parsed_dir> \
  --readability_csv <readability_csv> \
  --out_dir <out_dir>
```

Output: `syntax_level_summary.csv` in `<out_dir>`

---

### Surface Features

```bash
python feature_extraction/surface_feats.py \
  --input_csv <input_csv> \
  --out_dir <out_dir> \
  --levels_csv <levels_csv>   # optional: CSV with Document_ID, CEFR (if not in input)
```

Outputs written to `<out_dir>`: `sentence_level_metrics.csv`, `essay_level_summary.csv`,
`surface_level_summary.csv`, `cefr_words_metrics_plot.png`, `cefr_sentence_word_metrics_plot.png`

---

### Lexical Features (CAMeL disambiguation)

```bash
export CAMEL_MORPH_DB=calima-msa-s31.db

python feature_extraction/cefr_sentence_camel_features.py \
  --input_csv <input_csv> \
  --levels_csv <levels_csv> \
  --out_dir <out_dir>
```

To skip CAMeL processing and reuse an existing sentence features CSV (e.g. to redo only essay/level aggregation):

```bash
python feature_extraction/cefr_sentence_camel_features.py \
  --sentence_csv <existing_sentence_features_camel.csv> \
  --levels_csv <levels_csv> \
  --out_dir <out_dir>
```

---

## Profile Filtering

We keep only **progression-sensitive** features — those that increase
monotonically across CEFR levels (A1 → C2).

### Select monotonic features

Takes the three feature summary CSVs and outputs a single filtered feature list:

```bash
python utils/profile_creator.py \
  --surface_csv <surface_csv> \
  --camel_csv <camel_csv> \
  --syntax_csv <syntax_csv> \
  --trend increasing \
  --out_csv <out_csv>
```

The output serves as the **reference profile** for evaluation.

## Vocabulary List Construction

### Step 1 — Generate candidate vocab per prompt (GPT)

```bash
export OPENAI_API_KEY=your_key_here

python vocabulary_construction/generate_prompts_vocabs.py \
  --input_csv <input_csv> \ # prompt with their level
  --out_jsonl <out_jsonl> \
  --out_csv <out_csv>
```

### Step 2 — Filter vocab using SAMER readability lexicon
```bash
python vocabulary_construction/gpt_tosamer.py \
  --input_jsonl <input_jsonl> \
  --samer_tsv <samer_tsv> \
  --out_dir <out_dir> \
  --levels 5   # or 3 or 5
```

### Step 3 — Extract Low vocab prompts
```bash
python vocabulary_construction/low_vocabs.py \
  --filtered_vocab_csv <filtered_vocab_csv> \
  --prompts_csv <prompts_csv> \
  --out_csv <out_csv> \
  --min_words 4
```

### Step 4 — Expand low-count lists using relevance model (optional)

```bash
python vocabulary_construction/vocab_relevance.py \
  --prompts_csv <prompts_csv> \
  --lexicon_path <lexicon_path> \
  --out_csv <out_csv> \
  --threshold 0.50 \
  --fallback_threshold 0.35 \
  --top_k 40 \
  --device auto
```

### Step 4 — Merge vocab sources

```bash
python vocabulary_construction/merge_vocabs.py \
  --gpt_csv <gpt_csv> \
  --relevant_csv <relevant_csv> \
  --out_csv <out_csv>
```

---

## Essay Generation (OpenAI Batch)

Essays are generated under 5 prompting conditions:

| ID | Description |
|----|-------------|
| P1 | Topic only (unconstrained baseline) |
| P2 | Topic + CEFR level |
| P3 | Topic + CEFR level + linguistic profile specs |
| P4 | Topic + CEFR level + vocabulary list |
| P5 | Topic + CEFR level + profile specs + vocabulary list |

### Step 1 — Create batch input

# P1 — no level, no vocab, no profile
python generation/build_batch.py \
  --prompts_csv <prompts_csv> --out_jsonl <out_jsonl> \
  --prompt_version P1 --custom_id_suffix 6levels_p1

# P2 — level only
python generation/build_batch.py \
  --prompts_csv <prompts_csv> --out_jsonl <out_jsonl> \
  --prompt_version P2 --custom_id_suffix 6levels_p2

# P3 — level + profile
python generation/build_batch.py \
  --prompts_csv <prompts_csv> --out_jsonl <out_jsonl> \
  --prompt_version P3 --profile_csv <profile_csv> --custom_id_suffix 6levels_p3

# P4 — level + vocab
python generation/build_batch.py \
  --prompts_csv <prompts_csv> --out_jsonl <out_jsonl> \
  --prompt_version P4 --vocab_csv <vocab_csv> --custom_id_suffix 6levels_p4

# P5 — level + profile + vocab
python generation/build_batch.py \
  --prompts_csv <prompts_csv> --out_jsonl <out_jsonl> \
  --prompt_version P5 --profile_csv <profile_csv> --vocab_csv <vocab_csv> \
  --custom_id_suffix 5levels_p5

### Step 2 — Upload and run batch
```bash
export OPENAI_API_KEY=your_key_here

python generation/upload_run.py \
  --batch_input <batch_input_jsonl> \
  --out_tsv <out_tsv>
```
---

## Evaluation Against Profiles

After generation, run the same feature extraction steps on the generated essays
(Steps 1–5 of [Linguistic Profiling](#linguistic-profiling)), then filter to the
selected feature set and evaluate:

### Filter

```bash
python utils/features_filter.py \
  --selected_features_csv <selected_features_csv> \ # profile
  --surface_csv <surface_csv> \
  --camel_csv <camel_csv> \
  --syntax_csv <syntax_csv> \
  --out_csv <out_csv>
```

### Evaluate
```bash
python utils/cosine_similarity.py \
  --reference_features_csv <reference_features_csv> \
  --system_features_csv <system_features_csv> \
  --out_dir <out_dir> \
  --include_readability
```