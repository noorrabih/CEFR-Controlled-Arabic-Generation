# Data Sources

In this work, we leverage two learner corpora:

- **ZAEBUC**
- **ARWI**

Both datasets are preprocessed before running the experiments.

---

## ZAEBUC

In ZAEBUC, corrected essays are presented **one word per row**.  
Therefore, we merge them into full essays before further processing.

### Preprocessing

**Script:**  
`preprocessing/zaebuc_merge_correct_essays.py`

**Functionality:**
- Merges corrected essays into full essays  
- Outputs: `full_texts.tsv`

---

## ARWI

The ARWI dataset is already formatted as full essays and does not require merging.

---

# Readability Alignment

We predict the **BAREC (Taha-19) readability level** for the processed essays.

---

## 1. Readability Preprocessing (D3 Tokenization)

**Script:**  
`utils/essays_d3tok.py`

**Input:**  
`full_texts.tsv`

**Functionality:**
- Splits essays into sentences
- Applies D3 tokenization
- Computes word counts

**Output:**  
CSV file containing:
- Essay text
- D3-tokenized text
- CEFR level
- Word count

---

## 2. Readability Prediction (BAREC Model)

**Script:**  
`utils/add_readability_levels.py`

**Input:**  
CSV file with D3 tokenization

**Output:**  
CSV file with predicted BAREC (Taha-19) readability levels

---

## 3. Correlation Analysis

We calculate the correlation between Taha-19 levels and CEFR levels.

**Script:**  
`utils/barec_correlation.py`

---

## 4. Visualization

**Script:**  
`utils/readability_visualization.py`

**Input:**  
CSV file containing CEFR and BAREC levels

**Output:**  
Heatmap visualization showing alignment between CEFR and BAREC levels

---

# Profiling

We create linguistic profiles for each CEFR level based on the processed data.

Each profile consists of:
- Syntactic features
- Lexical features
- Surface-level features

---

## Syntactic Features

### Preprocessing

1. Sentence splitting  
   `utils/split_sentences.py`

2. Dependency parsing  
   `feature_extraction/tree_parsing.py`

---

### Feature Extraction

1. POS tags, dependency relations, and Dep(POS) combinations  
   Extracted using CamelParser  
   `feature_extraction/syntactic_features.py`

2. Tree depth statistics  
   - `feature_extraction/tree_stats.py`  
   - `feature_extraction/tree_stats_postprocess.py`

---

## Lexical and Surface-Level Features

1. Surface-level features  
   `feature_extraction/surface_feats.py`

2. Features requiring morphological disambiguation  
   `feature_extraction/surface_disambig_features.py`

---

# Profile Filtering

We filter features based on their trends across CEFR levels to retain progression-sensitive and discriminative features.

## 1. Profile Creation

**Script:**  
`profile_creator.py`

**Input:**  
Feature summaries from ZAEBUC and ARWI combined

**Output:**  
- Profiles of selected features  
- Visualization plots located in:  
  `profiling_data/ZAEBUC+ARWI/6levels`

---

# Vocabulary List Construction

For each prompt, we construct a vocabulary list suitable for the target level.

## 1. Generate vocabulary list per prompt  
`vocabulary_construction/generate_prompts_vocabs.py`

## 2. Map and filter generated words to their corresponding SAMER level  
`vocabulary_construction/gpt_tosamer.py`

## 3. For prompts with low vocabulary count, use a vocabulary relevance model to expand the list  
`vocabulary_construction/vocab_relevance.py`

## 4. Merge outputs from steps 2 and 3  
`vocabulary_construction/merge_vocabs.py`

---

# Generation

We prompt GPT-4o to generate essays using topics from the ARWI dataset.  
The topics are categorized into:

- Beginner
- Intermediate
- Advanced

Prompts are structured with different level granularities:

### 3 Levels
- Beginner → A  
- Intermediate → B  
- Advanced → C  

### 5 Levels
- Beginner → A  
- Intermediate → B1, B2  
- Advanced → C1, C2  

### 6 Levels
- Beginner → A1, A2  
- Intermediate → B1, B2  
- Advanced → C1, C2  

We use batch mode for generation.

## 1. Batch creation  
`Generation/batch_creation.py`

## 2. Upload and run  
`Generation/upload_run.py`

---

# Generation Conditions

We generate essays under five conditions:

### P1
Given topic only (no level specified).  
Evaluation: 3 levels.

### P2
Given topic and level.  
Evaluation: 6 levels.

### P3
Given topic, level, and profile specifications based on the level.  
Evaluation: 6 levels.

### P4
Given topic, level, and vocabulary list.  
Evaluation: 5 levels.

### P5
Given topic, level, profile specifications, and vocabulary list.  
Evaluation: 5 levels.

---

# Post-Generation Processing

After generation:

1. We extract linguistic features (same pipeline as Feature Extraction section).
2. We filter features based on the selected profile features:  
   `utils/features_filter.py`
3. We evaluate generated essays against CEFR profiles using cosine similarity:  
   `utils/cosine_similarity.py`