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

We predict the **BAREC readability level** for the processed essays.

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
CSV file with predicted BAREC readability levels

---

## 3. corelation
We calculate the corelation between Taha-19 levels and CEFR
    utils/barec_corelation.py

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

1. Create profiles:
    profile_creator.py
    - input: feature summaries from zaebuc and arwi combined
    - output:profiles of the selected features + visualization plots found in profiling_data/ZAEBUC+ARWI/6levels

# vocabulary list construction:
for every prompt we create a list of vocabs that are suitable for that level.

1. Generate vocabs list per prompt
   - vocabulary_construction/generate_prompts_vocabs.py

2. map and filter the generated words to their corresponding SAMER level
   - vocabulary_construction/gpt_tosamer.py

3. for those with low vocabs count, use vocab relevance model to get more vocabs
   - vocabulary_construction/vocab-relavance.py

4. merge both from 2 and 3 
   - vocabulary_construction/merge_vocabs.py

# Generation:
We prompt gpt-4o and ask it to generate essays of topics from the ARWI dataset. those topics are categorized into Begginer, Intermediate, and Advanced.

prompts are set up with 3, 5 and 6 levels
 
3 levels:  Begginer ->A, Intermediate -> B, and Advanced -> C
5 levels: Begginer ->A, Intermediate -> B1, B2 , Advanced -> C1, C2
6 levels: Begginer -> A1, A2 , Intermediate -> B1, B2 , Advanced -> C1, C2


We use batch mode. 

1. batch data 
    Generation/batch_creation.py
2. upload and run
    Generation/upload_run.py

We generate under 5 conditions:
P1: No given level, given just the topic.
The evaluation of this one is done on 3 levels.

P2: given topic and level
The evaluation of this one is done on 6 levels.

P3: given topic, level and profile specifications based on the level.
The evaluation of this one is done on 6 levels.

P4: given topic, level, and vocabs list.
The evaluation of this one is done on 5 levels.
Details on how the Vocabs list formulation below

P5: given topic, level, profile specifications, and vocabs list.
The evaluation of this one is done on 5 levels.

After generation we extract the features just like we did in Feature Extraction section. 
then we filter based on the chosen ones in the profile:
    - utils/features_filter.py

Then we evaluate against the profiles:
    - utils/cosine_similarity.py

