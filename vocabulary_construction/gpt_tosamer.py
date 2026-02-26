from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.tokenizers.word import simple_word_tokenize
import json
import pandas as pd

directory = '/home/nour.rabih/arwi/vocab/3levels/'

vocab_path = f'{directory}full_vocab_prompts_3levels.jsonl'
gpt_vocab_path = f'{directory}full_vocab_prompts_3levels.jsonl'


unique_vocab_output_path = f'{directory}lowvocab_gpt_vocab_responses_unique.csv'
disambig_output_path = f'{directory}lowvocab_gpt_vocab_responses_unique_disambig.csv'
filtered_vocab_output_path = f'{directory}lowvocab_gpt_vocab_responses_filtered_by_samer_5levels_withpos_gloss.csv'
prompts_path = '/home/nour.rabih/arwi/generation/prompts.csv'
low_words_output_path = f'{directory}lowvocab_prompts.csv'

# level_mapping = {'A1': 1, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5} # 5 loevels
level_mapping = {'A': 1, 'B': [2,3], 'C': [4,5]} # 3 levels


samer_path = 'SAMER_Readability_Levels_camelmorph.tsv'
# db = MorphologyDB("/home/nour.rabih/Readability-morph/camel_morph_msa_v1.0.db", "a")

# # Initialize the analyzer
# analyzer = Analyzer(db, 'NONE', cache_size=100000)

# # # Load the pretrained BERT model
# bert = BERTUnfactoredDisambiguator.pretrained(model_name='msa', pretrained_cache=False)

# # # Set the analyzer
# bert._analyzer = analyzer

# def disambiguate_sentence(sentence_text):
#     # Tokenize the sentence
#     sentence = simple_word_tokenize(sentence_text)
    
#     # Tag and disambiguate the sentence
#     disambig = bert.tag_sentence(sentence)
    
#     return disambig

# data = []
# with open(vocab_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         if line.strip():  # skip empty lines
#             data.append(json.loads(line))

# print(f"Loaded {len(data)} entries.")

# # create a csv with columns: prompt_id, topic_id, level, words.
# # words is a list so create a row for each word with the same prompt_id, topic_id, level.
# rows = []
# for entry in data:
#     for word in entry['words']:
#         rows.append({'word': word})

# #keep only unique words
# unique_words = list(set([row['word'] for row in rows]))
# print(f"Unique words: {len(unique_words)}")
# rows = [{'word': word} for word in unique_words]
# df = pd.DataFrame(rows)
# df.to_csv(unique_vocab_output_path, index=False)

# # run on unique words only and map back to dataframe
# from functools import lru_cache

# # --- helpers ---
# _DIACRITICS = {'َ','ِ','ُ','ً','ٍ','ٌ','ْ','ّ'}

# #remove diacritics from the end of the lemma
# def clean_lemma(lemma: str) -> str:
#     return lemma[:-1] if lemma and lemma[-1] in _DIACRITICS else lemma

# @lru_cache(maxsize=None)
# def analyze_token(token: str):
#     """Run disambiguation (+ possible re-analysis) once per unique token."""
#     disambig = disambiguate_sentence(token)
#     if not disambig:
#         return ('N/A', 'N/A', 'N/A')

#     best = disambig[0]
#     pos = best.get('pos', 'N/A')

#     # Re-analyze special POS
#     if pos in ('noun_prop', 'pron_rel'):
#         analyses = analyzer.analyze(token)
#         noun_analyses = [a for a in analyses if a.get('pos') == 'noun']
#         adj_analyses  = [a for a in analyses if a.get('pos') == 'adj']
#         if noun_analyses:
#             best = noun_analyses[0]
#         elif pos == 'noun_prop' and adj_analyses:
#             best = adj_analyses[0]
#         pos = best.get('pos', pos)

#     lemma = clean_lemma(best.get('lex', 'N/A'))
#     stemgloss = best.get('stemgloss', 'N/A')
#     return (lemma or 'N/A', pos or 'N/A', stemgloss or 'N/A')

# # --- main function ---
# def disambig_df(df_in: pd.DataFrame) -> pd.DataFrame:
#     if 'word' not in df_in.columns:
#         raise ValueError("Input DataFrame must contain a 'word' column.")

#     # 1️⃣ Split multi-word phrases -> tokens, keep original phrase
#     df = df_in.copy()
#     df['subwords'] = df['word'].astype(str).str.split()
#     df = df.explode('subwords', ignore_index=True).rename(columns={'subwords': 'subword'})
#     df['subword'] = df['subword'].str.strip()
#     df = df[df['subword'].ne('')]

#     # 2️⃣ Analyze each unique token once, map back to df
#     uniq = df['subword'].unique()
#     lookup = {tok: analyze_token(tok) for tok in uniq}

#     # Convert mapping dict to a DataFrame for clean merge
#     df_lookup = (
#         pd.DataFrame.from_dict(lookup, orient='index', columns=['lemma', 'pos', 'stemgloss'])
#         .reset_index()
#         .rename(columns={'index': 'subword'})
#     )

#     # Merge back
#     df = df.merge(df_lookup, on='subword', how='left')

#     return df[['word', 'subword', 'lemma', 'pos', 'stemgloss']]

# # --- usage ---
# df_results = disambig_df(df)

# # Save
# df_results.to_csv(disambig_output_path, index=False)


# # if last char is diacritic, remove it in lex_CM
# import pandas as pd
# samer_df = pd.read_csv(samer_path, sep='\t')    

# # use clean_lemma on lex_CM
# samer_df['lex_CM'] = samer_df['lex_CM'].apply(clean_lemma)

# # Get the matching ones between df_results and samer_df, add readability if available
# # we're matching on lemma and pos
# merged_df = pd.merge(df_results, samer_df[['lex_CM', 'pos_CM', 'stemgloss_CM', 'readability (rounded average)']], left_on=['lemma', 'pos', 'stemgloss'], right_on=['lex_CM', 'pos_CM', 'stemgloss_CM'], how='left')
# merged_df = merged_df.drop(columns=['lex_CM', 'pos_CM'])
# merged_df = merged_df.rename(columns={'readability (rounded average)': 'samer_readability_level'})
# merged_df['samer_readability_level'] = merged_df['samer_readability_level'].fillna(5)
# merged_df.to_csv(f'{directory}gpt_vocab_responses_unique_disambig_samer_readability.csv', index=False)
merged_df = pd.read_csv(f'{directory}gpt_vocab_responses_unique_disambig_samer_readability.csv')
# multi word treatment 5 levels
# open /Users/nour.rabih/Desktop/arwi/gpt_vocab_responses.json
gpt_vocab = pd.read_json(gpt_vocab_path, lines=True)

# map gpt_vocab A1 -> 1, A2 -> 1, B1 -> 2, B2 -> 3, C1 -> 4, C2 -> 5

gpt_vocab['level_num'] = gpt_vocab['level'].map(level_mapping)
new_rows = []


for row in gpt_vocab.itertuples():
    prompt_id = row.prompt_id
    topic_id = row.topic_id
    level = row.level_num
    words = row.words
    matched_words = []

    for word in words:
        subwords = word.split(' ') if ' ' in word else [word]

        # case 1: multi-word phrase
        if len(subwords) > 1:
            levels = []
            for subword in subwords:
                # print("Checking subword:", subword)
                match = merged_df[(merged_df['subword'] == subword) & (merged_df['word'] == word)]
                if not match.empty:
                    length = len(match['samer_readability_level'])
                    if length > 1 :
                        print('more than one match')
                        print(subword)

                    samer_level = match['samer_readability_level'].values[0]
                    if pd.isna(samer_level):
                        print("Samer level is NaN, assigning level 5")
                        samer_level = level_mapping['C']
                    # print("Samer level:", samer_level)
                    levels.append(samer_level)

            if levels:
                max_level = max(levels)
                # print("Max level:", max_level)
                # check if CEFR level matches the highest Samer level
                if isinstance(level, list):
                    if max_level in level:
                        matched_words.append((word, match['pos'].values[0], match['stemgloss_CM'].values[0]) )  # add the phrase as a whole
                else:
                    if level == max_level:
                        matched_words.append((word, match['pos'].values[0], match['stemgloss_CM'].values[0]) )  # add the phrase as a whole

        # case 2: single word
        else:
            match = merged_df[merged_df['subword'] == word]
            if not match.empty:
                samer_level = match['samer_readability_level'].values[0]
                # barec_level = match['barec_readability_level'].values[0]
                # if level is list (e.g., B -> [2,3]), check if samer_level in that list
                if isinstance(level, list):
                    if samer_level in level:
                        matched_words.append((word, match['pos'].values[0], match['stemgloss_CM'].values[0]))
                else:
                    if level == samer_level:
                        matched_words.append((word, match['pos'].values[0], match['stemgloss_CM'].values[0]))

    new_rows.append({
        'prompt_id': prompt_id,
        'topic_id': topic_id,
        'CEFR_level': row.level,
        'level': level,
        'words': matched_words
    })

df = pd.DataFrame(new_rows)
df.to_csv(filtered_vocab_output_path, index=False)


#get the lengths of words in new_rows
lengths = [len(row['words']) for row in new_rows]

# count the frequency of each length
from collections import Counter
length_counts = Counter(lengths)
# sort by length
length_counts = dict(sorted(length_counts.items()))
print("Word count distribution after filtering by Samer readability levels:")
print(length_counts)

import pandas as pd
import ast

# those that have 0, look into their words
prompt_vocabs = pd.read_csv(filtered_vocab_output_path)
s = prompt_vocabs['words'][0]
words = ast.literal_eval(s)

# import pdb; pdb.set_trace()
# read row['words'] as a list
import re

_nan_token = re.compile(r'(?<![\w])nan(?![\w])', flags=re.IGNORECASE)

def parse_words_cell(s):
    s = str(s).strip()
    # Replace bare nan with None so it's a valid Python literal
    s = _nan_token.sub("None", s)
    return ast.literal_eval(s)

prompt_vocabs["words"] = prompt_vocabs["words"].apply(parse_words_cell)

#get the lengths of words in new_rows
lengths = [len(row['words']) for index, row in prompt_vocabs.iterrows()]

# count the frequency of each length
from collections import Counter
length_counts = Counter(lengths)
# sort by length
length_counts = dict(sorted(length_counts.items()))
length_counts

#get all with 0, 1, 2, 3, 4  words
zero_word_prompts = prompt_vocabs[prompt_vocabs['words'].apply(len).isin([0, 1, 2, ])] #3, 4

# merge it with prompt on prompt_id CEFR_level
prompts = pd.read_csv(prompts_path)

low_words = pd.merge(zero_word_prompts, prompts, on=['prompt_id'], how='left')
low_words.to_csv(low_words_output_path)