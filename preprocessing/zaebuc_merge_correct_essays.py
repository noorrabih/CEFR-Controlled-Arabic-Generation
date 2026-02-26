# merges the corrected essays from zaebuc into one row per essay
import pandas as pd

# read in the data
df = pd.read_csv('/home/nour.rabih/arwi/readability_controlled_generation/ZAEBUC-v1.01/AR-all.alignment-FINAL.tsv', sep='\t')

# group by text
# grouped = df.groupby('Document')

# get the raw cell and group it by Document and name it Raw and join the list to a string
# ignore float errors
df['Corrected'] = df['Corrected'].astype(str)
grouped = df.groupby('Document')['Corrected'].apply(lambda x: ' '.join(x)).reset_index()

# save to csv
# grouped.to_csv('/mnt/data/users/nour.rabih/barec/ZAEBUC-v1.0/full_texts.tsv', sep='\t', index=False)

ZAEBUC_AR_COR = pd.read_csv('/home/nour.rabih/arwi/readability_controlled_generation/ZAEBUC-v1.01/AR-all.extracted.corrected.analyzed.corrected-FINAL.tsv', encoding='utf_8',sep='\t')

import xmltodict
import numpy as np
docs = ZAEBUC_AR_COR['Document'].apply(lambda x: x if x.startswith('<') else np.nan).dropna()

grades = []
word_count = []

for xml in docs:
    if xml != "</doc>":
        doc = xmltodict.parse(xml)
        grades.append(doc["doc"]["@CEFR"])
        word_count.append(doc["doc"]["@word_count"])


grouped['CEFR'] = grades
grouped.to_csv('/home/nour.rabih/arwi/readability_controlled_generation/ZAEBUC-v1.01/corrected_full_texts.tsv', sep='\t', index=False)

# to do remove the nan words