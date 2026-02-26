# open /home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/selected_readability_monotonic_features.csv
import pandas as pd

selected_features_df = pd.read_csv("/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/selected_readability_monotonic_features.csv")

print("Selected features:")
print(len(selected_features_df.columns))
directory = "/home/nour.rabih/arwi/readability_controlled_generation/generation/vocabs_prompt/5levels"
# open words /home/nour.rabih/arwi/readability_controlled_generation/generation/plain_prompt/6levels/surface_feats/level_summary.csv
words_df = pd.read_csv(f"{directory}/surface_feats/level_summary.csv")

# open camel /home/nour.rabih/arwi/readability_controlled_generation/generation/plain_prompt/6levels/surface_feats/level_features_camel.csv
camel_df = pd.read_csv(f"{directory}/surface_feats/level_features_camel.csv")

# syntax  /home/nour.rabih/arwi/readability_controlled_generation/generation/plain_prompt/6levels/syntax/level_syntax_stats.csv
# syntax_df = pd.read_csv(f"{directory}/syntax/level_syntax_stats.csv")
syntax_df = pd.read_csv(f"{directory}/syntax/level_syntax_stats.csv")
# import pdb; pdb.set_trace()
# merge all dataframes on 'level' column
merged_df = words_df.merge(camel_df, on='level').merge(syntax_df, on='level')

# filter merged_df to keep only columns in selected_features_df.columns
# if column not in index assign it nan

filtered_df = merged_df[[col for col in selected_features_df.columns if col in merged_df.columns]]
print("Filtered features:")
print(len(filtered_df.columns))
# if any columns are missing, assign them nan values
for col in selected_features_df.columns:
    if col not in filtered_df.columns:
        filtered_df[col] = float('nan')

# filtered_df = merged_df[selected_features_df.columns]


print("Filtered features:")
print(len(filtered_df.columns))

filtered_df.to_csv(f"{directory}/filtered_monotonic_features.csv", index=False)
# "['CEFR', 'depPOS_SBJ(CCONJ)_mean', 'depPOS_PRD(X)_mean', 'depPOS_IDF(PART)_sum', 'depPOS_OBJ(INTJ)_sum', 'depPOS_OBJ(PUNCT)_sum', 'depPOS_TPC(NUM)_sum', 'depPOS_PRD(X)_sum']