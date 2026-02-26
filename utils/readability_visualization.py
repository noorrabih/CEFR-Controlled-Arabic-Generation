import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
# file_path = "/home/nour.rabih/arwi/readability_controlled_generation/arabic-aes-bea25/Data/arwi_essays_readability.csv"
# df1 = pd.read_csv(file_path)

file_path2 = "/home/nour.rabih/arwi/readability_controlled_generation/generation/syntax_vocab_prompt/5levels/generated_essays_readability.csv"

df2 = pd.read_csv(file_path2)
# remove CEFR unassessable
df2 = df2[df2['CEFR'] != 'Unassessable']
# df = pd.concat([df1, df2], ignore_index=True)
df = df2.copy()
# Extract CEFR (e.g., 1_A1_1 → A1)
# df['CEFR'] = df.iloc[:, 0].str.extract(r'_(A1|A2|B1|B2|C1|C2)_')[0]

# Drop missing values
df = df.dropna(subset=['avg_level', 'CEFR'])

# Round avg_prediction
df['avg_prediction_rounded'] = df['avg_level'].round().astype(int)
# Create crosstab
ct = pd.crosstab(df['avg_prediction_rounded'], df['CEFR']).fillna(0).astype(int)

# Plot heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Count'}, linewidths=0.3)
# plt.title('Relation between average readability and CEFR')
plt.xlabel('CEFR')
plt.ylabel('avg_readability')

# Save heatmap
out_path = "/home/nour.rabih/arwi/readability_controlled_generation/generation/syntax_vocab_prompt/5levels/generated_essays_readability_vs_CEFR_heatmap_mean.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()

# do the same with max_level
df['max_prediction_rounded'] = df['max_level'].round().astype(int)
ct_max = pd.crosstab(df['max_prediction_rounded'], df['CEFR']).fillna(0).astype(int)    
plt.figure(figsize=(10, 7))
sns.heatmap(ct_max, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Count'}, linewidths=0.3)
# plt.title('Relation between max readability and CEFR')
plt.xlabel('CEFR')
plt.ylabel('max_readability')

# Save heatmap
out_path_max = "/home/nour.rabih/arwi/readability_controlled_generation/generation/syntax_vocab_prompt/5levels/generated_essays_readability_vs_CEFR_heatmap_max.png"
plt.savefig(out_path_max, dpi=200, bbox_inches='tight')
plt.close()