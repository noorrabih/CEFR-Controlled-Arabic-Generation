import pandas as pd
from scipy.stats import spearmanr, pearsonr

# Load your data
bea = pd.read_csv("/home/nour.rabih/arwi/readability_controlled_generation/bea_data/arwi_essays_readability.csv")
zaebuc = pd.read_csv("/home/nour.rabih/arwi/readability_controlled_generation/ZAEBUC-v1.01/corrected_essays_readability.csv")
df = pd.concat([bea, zaebuc], ignore_index=True)
# Encode CEFR levels as ordered integers
cefr_mapping = {
    "A1": 1,
    "A2": 2,
    "B1": 3,
    "B2": 4,
    "C1": 5,
    "C2": 6
}

df["CEFR_numeric"] = df["CEFR"].map(cefr_mapping)

# Drop any rows with missing values (just in case)
df = df.dropna(subset=["CEFR_numeric", "avg_level"])

# Compute Spearman correlation (recommended)
spearman_corr, spearman_p = spearmanr(df["CEFR_numeric"], df["avg_level"])

# Compute Pearson correlation (optional)
pearson_corr, pearson_p = pearsonr(df["CEFR_numeric"], df["avg_level"])

print("Spearman correlation (recommended):")
print(f"ρ = {spearman_corr:.4f}, p-value = {spearman_p:.6f}")

print("\nPearson correlation:")
print(f"r = {pearson_corr:.4f}, p-value = {pearson_p:.6f}")
