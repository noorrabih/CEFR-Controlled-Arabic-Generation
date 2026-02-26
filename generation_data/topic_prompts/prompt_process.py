import pandas as pd

# Load your CSV
df = pd.read_csv("/home/nour.rabih/arwi/generation/prompts.csv")

# Mapping of Level → CEFR levels (list)
mapping = {
    "Beginner": ["A"],
    "Intermediate": ["B"],
    "Advanced": ["C"]
}

# Expand rows
expanded_rows = []

for _, row in df.iterrows():
    level = row["Level"]
    if level in mapping:
        for cefr in mapping[level]:
            new_row = row.copy()
            new_row["CEFR_level"] = cefr
            expanded_rows.append(new_row)
    else:
        # In case there are unexpected values
        new_row = row.copy()
        new_row["CEFR_level"] = None
        expanded_rows.append(new_row)

# Create new dataframe
df_expanded = pd.DataFrame(expanded_rows)

# Save to new CSV
df_expanded.to_csv("prompts_with_3cefr_levels.csv", index=False)

df_expanded.head()
