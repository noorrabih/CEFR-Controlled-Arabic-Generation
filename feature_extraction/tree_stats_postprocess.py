# merge all tree stats files into one
# /l/users/nour.rabih/arwi_data_corrected_essays/clean_essays_readability_avg.csv for levels
# new csv that merges each sentence , get the avg, max of each stat, and contains a list of all stats per sentence 
# merge with levels
import pandas as pd
import os
from glob import glob

# ========= 1. Load & merge all tree stats files =========
stats_dir = "/home/nour.rabih/arwi/readability_controlled_generation/generation/no_level/essays_parsed_Text_100/tree_stats"
readability_path = "/home/nour.rabih/arwi/readability_controlled_generation/generation/no_level/generated_essays_p4_readability.csv"


output_path = "/home/nour.rabih/arwi/readability_controlled_generation/generation/no_level/tree_graphs/tree_stats_aggregated_with_readability.csv"

directory =  "/home/nour.rabih/arwi/readability_controlled_generation/generation/no_level/tree_graphs"
all_files = glob(os.path.join(stats_dir, "batch_*.xlsx"))

dfs = []
for f in all_files:
    tmp = pd.read_excel(f)
    dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)

# ========= 2. Create essay_id from sentence_id =========
# df["essay_id"] = df["sentence_id"].astype(str).str.split("-").str[2]
df["essay_id"] = df["sentence_id"].astype(str).str.rsplit("-", n=1).str[0]
print(df["essay_id"].head())

# ========= 3. Aggregate stats per essay =========
# If you want ALL numeric stats, not just depth/breadth/max_branching_factor:
include_cols = ["depth", "breadth", "total_nodes", "max_branching_factor"]
stat_cols = [c for c in df.columns if c  in include_cols and pd.api.types.is_numeric_dtype(df[c])]

agg_funcs = {col: ["mean", "max", list] for col in stat_cols}

grouped = df.groupby("essay_id").agg(agg_funcs)

# Flatten MultiIndex columns
grouped.columns = [f"{c[0]}_{c[1]}" for c in grouped.columns.to_flat_index()]
grouped = grouped.reset_index()

# Convert list columns to comma-separated strings for CSV compatibility
for col in grouped.columns:
    if col.endswith("_list"):
        grouped[col] = grouped[col].apply(lambda x: ",".join(str(v) for v in x))

# ========= 4. Merge with readability levels (CEFR, avg_readability_level) =========
read_df = pd.read_csv(readability_path)

# Make sure both keys have the SAME dtype (string)
grouped["essay_id"] = grouped["essay_id"].astype(str)
read_df["Document_ID"] = read_df["ID"].astype(str)
# id: 1_A1_1  i want CEFR to be A1
if 'CEFR' not in read_df.columns:
    read_df['CEFR'] =  read_df['ID'].str.split('_').str[1]

merged = grouped.merge(
    read_df[["Document_ID", "CEFR"]], # avg_level
    left_on="essay_id",
    right_on="Document_ID",
    how="left",
)

merged = merged.drop(columns=["Document_ID"])

# ========= 5. Save =========
merged.to_csv(output_path, index=False)
print("Saved to:", output_path)
print(merged.head())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# output_path = "/home/nour.rabih/arwi/readability_controlled_generation/ZAEBUC-v1.01/tree_graphs/tree_stats_aggregated_with_readability.csv"
# directory =  "/home/nour.rabih/arwi/readability_controlled_generation/ZAEBUC-v1.01/tree_graphs"
# # bea = pd.read_csv("/home/nour.rabih/arwi/readability_controlled_generation/arabic-aes-bea25/Data/tree_graphs/tree_stats_aggregated_with_readability.csv")
# zaebuc = pd.read_csv("/home/nour.rabih/arwi/readability_controlled_generation/ZAEBUC-v1.01/tree_graphs/tree_stats_aggregated_with_readability.csv")
# zaebuc = zaebuc[zaebuc['CEFR'] != 'Unassessable']
# df = zaebuc.copy()
# df = pd.concat([bea, zaebuc], ignore_index=True)
# df.to_csv(output_path, index=False)
# Load data
df = merged

# Round depth_mean
df['depth_mean_rounded'] = df['depth_mean'].round(0).astype(int)

# Pivot
pivot = df.pivot_table(
    index='depth_mean_rounded',
    columns='CEFR',
    aggfunc='size',
    fill_value=0
)

plt.figure(figsize=(10,7))

# Use the same colormap as your example: YlGnBu
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")

plt.title("Relation between depth_mean and CEFR")
plt.xlabel("CEFR")
plt.ylabel("depth_mean (rounded)")

plt.tight_layout()
plt.show()

plt.savefig(f"{directory}/depth_mean_vs_CEFR_heatmap.png")


# Pivot depth_max
pivot = df.pivot_table(
    index='depth_max',
    columns='CEFR',
    aggfunc='size',
    fill_value=0
)

plt.figure(figsize=(10,7))

# Use the same colormap as your example: YlGnBu
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")

plt.title("Relation between depth_max and CEFR")
plt.xlabel("CEFR")
plt.ylabel("depth_max")

plt.tight_layout()
plt.show()

plt.savefig(f"{directory}/depth_max_vs_CEFR_heatmap.png")


# Pivot breadth_mean

df['breadth_mean_rounded'] = df['breadth_mean'].round(0).astype(int)

pivot = df.pivot_table(
    index='breadth_mean_rounded',
    columns='CEFR',
    aggfunc='size',
    fill_value=0
)

plt.figure(figsize=(10,7))

# Use the same colormap as your example: YlGnBu
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")

plt.title("Relation between breadth_mean and CEFR")
plt.xlabel("CEFR")
plt.ylabel("breadth_mean")

plt.tight_layout()
plt.show()

plt.savefig(f"{directory}/breadth_mean_vs_CEFR_heatmap.png")

# Pivot breadth_max
pivot = df.pivot_table(
    index='breadth_max',
    columns='CEFR',
    aggfunc='size',
    fill_value=0
)

plt.figure(figsize=(10,7))

# Use the same colormap as your example: YlGnBu
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")

plt.title("Relation between breadth_max and CEFR")
plt.xlabel("CEFR")
plt.ylabel("breadth_max")

plt.tight_layout()
plt.show()

plt.savefig(f"{directory}/breadth_max_vs_CEFR_heatmap.png")



# Round total_nodes_mean
df['total_nodes_mean_rounded'] = df['total_nodes_mean'].round(0).astype(int)

# Pivot
pivot = df.pivot_table(
    index='total_nodes_mean_rounded',
    columns='CEFR',
    aggfunc='size',
    fill_value=0
)

plt.figure(figsize=(10,7))

# Use the same colormap as your example: YlGnBu
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")

plt.title("Relation between total_nodes_mean and CEFR")
plt.xlabel("CEFR")
plt.ylabel("total_nodes_mean (rounded)")

plt.tight_layout()
plt.show()

plt.savefig(f"{directory}/total_nodes_mean_vs_CEFR_heatmap.png")

# Pivot total_nodes_max
pivot = df.pivot_table(
    index='total_nodes_max',
    columns='CEFR',
    aggfunc='size',
    fill_value=0
)

plt.figure(figsize=(10,7))

# Use the same colormap as your example: YlGnBu
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")

plt.title("Relation between total_nodes_max and CEFR")
plt.xlabel("CEFR")
plt.ylabel("total_nodes_max")

plt.tight_layout()
plt.show()

plt.savefig(f"{directory}/total_nodes_max_vs_CEFR_heatmap.png")