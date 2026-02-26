# Fix: Use independent scalers per cluster and per feature set

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# (optional, but nice) for Spearman correlation
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# reference_df
# -----------------------------
reference_df = pd.read_csv('/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/5levels/selected_readability_monotonic_features.csv')

average_readability_df = pd.read_csv('/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/5levels/average_barec_level_by_CEFR.csv')
average_depth_df = pd.read_csv('/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/5levels/average_depth_by_CEFR.csv')

# Safety checks to avoid duplicate merges
assert average_readability_df["CEFR"].is_unique, "Reference avg_level file has duplicate CEFR rows."
assert average_depth_df["CEFR"].is_unique, "Reference depth file has duplicate CEFR rows."

reference_df = reference_df.merge(average_readability_df, on="CEFR", how="left")
reference_df = reference_df.merge(average_depth_df, on="CEFR", how="left")
assert reference_df["CEFR"].is_unique, "Reference DF has duplicate CEFR rows after merge."


# -----------------------------
# system_df
# -----------------------------
directory = '/home/nour.rabih/arwi/readability_controlled_generation/generation/vocabs_prompt/5levels/'
system_df = pd.read_csv(f'{directory}filtered_monotonic_features.csv')

system_average_readability_df = pd.read_csv(f'{directory}average_barec_level_by_CEFR.csv')
system_average_depth_df = pd.read_csv(f'{directory}average_depth_by_CEFR.csv')

if "CEFR" not in system_average_readability_df.columns:
    system_average_readability_df.rename(columns={system_average_readability_df['level']: "CEFR"}, inplace=True)
assert system_average_readability_df["CEFR"].is_unique, "System avg_level file has duplicate CEFR rows."
assert system_average_depth_df["CEFR"].is_unique, "System depth file has duplicate CEFR rows."

system_df = system_df.merge(system_average_readability_df, on="CEFR", how="left")
system_df = system_df.merge(system_average_depth_df, on="CEFR", how="left")
assert system_df["CEFR"].is_unique, "System DF has duplicate CEFR rows after merge."


# -----------------------------
# Feature clusters
# -----------------------------
SURFACE = [
    "total_words","avg_words_per_sentence","total_unique_words","avg_unique_words",
    "overall_avg_word_len","overall_max_word_len","avg_syllables_per_word",
    "max_max_syllables","avg_unique_lemmas","avg_unique_lemma_pos","avg_camel_tokens"
]

DEP = [
    "dep_IDF_mean","dep_MOD_mean","dep_OBJ_mean","dep_SBJ_mean"
]

POS = [
    "pos_ADJ_mean","pos_ADP_mean","pos_AUX_mean","pos_CCONJ_mean",
    "pos_DET_mean","pos_NOUN_mean","pos_PUNCT_mean","pos_VERB_mean"
]

DEPPOS = [
    "depPOS_IDF(NOUN)_mean","depPOS_MOD(ADJ)_mean","depPOS_MOD(ADP)_mean",
    "depPOS_MOD(AUX)_mean","depPOS_MOD(CCONJ)_mean","depPOS_MOD(DET)_mean",
    "depPOS_MOD(NOUN)_mean","depPOS_MOD(PRON)_mean","depPOS_MOD(PROPN)_mean",
    "depPOS_MOD(PUNCT)_mean","depPOS_MOD(VERB)_mean","depPOS_OBJ(ADJ)_mean",
    "depPOS_OBJ(ADV)_mean","depPOS_OBJ(DET)_mean","depPOS_OBJ(NOUN)_mean",
    "depPOS_SBJ(NOUN)_mean","depPOS_SBJ(SCONJ)_mean", "depth_mean"
]

READABILITY = ["avg_level"]

ALL_FEATURES = SURFACE + DEP + POS + DEPPOS  + READABILITY


# -----------------------------
# Align & sort (IMPORTANT)
# -----------------------------
# Ensure both sets contain the same CEFR labels and align rows exactly
assert set(reference_df["CEFR"]) == set(system_df["CEFR"]), "Reference/System CEFR sets do not match."

reference_df = reference_df[["CEFR"] + ALL_FEATURES].sort_values("CEFR").reset_index(drop=True)
system_df = system_df[["CEFR"] + ALL_FEATURES].set_index("CEFR").loc[reference_df["CEFR"]].reset_index()


# -------------------------
# Overall cosine similarity (all features)
# -------------------------
scaler_all = StandardScaler()
ref_all = scaler_all.fit_transform(reference_df[ALL_FEATURES])
sys_all = scaler_all.transform(system_df[ALL_FEATURES])

overall_similarity = cosine_similarity(ref_all.reshape(1,-1), sys_all.reshape(1,-1))[0][0]
overall_df = pd.DataFrame({"Metric": ["Overall Cosine Similarity"], "Value": [overall_similarity]})


# -------------------------
# Cluster cosine similarity (overall)
# -------------------------
def cluster_cosine_similarity(features):
    scaler = StandardScaler()
    ref = scaler.fit_transform(reference_df[features])
    sys = scaler.transform(system_df[features])
    return cosine_similarity(ref.reshape(1,-1), sys.reshape(1,-1))[0][0]

cluster_df = pd.DataFrame({
    "Cluster": ["Surface","Dependency","POS","DepPOS"],
    "Cosine Similarity": [
        cluster_cosine_similarity(SURFACE),
        cluster_cosine_similarity(DEP),
        cluster_cosine_similarity(POS),
        cluster_cosine_similarity(DEPPOS),
        # cluster_cosine_similarity(READABILITY)
    ]
})


# -------------------------
# Per level cosine similarity (all features)
# -------------------------
level_similarity = []
for i, level in enumerate(reference_df["CEFR"]):
    sim = cosine_similarity(ref_all[i].reshape(1,-1), sys_all[i].reshape(1,-1))[0][0]
    level_similarity.append((level, sim))

level_df = pd.DataFrame(level_similarity, columns=["CEFR Level","Cosine Similarity"])


# -------------------------
# Per feature cosine similarity (vector across levels)
# -------------------------
feature_similarity = []
for feature in ALL_FEATURES:
    scaler = StandardScaler()
    ref = scaler.fit_transform(reference_df[[feature]])
    sys = scaler.transform(system_df[[feature]])
    sim = cosine_similarity(ref.reshape(1,-1), sys.reshape(1,-1))[0][0]
    feature_similarity.append((feature, sim))

feature_df = (
    pd.DataFrame(feature_similarity, columns=["Feature","Cosine Similarity"])
    .sort_values("Cosine Similarity", ascending=False)
    .reset_index(drop=True)
)


# ============================================================
# ADDITION: avg_level evaluation with MAE / RMSE / Spearman
# ============================================================
ref_lvl = reference_df["avg_level"].astype(float).values
sys_lvl = system_df["avg_level"].astype(float).values

avg_level_mae = mean_absolute_error(ref_lvl, sys_lvl)
mse = mean_squared_error(ref_lvl, sys_lvl)
avg_level_rmse = np.sqrt(mse)

# Spearman correlation (trend across levels)
# (If only 2 unique levels, spearman can be undefined; with 3+ it's fine.)
spearman_rho, spearman_p = spearmanr(ref_lvl, sys_lvl)

avg_level_metrics_df = pd.DataFrame([{
    "Metric": "avg_level MAE",
    "Value": avg_level_mae
}, {
    "Metric": "avg_level RMSE",
    "Value": avg_level_rmse
}, {
    "Metric": "avg_level Spearman rho",
    "Value": spearman_rho
}, {
    "Metric": "avg_level Spearman p-value",
    "Value": spearman_p
}])


# ============================================================
# Per cluster per level (cosine)
# ============================================================
CLUSTERS = {
    "Surface": SURFACE,
    "Dependency": DEP,
    "POS": POS,
    "DepPOS": DEPPOS,
}

cluster_level_rows = []
for cluster_name, feats in CLUSTERS.items():
    scaler = StandardScaler()
    ref_mat = scaler.fit_transform(reference_df[feats])
    sys_mat = scaler.transform(system_df[feats])

    for i, level in enumerate(reference_df["CEFR"]):
        sim = cosine_similarity(ref_mat[i].reshape(1,-1), sys_mat[i].reshape(1,-1))[0][0]
        cluster_level_rows.append({
            "CEFR Level": level,
            "Cluster": cluster_name,
            "Cosine Similarity": sim
        })

cluster_level_df = (
    pd.DataFrame(cluster_level_rows)
    .sort_values(["CEFR Level", "Cluster"])
    .reset_index(drop=True)
)


# -------------------------
# Print quick summary
# -------------------------
print("OVERALL COSINE SIMILARITY:", overall_similarity)
print("\nCluster similarity:\n", cluster_df)
print("\nPer level similarity:\n", level_df)
print("\navg_level metrics (MAE/RMSE/Spearman):\n", avg_level_metrics_df)
print("\nTop 10 best matching features:\n", feature_df.head(10))
print("\nWorst 10 matching features:\n", feature_df.tail(10))


# -------------------------
# Save files
# -------------------------
out_dir = f"{directory}cosine_eval2/"
os.makedirs(out_dir, exist_ok=True)

excel_path = f"{out_dir}readability_similarity_report.xlsx"

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    overall_df.to_excel(writer, sheet_name="Overall", index=False)
    cluster_df.to_excel(writer, sheet_name="Cluster", index=False)
    level_df.to_excel(writer, sheet_name="Per_Level", index=False)
    feature_df.to_excel(writer, sheet_name="Per_Feature", index=False)
    cluster_level_df.to_excel(writer, sheet_name="Cluster_x_Level", index=False)

    # NEW: avg_level error metrics
    avg_level_metrics_df.to_excel(writer, sheet_name="avg_level_metrics", index=False)

print("\nSaved Excel report:", excel_path)
