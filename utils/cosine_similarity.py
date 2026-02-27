"""Evaluate generated profiles against reference profiles using cosine similarity.

This script compares:
- reference feature table (selected features per CEFR)
- system feature table (same selected features per CEFR)

It also merges in avg readability and avg depth (optional) and reports:
- overall cosine similarity
- per cluster cosine similarity
- per level cosine similarity
- per feature cosine similarity
- avg_level MAE/RMSE/Spearman (if avg_level is present)

Example:
  python utils/cosine_similarity.py \
    --reference_features_csv ref_selected_features.csv \
    --system_features_csv sys_filtered_features.csv \
    --out_dir results/cosine_eval \
    --reference_avg_readability_csv ref_avg_level.csv \
    --system_avg_readability_csv sys_avg_level.csv \
    --reference_avg_depth_csv ref_avg_depth.csv \
    --system_avg_depth_csv sys_avg_depth.csv
"""

from __future__ import annotations

import argparse
import os
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr


# -----------------------------
# Feature clusters (defaults)
# -----------------------------
SURFACE = [
    "total_words", "avg_words_per_sentence", "total_unique_words", "avg_unique_words",
    "overall_avg_word_len", "overall_max_word_len", "avg_syllables_per_word",
    "max_max_syllables", "avg_unique_lemmas", "avg_unique_lemma_pos", "avg_camel_tokens"
]

DEP = ["dep_IDF_mean", "dep_MOD_mean", "dep_OBJ_mean", "dep_SBJ_mean"]

POS = [
    "pos_ADJ_mean", "pos_ADP_mean", "pos_AUX_mean", "pos_CCONJ_mean",
    "pos_DET_mean", "pos_NOUN_mean", "pos_PUNCT_mean", "pos_VERB_mean"
]

DEPPOS = [
    "depPOS_IDF(NOUN)_mean", "depPOS_MOD(ADJ)_mean", "depPOS_MOD(ADP)_mean",
    "depPOS_MOD(AUX)_mean", "depPOS_MOD(CCONJ)_mean", "depPOS_MOD(DET)_mean",
    "depPOS_MOD(NOUN)_mean", "depPOS_MOD(PRON)_mean", "depPOS_MOD(PROPN)_mean",
    "depPOS_MOD(PUNCT)_mean", "depPOS_MOD(VERB)_mean", "depPOS_OBJ(ADJ)_mean",
    "depPOS_OBJ(ADV)_mean", "depPOS_OBJ(DET)_mean", "depPOS_OBJ(NOUN)_mean",
    "depPOS_SBJ(NOUN)_mean", "depPOS_SBJ(SCONJ)_mean", "depth_mean"
]

READABILITY = ["avg_level"]

CLUSTERS: Dict[str, List[str]] = {
    "Surface": SURFACE,
    "Dependency": DEP,
    "POS": POS,
    "DepPOS": DEPPOS,
}


def _maybe_merge_aux(df: pd.DataFrame, aux_path: str | None, key: str = "CEFR") -> pd.DataFrame:
    if not aux_path:
        return df
    aux = pd.read_csv(aux_path)
    if key not in aux.columns:
        raise ValueError(f"Aux file {aux_path} must include '{key}' column. Found: {list(aux.columns)}")
    if not aux[key].is_unique:
        raise ValueError(f"Aux file {aux_path} has duplicate '{key}' rows.")
    return df.merge(aux, on=key, how="left")


def cluster_cosine_similarity(reference_df: pd.DataFrame, system_df: pd.DataFrame, features: List[str]) -> float:
    feats = [f for f in features if f in reference_df.columns and f in system_df.columns]
    if not feats:
        return float('nan')
    scaler = StandardScaler()
    ref = scaler.fit_transform(reference_df[feats])
    sys = scaler.transform(system_df[feats])
    return float(cosine_similarity(ref.reshape(1, -1), sys.reshape(1, -1))[0][0])


def main(args):
    ref = pd.read_csv(args.reference_features_csv)
    sys = pd.read_csv(args.system_features_csv)

    # Optional merges
    ref = _maybe_merge_aux(ref, args.reference_avg_readability_csv)
    ref = _maybe_merge_aux(ref, args.reference_avg_depth_csv)

    sys = _maybe_merge_aux(sys, args.system_avg_readability_csv)
    sys = _maybe_merge_aux(sys, args.system_avg_depth_csv)

    key = args.key
    if key not in ref.columns or key not in sys.columns:
        raise ValueError(f"Both reference and system must contain key column '{key}'.")

    # Determine feature set
    all_features = []
    for feats in CLUSTERS.values():
        all_features.extend(feats)
    if args.include_readability and all(f in ref.columns for f in READABILITY) and all(f in sys.columns for f in READABILITY):
        all_features += READABILITY

    all_features = [f for f in all_features if f in ref.columns and f in sys.columns]
    if not all_features:
        raise ValueError("No overlapping features found between reference and system.")

    # Align & sort
    if set(ref[key]) != set(sys[key]):
        raise ValueError("Reference/System CEFR sets do not match.")

    ref = ref[[key] + all_features].sort_values(key).reset_index(drop=True)
    sys = sys[[key] + all_features].set_index(key).loc[ref[key]].reset_index()

    # Overall similarity
    scaler_all = StandardScaler()
    ref_all = scaler_all.fit_transform(ref[all_features])
    sys_all = scaler_all.transform(sys[all_features])

    overall_similarity = float(cosine_similarity(ref_all.reshape(1, -1), sys_all.reshape(1, -1))[0][0])
    overall_df = pd.DataFrame({"Metric": ["Overall Cosine Similarity"], "Value": [overall_similarity]})

    # Cluster similarity
    cluster_df = pd.DataFrame({
        "Cluster": list(CLUSTERS.keys()),
        "Cosine Similarity": [cluster_cosine_similarity(ref, sys, feats) for feats in CLUSTERS.values()],
    })

    # Per-level similarity
    level_rows = []
    for i, level in enumerate(ref[key]):
        sim = float(cosine_similarity(ref_all[i].reshape(1, -1), sys_all[i].reshape(1, -1))[0][0])
        level_rows.append((level, sim))
    level_df = pd.DataFrame(level_rows, columns=[key, "Cosine Similarity"])

    # Per-feature similarity
    feat_rows = []
    for feature in all_features:
        scaler = StandardScaler()
        ref_vec = scaler.fit_transform(ref[[feature]])
        sys_vec = scaler.transform(sys[[feature]])
        sim = float(cosine_similarity(ref_vec.reshape(1, -1), sys_vec.reshape(1, -1))[0][0])
        feat_rows.append((feature, sim))
    feature_df = pd.DataFrame(feat_rows, columns=["Feature", "Cosine Similarity"]).sort_values("Cosine Similarity", ascending=False)

    # Optional avg_level metrics
    avg_level_metrics_df = None
    if args.include_readability and "avg_level" in ref.columns and "avg_level" in sys.columns:
        ref_lvl = pd.to_numeric(ref["avg_level"], errors="coerce").values
        sys_lvl = pd.to_numeric(sys["avg_level"], errors="coerce").values
        mask = ~np.isnan(ref_lvl) & ~np.isnan(sys_lvl)
        if mask.sum() >= 3:
            mae = mean_absolute_error(ref_lvl[mask], sys_lvl[mask])
            rmse = np.sqrt(mean_squared_error(ref_lvl[mask], sys_lvl[mask]))
            rho, p = spearmanr(ref_lvl[mask], sys_lvl[mask])
            avg_level_metrics_df = pd.DataFrame([
                {"Metric": "avg_level MAE", "Value": float(mae)},
                {"Metric": "avg_level RMSE", "Value": float(rmse)},
                {"Metric": "avg_level Spearman rho", "Value": float(rho)},
                {"Metric": "avg_level Spearman p-value", "Value": float(p)},
            ])

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    excel_path = os.path.join(args.out_dir, "readability_similarity_report.xlsx")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        overall_df.to_excel(writer, sheet_name="Overall", index=False)
        cluster_df.to_excel(writer, sheet_name="Cluster", index=False)
        level_df.to_excel(writer, sheet_name="Per_Level", index=False)
        feature_df.to_excel(writer, sheet_name="Per_Feature", index=False)
        if avg_level_metrics_df is not None:
            avg_level_metrics_df.to_excel(writer, sheet_name="avg_level_metrics", index=False)

    print("OVERALL COSINE SIMILARITY:", overall_similarity)
    print(f"Saved report: {excel_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cosine similarity evaluation between reference and system profiles.")
    ap.add_argument("--reference_features_csv", required=True)
    ap.add_argument("--system_features_csv", required=True)
    ap.add_argument("--reference_avg_readability_csv", default=None)
    ap.add_argument("--system_avg_readability_csv", default=None)
    ap.add_argument("--reference_avg_depth_csv", default=None)
    ap.add_argument("--system_avg_depth_csv", default=None)
    ap.add_argument("--key", default="CEFR", help="Join/sort key (default: CEFR).")
    ap.add_argument("--include_readability", action="store_true", help="Include avg_level in evaluation if present.")
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())
