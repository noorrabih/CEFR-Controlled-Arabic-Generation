import pandas as pd
import string
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.charsets import UNICODE_PUNCT_CHARSET
# -----------------------------
# Your unique-word function
# -----------------------------
def count_unique_words(tokenized_sentence):

    unique_words = set(tokenized_sentence)

    return len(unique_words)

# -----------------------------
# Sentence-level metrics
# -----------------------------
def sentence_metrics(sentence):
    tokens = simple_word_tokenize(sentence)
    # remove punctuation tokens from tokens UNICODE_PUNCT_CHARSET
    tokens = [t for t in tokens if all(c not in UNICODE_PUNCT_CHARSET for c in t)]

    word_count = len(tokens)
    unique_word_count = count_unique_words(tokens)

    if word_count == 0:
        return word_count, unique_word_count, 0.0, 0, 0
    lengths = [len(w) for w in tokens]

    return (
        word_count,
        unique_word_count,
        sum(lengths) / word_count,
        min(lengths),
        max(lengths),
    )

# -----------------------------
# Main
# -----------------------------
def main(input_csv, directory, levels_csv=None):
    df = pd.read_csv(input_csv)

    # Extract group_id from ID (before last dash)
    # df["group_id"] = df["ID"].str.rsplit("-", n=1).str[0]
    # 1_A1_6levels_p1-1
    # df["group_id"] = df["ID"].astype(str).str.rsplit("_", n=1).str[1]
    df["group_id"] = df["anon_id"]
    metrics = df["Sentence"].apply(sentence_metrics)
    df[
        ["word_count", "unique_word_count", "avg_word_len", "min_word_len", "max_word_len"]
    ] = pd.DataFrame(metrics.tolist(), index=df.index)

    # Save outputs
    df.to_csv(f"{directory}/sentence_level_metrics.csv", index=False, encoding="utf-8-sig")
    print("Saved sentence-level metrics to:", f"{directory}/sentence_level_metrics.csv")

    group_summary = (
        df.groupby("group_id")
        .agg(
            sentences=("ID", "count"),
            total_words=("word_count", "sum"),
            avg_words_per_sentence=("word_count", "mean"),
            total_unique_words=("unique_word_count", "sum"),
            avg_unique_words=("unique_word_count", "mean"),
            overall_avg_word_len=("avg_word_len", "mean"),
            overall_max_word_len=("max_word_len", "max"),

        )
        .reset_index()
    )

    group_summary.to_csv(f"{directory}/essaylevel_summary.csv", index=False, encoding="utf-8-sig")

    print("Saved:")
    print("- sentence_level_metrics.csv")
    print("- essaylevel_summary.csv")


    # bea = pd.read_csv("/home/nour.rabih/arwi/readability_controlled_generation/arabic-aes-bea25/Data/surface_feats/essaylevel_summary.csv")
    # # AR_gpt-4o_1_1_1
    # zaebuc = pd.read_csv("/home/nour.rabih/arwi/readability_controlled_generation/ZAEBUC-v1.01/surface_feats/essaylevel_summary.csv")
    # # AR-030-268469 split and get last -
    # zaebuc['group_id'] = zaebuc['group_id'].str.split('-').str[-1]
    # print(zaebuc["group_id"].head())

    # bea_levels = pd.read_csv("/home/nour.rabih/arwi/readability_controlled_generation/arabic-aes-bea25/Data/arwi_cefr_levels.csv")
    # # AR_gpt-4o_1_1_1
    # zaebuc_levels = pd.read_csv("/home/nour.rabih/arwi/readability_controlled_generation/ZAEBUC-v1.01/corrected_essays_readability.csv")
    # # zaebuc_levels = zaebuc_levels[zaebuc_levels['CEFR'] != 'Unassessable']

    # # 268469
    # df = pd.concat([bea, zaebuc], ignore_index=True)
    # levels = pd.concat([bea_levels, zaebuc_levels], ignore_index=True)
    df = group_summary

    # df = pd.read_csv(f"{directory}/essaylevel_summary.csv")
    # Extract CEFR
    if "CEFR"  not in df.columns:
        df["CEFR"] = df["group_id"].str.split("_").str[1]
    print(df["CEFR"].value_counts())
    if df["CEFR"].nunique() < 2:
        print("Warning: Some CEFR levels could not be extracted.")
        # cefr level mapping file
        levels = pd.read_csv(levels_csv)
        levels["essay_id"] = levels["ID"].astype(str)
        cefr_dict = dict(zip(levels["essay_id"], levels["CEFR"]))
        def get_cefr_from_dict(group_id):
            # essay_id = group_id.split("-")[2]
            return cefr_dict.get(group_id, "NA")
        df["CEFR"] = df["group_id"].apply(get_cefr_from_dict)
        df = df[df["CEFR"] != "Unassessable"]

    else:
        print("All CEFR levels extracted successfully.")
        if df["CEFR"].nunique() == 3:
            cefr_order = ["A", "B", "C"]
        elif df["CEFR"].nunique() < 6:
             cefr_order = ["A", "B1", "B2", "C1", "C2"]
        else:
            cefr_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
        df["CEFR"] = pd.Categorical(df["CEFR"], categories=cefr_order, ordered=True)

    # Aggregate per CEFR
    cefr_summary = (
        df.groupby("CEFR")
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("CEFR")
    )
    #rename level column to CEFR
    cefr_summary = cefr_summary.rename(columns={"CEFR": "level"})
    cefr_summary.to_csv(f"{directory}/level_summary.csv", index=False, encoding="utf-8-sig")

    # -------- Plot 1: Total words & total unique words --------
    plt.figure(figsize=(8, 5))
    plt.plot(cefr_summary["level"], cefr_summary["total_words"], marker="o", label="total_words")
    plt.plot(cefr_summary["level"], cefr_summary["total_unique_words"], marker="o", label="total_unique_words")
    plt.xlabel("CEFR Level")
    plt.ylabel("Count")
    plt.title("Total Words & Total Unique Words vs CEFR")
    plt.legend()
    plt.tight_layout()
    # plt.show()

    # save plot
    plt.savefig(f"{directory}/cefr_words_metrics_plot.png")

    # -------- Plot 2: Remaining metrics --------
    plt.figure(figsize=(10, 5))
    metrics_small = [
        "sentences",
        "avg_words_per_sentence",
        "avg_unique_words",
        "overall_avg_word_len",
        "overall_max_word_len",
    ]

    for metric in metrics_small:
        plt.plot(cefr_summary["level"], cefr_summary[metric], marker="o", label=metric)

    plt.xlabel("CEFR Level")
    plt.ylabel("Average Value")
    plt.title("Sentence & Word-Level Metrics vs CEFR")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    # save plot
    plt.savefig(f"{directory}/cefr_sentence_word_metrics_plot.png")
if __name__ == "__main__":
    directory = "/home/nour.rabih/arwi/readability_controlled_generation/generation/vocabs_prompt/5levels"
    main(f"{directory}/generated_essays_p3_sentences.csv", f"{directory}/surface_feats", levels_csv=f"{directory}/generated_essays_p3_readability.csv")

