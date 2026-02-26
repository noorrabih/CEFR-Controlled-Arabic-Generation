import pandas as pd

def get_monotonic_columns(
    csv_path,
    trend="increasing",
    cefr_order=["A1", "A2", "B1", "B2", "C1", "C2"],
    exclude_pattern=r'ratio|---|_median|std|count'
):
    df = pd.read_csv(csv_path)

    # Detect level column
    if "level" in df.columns:
        level_col = "level"
    elif "CEFR" in df.columns:
        level_col = "CEFR"
    else:
        raise ValueError("CSV must contain a 'level' or 'CEFR' column")

    # Keep only wanted columns
    keep_cols = df.columns[
        ~df.columns.str.contains(exclude_pattern, case=False, regex=True)
    ].tolist()

    if level_col not in keep_cols:
        keep_cols = [level_col] + keep_cols

    level_df = df.loc[:, keep_cols].copy()
    level_df[level_col] = pd.Categorical(level_df[level_col], categories=cefr_order, ordered=True)

    # Mean per level
    feat_cols = level_df.columns.drop(level_col)
    level_means = (
        level_df.groupby(level_col, observed=True)[feat_cols]
        .mean()
        .reindex(cefr_order)
    )

    # Trend selection
    if trend == "increasing":
        cols = level_means.columns[
            level_means.apply(lambda s: s.is_monotonic_increasing)
        ]
    elif trend == "decreasing":
        cols = level_means.columns[
            level_means.apply(lambda s: s.is_monotonic_decreasing)
        ]
    else:
        raise ValueError("trend must be 'increasing' or 'decreasing'")

    return cols.tolist()

if __name__ == "__main__":
    # surface features 
    csv_path = "/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/surface_feats/level_summary.csv"
    surface_df = pd.read_csv(csv_path)
    inc_cols = get_monotonic_columns(csv_path, trend="increasing")
    dec_cols = get_monotonic_columns(csv_path, trend="decreasing", cefr_order=["B1", "B2", "C1", "C2"],)
    surface_df = surface_df[['CEFR'] + inc_cols + dec_cols]
    # remove those that include total in name
    inc_cols = [col for col in inc_cols if 'total' not in col]
    dec_cols = [col for col in dec_cols if 'total' not in col]
    print("Number of selected surface features:")
    print(len(inc_cols),  len(dec_cols))


    camel_path = "/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/surface_feats/level_features_camel.csv"
    camel_df = pd.read_csv(camel_path)
    camel_inc_cols = get_monotonic_columns(camel_path, trend="increasing")
    camel_dec_cols = get_monotonic_columns(camel_path, trend="decreasing", cefr_order=["B1", "B2", "C1", "C2"],)
    camel_df = camel_df[['CEFR'] + camel_inc_cols + camel_dec_cols]
    print("Number of selected camel features:")
    print(len(camel_inc_cols), len(camel_dec_cols))

    # syntax
    syntax_path = "/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/syntax_graphs/level_syntax_stats.csv"
    syntax_df = pd.read_csv(syntax_path)
    syntax_inc_cols = get_monotonic_columns(syntax_path, trend="increasing")
    # remove syntax_inc_cols that include (X) or depPOS_SBJ(ADV)_mean or depPOS_SBJ(CCONJ)_mean in name
    syntax_inc_cols = [col for col in syntax_inc_cols if '(X)' not in col and 'depPOS_SBJ(ADV)_mean' not in col and 'depPOS_SBJ(CCONJ)_mean' not in col]
    # remove syntax_inc_cols _sum columns,
    syntax_inc_cols = [col for col in syntax_inc_cols if 'sum' not in col]
    print("selected syntax_inc_cols syntax features:")
    print(syntax_inc_cols)
    # syntax_dec_cols = get_monotonic_columns(syntax_path, trend="decreasing", cefr_order=["B1", "B2", "C1", "C2"],)
    # remove syntax_dec_cols _mean columns
    # syntax_dec_cols = [col for col in syntax_dec_cols if 'mean' not in col]
    # print("selected syntax_dec_cols syntax features:")
    # print(syntax_dec_cols)
    syntax_df = syntax_df[['level'] + syntax_inc_cols ] # + syntax_dec_cols
    print("Number of selected syntax features:")
    print(len(syntax_inc_cols)) # ,  len(syntax_dec_cols)

    # merge all
    merged = surface_df.merge(
        camel_df,
        on="CEFR",
        how="inner",
    ).merge(    
        syntax_df,
        left_on="CEFR",
        right_on="level",
        how="inner",
    )


    print("Total number of selected features:", merged.shape)  # exclude CEFR
    out_path = "/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/selected_readability_monotonic_features.csv"
    merged.to_csv(out_path, index=False)
    print("Saved to:", out_path)    

    # merged = pd.read_csv("/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_readability_monotonic_features.csv")
    # plot the increasing features
    import matplotlib.pyplot as plt
    import seaborn as sns
    # + syntax_inc_cols inc_cols + 
    melt_df = merged.melt(id_vars=['CEFR'], value_vars=camel_inc_cols , var_name='feature', value_name='value')
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=melt_df, x='CEFR', y='value', hue='feature', marker='o')
    plt.title('Increasing Camel Features across CEFR Levels')
    plt.xlabel('CEFR Level')
    plt.ylabel('Feature Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = "/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/camel_increasing_features_across_cefr.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print("Plot saved to:", plot_path)

    melt_df = merged.melt(id_vars=['CEFR'], value_vars=inc_cols, var_name='feature', value_name='value')
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=melt_df, x='CEFR', y='value', hue='feature', marker='o')
    # plt.title('Increasing Words Features across CEFR Levels')
    plt.xlabel('CEFR Level')
    plt.ylabel('Feature Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = "/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/words_increasing_features_across_cefr.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print("Plot saved to:", plot_path)

    melt_df = merged.melt(id_vars=['CEFR'], value_vars=inc_cols + camel_inc_cols, var_name='feature', value_name='value')
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=melt_df, x='CEFR', y='value', hue='feature', marker='o')
    # plt.title('Increasing Words Features across CEFR Levels')
    plt.xlabel('CEFR Level')
    plt.ylabel('Feature Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = "/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/camelandwords_increasing_features_across_cefr.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print("Plot saved to:", plot_path)

    mean_syntax_inc_cols = [col for col in syntax_inc_cols if 'mean' in col]
    pos_syntax_inc_cols = [col for col in syntax_inc_cols if 'pos_' in col]
    melt_df = merged.melt(id_vars=['CEFR'], value_vars=pos_syntax_inc_cols , var_name='feature', value_name='value')
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=melt_df, x='CEFR', y='value', hue='feature', marker='o')
    # plt.title('Increasing POS Features across CEFR Levels')
    plt.xlabel('CEFR Level')
    plt.ylabel('Feature Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = "/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/pos_increasing_features_across_cefr.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print("Plot saved to:", plot_path)

    dep_syntax_inc_cols = [col for col in syntax_inc_cols if 'dep_' in col]
    melt_df = merged.melt(id_vars=['CEFR'], value_vars=dep_syntax_inc_cols , var_name='feature', value_name='value')
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=melt_df, x='CEFR', y='value', hue='feature', marker='o')
    # plt.title('Increasing Dependency Features across CEFR Levels')
    plt.xlabel('CEFR Level')
    plt.ylabel('Feature Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = "/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/dep_increasing_features_across_cefr.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print("Plot saved to:", plot_path)

    deppos_syntax_inc_cols = [col for col in syntax_inc_cols if 'depPOS_' in col]
    melt_df = merged.melt(id_vars=['CEFR'], value_vars=deppos_syntax_inc_cols , var_name='feature', value_name='value')
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=melt_df, x='CEFR', y='value', hue='feature', marker='o')
    # plt.title('Increasing Dependency(POS) Features across CEFR Levels')
    plt.xlabel('CEFR Level')
    plt.ylabel('Feature Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = "/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/depPOS_increasing_features_across_cefr.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print("Plot saved to:", plot_path)

    # plot the decreasing features
    # mean_syntax_dec_cols = [col for col in syntax_dec_cols if 'mean' in col]
    # # remove depPOS_OBJ(SCONJ)_mean because it is constant 0
    # if 'depPOS_OBJ(SCONJ)_mean' in mean_syntax_dec_cols:
    #     mean_syntax_dec_cols.remove('depPOS_OBJ(SCONJ)_mean')
    # print("Decreasing syntax mean cols:", mean_syntax_dec_cols)
    # melt_df_dec = decreasing_df.melt(id_vars=['CEFR'], value_vars= mean_syntax_dec_cols , var_name='feature', value_name='value')
    # plt.figure(figsize=(12, 8))
    # sns.lineplot(data=melt_df_dec, x='CEFR', y='value', hue='feature', marker='o')
    # plt.title('Decreasing Features across CEFR Levels')
    # plt.xlabel('CEFR Level')
    # plt.ylabel('Feature Value')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plot_path_dec = "/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/decreasing_features_across_cefr.png"
    # plt.savefig(plot_path_dec, dpi=200)
    # plt.close()
    # print("Plot saved to:", plot_path_dec)

    # decreasing graphs start at B1
    # decreasing_df = merged[merged['CEFR'].isin(['B1', 'B2', 'C1', 'C2'])]
    # sum_syntax_dec_cols = [col for col in syntax_dec_cols if 'sum' in col]
    # # remove depPOS_SBJ(DET)_sum from sum_syntax_dec_cols
    # if 'depPOS_SBJ(DET)_sum' in sum_syntax_dec_cols:
    #     sum_syntax_dec_cols.remove('depPOS_SBJ(DET)_sum')
    # print("Decreasing syntax sum cols:", sum_syntax_dec_cols)
    # if 'depPOS_OBJ(SCONJ)_sum' in sum_syntax_dec_cols:
    #     sum_syntax_dec_cols.remove('depPOS_OBJ(SCONJ)_sum')
    # melt_df_dec = decreasing_df.melt(id_vars=['CEFR'], value_vars= sum_syntax_dec_cols , var_name='feature', value_name='value')
    # plt.figure(figsize=(12, 8))
    # sns.lineplot(data=melt_df_dec, x='CEFR', y='value', hue='feature', marker='o')
    # plt.title('Decreasing Features across CEFR Levels')
    # plt.xlabel('CEFR Level')
    # plt.ylabel('Feature Value')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plot_path_dec = "/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/syntax_sum_decreasing_features_across_cefr.png"
    # plt.savefig(plot_path_dec, dpi=200)
    # plt.close()
    # print("Plot saved to:", plot_path_dec)






# if mean < 0, look at count. this can indicate the existence of this feature in this level in general.