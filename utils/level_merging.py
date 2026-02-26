# open /home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/selected_readability_monotonic_features.csv

# for levels A1 and A2, add both to a new row call it A, and add the average of the two values in the new row

import pandas as pd

# read the csv file
df = pd.read_csv('/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/selected_readability_monotonic_features.csv')
# create a new row for A by averaging A1 and A2
print(df)
# compute the average of A1 and A2
merged_A1_A2 = (
    df[df["level"].isin(["A1", "A2"])]
    .mean(numeric_only=True)
    .to_frame()
    .T
)

# assign the new level name
merged_A1_A2["level"] = "A"
merged_A1_A2["CEFR"] = "A"

# remove original A1 and A2 rows
df = df[~df["level"].isin(["A1", "A2"])]

# append the merged row
df = pd.concat([df, merged_A1_A2], ignore_index=True)

print(df)


# save the new csv file
df.to_csv('/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/5levels/selected_readability_monotonic_features.csv', index=False)

# now B1 and B2
# compute the average of B1 and B2
merged_B1_B2 = (
    df[df["level"].isin(["B1", "B2"])]
    .mean(numeric_only=True)
    .to_frame()
    .T
)
# assign the new level name
merged_B1_B2["level"] = "B"
merged_B1_B2["CEFR"] = "B"
# remove original B1 and B2 rows
df = df[~df["level"].isin(["B1", "B2"])]
# append the merged row
df = pd.concat([df, merged_B1_B2], ignore_index=True)
print(df)

# now C

# compute the average of C1 and C2
merged_C1_C2 = (
    df[df["level"].isin(["C1", "C2"])]
    .mean(numeric_only=True)
    .to_frame()
    .T
)
# assign the new level name
merged_C1_C2["level"] = "C"
merged_C1_C2["CEFR"] = "C"
# remove original C1 and C2 rows
df = df[~df["level"].isin(["C1", "C2"])]
# append the merged row
df = pd.concat([df, merged_C1_C2], ignore_index=True)
print(df)

# save the new csv file
df.to_csv('/home/nour.rabih/arwi/readability_controlled_generation/zaebuc+bea/selected_features_graphs/3levels/selected_readability_monotonic_features.csv', index=False)