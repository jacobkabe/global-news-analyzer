
# ============================================================
# SENTIMENT MODEL COMPARISON
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from IPython.display import display

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Full per-article results (contain article text) stay in data/raw/ (ignored by Git).
output_dir = "data/raw/outputs/"

# Small aggregate tables and plots that we commit.
results_dir = "results/"

# Small anonymised sample that we commit (no summary, text, url or author names).
sample_path = "data/sample/sentiment_sample.csv"
sample_size = 50

models = {
    "DistilBERT": "distilbert_results.csv",
    "SiEBERT": "siebert_results.csv",
    "RoBERTa": "roberta_results.csv",
    "TextBlob": "textblob_results.csv",
    "VADER": "vader_results.csv"
}

sentiments = ["negative", "neutral", "positive"]

os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.dirname(sample_path), exist_ok=True)


# ============================================================
# 2. LOAD RESULTS FROM ALL MODELS
# ============================================================

print("Loading sentiment results...")

model_results = {}

for model_name, filename in models.items():

    file_path = os.path.join(output_dir, filename)

    df = pd.read_csv(file_path)

    model_results[model_name] = df

    print(model_name, "loaded:", len(df), "headlines")


# ============================================================
# 3. COMBINE MODEL PREDICTIONS
# ============================================================

# We match the models on the article "id" (unique per article),
# not on the title: two articles can have the same headline.

comparison_df = model_results["DistilBERT"][
    ["id", "source", "published_at", "title"]
].copy()

for model_name, df in model_results.items():

    if df["id"].duplicated().any():
        raise ValueError(f"{model_name} has duplicate article ids.")

    if set(df["id"]) != set(comparison_df["id"]):
        raise ValueError(
            f"{model_name} contains different articles than DistilBERT."
        )

    column = model_name.lower() + "_sentiment"

    labels = df[["id", column]].copy()

    labels[column] = (
        labels[column]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    comparison_df = comparison_df.merge(labels, on="id", how="left")


print("\nAll model predictions combined successfully!")

display(comparison_df.head(10))


# ============================================================
# 4. SENTIMENT DISTRIBUTION TABLE
# ============================================================

print("\nSENTIMENT DISTRIBUTION")

distribution = {}

for model_name in models:

    column = model_name.lower() + "_sentiment"

    counts = comparison_df[column].value_counts()

    percentages = (
        comparison_df[column]
        .value_counts(normalize=True)
        .mul(100)
    )

    distribution[model_name] = {
        sentiment: (
            f"{counts.get(sentiment, 0)} "
            f"({percentages.get(sentiment, 0):.2f}%)"
        )
        for sentiment in sentiments
    }


distribution_df = pd.DataFrame(distribution).T

print(distribution_df)

display(distribution_df)

distribution_df.to_csv(
    results_dir + "sentiment_distribution_comparison.csv"
)


# ============================================================
# 5. GRAPH 1: SENTIMENT DISTRIBUTION
# ============================================================


percentage_data = pd.DataFrame(
    index=models.keys(),
    columns=sentiments,
    dtype=float
)

for model_name in models:

    column = model_name.lower() + "_sentiment"

    percentages = (
        comparison_df[column]
        .value_counts(normalize=True)
        .mul(100)
    )

    percentage_data.loc[model_name] = (
        percentages.reindex(sentiments, fill_value=0).values
    )

ax = percentage_data.plot(
    kind="bar",
    figsize=(12, 6),
    color=["#E76F51", "#A8A8A8", "#2A9D8F"]
)

plt.title("Sentiment Distribution Across Models", fontsize=15)
plt.xlabel("Sentiment Models")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.legend(title="Sentiment")
plt.tight_layout()

plt.savefig(
    results_dir + "sentiment_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. GRAPH 2: SENTIMENT COMPOSITION
# ============================================================

ax = percentage_data.plot(
    kind="barh",
    stacked=True,
    figsize=(11, 6),
    color=["#E76F51", "#A8A8A8", "#2A9D8F"]
)

plt.title("Sentiment Composition by Model", fontsize=15)
plt.xlabel("Percentage (%)")
plt.ylabel("Model")
plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1))
plt.tight_layout()

plt.savefig(
    results_dir + "sentiment_composition.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. MODEL AGREEMENT TABLE
# ============================================================

print("\nMODEL AGREEMENT")

model_columns = {
    name: name.lower() + "_sentiment"
    for name in models
}

agreement = pd.DataFrame(
    index=models.keys(),
    columns=models.keys(),
    dtype=float
)

for model_1, column_1 in model_columns.items():

    for model_2, column_2 in model_columns.items():

        agreement.loc[model_1, model_2] = (
            comparison_df[column_1] ==
            comparison_df[column_2]
        ).mean() * 100


agreement = agreement.round(2)

print(agreement)

display(agreement)

agreement.to_csv(
    results_dir + "model_agreement.csv"
)


# ============================================================
# 8. GRAPH 3: MODEL AGREEMENT HEATMAP
# ============================================================

plt.figure(figsize=(9, 7))

sns.heatmap(
    agreement.astype(float),
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    vmin=0,
    vmax=100,
    linewidths=0.5
)

plt.title("Agreement Between Sentiment Models (%)", fontsize=15)
plt.tight_layout()

plt.savefig(
    results_dir + "model_agreement_heatmap.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. OVERALL MODEL AGREEMENT
# ============================================================

print("\nOVERALL MODEL AGREEMENT")

agreement_values = agreement.values[
    np.triu_indices(len(models), k=1)
]

overall_agreement = agreement_values.mean()

print(
    "Average pairwise agreement:",
    round(overall_agreement, 2),
    "%"
)


# ============================================================
# 10. SAVE COMBINED PREDICTIONS
# ============================================================

comparison_df.to_csv(
    output_dir + "all_models_comparison.csv",
    index=False
)

# ============================================================
# 11. SAVE A SMALL SAMPLE THAT WE CAN COMMIT
# ============================================================

# Guardian headlines end with " | Author Name": remove it (no author names in Git).
author_suffix = re.compile(r"\s+\|\s+[^|]+$")

sample_df = comparison_df.sample(
    n=min(sample_size, len(comparison_df)),
    random_state=42
).sort_values("id")

sample_df["title"] = (
    sample_df["title"]
    .fillna("")
    .astype(str)
    .str.replace(author_suffix, "", regex=True)
)

sample_df.to_csv(sample_path, index=False)

print("\nAll comparison files saved successfully!")

print("\nGenerated files:")
print("In", results_dir, "(committed):")
print("1. sentiment_distribution_comparison.csv")
print("2. model_agreement.csv")
print("3. sentiment_distribution.png")
print("4. sentiment_composition.png")
print("5. model_agreement_heatmap.png")
print("In", output_dir, "(not committed):")
print("6. all_models_comparison.csv")
print("Sample (committed):", sample_path)

print("\nCOMPARISON COMPLETED!")
