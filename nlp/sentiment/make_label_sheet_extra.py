
# ============================================================
# MAKE A SECOND SHEET: HEADLINES THAT ARE PROBABLY POSITIVE
# ============================================================
#
# The first 150 hand labels have only 8 positive headlines, too few to
# measure the models on the positive class. This script picks 80 more
# headlines that at least one model thinks are positive:
#   RoBERTa says positive, or VADER and TextBlob both say positive.
# Headlines that are already labelled are skipped.
#
# Like the first sheet, it hides the outlet and the model labels.
# Write in the report that these 80 were chosen this way (not at random).
#
# Run from the project folder:  python nlp\sentiment\make_label_sheet_extra.py
# Needs data/raw/outputs/all_models_comparison.csv (made by comparison.py)
# and the first labels in data/sample/sentiment_gold.csv.

import os
import re
import sys

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

input_path = "data/raw/outputs/all_models_comparison.csv"

gold_path = "data/sample/sentiment_gold.csv"

# Excel file, so it opens with a double click (no import needed).
output_path = "data/sample/sentiment_to_label_extra.xlsx"

sample_size = 80

# Guardian headlines end with " | Author Name": remove it (no author names in Git).
author_suffix = re.compile(r"\s+\|\s+[^|]+$")


# ============================================================
# 2. DO NOT OVERWRITE LABELS
# ============================================================

if os.path.exists(output_path):
    sys.exit(
        f"{output_path} already exists. It may contain your labels, "
        "so it is not overwritten. Rename or delete it first."
    )


# ============================================================
# 3. PICK THE HEADLINES
# ============================================================

df = pd.read_csv(input_path)
gold = pd.read_csv(gold_path)

df = df[~df["id"].isin(gold["id"])]
df = df[df["title"].fillna("").str.strip() != ""]

probably_positive = (
    (df["roberta_sentiment"] == "positive")
    | ((df["vader_sentiment"] == "positive") & (df["textblob_sentiment"] == "positive"))
)
candidates = df[probably_positive]

# The small outlets have only a few candidates: take all of them,
# fill the rest with Guardian and NYT.
small = candidates[~candidates["source"].isin(["guardian", "nyt"])]
large = candidates[candidates["source"].isin(["guardian", "nyt"])]

sheet = pd.concat([
    small,
    large.sample(n=sample_size - len(small), random_state=42),
])

# Shuffle, so the outlets are mixed.
sheet = sheet.sample(frac=1, random_state=42)

sheet["title"] = sheet["title"].str.replace(author_suffix, "", regex=True)

sheet = sheet[["id", "title"]]
sheet["my_label"] = ""
sheet["note"] = ""


# ============================================================
# 4. SAVE
# ============================================================

sheet.to_excel(output_path, index=False)

print(len(sheet), "headlines written to", output_path)
print(df[df["id"].isin(sheet["id"])]["source"].value_counts())
