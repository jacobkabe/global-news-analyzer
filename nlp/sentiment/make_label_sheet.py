
# ============================================================
# MAKE A SHEET OF HEADLINES TO LABEL BY HAND
# ============================================================
#
# Picks 150 headlines, spread across the outlets, and writes them to
# data/sample/sentiment_to_label.csv with an empty "my_label" column.
#
# The sheet does NOT show the model labels or the outlet, so they do
# not influence the person labelling. We join them back later by "id".
#
# Run from the project folder:  python nlp\sentiment\make_label_sheet.py
# Needs data/raw/outputs/all_models_comparison.csv (made by comparison.py).

import os
import re
import sys

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

input_path = "data/raw/outputs/all_models_comparison.csv"

output_path = "data/sample/sentiment_to_label.csv"

# How many headlines per outlet. Small outlets get more than their share,
# so that every outlet has enough headlines to check the models on.
per_source = {
    "guardian": 38,
    "nyt": 37,
    "aljazeera": 20,
    "bbc": 20,
    "foxnews": 20,
    "npr": 15,
}

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

df = df[df["title"].fillna("").str.strip() != ""]

picked = []

for source, n in per_source.items():

    rows = df[df["source"] == source]

    if len(rows) < n:
        print(f"{source}: only {len(rows)} headlines, taking all of them")
        n = len(rows)

    picked.append(rows.sample(n=n, random_state=42))

sheet = pd.concat(picked)

# Shuffle, so the outlets are mixed and nobody labels one outlet in a row.
sheet = sheet.sample(frac=1, random_state=42)

sheet["title"] = sheet["title"].str.replace(author_suffix, "", regex=True)

sheet = sheet[["id", "title"]]
sheet["my_label"] = ""
sheet["note"] = ""


# ============================================================
# 4. SAVE
# ============================================================

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# utf-8-sig so that Excel shows quotes and accents correctly.
sheet.to_csv(output_path, index=False, encoding="utf-8-sig")

print(len(sheet), "headlines written to", output_path)
print(pd.concat(picked)["source"].value_counts())
