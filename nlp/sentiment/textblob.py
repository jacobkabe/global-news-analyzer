
# ============================================================
# TEXTBLOB SENTIMENT ANALYSIS
# ============================================================

import pandas as pd
import time
import os

from textblob import TextBlob
from tqdm.auto import tqdm


# ============================================================
# 1. ANALYZE SENTIMENT
# ============================================================

def analyze_titles(texts):

    results = []

    for title in tqdm(texts, desc="TextBlob analysis"):

        blob = TextBlob(title)

        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0:
            sentiment = "positive"

        elif polarity < 0:
            sentiment = "negative"

        else:
            sentiment = "neutral"

        results.append({
            "sentiment": sentiment,
            "polarity": polarity,
            "subjectivity": subjectivity
        })

    return pd.DataFrame(results)


# ============================================================
# 2. RUN COMPLETE ANALYSIS ON CSV
# ============================================================

def run_textblob(
    input_path,
    output_path,
    text_column="title"
):

    print("\nLoading dataset...")

    df = pd.read_csv(input_path)

    all_texts = (
        df[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print("Total titles:", len(all_texts))

    print("\nStarting TextBlob analysis...")

    start_time = time.time()

    textblob_results = analyze_titles(all_texts)

    elapsed = time.time() - start_time

    print("\nTextBlob completed!")
    print("Time:", round(elapsed, 2), "seconds")

    df["textblob_sentiment"] = textblob_results["sentiment"].values
    df["textblob_polarity"] = textblob_results["polarity"].values
    df["textblob_subjectivity"] = textblob_results["subjectivity"].values

    display(df.head(10)) if "display" in globals() else print(df.head(10))

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nTextBlob results saved!")
    print("Output:", os.path.abspath(output_path))

    return df