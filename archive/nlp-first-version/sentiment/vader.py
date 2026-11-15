
# ============================================================
# VADER SENTIMENT ANALYSIS
# ============================================================

import pandas as pd
import time
import os

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm.auto import tqdm


# ============================================================
# 1. LOAD VADER
# ============================================================

def load_vader():

    analyzer = SentimentIntensityAnalyzer()

    print("VADER loaded successfully!")

    return analyzer


# ============================================================
# 2. ANALYZE SENTIMENT
# ============================================================

def analyze_titles(texts, analyzer):

    results = []

    for title in tqdm(texts, desc="VADER analysis"):

        scores = analyzer.polarity_scores(title)

        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "positive"

        elif compound <= -0.05:
            sentiment = "negative"

        else:
            sentiment = "neutral"

        results.append({
            "sentiment": sentiment,
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "compound": compound
        })

    return pd.DataFrame(results)


# ============================================================
# 3. RUN COMPLETE ANALYSIS ON CSV
# ============================================================

def run_vader(
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

    analyzer = load_vader()

    print("\nStarting VADER analysis...")

    start_time = time.time()

    vader_results = analyze_titles(all_texts, analyzer)

    elapsed = time.time() - start_time

    print("\nVADER completed!")
    print("Time:", round(elapsed, 2), "seconds")

    df["vader_sentiment"] = vader_results["sentiment"].values
    df["vader_positive"] = vader_results["positive"].values
    df["vader_negative"] = vader_results["negative"].values
    df["vader_neutral"] = vader_results["neutral"].values
    df["vader_compound"] = vader_results["compound"].values

    display(df.head(10)) if "display" in globals() else print(df.head(10))

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nVADER results saved!")
    print("Output:", os.path.abspath(output_path))

    return df
