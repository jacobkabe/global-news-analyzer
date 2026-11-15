
# ============================================================
# RUN ALL SENTIMENT ANALYSIS MODELS
# ============================================================

import pandas as pd
import time
import gc
import torch

from nlp.sentiment.distilbert import run_distilbert
from nlp.sentiment.siebert import run_siebert
from nlp.sentiment.roberta import run_roberta
from nlp.sentiment.textblob import run_textblob
from nlp.sentiment.vader import run_vader


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Made by src/collect/export_csv.py
input_path = "data/raw/news.csv"

# Full results contain the article text, so they stay in data/raw/ (ignored by Git).
output_dir = "data/raw/outputs/"

text_column = "title"

print("Starting sentiment analysis...")
print("Input dataset:", input_path)


# ============================================================
# 2. DISTILBERT
# ============================================================

print("\n" + "=" * 50)
print("1. DISTILBERT")
print("=" * 50)

start = time.time()

distilbert_df = run_distilbert(
    input_path=input_path,
    output_path=output_dir + "distilbert_results.csv",
    text_column=text_column,
    batch_size=8
)

print("DistilBERT time:", round((time.time() - start) / 60, 2), "minutes")

del distilbert_df
gc.collect()


# ============================================================
# 3. SIEBERT
# ============================================================

print("\n" + "=" * 50)
print("2. SIEBERT")
print("=" * 50)

start = time.time()

siebert_df = run_siebert(
    input_path=input_path,
    output_path=output_dir + "siebert_results.csv",
    text_column=text_column,
    batch_size=4
)

print("SiEBERT time:", round((time.time() - start) / 60, 2), "minutes")

del siebert_df
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 4. ROBERTA
# ============================================================

print("\n" + "=" * 50)
print("3. ROBERTA")
print("=" * 50)

start = time.time()

roberta_df = run_roberta(
    input_path=input_path,
    output_path=output_dir + "roberta_results.csv",
    text_column=text_column,
    batch_size=8
)

print("RoBERTa time:", round((time.time() - start) / 60, 2), "minutes")

del roberta_df
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 5. TEXTBLOB
# ============================================================

print("\n" + "=" * 50)
print("4. TEXTBLOB")
print("=" * 50)

start = time.time()

textblob_df = run_textblob(
    input_path=input_path,
    output_path=output_dir + "textblob_results.csv",
    text_column=text_column
)

print("TextBlob time:", round(time.time() - start, 2), "seconds")

del textblob_df
gc.collect()


# ============================================================
# 6. VADER
# ============================================================

print("\n" + "=" * 50)
print("5. VADER")
print("=" * 50)

start = time.time()

vader_df = run_vader(
    input_path=input_path,
    output_path=output_dir + "vader_results.csv",
    text_column=text_column
)

print("VADER time:", round(time.time() - start, 2), "seconds")

del vader_df
gc.collect()


# ============================================================
# 7. COMPLETED
# ============================================================

print("\n" + "=" * 50)
print("ALL SENTIMENT ANALYSES COMPLETED!")
print("=" * 50)

print("\nResults saved in:", output_dir)

print("""
Generated files:

1. distilbert_results.csv
2. siebert_results.csv
3. roberta_results.csv
4. textblob_results.csv
5. vader_results.csv
""")
