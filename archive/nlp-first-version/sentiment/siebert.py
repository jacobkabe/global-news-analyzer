
# ============================================================
# SIEBERT SENTIMENT ANALYSIS
# ============================================================

import pandas as pd
import torch
import time
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from tqdm.auto import tqdm


# ============================================================
# 1. LOAD MODEL
# ============================================================

def load_siebert():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_id = "siebert/sentiment-roberta-large-english"

    print("Using device:", device)
    print("Loading SiEBERT...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id
    ).to(device)

    model.eval()

    print("SiEBERT loaded successfully!")

    return model, tokenizer, device


# ============================================================
# 2. ANALYZE SENTIMENT
# ============================================================

def analyze_titles(texts, model, tokenizer, device, batch_size=4):

    results = []

    id2label = model.config.id2label

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="SiEBERT analysis"
    ):

        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)

        predictions = torch.argmax(probabilities, dim=1)

        for j in range(len(batch)):

            pred_id = predictions[j].item()

            results.append({
                "sentiment": id2label[pred_id].lower(),
                "confidence": probabilities[j][pred_id].item()
            })

    return pd.DataFrame(results)


# ============================================================
# 3. RUN COMPLETE ANALYSIS ON CSV
# ============================================================

def run_siebert(
    input_path,
    output_path,
    text_column="title",
    batch_size=4
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

    model, tokenizer, device = load_siebert()

    print("\nStarting SiEBERT analysis...")

    start_time = time.time()

    siebert_results = analyze_titles(
        all_texts,
        model,
        tokenizer,
        device,
        batch_size=batch_size
    )

    elapsed = time.time() - start_time

    print("\nSiEBERT completed!")
    print("Time:", round(elapsed / 60, 2), "minutes")

    df["siebert_sentiment"] = siebert_results["sentiment"].values
    df["siebert_confidence"] = siebert_results["confidence"].values

    display(df.head(10)) if "display" in globals() else print(df.head(10))

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nSiEBERT results saved!")
    print("Output:", os.path.abspath(output_path))

    del model
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df
