
# ============================================================
# DISTILBERT SENTIMENT ANALYSIS
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

def load_distilbert():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    print("Using device:", device)
    print("Loading DistilBERT...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id
    ).to(device)

    model.eval()

    print("DistilBERT loaded successfully!")

    return model, tokenizer, device


# ============================================================
# 2. ANALYZE SENTIMENT
# ============================================================

def analyze_titles(texts, model, tokenizer, device, batch_size=8):

    results = []

    id2label = model.config.id2label

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="DistilBERT analysis"
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

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

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

def run_distilbert(
    input_path,
    output_path,
    text_column="title",
    batch_size=8
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

    # Load model
    model, tokenizer, device = load_distilbert()

    # Run sentiment analysis
    print("\nStarting DistilBERT analysis...")

    start_time = time.time()

    distilbert_results = analyze_titles(
        all_texts,
        model,
        tokenizer,
        device,
        batch_size=batch_size
    )

    elapsed = time.time() - start_time

    print("\nDistilBERT completed!")
    print("Time:", round(elapsed / 60, 2), "minutes")

    # Add results to original dataset
    df["distilbert_sentiment"] = (
        distilbert_results["sentiment"].values
    )

    df["distilbert_confidence"] = (
        distilbert_results["confidence"].values
    )

    # Display results
    display(df.head(10)) if "display" in globals() else print(df.head(10))

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nDistilBERT results saved!")
    print("Output:", output_path)

    # Release memory
    del model
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df
