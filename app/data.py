"""
data.py - loads the pipeline outputs for the dashboard and joins them into tables that are easy to show.

Everything comes from data/raw/outputs/ (made by the pipeline, see pipeline/README.md):
  articles_clean.csv      one row per article (step 1, clean.py)
  sentiment.csv           one row per article and model (step 2, sentiment.py run)
  sentiment_explain.csv   the driving words of each label (step 2, sentiment.py explain)
  entities.csv            people, places, organisations... after name normalisation (step 3, ner.py)
and the tuned cut-offs from results/sentiment_cutoffs.csv (sentiment.py evaluate).

Which sentiment label we show (the reasons are in pipeline/README.md, step 2):
  roberta     its own label. This is THE label of the dashboard (best accuracy on our hand labels).
  vader       score with our tuned cut-offs (own labels: 57% accuracy, tuned: 70%).
  bart_mnli   score with our tuned cut-offs (its own labels almost never say "neutral").
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "data" / "raw" / "outputs"
CUTOFFS = ROOT / "results" / "sentiment_cutoffs.csv"

# model -> (name shown on the dashboard, what kind of model, which label we use)
MODELS = {
    "roberta": ("RoBERTa", "transformer", "own label"),
    "vader": ("VADER", "word list (baseline)", "tuned cut-offs"),
    "bart_mnli": ("BART-MNLI", "zero-shot", "tuned cut-offs"),
}

# entity type -> short tag on the chips (like the mockup in the project proposal)
ENTITY_TAGS = {"person": "PER", "place": "LOC", "organisation": "ORG", "group": "GRP", "event": "EVT"}

# entities.csv column "how" -> which rule of ner.py gave the final name, in plain words
HOW_TEXT = {
    "same": "name as found in the text",
    "alias": "our alias list (pipeline/entity_aliases.csv)",
    "article": "short name -> full name used in the same article",
    "corpus": "surname -> the most common full name in all articles",
}


def check_files():
    """Return the list of pipeline files that are missing (empty list = all there)."""
    needed = ["articles_clean.csv", "sentiment.csv", "sentiment_explain.csv", "entities.csv"]
    return [name for name in needed if not (OUTPUTS / name).exists()]


def label_from_score(score, low, high):
    """Our tuned rule: negative if score <= low, positive if score >= high, otherwise neutral."""
    if score <= low:
        return "negative"
    if score >= high:
        return "positive"
    return "neutral"


def parse_words(text):
    """'killed:-0.88; crash:-0.42' -> [('killed', -0.88), ('crash', -0.42)]"""
    if not isinstance(text, str) or not text:
        return []
    words = []
    for part in text.split("; "):
        word, _, effect = part.rpartition(":")
        words.append((word, float(effect)))
    return words


def load_articles():
    """One row per article, with the sentiment of each model and the driving words as extra columns."""
    articles = pd.read_csv(OUTPUTS / "articles_clean.csv")
    articles["date"] = pd.to_datetime(articles["date"]).dt.date

    # sentiment.csv has one row per (article, model): make one column per model instead
    sentiment = pd.read_csv(OUTPUTS / "sentiment.csv")
    cutoffs = pd.read_csv(CUTOFFS).set_index("model")
    for model in MODELS:
        rows = sentiment[sentiment["model"] == model].set_index("article_id")
        articles[f"{model}_score"] = articles["article_id"].map(rows["score"])
        if model == "roberta":
            articles[f"{model}_label"] = articles["article_id"].map(rows["label"])
            for p in ["p_neg", "p_neu", "p_pos"]:          # its three probabilities, to explain the label
                articles[f"roberta_{p}"] = articles["article_id"].map(rows[p])
        else:
            low = cutoffs.loc[model, "negative_if_score_at_most"]
            high = cutoffs.loc[model, "positive_if_score_at_least"]
            articles[f"{model}_label"] = articles[f"{model}_score"].apply(
                lambda s: label_from_score(s, low, high) if pd.notna(s) else None)

    # driving words: only RoBERTa and VADER have them (see "explain" in sentiment.py)
    explain = pd.read_csv(OUTPUTS / "sentiment_explain.csv", keep_default_na=False)
    for model in ["roberta", "vader"]:
        rows = explain[explain["model"] == model].set_index("article_id")
        articles[f"{model}_words"] = articles["article_id"].map(rows["top_words"]).apply(parse_words)

    # headlines without a RoBERTa label (not scored yet) are shown as "not analysed"
    articles["roberta_label"] = articles["roberta_label"].fillna("not analysed")
    return articles


def load_entities():
    """One row per (article, entity): an entity mentioned twice in one article counts once."""
    entities = pd.read_csv(OUTPUTS / "entities.csv")
    entities = entities.drop_duplicates(["article_id", "entity"])
    entities["tag"] = entities["type"].map(ENTITY_TAGS)
    return entities[["article_id", "entity", "type", "tag", "text", "how"]]
