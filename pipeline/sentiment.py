"""
sentiment.py - step 2 of the pipeline: headline sentiment with 4 models, evaluated on our hand labels.

Commands
  run        score headlines with the models -> data/raw/outputs/sentiment.csv
  evaluate   compare the models with data/sample/sentiment_gold.csv -> results/, failures/
  explain    "driving words" for each headline -> data/raw/outputs/sentiment_explain.csv

Examples (from the project folder)
  python pipeline\sentiment.py run --gold-only          only the 230 hand-labelled headlines (a few minutes)
  python pipeline\sentiment.py run                      all headlines (BART-MNLI takes the longest)
  python pipeline\sentiment.py run --models vader roberta
  python pipeline\sentiment.py evaluate
  python pipeline\sentiment.py explain --gold-only      (about 10 minutes for all headlines)

How "explain" works (each label is explained by the model that gave it)
  roberta     occlusion: remove one word, score the headline again; the change is that word's effect.
              "flood disaster" -> removing "disaster" moves the score from -0.62 to -0.20: disaster -0.42
  vader       the word's value in VADER's word list (VADER also applies rules for "not", "but", CAPS, "!",
              so the words explain most, but not all, of its score)
  Saved as text, strongest first, e.g.  "disaster:-0.42; rescuers:+0.15". Filler words are not shown.
  A RoBERTa effect can be above 1: removing one word can flip the score from +0.7 to -0.7 (a change of 1.4).
  Limit: if two words say the same ("killed in crash"), removing one changes little, so both look weak.

Models (every model gives p_neg, p_neu, p_pos and score = p_pos - p_neg, from -1 to +1)
  vader       baseline: word list with scores; p_* are VADER's word shares, score is its "compound"
  roberta     cardiffnlp/twitter-roberta-base-sentiment-latest: trained on tweets, has neutral
  bart_mnli   facebook/bart-large-mnli zero-shot: asked our codebook question, "this headline reports
              something bad / something good / neither good nor bad", with no sentiment training
  distilbert  distilbert-base-uncased-finetuned-sst-2-english: trained on movie reviews, has NO neutral.
              Kept as a failure example.

Two ways to turn a score into a label (both are evaluated)
  default     the model's own label (VADER: compound >= 0.05 positive, <= -0.05 negative)
  tuned       neutral if the score is between two cut-offs, tuned on our labels to get the best macro-F1.
              The reported "tuned" numbers use 5-fold cross-validation: the cut-offs are always tuned on
              other headlines than the ones they are tested on.

Resumable: run skips (article, model) pairs that are already in sentiment.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTICLES = ROOT / "data" / "raw" / "outputs" / "articles_clean.csv"
OUTPUT = ROOT / "data" / "raw" / "outputs" / "sentiment.csv"
EXPLAIN = ROOT / "data" / "raw" / "outputs" / "sentiment_explain.csv"
GOLD = ROOT / "data" / "sample" / "sentiment_gold.csv"
RESULTS = ROOT / "results"
FAILURES = ROOT / "failures"

MODELS = ["vader", "roberta", "bart_mnli", "distilbert"]
LABELS = ["negative", "neutral", "positive"]
COLS = ["article_id", "model", "p_neg", "p_neu", "p_pos", "score", "label"]
SAVE_EVERY = 256   # rows; the file is saved this often, so a stopped run loses little
EXPLAIN_MODELS = ["vader", "roberta"]
MIN_EFFECT = 0.05  # words that change the score by less than this are not shown
TOP_WORDS = 4

ZERO_SHOT_LABELS = {   # our codebook question, as the zero-shot model's choices
    "something bad": "negative",
    "neither good nor bad": "neutral",
    "something good": "positive",
}


# ---- models: each returns a list of (p_neg, p_neu, p_pos, label) for a list of headlines ----------
def load_model(name):
    """Return a function headlines -> results. Big libraries are imported only when needed."""
    if name == "vader":
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

        def vader(texts):
            out = []
            for t in texts:
                s = analyzer.polarity_scores(t)
                label = "positive" if s["compound"] >= 0.05 else "negative" if s["compound"] <= -0.05 else "neutral"
                out.append((s["neg"], s["neu"], s["pos"], label, s["compound"]))
            return out
        return vader

    import torch
    from transformers import pipeline

    if name == "bart_mnli":
        clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

        def bart(texts):
            out = []
            for r in clf(list(texts), candidate_labels=list(ZERO_SHOT_LABELS),
                         hypothesis_template="This headline reports {}.", batch_size=8):
                p = {ZERO_SHOT_LABELS[l]: s for l, s in zip(r["labels"], r["scores"])}
                out.append(_from_probs(p))
            return out
        return bart

    model_id = {
        "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "distilbert": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    }[name]
    clf = pipeline("text-classification", model=model_id, top_k=None, device=-1)

    def transformer(texts):
        with torch.no_grad():
            results = clf(list(texts), batch_size=16, truncation=True)
        return [_from_probs({r["label"].lower(): r["score"] for r in rows}) for rows in results]
    return transformer


def _from_probs(p):
    """{'negative': .., 'neutral': .., 'positive': ..} (neutral may be missing) -> one result row."""
    p_neg, p_neu, p_pos = p.get("negative", 0.0), p.get("neutral", 0.0), p.get("positive", 0.0)
    label = LABELS[int(np.argmax([p_neg, p_neu, p_pos]))]
    return (p_neg, p_neu, p_pos, label, p_pos - p_neg)


# ---- run --------------------------------------------------------------------------------------------
def read_existing():
    if OUTPUT.exists():
        return pd.read_csv(OUTPUT)
    return pd.DataFrame(columns=COLS)


def run(models, gold_only):
    if not ARTICLES.exists():
        sys.exit(f"{ARTICLES} not found. Run python pipeline\\clean.py first.")
    articles = pd.read_csv(ARTICLES, keep_default_na=False)
    if gold_only:
        gold_ids = set(pd.read_csv(GOLD)["id"])
        articles = articles[articles["article_id"].isin(gold_ids)]

    done = read_existing()
    rows = done.to_dict("records")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for name in models:
        already = set(done.loc[done["model"] == name, "article_id"])
        todo = articles[~articles["article_id"].isin(already)]
        print(f"\n{name}: {len(todo)} headlines to score ({len(already)} already done)")
        if todo.empty:
            continue
        predict = load_model(name)
        start = time.time()
        for i in range(0, len(todo), SAVE_EVERY):
            part = todo.iloc[i:i + SAVE_EVERY]
            for article_id, (p_neg, p_neu, p_pos, label, score) in zip(part["article_id"], predict(part["headline"])):
                rows.append({"article_id": article_id, "model": name, "p_neg": round(p_neg, 4),
                             "p_neu": round(p_neu, 4), "p_pos": round(p_pos, 4),
                             "score": round(score, 4), "label": label})
            pd.DataFrame(rows, columns=COLS).to_csv(OUTPUT, index=False)
            n = min(i + SAVE_EVERY, len(todo))
            print(f"  {n}/{len(todo)}  ({time.time() - start:.0f} s)")

    print(f"\nsaved to {OUTPUT}")


# ---- explain ----------------------------------------------------------------------------------------
def words_of(headline):
    """The headline's words, without punctuation around them ('"Thrilled":' -> 'Thrilled')."""
    return [w.strip(".,:;!?'\"()[]") for w in headline.split()]


def as_text(effects):
    """[('disaster', -0.42), ...] -> 'disaster:-0.42; ...', strongest first.
    Small effects and filler words ('to', 'is', 'and') are left out: removing them changes the
    score a little, but they are not a reason a person would accept."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    effects = [(w, e) for w, e in effects
               if w and w.lower() not in ENGLISH_STOP_WORDS and abs(e) >= MIN_EFFECT]
    effects.sort(key=lambda we: -abs(we[1]))
    return "; ".join(f"{w}:{e:+.2f}" for w, e in effects[:TOP_WORDS])


def explain_vader(headlines):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    lexicon = SentimentIntensityAnalyzer().lexicon
    # VADER's word values go from -4 to +4; divide by 4 to show them on the same -1..+1 scale as the score
    return [as_text([(w, lexicon.get(w.lower(), 0.0) / 4) for w in words_of(h)]) for h in headlines]


def explain_roberta(headlines):
    """Occlusion: score the headline once, then once without each word."""
    predict = load_model("roberta")
    out = []
    for h in headlines:
        tokens = h.split()
        variants = [h] + [" ".join(tokens[:i] + tokens[i + 1:]) for i in range(len(tokens))]
        scores = [r[4] for r in predict(variants)]
        full = scores[0]
        out.append(as_text([(w, full - s) for w, s in zip(words_of(h), scores[1:])]))
    return out


def explain(gold_only, limit):
    articles = pd.read_csv(ARTICLES, keep_default_na=False)
    if gold_only:
        articles = articles[articles["article_id"].isin(set(pd.read_csv(GOLD)["id"]))]
    if limit:
        articles = articles.head(limit)

    cols = ["article_id", "model", "top_words"]
    done = pd.read_csv(EXPLAIN, keep_default_na=False) if EXPLAIN.exists() else pd.DataFrame(columns=cols)
    rows = done.to_dict("records")

    for name, explain_fn in [("vader", explain_vader), ("roberta", explain_roberta)]:
        already = set(done.loc[done["model"] == name, "article_id"])
        todo = articles[~articles["article_id"].isin(already)]
        print(f"\n{name}: {len(todo)} headlines to explain ({len(already)} already done)")
        start = time.time()
        for i in range(0, len(todo), SAVE_EVERY):
            part = todo.iloc[i:i + SAVE_EVERY]
            for article_id, words in zip(part["article_id"], explain_fn(part["headline"])):
                rows.append({"article_id": article_id, "model": name, "top_words": words})
            pd.DataFrame(rows, columns=cols).to_csv(EXPLAIN, index=False)
            print(f"  {min(i + SAVE_EVERY, len(todo))}/{len(todo)}  ({time.time() - start:.0f} s)")

    print(f"\nsaved to {EXPLAIN}")


# ---- evaluate ---------------------------------------------------------------------------------------
def labels_from_score(score, low, high):
    return np.where(score >= high, "positive", np.where(score <= low, "negative", "neutral"))


def best_cutoffs(score, gold):
    """Try all cut-off pairs on a grid and keep the one with the best macro-F1."""
    from sklearn.metrics import f1_score
    grid = np.round(np.arange(0.0, 0.951, 0.05), 2)
    best = (-1.0, -0.05, 0.05)
    for low in -grid:
        for high in grid:
            f1 = f1_score(gold, labels_from_score(score, low, high), labels=LABELS, average="macro", zero_division=0)
            if f1 > best[0]:
                best = (f1, low, high)
    return best[1], best[2]


def tuned_cross_validated(score, gold, folds=5):
    """Labels for every headline, using cut-offs tuned on the other folds only."""
    from sklearn.model_selection import StratifiedKFold
    pred = np.empty(len(gold), dtype=object)
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=42).split(score, gold):
        low, high = best_cutoffs(score[train], gold[train])
        pred[test] = labels_from_score(score[test], low, high)
    return pred


def metrics(gold, pred, rng):
    """Accuracy, macro-F1 with a 95% bootstrap interval, and F1 for each class."""
    from sklearn.metrics import accuracy_score, f1_score
    macro = lambda g, p: f1_score(g, p, labels=LABELS, average="macro", zero_division=0)
    boot = []
    for _ in range(1000):
        idx = rng.integers(0, len(gold), len(gold))
        boot.append(macro(gold[idx], pred[idx]))
    per_class = f1_score(gold, pred, labels=LABELS, average=None, zero_division=0)
    return {
        "n": len(gold),
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": macro(gold, pred),
        "macro_f1_ci_low": np.percentile(boot, 2.5),
        "macro_f1_ci_high": np.percentile(boot, 97.5),
        "f1_negative": per_class[0], "f1_neutral": per_class[1], "f1_positive": per_class[2],
        "predicted_positive": int((pred == "positive").sum()),
    }


def evaluate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    gold = pd.read_csv(GOLD, keep_default_na=False)
    articles = pd.read_csv(ARTICLES, keep_default_na=False)[["article_id", "source", "headline"]]
    pred = read_existing()
    data = gold.merge(articles, left_on="id", right_on="article_id")
    models = [m for m in MODELS if m in set(pred["model"])]
    if not models:
        sys.exit("No predictions yet. Run: python pipeline\\sentiment.py run --gold-only")

    rng = np.random.default_rng(42)
    table, by_outlet, cutoffs = [], [], []
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4))
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, models):
        p = data.merge(pred[pred["model"] == name], on="article_id", how="left")
        if p["label"].isna().any():
            sys.exit(f"{name} is missing some gold headlines. Run: python pipeline\\sentiment.py run --gold-only")
        g = p["my_label"].to_numpy()
        score = p["score"].to_numpy()
        variants = {"default": p["label"].to_numpy(), "tuned": tuned_cross_validated(score, g)}
        low, high = best_cutoffs(score, g)   # final cut-offs, tuned on all 230, for the dashboard
        cutoffs.append({"model": name, "negative_if_score_at_most": low, "positive_if_score_at_least": high})

        for variant, labels in variants.items():
            data[f"{name}_{variant}"] = labels
            for eval_set, mask in [("random150", p["sample"] == "random"), ("all230", p["sample"] != "")]:
                row = metrics(g[mask], labels[mask], rng)
                table.append({"model": name, "labels": variant, "eval_set": eval_set, **row})
            for source, part in p[p["sample"] == "random"].groupby("source"):
                by_outlet.append({"model": name, "labels": variant, "source": source, "n": len(part),
                                  "accuracy": (labels[part.index] == g[part.index]).mean()})
        data[f"{name}_score"] = score

        # confusion matrix: tuned labels on the random 150
        m = (p["sample"] == "random").to_numpy()
        cm = confusion_matrix(g[m], variants["tuned"][m], labels=LABELS)
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("model (tuned)")
        ax.set_ylabel("our label")
        ax.set_title(name)

    RESULTS.mkdir(exist_ok=True)
    FAILURES.mkdir(exist_ok=True)
    fig.suptitle("Sentiment: our labels vs. models (random 150 headlines)")
    fig.tight_layout()
    fig.savefig(RESULTS / "sentiment_confusion.png", dpi=150)

    table = pd.DataFrame(table).round(3)
    table.to_csv(RESULTS / "sentiment_metrics.csv", index=False)
    pd.DataFrame(by_outlet).round(3).to_csv(RESULTS / "sentiment_by_outlet.csv", index=False)
    pd.DataFrame(cutoffs).to_csv(RESULTS / "sentiment_cutoffs.csv", index=False)

    # failure examples: every gold headline with all model labels, the ones most models get wrong first
    tuned_cols = [f"{m}_tuned" for m in models]
    data["n_models_wrong"] = sum((data[c] != data["my_label"]).astype(int) for c in tuned_cols)
    cols = ["id", "sample", "source", "headline", "my_label", "n_models_wrong"] + \
           [c for m in models for c in (f"{m}_tuned", f"{m}_score")]
    if EXPLAIN.exists():   # add the driving words, so a failure also shows why the model decided so
        words = pd.read_csv(EXPLAIN, keep_default_na=False)
        for m in EXPLAIN_MODELS:
            w = words[words["model"] == m].set_index("article_id")["top_words"].rename(f"{m}_words")
            data = data.join(w, on="article_id")
            cols.append(f"{m}_words")
    data[cols].sort_values(["n_models_wrong", "id"], ascending=[False, True]).round(3) \
        .to_csv(FAILURES / "sentiment_gold_vs_models.csv", index=False)

    show = table[table["eval_set"] == "random150"][["model", "labels", "accuracy", "macro_f1",
                                                    "macro_f1_ci_low", "macro_f1_ci_high",
                                                    "f1_negative", "f1_neutral", "f1_positive"]]
    print("\nRandom 150 headlines (main result):\n" + show.to_string(index=False))
    print(f"\nsaved: results/sentiment_metrics.csv, sentiment_by_outlet.csv, sentiment_cutoffs.csv, "
          f"sentiment_confusion.png; failures/sentiment_gold_vs_models.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="score headlines")
    p_run.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p_run.add_argument("--gold-only", action="store_true", help="only the hand-labelled headlines")
    sub.add_parser("evaluate", help="compare with our hand labels")
    p_explain = sub.add_parser("explain", help="driving words for VADER and RoBERTa")
    p_explain.add_argument("--gold-only", action="store_true", help="only the hand-labelled headlines")
    p_explain.add_argument("--limit", type=int, help="only the first N headlines (quick test)")
    args = parser.parse_args()

    if args.command == "run":
        run(args.models, args.gold_only)
    elif args.command == "explain":
        explain(args.gold_only, args.limit)
    else:
        evaluate()
