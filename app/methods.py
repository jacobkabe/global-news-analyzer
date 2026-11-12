"""
methods.py - the "How the labels work" page of the dashboard: every rule behind a label, and why we chose it.

The numbers (cut-offs, accuracy, how often each name rule was used) are read from the pipeline results,
so this page stays correct when the pipeline is run again.
"""
import altair as alt
import pandas as pd
import streamlit as st

import data

RESULTS = data.ROOT / "results"


def show(articles, entities, colours):
    st.markdown("""
Every label on this dashboard comes from a model or a fixed rule, and every rule is described here.
Our "no black box" rule: for each label you can see **which model gave it** and **which words or rules
decided it** (the "Why these labels?" box on the Articles page).

| What you see | Where it comes from | Pipeline step |
|---|---|---|
| Sentiment label and score | RoBERTa on the **headline only** | 2, `pipeline/sentiment.py` |
| Driving words | RoBERTa itself (remove a word, see how the score changes); VADER's word list | 2, `sentiment.py explain` |
| People, places, organisations | spaCy transformer model + our name rules | 3, `pipeline/ner.py` |
| Topic | the outlet's own section (`U.S.` and `US news` -> `US`), no model yet | 1, `pipeline/clean.py` |
| "Only news" | the article type from its section or headline (opinion, briefing, letter, live...) | 1, `clean.py` |
""")

    # ---- sentiment ----
    st.header("1. Sentiment")
    st.markdown("""
**The question every model answers** (our codebook, `data/sample/README.md`): does the headline report something
bad (negative), something good (positive), or neither (neutral)? We judge the **event**, not the writing style.

Every model gives three probabilities (negative, neutral, positive) and a **score = positive − negative**,
from −1 (surely negative) to +1 (surely positive).

| Model | What it is | Why it is here |
|---|---|---|
| **RoBERTa** (`twitter-roberta-base-sentiment-latest`) | transformer trained on tweets, has a neutral class | **gives the label of the dashboard** |
| **VADER** | a fixed word list with a value for each word | the baseline the course asks for; the most transparent |
| **BART-MNLI** (zero-shot) | never trained on sentiment; we only ask it our codebook question | a third, independent opinion |
| DistilBERT SST-2 | trained on movie reviews, has no neutral class | not shown: kept as a failure example |
""")

    st.subheader("From a score to a label: the cut-offs")
    cutoffs = pd.read_csv(RESULTS / "sentiment_cutoffs.csv").set_index("model")
    v_low, v_high = cutoffs.loc["vader"]
    b_low, b_high = cutoffs.loc["bart_mnli"]
    st.markdown(f"""
A model's own label is not always useful. So we tried a second rule for every model and **tuned it on our
230 hand-labelled headlines**: neutral if the score is between two cut-offs, chosen to match our labels best
(best macro-F1; checked with 5-fold cross-validation, so the cut-offs are always tested on other headlines
than the ones they were tuned on).

| Model | Label on the dashboard | Negative if score ≤ | Positive if score ≥ |
|---|---|---|---|
| RoBERTa | its own label (the class with the highest probability) | – | – |
| VADER | tuned cut-offs | {v_low:+.2f} | {v_high:+.2f} |
| BART-MNLI | tuned cut-offs | {b_low:+.2f} | {b_high:+.2f} |

**Why BART-MNLI can say "Neutral" at −0.92.** BART answers "something bad" with high confidence for almost
every news headline: {(articles['bart_mnli_score'] < 0).mean():.0%} of our headlines get a score below zero,
and its own labels called almost none of them neutral. Its score is still useful for **ranking** (a worse event gets a lower score),
but its zero is in the wrong place. On our hand labels, only headlines at {b_low:+.2f} or lower were really
negative. So for BART, −0.92 means "ordinary news", about what −0.1 means for RoBERTa.
The charts show it: each model's scores for all headlines, with the cut-offs as lines.
""")

    rows = []
    for model, (name, _, _) in data.MODELS.items():
        scores = articles[f"{model}_score"].dropna()
        rows.append(pd.DataFrame({"model": name, "score": scores}))
    scores = pd.concat(rows)
    lines = pd.DataFrame([
        {"model": "VADER", "cut": v_low}, {"model": "VADER", "cut": v_high},
        {"model": "BART-MNLI", "cut": b_low}, {"model": "BART-MNLI", "cut": b_high},
    ])
    # one histogram per model, with the cut-offs drawn as yellow lines
    chart = alt.Chart(scores).mark_bar(color=colours["neutral"]).encode(
        x=alt.X("score:Q", bin=alt.Bin(step=0.05, extent=[-1, 1]), scale=alt.Scale(domain=[-1, 1]),
                title="score (positive − negative)"),
        y=alt.Y("count():Q", title="headlines"),
    ).properties(height=150, width=280)
    charts = []
    for name in ["RoBERTa", "VADER", "BART-MNLI"]:
        part = chart.transform_filter(alt.datum.model == name).properties(title=name)
        cuts = lines[lines["model"] == name]
        if not cuts.empty:
            part = part + alt.Chart(cuts).mark_rule(color="#FFDE59", strokeWidth=2).encode(x="cut:Q")
        charts.append(part)
    st.altair_chart(alt.hconcat(*charts))
    st.caption("Yellow lines: our tuned cut-offs (left of the left line = negative, right of the right line = positive). "
               "RoBERTa has no lines: it uses its own label.")

    st.subheader("How good is each model? (our 150 random hand-labelled headlines)")
    metrics = pd.read_csv(RESULTS / "sentiment_metrics.csv")
    metrics = metrics[metrics["eval_set"] == "random150"]
    table = metrics.pivot(index="model", columns="labels", values=["accuracy", "macro_f1"])
    table.columns = [f"{'Accuracy' if m == 'accuracy' else 'Macro-F1'}, {'own label' if l == 'default' else 'tuned'}"
                     for m, l in table.columns]
    table = table.loc[["roberta", "vader", "bart_mnli", "distilbert"]].rename(
        index={"vader": "VADER", "roberta": "RoBERTa", "bart_mnli": "BART-MNLI", "distilbert": "DistilBERT"})
    table.index.name = "Model"
    st.dataframe(table.style.format("{:.2f}"))
    st.markdown("""
- **Accuracy**: share of headlines where the model agrees with us. **Macro-F1**: the average F1 of the three
  labels, so the rare positive label counts as much as the common neutral one.
- **Why RoBERTa gives the label:** highest accuracy, a real neutral class (most of our headlines are neutral),
  and about 20x faster than BART-MNLI. VADER-tuned and BART-tuned have a similar macro-F1, but 150 headlines
  are too few to tell them apart (the confidence intervals overlap).
- **Known weakness of every model:** good news told with bad words, e.g. *"Rescuers find two more people alive
  after flash flood disaster"* is called negative. Positive news is rare (8 of 150) and often missed.
  All failures: `failures/sentiment_gold_vs_models.csv`.
""")

    st.subheader("Driving words: why this label?")
    st.markdown("""
**Each label is explained by the model that gave it**, so the explanation never contradicts the label.

- **RoBERTa – occlusion.** We score the headline again without one word at a time. The change in the score is that
  word's effect. Example: *"flood disaster"* scores −0.62; without *"disaster"* it scores −0.20, so
  *disaster* = **−0.42** (it pushed the headline towards negative). In the headline, orange words push towards
  negative, blue words towards positive; the stronger the colour, the bigger the effect.
- **VADER – its word list.** Each word has a fixed value from −4 to +4 in VADER's list; we divide by 4 to show it on
  the same −1..+1 scale. VADER also has rules for "not", "but", CAPITALS and "!", so the words explain most,
  but not all, of its score.
- We show at most **4 words**, only words that change the score by **at least 0.05**, and no filler words
  (*to, is, and...*).
- **Limit:** if two words say the same thing (*"killed in crash"*), removing one changes little, so both look weak.
""")

    # ---- entities ----
    st.header("2. People, places and organisations")
    how = entities["how"].value_counts()
    st.markdown(f"""
We look for names in the **headline and the summary** with two spaCy models and compare them on our labels:
`en_core_web_sm` (small, fast baseline; relies on capital letters, so NYT Title Case headlines give mistakes like
"Trump Says" as an organisation) and `en_core_web_trf` (a transformer that reads the whole sentence).
**The dashboard uses the transformer.**

Types: **PER** person · **LOC** place (countries, cities, buildings) · **ORG** organisation ·
**GRP** group ("Republicans", "Russian") · **EVT** event.

**Name rules**: the same person or place should always have the same name, otherwise "Trump", "Donald Trump" and
"President Trump's" would be three different people. The "Why these labels?" box shows which rule gave each name.

| Rule | What it does | Example | Names decided by it |
|---|---|---|---|
| clean | removes "the", 's, titles (President, Mr, Sen.), dots and capitals | "President Trump's" -> "Trump" | (always applied first) |
| {data.HOW_TEXT['alias']} | our own list of {len(pd.read_csv(data.ROOT / 'pipeline' / 'entity_aliases.csv', comment='#'))} names; wins over the rules below | "US" -> "United States" | {how.get('alias', 0):,} |
| {data.HOW_TEXT['article']} | a short person name -> the longer one in the same article, if only one fits | "Trump" -> "Donald Trump" | {how.get('article', 0):,} |
| {data.HOW_TEXT['corpus']} | only if that full name has at least 80% of the mentions | "Trump" -> "Donald Trump", not "Melania Trump" | {how.get('corpus', 0):,} |
| {data.HOW_TEXT['same']} | no rule needed | "Yemen" | {how.get('same', 0):,} |

If one name gets different types ("Trump" as a person and as an organisation), the most common type wins.

**Limits:** "Trump" is always Donald Trump unless another Trump is named in the article. Spelling variants not
in our list stay separate. "Russian" (group) and "Russia" (place) stay separate on purpose.
The evaluation on our hand labels is still to come.
""")

    # ---- the rest ----
    st.header("3. Other parts of the dashboard")
    st.markdown("""
- **Trending entities:** names in the 7 days up to the end of the chosen date range, compared with the 7 days
  before. An entity needs at least 3 articles to count; "new" means it was not mentioned in the week before.
  Each article counts once, even if it mentions the name twice.
- **Outlets compared:** the gap is an outlet's average RoBERTa score minus the average of all outlets
  (with the current filters). **Careful:** outlets cover different topics, so a gap can come from the topic mix,
  not the tone. The fair comparison (the same story in every outlet) comes with story clustering (step 4).
- **Few articles = unreliable numbers:** the RSS outlets (BBC, Al Jazeera, Fox News, NPR) only started in October
  and give a few articles a day; the Guardian and the NYT have data from 31 August.
- **Coming later:** stories (articles about the same event grouped together), a story summary, a topic model.
""")
