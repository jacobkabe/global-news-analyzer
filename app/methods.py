"""
methods.py - the "How the labels work" page of the dashboard.

First a diagram of the whole pipeline (sources -> collect -> clean -> NLP -> evaluate -> dashboard),
then one short visual section per step. Longer explanations sit in "Details" boxes.
All numbers (article counts, accuracy, cut-offs, how often each name rule was used) are read from the
data and from results/, so the page stays correct when the pipeline is run again.
"""
import altair as alt
import pandas as pd
import streamlit as st

import data

RESULTS = data.ROOT / "results"
GOLD = data.ROOT / "data" / "sample" / "sentiment_gold.csv"
YELLOW = "#FFDE59"


# ---- the pipeline diagram ------------------------------------------------------------------------------
def pipeline_diagram(articles):
    """Graphviz DOT text: every box is a script or a file, every arrow says what flows along it."""
    n = articles["source"].value_counts()
    rss = sum(n.get(s, 0) for s in ["bbc", "aljazeera", "npr", "foxnews"])
    n_gold = len(pd.read_csv(GOLD, keep_default_na=False))
    return f"""
digraph pipeline {{
  rankdir=TB; bgcolor="transparent"; compound=true; nodesep=0.3; ranksep=0.4; newrank=true;
  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=15 fontcolor="#EDE6E3"
        color="#5A4F4B" fillcolor="#2A2323" margin="0.18,0.1"];
  edge [color="#9A8F8B" fontcolor="#BDB3AF" fontname="Helvetica" fontsize=12 arrowsize=0.7];
  graph [fontname="Helvetica-Bold" fontsize=14 fontcolor="{YELLOW}" color="#4A4040" style="rounded"];

  subgraph cluster_sources {{ label="1 · SOURCES";
    guardian [label="Guardian API
full text
{n.get('guardian', 0):,} articles" fillcolor="#2E2F5E"];
    nyt [label="NYT API
headline + abstract
{n.get('nyt', 0):,} articles" fillcolor="#2E2F5E"];
    rss [label="RSS: BBC, Al Jazeera,
NPR, Fox News
{rss:,} articles" fillcolor="#2E2F5E"];
    newsapi [label="The News API
8 more outlets
(coming)" fillcolor="#2E2F5E" style="rounded,filled,dashed"];
  }}
  collect [label="2 · COLLECT
collect.py, twice a day
↓
news.db (SQLite)
{len(articles):,} articles" fillcolor="#3A2F1F"];
  clean [label="3 · CLEAN
clean.py
headline · topic · type
↓
articles_clean.csv" fillcolor="#3A2F1F"];

  subgraph cluster_sentiment {{ label="4a · SENTIMENT (headline)";
    roberta [label="RoBERTa
→ the label" color="{YELLOW}" penwidth=2.5];
    vader [label="VADER
baseline"];
    bart [label="BART-MNLI
zero-shot"];
  }}
  subgraph cluster_entities {{ label="4b · ENTITIES (headline + summary)";
    trf [label="spaCy transformer" color="{YELLOW}" penwidth=2.5];
    sm [label="spaCy small
baseline"];
  }}
  stories [label="4c · STORIES
(later)" style="rounded,filled,dashed"];

  explain [label="driving words
why this label?"];
  norm [label="name rules
'Trump' → Donald Trump"];
  subgraph cluster_eval {{ label="5 · EVALUATE";
    gold [label="{n_gold} hand labels
(codebook)" shape=note fillcolor="#1C2E52"];
    evaluate [label="accuracy · macro-F1
tuned cut-offs" fillcolor="#1C2E52"];
  }}
  dash [label="6 · DASHBOARD" fillcolor="{YELLOW}" fontcolor="#2E2A55" fontname="Helvetica-Bold" fontsize=17];

  guardian -> collect; nyt -> collect; rss -> collect; newsapi -> collect [style=dashed];
  collect -> clean;
  clean -> vader [lhead=cluster_sentiment];
  clean -> sm [lhead=cluster_entities];
  clean -> stories [style=dashed];
  roberta -> explain; vader -> explain;
  trf -> norm;
  bart -> evaluate [ltail=cluster_sentiment label="scores"];
  gold -> evaluate;
  roberta -> dash [label="label"];
  explain -> dash; norm -> dash;
  evaluate -> dash [label="cut-offs"];
  stories -> dash [style=dashed];
}}"""


def card(title, lines, highlight=False):
    """A small bordered box with a title and a few lines (used for sources, models and rules)."""
    with st.container(border=True):
        colour = YELLOW if highlight else "#EDE6E3"
        st.markdown(f'<p style="font-weight:700;color:{colour};margin:0">{title}</p>'
                    + "".join(f'<p class="muted" style="margin:.1rem 0">{line}</p>' for line in lines),
                    unsafe_allow_html=True)


def section(title, note=""):
    st.markdown(f'<p class="gna-h" style="margin-top:1.5rem">{title}</p>'
                + (f'<p class="muted">{note}</p>' if note else ""), unsafe_allow_html=True)


# ---- the page ------------------------------------------------------------------------------------------
def show(articles, entities, colours):
    section("How it works", "From the news outlets to every label on this dashboard. Each box is one of our scripts "
            "or files; each arrow says what flows along it. Dashed = planned.")
    st.graphviz_chart(pipeline_diagram(articles), width="stretch")

    metrics = pd.read_csv(RESULTS / "sentiment_metrics.csv")
    metrics = metrics[metrics["eval_set"] == "random150"].set_index(["model", "labels"])
    cutoffs = pd.read_csv(RESULTS / "sentiment_cutoffs.csv").set_index("model")
    gold = pd.read_csv(GOLD, keep_default_na=False)
    random = gold[gold["sample"] == "random"]

    # ---- 1-3 sources, collect, clean ----
    section("1–3 · Sources, collecting, cleaning")
    counts = articles["outlet"].value_counts()
    first = articles.groupby("outlet")["date"].min()
    text_kind = {"The Guardian": "full text (API)", "The New York Times": "headline + abstract (API)"}
    for col, (outlet, count) in zip(st.columns(len(counts)), counts.items()):
        with col:
            card(outlet, [f"{count:,} articles", f"since {first[outlet]:%d %b}",
                          text_kind.get(outlet, "title + summary (RSS)")])
    with st.expander("Details: what each step does"):
        st.markdown("""
- **Collect** (`src/collect/collect.py`, twice a day): the Guardian and NYT APIs fetch everything since the newest
  article we have, so a missed day is filled in; RSS feeds only show the latest items, so a missed day is lost.
  Everything goes into one SQLite table (`news.db`): dates in UTC, URLs without tracking parameters, each URL once.
- **We do not scrape outlet websites:** their terms forbid it or are unclear (`docs/data-sources.md`).
  Only the Guardian API gives full text; for the others we have the headline and a summary.
- **Clean** (`pipeline/clean.py`): the **headline** the models read has no HTML, no Guardian "| Author" ending,
  no series labels ("LIVE:", "Ukraine war briefing:"), and straight quotes (VADER needs them).
  Each article gets a **topic** (the outlet's section, merged: "U.S." and "US news" -> US) and an **article type**
  (news, opinion, briefing, letter, live...). "Only news" on the dashboard hides everything that is not news.
""")

    # ---- 4 sentiment models ----
    section("4 · Three sentiment models, one question",
            "Each model reads only the headline and answers our codebook question: does it report something bad "
            "(negative), something good (positive), or neither (neutral)? Each gives a <b>score</b> from −1 to +1.")

    def acc(model, labels):
        return f"{metrics.loc[(model, labels), 'accuracy']:.0%}"

    v_low, v_high = cutoffs.loc["vader"]
    b_low, b_high = cutoffs.loc["bart_mnli"]
    c1, c2, c3 = st.columns(3)
    with c1:
        card("RoBERTa · gives the label", [
            "transformer trained on tweets, has a neutral class",
            "label: the most likely of its 3 classes",
            f"accuracy {acc('roberta', 'default')}",
            "why: real neutral class, fast, explains itself"], highlight=True)
    with c2:
        card("VADER · baseline", [
            "a fixed word list: every word has a value",
            f"label: negative ≤ {v_low:+.2f} · positive ≥ {v_high:+.2f}",
            f"accuracy {acc('vader', 'tuned')} (own rule {acc('vader', 'default')})",
            "why: the most transparent model"])
    with c3:
        card("BART-MNLI · zero-shot", [
            "never trained on sentiment; we just ask our question",
            f"label: negative ≤ {b_low:+.2f} · positive ≥ {b_high:+.2f}",
            f"accuracy {acc('bart_mnli', 'tuned')} (own rule {acc('bart_mnli', 'default')})",
            "why: an independent third opinion"])

    st.markdown('<p class="muted" style="margin-top:.6rem"><b>Why can BART say "Neutral" at −0.92?</b> '
                "Each model's score has its own scale. BART puts almost every news headline near −1, so on our hand "
                f"labels only scores at {b_low:+.2f} or lower were really negative. The yellow lines are the cut-offs "
                "we tuned on our labels (RoBERTa has none: it uses its own label).</p>", unsafe_allow_html=True)
    scores = pd.concat([pd.DataFrame({"model": name, "score": articles[f"{m}_score"].dropna()})
                        for m, (name, _, _) in data.MODELS.items()])
    lines = pd.DataFrame([{"model": "VADER", "cut": v_low}, {"model": "VADER", "cut": v_high},
                          {"model": "BART-MNLI", "cut": b_low}, {"model": "BART-MNLI", "cut": b_high}])
    base = alt.Chart(scores).mark_bar(color=colours["neutral"]).encode(
        x=alt.X("score:Q", bin=alt.Bin(step=0.05, extent=[-1, 1]), scale=alt.Scale(domain=[-1, 1]), title="score"),
        y=alt.Y("count():Q", title="headlines")).properties(height=130, width=270)
    charts = []
    for name in ["RoBERTa", "VADER", "BART-MNLI"]:
        part = base.transform_filter(alt.datum.model == name).properties(title=name)
        cuts = lines[lines["model"] == name]
        if not cuts.empty:
            part = part + alt.Chart(cuts).mark_rule(color=YELLOW, strokeWidth=2).encode(x="cut:Q")
        charts.append(part)
    st.altair_chart(alt.hconcat(*charts))

    # ---- 5 evaluation ----
    section("5 · How good are they? Our hand labels",
            f"{len(random)} headlines picked at random (spread across outlets) and labelled by hand with our codebook. "
            "Bars: macro-F1 (all three labels count equally). Lines: 95% interval. "
            "Orange dashed line: a \"model\" that always says neutral.")
    rows = []
    for (model, labels), r in metrics.iterrows():
        if model in data.MODELS:
            rows.append({"model": data.MODELS[model][0], "labels": "own rule" if labels == "default" else "tuned",
                         "macro_f1": r.macro_f1, "low": r.macro_f1_ci_low, "high": r.macro_f1_ci_high,
                         "accuracy": r.accuracy})
    table = pd.DataFrame(rows)
    share_neutral = (random["my_label"] == "neutral").mean()
    # "always neutral": precision = share of neutral, recall = 1, so F1 = 2p/(1+p); the other two classes get 0
    always_neutral_f1 = 2 * share_neutral / (1 + share_neutral) / 3
    bars = alt.Chart().mark_bar().encode(
        x=alt.X("labels:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("macro_f1:Q", title="macro-F1", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("labels:N", scale=alt.Scale(domain=["own rule", "tuned"], range=["#8A8080", YELLOW]), legend=None),
        tooltip=["model", "labels", alt.Tooltip("macro_f1:Q", format=".2f"), alt.Tooltip("accuracy:Q", format=".0%")])
    errors = alt.Chart().mark_errorbar(color="#EDE6E3").encode(x="labels:N", y=alt.Y("low:Q", title="macro-F1"), y2="high:Q")
    baseline = alt.Chart().mark_rule(strokeDash=[4, 4], color="#FF8A3D").encode(y=alt.datum(always_neutral_f1))
    st.altair_chart(alt.layer(bars, errors, baseline, data=table).properties(width=200, height=220)
                    .facet(column=alt.Column("model:N", sort=["RoBERTa", "VADER", "BART-MNLI"], title=None)))
    st.markdown(f"""
- **No model is clearly the best:** the intervals overlap, so {len(random)} headlines cannot separate them.
  RoBERTa gives the label for practical reasons: a real neutral class, fast, and it explains its own decisions.
- **Accuracy alone flatters:** always saying "neutral" is already right {share_neutral:.0%} of the time.
- **Positive labels are unreliable:** good news is rare ({(random['my_label'] == 'positive').sum()} of {len(random)})
  and models confuse positive *words* with good *news* ("Who's Winning the Race for Congress?").
""")
    with st.expander("Details: the full results table, the cut-offs, our labels"):
        full = metrics.reset_index()[["model", "labels", "accuracy", "macro_f1", "f1_negative", "f1_neutral", "f1_positive"]]
        st.dataframe(full.style.format({c: "{:.2f}" for c in full.columns[2:]}), hide_index=True)
        changed = (gold["my_label"] != gold["label_before_review"]).sum() if "label_before_review" in gold else 0
        st.markdown(f"""
- **Tuned cut-offs:** neutral if the score is between two cut-offs, chosen to match our labels best (best macro-F1).
  The "tuned" results use 5-fold cross-validation: cut-offs are always tested on headlines they were not tuned on.
  The dashboard uses cut-offs tuned on all {len(gold)} labelled headlines.
- **The {len(random)} "random" headlines are spread evenly over outlets** (20 each for the small RSS outlets), while
  most of our data is from the Guardian and the NYT.
- **Our labels:** one annotator, then a second pass against the codebook (12 Nov 2026) changed {changed} labels,
  mostly war, violence and threats that had been labelled neutral. The old label is kept in
  `data/sample/sentiment_gold.csv`; the codebook is in `data/sample/README.md`.
- {len(gold) - len(random)} more headlines were picked because a model called them positive (to test the positive
  class); they are not random, so the numbers above do not use them. Failures: `failures/sentiment_gold_vs_models.csv`.
- DistilBERT (trained on movie reviews, no neutral class) is kept as a failure example and not shown on the dashboard.
""")

    # ---- driving words ----
    section("Why this label? Driving words")
    w1, w2 = st.columns(2)
    with w1:
        card("RoBERTa: hide one word, score again", [
            "“flood disaster” scores −0.62; “flood &lt;mask&gt;” scores −0.20",
            "so “disaster” = −0.62 − (−0.20) = <b>−0.42</b>: it pushed the headline towards negative",
            "orange = towards negative, blue = towards positive"])
    with w2:
        card("VADER: its word list", [
            "every word has a fixed value from −4 to +4 (shown divided by 4)",
            "“killed” = −0.88, “win” = +0.70",
            "a word that is not in the list counts 0 (e.g. “deadly”)"])
    with st.expander("Details: limits of the driving words"):
        st.markdown("""
- At most 4 words are shown, only words that change the score by at least 0.05, and no filler words
  (*the, of, to…*; "not", "no", "never", "without" are always shown, they often decide the meaning).
- **Why a mask and not deleting the word:** deleting can break the sentence. "US workers facing deadly heat borrow
  from…" without "heat" reads "facing deadly borrow from…", and "heat" looked positive (+0.33) although it is not.
  With a mask the sentence keeps its shape: *win* +0.36 and *deadly* −0.20 are the words that matter.
  On our 230 hand-labelled headlines the masked words matter more in 45% of headlines, the deleted words in 25%
  (`docs/findings.md`).
- If two words say the same thing ("killed in crash"), hiding one changes little, so both look weak.
- The words explain the **score**, not the label: the label is the most likely of the three probabilities, which
  the "Why these labels?" box shows above the words.
""")

    # ---- entities ----
    section("People, places and organisations",
            "spaCy's transformer model finds names in the headline + summary; our rules then give the same person or "
            "place always the same name. The \"Why these labels?\" box shows the rule for each name.")
    how = entities["how"].value_counts()
    steps = [("found as written", "same", "“Yemen”"),
             ("our alias list", "alias", "“US” → United States"),
             ("full name in the same article", "article", "“Trump” → Donald Trump"),
             ("most common full name overall", "corpus", "“Trump” → Donald Trump")]
    for col, (title, key, example) in zip(st.columns(4), steps):
        with col:
            card(title, [example, f"{how.get(key, 0):,} names"])
    with st.expander("Details: entity types and rules"):
        st.markdown("""
- Types: **PER** person · **LOC** place (countries, cities, buildings) · **ORG** organisation ·
  **GRP** group ("Republicans", "Russian") · **EVT** event.
- Before the rules every name is cleaned: no "the", no 's, no titles (President, Mr, Sen.), dots and capitals ignored.
- The alias list (`pipeline/entity_aliases.csv`) wins over the other rules. "Surname → full name" is used only if
  one full name has at least 80% of the mentions (so "Trump" is Donald Trump, not Melania Trump).
- Compared with the small spaCy model, which relies on capitals (NYT Title Case gives "Trump Says" as an organisation).
  The evaluation on our hand labels is still to come.
""")

    with st.expander("Other parts of the dashboard: trending, outlets compared"):
        st.markdown("""
- **Trending entities:** articles mentioning a name in the 7 days up to the end of the date range, against the
  7 days before. At least 3 articles; "new" = not mentioned the week before.
- **Outlets compared:** an outlet's average RoBERTa score minus the average of all outlets. A gap can come from the
  topics an outlet covers, and RoBERTa is more accurate on some outlets than others. The fair comparison, the same
  story in every outlet, comes with story clustering.
""")
