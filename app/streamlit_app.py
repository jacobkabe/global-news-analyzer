"""
streamlit_app.py - the Global News Analyzer dashboard (version 2, reads the pipeline outputs).

Run from the project folder:
    streamlit run app\streamlit_app.py

Layout follows the mockup in our project proposal (slide 4):
  left    filters (topic, sentiment, outlets, people/places, dates)
  middle  article cards (stories come later, when pipeline step 4 groups articles into stories)
  right   "Why these labels?" for the selected article, sentiment over time, trending entities
A second tab compares the outlets; a third explains every rule behind the labels (methods.py).

The first version (app.py in the project folder, by Tala) stays as it is; ideas taken from it:
the outlet filter, the keyword search and the charts of sentiment and articles over time.
"""
import html
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

import data
import methods

st.set_page_config(page_title="Global News Analyzer", page_icon="🌍", layout="wide")

# Colours of the mockup: yellow accent, blue = positive, grey = neutral, orange = negative
YELLOW = "#FFDE59"
SENTIMENT_COLOURS = {"positive": "#6B9BFF", "neutral": "#8A8080", "negative": "#FF8A3D"}
CARDS_PER_STEP = 20

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&display=swap');
.gna-title {{ background:{YELLOW}; color:#2E2A55; font-family:Anton, sans-serif; font-size:1.9rem; white-space:nowrap;
              padding:0 14px; display:inline-block; letter-spacing:.5px; }}
.gna-h {{ font-family:Anton, sans-serif; font-size:1.45rem; letter-spacing:.5px; text-transform:uppercase;
          margin:0 0 .2rem 0; }}
.muted {{ color:#9A8F8B; font-size:.8rem !important; margin:.15rem 0 !important; }}
[data-testid="stMainBlockContainer"] {{ padding-top:2.5rem; }}
.badge {{ border:1px solid {YELLOW}; color:{YELLOW}; border-radius:999px; padding:2px 10px;
          font-size:.72rem; font-weight:700; letter-spacing:.5px; white-space:nowrap; }}
.tag {{ border-radius:6px; padding:1px 8px; font-size:.78rem; margin-right:6px; white-space:nowrap; }}
.tag-topic {{ background:#2E2F5E; color:#D6D8FF; }}
.tag-negative {{ background:#4A2A1A; color:#FF9A57; }}
.tag-positive {{ background:#1C2E52; color:#8FB3FF; }}
.tag-neutral, .tag-not {{ background:#2E2A2A; color:#C8BFBF; }}
.headline {{ color:#F3ECE8; font-weight:700; font-size:1.12rem; }}
.open-link {{ float:right; font-size:.8rem; color:{YELLOW} !important; text-decoration:none;
              position:relative; z-index:2; }}
/* click anywhere on a card: the card's button is stretched invisibly over the whole card */
[class*="st-key-card_"] {{ position:relative; cursor:pointer; }}
[class*="st-key-card_"]:hover {{ border-color:#7A6A3A !important; }}
[class*="st-key-pick_"] {{ position:absolute !important; inset:0; width:auto !important; z-index:1; margin:0 !important; }}
[class*="st-key-pick_"] div, [class*="st-key-pick_"] button {{ width:100% !important; height:100% !important; }}
[class*="st-key-pick_"] button {{ opacity:0; cursor:pointer; }}
.lead {{ color:#BDB3AF; font-size:.9rem; margin:.25rem 0 .4rem 0; }}
.chip {{ background:#2A2323; border-radius:999px; padding:2px 10px; font-size:.8rem;
         margin:0 6px 6px 0; display:inline-block; }}
.chip b {{ font-size:.66rem; color:#9AD3C8; margin-right:4px; }}
.word {{ border-radius:4px; padding:0 3px; }}
.row {{ display:flex; justify-content:space-between; align-items:center; margin:.2rem 0; font-size:.9rem; }}
.bar {{ background:#2A2323; border-radius:4px; height:8px; width:90px; display:inline-block; margin:0 10px; }}
.bar span {{ background:{YELLOW}; border-radius:4px; height:8px; display:block; }}
</style>
""", unsafe_allow_html=True)


# ---- load the data -----------------------------------------------------------------------------------
@st.cache_data
def load():
    return data.load_articles(), data.load_entities()


missing = data.check_files()
if missing:
    st.error(f"Pipeline outputs not found in data/raw/outputs/: {', '.join(missing)}. "
             "Run the pipeline first (see pipeline/README.md).")
    st.stop()

articles, entities = load()
esc = html.escape   # every text from the outlets goes through this before it is put into HTML


# ---- small helpers that build HTML ---------------------------------------------------------------------
def sentiment_tag(label, score):
    if label == "not analysed":
        return '<span class="tag tag-not">Not analysed</span>'
    return f'<span class="tag tag-{label}">{label.capitalize()} · {score:+.2f}</span>'


def entity_chips(article_entities, limit=6):
    chips = [f'<span class="chip"><b>{row.tag}</b>{esc(row.entity)}</span>'
             for row in article_entities.head(limit).itertuples()]
    return "".join(chips)


def highlight(headline, words):
    """Colour the driving words in the headline: orange pulls towards negative, blue towards positive."""
    effects = dict(words)
    out = []
    for token in headline.split():
        word = token.strip(".,:;!?'\"()[]")   # the same rule as words_of() in pipeline/sentiment.py
        if word in effects:
            effect = effects.pop(word)          # highlight only the first time the word appears
            colour = SENTIMENT_COLOURS["negative"] if effect < 0 else SENTIMENT_COLOURS["positive"]
            strength = min(0.25 + abs(effect), 0.9)
            out.append(f'<span class="word" style="background:{colour}{int(strength * 255):02x}">{esc(token)}</span>')
        else:
            out.append(esc(token))
    return " ".join(out)


def word_chips(words):
    chips = []
    for word, effect in words:
        label = "negative" if effect < 0 else "positive"
        chips.append(f'<span class="tag tag-{label}">{esc(word)} {effect:+.2f}</span>')
    return " ".join(chips) or '<span class="muted">no word changes the score much</span>'


def probability_bar(probs, label):
    """One bar split into negative / neutral / positive, as wide as RoBERTa's probability for each."""
    parts = []
    for name, p in probs.items():
        text = f"{p:.0%} {name}" if p >= 0.25 else ""   # too narrow for text: the line below has it
        border = "outline:2px solid #FFDE59;outline-offset:-2px;" if name == label else ""
        parts.append(f'<div title="{name} {p:.0%}" style="width:{p * 100:.1f}%;background:{SENTIMENT_COLOURS[name]};'
                     f'{border}color:#140F0F;font-size:.75rem;font-weight:700;display:flex;align-items:center;'
                     f'justify-content:center;white-space:nowrap;overflow:hidden">{text}</div>')
    legend = " · ".join(f"<b>{p:.0%} {n}</b>" if n == label else f"{p:.0%} {n}" for n, p in probs.items())
    return (f'<div style="display:flex;height:24px;border-radius:6px;overflow:hidden;margin:.2rem 0">'
            f'{"".join(parts)}</div><p class="muted">{legend}</p>')


def heading(text, note=""):
    st.markdown(f'<p class="gna-h">{text}</p>' + (f'<p class="muted">{note}</p>' if note else ""),
                unsafe_allow_html=True)


# ---- filters (left sidebar) --------------------------------------------------------------------------
with st.sidebar:
    heading("Filters")

    topics = st.pills("Topic", sorted(articles["topic"].dropna().unique()), selection_mode="multi",
                      help="Nothing selected = all topics. The topic comes from the outlet's section.")
    sentiments = st.pills("Sentiment (RoBERTa)", ["positive", "neutral", "negative"], selection_mode="multi",
                          format_func=str.capitalize,
                          help="Nothing selected = all. Careful with Positive: on our hand labels only about "
                               "4 in 10 headlines RoBERTa calls positive are really good news.")

    st.markdown("**Sources**")
    outlet_counts = articles["outlet"].value_counts()
    outlets = [o for o, n in outlet_counts.items() if st.checkbox(f"{o} · {n:,}", value=True, key=f"src_{o}")]

    only_news = st.toggle("Only news", value=True,
                          help="Hide opinion, briefings, letters, columns, live blogs: they are not one news event.")

    entity_counts = entities["entity"].value_counts()
    entity_types = entities.drop_duplicates("entity").set_index("entity")["tag"]
    chosen_entities = st.multiselect(
        "Country, person or org", entity_counts[entity_counts >= 2].index,
        format_func=lambda e: f"{e} ({entity_types[e]}, {entity_counts[e]})",
        placeholder="e.g. Donald Trump", help="Articles that mention all chosen names.")

    first_day, last_day = articles["date"].min(), articles["date"].max()
    dates = st.date_input("Date range", value=(first_day, last_day), min_value=first_day, max_value=last_day)


# ---- header ------------------------------------------------------------------------------------------
title_col, search_col, info_col = st.columns([1.5, 1.6, 1.4], vertical_alignment="center")
title_col.markdown('<span class="gna-title">GLOBAL NEWS ANALYZER</span>', unsafe_allow_html=True)
query = search_col.text_input("Search", placeholder="🔍  Search headlines, people, places...",
                              label_visibility="collapsed")
info = info_col.empty()   # filled in below, when we know how many articles are left


# ---- apply the filters -------------------------------------------------------------------------------
base = articles[articles["outlet"].isin(outlets)]
if only_news:
    base = base[base["content_type"] == "news"]
if topics:
    base = base[base["topic"].isin(topics)]
if sentiments:
    base = base[base["roberta_label"].isin(sentiments)]
for name in chosen_entities:
    base = base[base["article_id"].isin(entities.loc[entities["entity"] == name, "article_id"])]
if query:
    in_text = (base["headline"].str.contains(query, case=False, regex=False)
               | base["lead"].fillna("").str.contains(query, case=False, regex=False))
    in_entities = base["article_id"].isin(
        entities.loc[entities["entity"].str.contains(query, case=False, regex=False), "article_id"])
    base = base[in_text | in_entities]

# "base" ignores the dates (trending entities need the week before); "shown" is inside the date range
if len(dates) == 2:
    start, end = dates
else:                       # while the user is still picking the second date
    start, end = dates[0], last_day
shown = base[(base["date"] >= start) & (base["date"] <= end)]

info.markdown(f'<span class="muted">{len(shown):,} articles · {shown["outlet"].nunique()} sources · '
              f'{start:%d %b} – {end:%d %b %Y}</span> &nbsp; <span class="badge">PIPELINE DATA</span>',
              unsafe_allow_html=True)


# ---- tab 1: articles ---------------------------------------------------------------------------------
tab_articles, tab_outlets, tab_methods = st.tabs(["Articles", "Outlets compared", "How the labels work"])

with tab_articles:
    middle, right = st.columns([2, 1.15], gap="large")

    with middle:
        head_col, sort_col = st.columns([1.1, 1], vertical_alignment="bottom")
        with head_col:
            heading("Latest articles", "Stories (articles about the same event grouped) come with pipeline step 4.")
        sort = sort_col.segmented_control("Sort", ["Newest", "Most negative", "Most positive"],
                                          default="Newest", label_visibility="collapsed")
        if sort == "Most negative":
            ordered = shown.sort_values("roberta_score")
        elif sort == "Most positive":
            ordered = shown.sort_values("roberta_score", ascending=False)
        else:
            ordered = shown.sort_values("published_at", ascending=False)

        # which article is explained on the right: the one clicked, otherwise the first card
        if st.session_state.get("selected") not in set(ordered["article_id"]):
            st.session_state["selected"] = ordered["article_id"].iloc[0] if len(ordered) else None
        selected = st.session_state["selected"]
        st.markdown(f"<style>.st-key-card_{selected}.st-key-card_{selected} {{ border:1px solid {YELLOW} !important; }}</style>",
                    unsafe_allow_html=True)

        if ordered.empty:
            st.info("No articles match the filters.")

        def select(article_id):
            st.session_state["selected"] = article_id

        def show_more():
            st.session_state["how_many"] = st.session_state.get("how_many", CARDS_PER_STEP) + CARDS_PER_STEP

        # The cards scroll inside this box, so "Why these labels?" on the right stays in view.
        # Each card has an invisible button over the whole card (see the CSS "click anywhere on a card"):
        # a click anywhere selects the article. Only the "Open article" link sits above the button.
        how_many = st.session_state.get("how_many", CARDS_PER_STEP)
        with st.container(height=820, border=False):
            for a in ordered.head(how_many).itertuples():
                with st.container(border=True, key=f"card_{a.article_id}"):
                    st.markdown(
                        f'<span class="tag tag-topic">{esc(str(a.topic))}</span>'
                        f'{sentiment_tag(a.roberta_label, a.roberta_score)}'
                        f'<span class="muted">{esc(a.outlet)} · {a.date:%d %b %Y}</span>'
                        f'<a class="open-link" href="{esc(a.url)}" target="_blank">Open article ↗</a>'
                        f'<div class="headline">{esc(a.headline)}</div>'
                        + (f'<div class="lead">{esc(a.lead)}</div>' if isinstance(a.lead, str) else "")
                        + entity_chips(entities[entities["article_id"] == a.article_id]),
                        unsafe_allow_html=True)
                    st.button("Show why", key=f"pick_{a.article_id}", on_click=select, args=(a.article_id,))

            if len(ordered) > how_many:
                st.caption(f"Showing {how_many} of {len(ordered):,}")
                st.button("Show more", on_click=show_more)

    with right:
        # -- why these labels? --
        with st.container(border=True):
            heading("Why these labels?")
            if selected is None:
                st.caption("Select an article.")
            else:
                a = articles.set_index("article_id").loc[selected]
                st.markdown(f'<p style="font-weight:700">{highlight(a.headline, a.roberta_words)}</p>',
                            unsafe_allow_html=True)

                # RoBERTa's three probabilities as one bar: the label is the biggest part
                if pd.notna(a.roberta_p_neu):
                    probs = {"negative": a.roberta_p_neg, "neutral": a.roberta_p_neu, "positive": a.roberta_p_pos}
                    st.markdown('<p class="muted">ROBERTA · HOW SURE IS IT?</p>' + probability_bar(probs, a.roberta_label)
                                + f'<p class="muted">Label = the biggest part · score = positive − negative = '
                                  f'{a.roberta_score:+.2f}</p>', unsafe_allow_html=True)

                st.markdown('<p class="muted" style="margin-top:.6rem">3 MODELS COMPARED</p>', unsafe_allow_html=True)
                rows = ""
                for model, (name, kind, _) in data.MODELS.items():
                    label, score = a[f"{model}_label"], a[f"{model}_score"]
                    tag = sentiment_tag(label, score) if pd.notna(score) else '<span class="muted">–</span>'
                    rows += f'<div class="row"><span>{name} <span class="muted">· {kind}</span></span>{tag}</div>'
                st.markdown(rows, unsafe_allow_html=True)

                st.markdown('<p class="muted" style="margin-top:.6rem">DRIVING WORDS · RoBERTa</p>'
                            + word_chips(a.roberta_words), unsafe_allow_html=True)
                st.markdown('<p class="muted">VADER WORD LIST</p>' + word_chips(a.vader_words), unsafe_allow_html=True)

                found = entities[entities["article_id"] == selected]
                if not found.empty:
                    lines = []
                    for e in found.itertuples():
                        if e.text != e.entity:      # only a renamed name needs an explanation
                            lines.append(f'<div class="row"><span class="chip"><b>{e.tag}</b>“{esc(e.text)}” → '
                                         f'{esc(e.entity)}</span><span class="muted">{data.HOW_TEXT.get(e.how, e.how)}'
                                         f'</span></div>')
                        else:
                            lines.append(f'<span class="chip"><b>{e.tag}</b>{esc(e.entity)}</span>')
                    st.markdown('<p class="muted" style="margin-top:.6rem">ENTITIES · spaCy transformer</p>'
                                + "".join(lines), unsafe_allow_html=True)

                st.markdown(f'<p class="muted" style="margin-top:.6rem">TOPIC · from the outlet\'s section: '
                            f'<b>{esc(str(a.topic))}</b> · STORY · not grouped yet</p>', unsafe_allow_html=True)

                notes = ["RoBERTa uses its own label; VADER and BART use cut-offs tuned on our hand labels.",
                         "Names are merged by our rules (“Trump” → Donald Trump); rules and limits: "
                         "tab “How the labels work”."]
                if a.roberta_label == "positive":
                    notes.append("⚠ Positive labels are often wrong: on our hand labels only about 4 in 10 are "
                                 "really good news (positive words ≠ good news).")
                st.caption("  \n".join(notes))

        # -- trending entities: the last 7 days of the range vs the 7 days before --
        with st.container(border=True):
            heading("Trending entities", f"Articles in the 7 days to {end:%d %b} vs the 7 days before")
            week = base[(base["date"] > end - timedelta(days=7)) & (base["date"] <= end)]
            before = base[(base["date"] > end - timedelta(days=14)) & (base["date"] <= end - timedelta(days=7))]
            now_counts = entities[entities["article_id"].isin(week["article_id"])]["entity"].value_counts()
            old_counts = entities[entities["article_id"].isin(before["article_id"])]["entity"].value_counts()
            trend = pd.DataFrame({"now": now_counts, "before": old_counts}).fillna(0)
            trend = trend[trend["now"] >= 3]   # fewer than 3 articles is too little to call a trend
            trend["growth"] = trend["now"] - trend["before"]
            trend = trend.sort_values(["growth", "now"], ascending=False).head(6)
            if trend.empty:
                st.caption("Not enough articles in this week.")
            for name, row in trend.iterrows():
                change = "new" if row.before == 0 else f"{(row.now / row.before - 1) * 100:+.0f}%"
                width = int(100 * row.now / trend["now"].max())
                st.markdown(f'<div class="row"><span><span class="chip"><b>{entity_types[name]}</b>{esc(name)}</span></span>'
                            f'<span><span class="muted">{int(row.now)} art.</span><span class="bar"><span style="width:{width}%"></span></span>'
                            f'<b style="color:{YELLOW}">{change}</b></span></div>', unsafe_allow_html=True)

        # -- sentiment over time --
        with st.container(border=True):
            heading("Sentiment over time", "Articles per day, RoBERTa label")
            per_day = (shown[shown["roberta_label"] != "not analysed"]
                       .groupby(["date", "roberta_label"]).size().reset_index(name="articles"))
            if per_day.empty:
                st.caption("No articles.")
            else:
                per_day["date"] = pd.to_datetime(per_day["date"])
                order = ["positive", "neutral", "negative"]
                chart = alt.Chart(per_day).mark_bar().encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("articles:Q", title=None),
                    color=alt.Color("roberta_label:N", title=None, sort=order,
                                    scale=alt.Scale(domain=order, range=[SENTIMENT_COLOURS[s] for s in order]),
                                    legend=alt.Legend(orient="bottom")),
                    order=alt.Order("order:Q"),
                    tooltip=["date:T", "roberta_label:N", "articles:Q"],
                ).transform_calculate(order=f"indexof({order}, datum.roberta_label)").properties(height=200)
                st.altair_chart(chart, width="stretch")


# ---- tab 2: outlets compared -------------------------------------------------------------------------
# ---- tab 3: how the labels work (written before tab 2, because tab 2 can stop the script) ---------------
with tab_methods:
    methods.show(articles, entities, SENTIMENT_COLOURS)


with tab_outlets:
    analysed = shown[shown["roberta_label"] != "not analysed"]
    if analysed.empty:
        st.info("No articles match the filters.")
        st.stop()

    heading("Sentiment mix per outlet", "Share of headlines per RoBERTa label (with the current filters)")
    mix = analysed.groupby(["outlet", "roberta_label"]).size().reset_index(name="articles")
    order = ["negative", "neutral", "positive"]
    chart = alt.Chart(mix).mark_bar().encode(
        y=alt.Y("outlet:N", title=None, axis=alt.Axis(labelLimit=220), sort=list(analysed["outlet"].value_counts().index)),
        x=alt.X("articles:Q", stack="normalize", title=None, axis=alt.Axis(format="%")),
        color=alt.Color("roberta_label:N", title=None, sort=order,
                        scale=alt.Scale(domain=order, range=[SENTIMENT_COLOURS[s] for s in order]),
                        legend=alt.Legend(orient="bottom")),
        order=alt.Order("order:Q"),
        tooltip=["outlet:N", "roberta_label:N", "articles:Q"],
    ).transform_calculate(order=f"indexof({order}, datum.roberta_label)").properties(height=45 * mix["outlet"].nunique())
    st.altair_chart(chart, width="stretch")

    average = analysed["roberta_score"].mean()
    table = analysed.groupby("outlet").agg(articles=("article_id", "size"), avg_score=("roberta_score", "mean"))
    table["gap_from_all"] = table["avg_score"] - average
    table = table.sort_values("articles", ascending=False).rename_axis("Outlet")
    st.dataframe(table, column_config={
        "articles": "Articles",
        "avg_score": st.column_config.NumberColumn("Average score", format="%+.2f"),
        "gap_from_all": st.column_config.NumberColumn(f"Gap from all outlets ({average:+.2f})", format="%+.2f"),
    })
    st.caption("Careful: outlets cover different topics, so a gap here can come from the topic mix, not the tone. "
               "The fair comparison (the same story in every outlet) comes with pipeline step 4. "
               "Outlets with few articles (RSS only started in October) give unreliable numbers. "
               "RoBERTa is also more accurate on some outlets than others (on our hand labels: NYT 84%, "
               "Fox News 55%), so part of a gap can be model error.")

    heading("Most mentioned per outlet", "Share of the outlet's articles that mention the name")
    columns = {}
    for outlet, group in analysed.groupby("outlet"):
        counts = entities[entities["article_id"].isin(group["article_id"])]["entity"].value_counts().head(10)
        columns[f"{outlet} ({len(group)})"] = [f"{name} ({n / len(group):.0%})" for name, n in counts.items()]
    st.dataframe(pd.DataFrame({k: pd.Series(v) for k, v in columns.items()}), hide_index=True)
