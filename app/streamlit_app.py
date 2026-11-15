"""
streamlit_app.py - the Global News Analyzer dashboard (version 2, reads the pipeline outputs).

Run from the project folder:
    streamlit run app\streamlit_app.py

Layout follows the mockup in our project proposal (slide 4):
  left    filters (topic, sentiment, outlets, people/places, dates)
  middle  article cards (story cards were planned; story clustering is future work)
  right   "Why these labels?" for the selected article, trending entities, sentiment over time
Views (buttons at the top): Articles · People & places (one name's coverage, entity_view.py) ·
Outlets compared · How the labels work (methods.py). Shared colours and HTML helpers: ui.py.

The first version (app.py in the project folder, by Tala) stays as it is; ideas taken from it:
the outlet filter, the keyword search and the charts of sentiment and articles over time.
"""
from datetime import timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import data
import entity_view
import methods
from ui import (SENTIMENT_COLOURS, YELLOW, entity_chips, esc, heading, highlight, probability_bar,
                sentiment_tag, word_chips)

st.set_page_config(page_title="Global News Analyzer", page_icon="🌍", layout="wide")

CARDS_PER_STEP = 20
REFERENCE_OUTLET = "The Guardian"   # Outlets compared: the only outlet collected on every day
MIN_OUTLET_ARTICLES = 30            # Outlets compared: fewer articles give too wide intervals
LIST_HEIGHT = 1310   # px: the article list ends about where the right column ("Sentiment over time") ends
VIEWS = ["Articles", "People & places", "Outlets compared", "How the labels work"]

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
/* the same for the rows of the trending list: a click opens the name's profile */
[class*="st-key-trend_"] {{ position:relative; cursor:pointer; border-radius:6px; }}
[class*="st-key-trend_"]:hover {{ background:#2A2323; }}
[class*="st-key-open_"] {{ position:absolute !important; inset:0; width:auto !important; z-index:1; margin:0 !important; }}
[class*="st-key-open_"] div, [class*="st-key-open_"] button {{ width:100% !important; height:100% !important; }}
[class*="st-key-open_"] button {{ opacity:0; cursor:pointer; }}
/* the "mentioned together with" names wrap onto more lines instead of being cut off */
.st-key-together_pick div {{ flex-wrap:wrap !important; height:auto !important; overflow:visible !important; }}
.lead {{ color:#BDB3AF; font-size:.9rem; margin:.25rem 0 .4rem 0; }}
.chip {{ background:#2A2323; border-radius:999px; padding:2px 10px; font-size:.8rem;
         margin:0 6px 6px 0; display:inline-block; }}
.chip b {{ font-size:.66rem; color:#9AD3C8; margin-right:4px; }}
.word {{ border-radius:4px; padding:0 3px; }}
.sec {{ color:#9A8F8B; font-size:.75rem !important; letter-spacing:.4px; border-top:1px solid #2E2727;
        padding-top:.55rem; margin:.8rem 0 .35rem 0 !important; }}
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
# ---- filters (left sidebar) --------------------------------------------------------------------------
with st.sidebar:
    heading("Filters")

    topics = st.pills("Topic", sorted(articles["topic"].dropna().unique()), selection_mode="multi",
                      help="Nothing selected = all topics. The topic comes from the outlet's section.")
    sentiments = st.pills("Sentiment (RoBERTa)", ["positive", "neutral", "negative"], selection_mode="multi",
                          format_func=str.capitalize,
                          help="Nothing selected = all. Used in Articles (Outlets compared measures the tone itself). Careful with "
                               "Positive: on our hand labels only about 4 in 10 are really good news.")

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
        placeholder="e.g. Donald Trump",
        help="Articles that mention all chosen names (Articles and Outlets compared). "
             "For the whole coverage of one name, open People & places.")

    first_day, last_day = articles["date"].min(), articles["date"].max()
    dates = st.date_input("Date range", value=(first_day, last_day), min_value=first_day, max_value=last_day)


# ---- header ------------------------------------------------------------------------------------------
title_col, search_col, info_col = st.columns([1.5, 1.6, 1.4], vertical_alignment="center")
title_col.markdown('<span class="gna-title">GLOBAL NEWS ANALYZER</span>', unsafe_allow_html=True)
in_articles = st.session_state.get("view", "Articles") == "Articles"
query = search_col.text_input(
    "Search", label_visibility="collapsed", key="query", disabled=not in_articles,
    placeholder="🔍  Search headlines, people, places..." if in_articles else "Search works in the Articles view")
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

# the People & places view measures the tone itself, so it ignores the sentiment, name and search filters
scope_all_dates = articles[articles["outlet"].isin(outlets)]
if only_news:
    scope_all_dates = scope_all_dates[scope_all_dates["content_type"] == "news"]
if topics:
    scope_all_dates = scope_all_dates[scope_all_dates["topic"].isin(topics)]
scope = scope_all_dates[(scope_all_dates["date"] >= start) & (scope_all_dates["date"] <= end)]

newest = " · ".join(f"{o}: {d:%d %b}" for o, d in articles.groupby("outlet")["date"].max().items())
info.markdown(f'<span class="muted">{len(shown):,} articles · {shown["outlet"].nunique()} sources · '
              f'{start:%d %b} – {end:%d %b %Y}</span> &nbsp; '
              f'<span class="badge" title="Newest article per outlet: {newest}">DATA TO {last_day:%d %b}</span>',
              unsafe_allow_html=True)


# ---- navigation: buttons instead of tabs, so a click elsewhere (a trending name) can switch the view ----
def keep_view():
    """A click on the active button would unselect it: keep the last view. Switching views unpins an article."""
    if st.session_state.get("view") is None:
        st.session_state["view"] = st.session_state.get("last_view", "Articles")
    st.session_state["last_view"] = st.session_state["view"]
    st.session_state.pop("focus", None)


st.session_state.setdefault("view", "Articles")
view = st.segmented_control("View", VIEWS, key="view", label_visibility="collapsed", on_change=keep_view) or "Articles"


# ---- view 1: articles --------------------------------------------------------------------------------
if view == "Articles":
    middle, right = st.columns([2, 1.15], gap="large")

    with middle:
        # the sort order is a small drop-down on the right: an option of the list, not another page switch
        head_col, sort_col = st.columns([2.2, 1], vertical_alignment="center")
        with head_col:
            heading("Latest articles")
        sort = sort_col.selectbox("Sort", ["Newest", "Most negative", "Most positive"], key="sort",
                                  format_func=lambda o: f"Sort: {o}", label_visibility="collapsed")
        if sort == "Most negative":
            ordered = shown.sort_values("roberta_score")
        elif sort == "Most positive":
            ordered = shown.sort_values("roberta_score", ascending=False)
        else:
            ordered = shown.sort_values("published_at", ascending=False)

        # an article opened with "Why?" in a profile: show only it, with a way back (the side panels stay as they are)
        focus = st.session_state.get("focus")
        if focus is not None:
            ordered = articles[articles["article_id"] == focus]
            from_name = st.session_state.get("focus_from", "")
            back_col, all_col = st.columns([1, 1])
            back_col.button(f"← Back to {from_name}", on_click=entity_view.open_entity, args=(from_name,),
                            width="stretch")
            all_col.button("Show all articles", on_click=lambda: st.session_state.pop("focus", None), width="stretch")

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
        with st.container(height=LIST_HEIGHT, border=False):
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
                    st.markdown('<p class="sec">ROBERTA · HOW SURE IS IT?</p>' + probability_bar(probs, a.roberta_label),
                                unsafe_allow_html=True)

                st.markdown('<p class="sec">3 MODELS COMPARED</p>', unsafe_allow_html=True)
                rows = ""
                for model, (name, kind, _) in data.MODELS.items():
                    label, score = a[f"{model}_label"], a[f"{model}_score"]
                    tag = sentiment_tag(label, score) if pd.notna(score) else '<span class="muted">–</span>'
                    rows += f'<div class="row"><span>{name} <span class="muted">· {kind}</span></span>{tag}</div>'
                st.markdown(rows, unsafe_allow_html=True)

                st.markdown('<p class="sec">DRIVING WORDS · RoBERTa</p>'
                            + word_chips(a.roberta_words), unsafe_allow_html=True)
                st.markdown('<p class="muted" style="margin-top:.5rem !important">VADER WORD LIST</p>' + word_chips(a.vader_words), unsafe_allow_html=True)

                found = entities[entities["article_id"] == selected]
                if not found.empty:
                    lines = []
                    for e in found.itertuples():
                        renamed = f"“{esc(e.text)}” → " if e.text != e.entity else ""   # e.g. "US" → United States
                        lines.append(f'<span class="chip"><b>{e.tag}</b>{renamed}{esc(e.entity)}</span>')
                    st.markdown('<p class="sec">ENTITIES · spaCy transformer</p>'
                                + "".join(lines), unsafe_allow_html=True)

                limits = ["The models read only the headline, not the article.",
                          "Positive labels are often wrong: about 4 in 10 are really good news "
                          "(positive words ≠ good news).",
                          "Driving words show what moved the score; two words that say the same thing both look weak."]
                if a.roberta_label == "positive":
                    limits[1] = "⚠ " + limits[1]
                st.markdown('<p class="sec">LIMITS</p><ul class="muted" style="margin:0;'
                            'padding-left:1.1rem">' + "".join(f"<li>{x}</li>" for x in limits) + "</ul>"
                            '<p class="muted">More: tab “How the labels work”.</p>', unsafe_allow_html=True)

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
            for i, (name, row) in enumerate(trend.iterrows()):
                change = "new" if row.before == 0 else f"{(row.now / row.before - 1) * 100:+.0f}%"
                width = int(100 * row.now / trend["now"].max())
                with st.container(key=f"trend_{i}"):     # a click anywhere on the row opens the profile
                    st.markdown(f'<div class="row"><span><span class="chip"><b>{entity_types[name]}</b>{esc(name)}</span></span>'
                                f'<span><span class="muted">{int(row.now)} art.</span><span class="bar"><span style="width:{width}%"></span></span>'
                                f'<b style="color:{YELLOW}">{change}</b></span></div>', unsafe_allow_html=True)
                    st.button("Open profile", key=f"open_{i}", on_click=entity_view.open_entity, args=(name,))
            if not trend.empty:
                st.caption("Click a name to see its whole coverage (People & places).")

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


# ---- view 2: people & places --------------------------------------------------------------------------
elif view == "People & places":
    entity_view.show(scope, scope_all_dates, entities, entity_types, end)


# ---- view 4: how the labels work ----------------------------------------------------------------------
elif view == "How the labels work":
    methods.show(articles, entities, SENTIMENT_COLOURS)


# ---- view 3: outlets compared -------------------------------------------------------------------------
elif view == "Outlets compared":
    # Outlets were collected over different periods (RSS only in October, The News API only in November), so an
    # outlet is compared with the Guardian on the SAME days: the Guardian is the only outlet we have for every day.
    # The sentiment filter is not used here (it would make the shares meaningless); topic, names and dates are.
    pool = articles[articles["roberta_label"] != "not analysed"]
    if only_news:
        pool = pool[pool["content_type"] == "news"]
    if topics:
        pool = pool[pool["topic"].isin(topics)]
    for name in chosen_entities:
        pool = pool[pool["article_id"].isin(entities.loc[entities["entity"] == name, "article_id"])]
    pool = pool[(pool["date"] >= start) & (pool["date"] <= end)]
    reference = pool[pool["outlet"] == REFERENCE_OUTLET]
    analysed = pool[pool["outlet"].isin(outlets)]
    if analysed.empty:
        st.info("No articles match the filters.")
        st.stop()

    heading(f"Compared with {REFERENCE_OUTLET} on the same days",
            "Share of negative headlines (RoBERTa): each outlet against the Guardian's headlines from the days that "
            "outlet was collected. Line: 95% interval. If it crosses 0, the difference may be chance.")
    rows, too_few = [], []
    for outlet, group in analysed.groupby("outlet"):
        if outlet == REFERENCE_OUTLET:
            continue
        if len(group) < MIN_OUTLET_ARTICLES:
            too_few.append(f"{outlet} ({len(group)})")
            continue
        same_days = reference[reference["date"].isin(set(group["date"]))]
        if len(same_days) < MIN_OUTLET_ARTICLES:
            too_few.append(f"{outlet} (only {len(same_days)} Guardian articles on its days)")
            continue
        neg, neg_ref = (group["roberta_label"] == "negative").mean(), (same_days["roberta_label"] == "negative").mean()
        diff = neg - neg_ref
        # 95% interval of a difference of two shares (normal approximation)
        margin = 1.96 * np.sqrt(neg * (1 - neg) / len(group) + neg_ref * (1 - neg_ref) / len(same_days))
        verdict = ("more often negative" if diff - margin > 0 else
                   "less often negative" if diff + margin < 0 else "no clear difference")
        rows.append({"Outlet": outlet, "Period": f"{group['date'].min():%d %b} – {group['date'].max():%d %b}",
                     "Days": group["date"].nunique(), "Articles": len(group), "Negative": neg,
                     "Guardian, same days": neg_ref, "Difference": diff * 100,
                     "low": (diff - margin) * 100, "high": (diff + margin) * 100, "Verdict": verdict})
    if rows:
        table = pd.DataFrame(rows).sort_values("Articles", ascending=False)
        points = alt.Chart(table).encode(
            y=alt.Y("Outlet:N", title=None, sort=list(table["Outlet"]), axis=alt.Axis(labelLimit=220)),
            tooltip=["Outlet", "Period", "Articles", alt.Tooltip("Negative:Q", format=".0%"),
                     alt.Tooltip("Guardian, same days:Q", format=".0%"),
                     alt.Tooltip("Difference:Q", format="+.0f", title="Difference (points)"), "Verdict"])
        chart = alt.layer(
            alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8A8080", strokeDash=[4, 4]).encode(x="x:Q"),
            points.mark_rule(color="#EDE6E3").encode(x=alt.X("low:Q", title="difference in negative share "
                                                              "(percentage points)"), x2="high:Q"),
            points.mark_point(filled=True, size=90, color=YELLOW).encode(x="Difference:Q"),
        ).properties(height=42 * len(table) + 30)
        st.altair_chart(chart, width="stretch")
        st.dataframe(table.drop(columns=["low", "high"]), hide_index=True, column_config={
            "Negative": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Guardian, same days": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Difference": st.column_config.NumberColumn("Difference (points)", format="%+.0f")})
    if too_few:
        st.caption(f"Not compared (fewer than {MIN_OUTLET_ARTICLES} articles with these filters): "
                   + ", ".join(too_few) + ".")
    st.caption("Still careful: outlets cover different topics (a gap can come from the topic mix, not the tone), "
               "the same day is not the same story (matching stories is future work), and RoBERTa is more accurate "
               "on some outlets than others (on our hand labels: NYT 84%, Fox News 55%).")

    heading("Sentiment mix per outlet", "Share of headlines per RoBERTa label, each outlet over its own period "
            "(see the table above)")
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

    heading("Most mentioned per outlet", "Share of the outlet's articles that mention the name "
            f"(outlets with at least {MIN_OUTLET_ARTICLES} articles)")
    columns = {}
    big = analysed.groupby("outlet").filter(lambda g: len(g) >= MIN_OUTLET_ARTICLES)
    for outlet, group in big.groupby("outlet"):
        counts = entities[entities["article_id"].isin(group["article_id"])]["entity"].value_counts().head(10)
        columns[f"{outlet} ({len(group)})"] = [f"{name} ({n / len(group):.0%})" for name, n in counts.items()]
    st.dataframe(pd.DataFrame({k: pd.Series(v) for k, v in columns.items()}), hide_index=True)
