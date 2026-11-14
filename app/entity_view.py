"""
entity_view.py - the "People & places" view: how one person, place or organisation is covered.

For the chosen name it shows
  - how many articles mention it, and whether their tone is more negative than the average article
  - coverage over time (articles per week, coloured by RoBERTa's label)
  - each outlet: how often it mentions the name (share of its articles) and in which tone
  - names that appear together with it (click one to open its profile)
  - its most negative and most positive headlines, each with the words that drove the label
Everything follows the sidebar filters (outlets, topic, "only news", dates), but not the sentiment filter:
the tone of the coverage is what this view measures.
"""
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

from ui import SENTIMENT_COLOURS, YELLOW, esc, heading, highlight, sentiment_tag, word_chips

MIN_ARTICLES = 3        # names with fewer articles in the current filters are not offered
MIN_OUTLET_ARTICLES = 100   # the "share of the outlet" needs enough articles of that outlet
TOP_HEADLINES = 3


def open_entity(name):
    """Callback: show the profile of a name (trending list, 'together with' pills, 'back' from an article)."""
    st.session_state["view"] = st.session_state["last_view"] = "People & places"
    st.session_state["entity"] = name
    st.session_state.pop("focus", None)


def open_together():
    """Callback of the 'mentioned together with' pills: open the clicked name, then clear the pills."""
    picked = st.session_state.get("together_pick")
    st.session_state["together_pick"] = None
    if picked:
        open_entity(picked)


def open_article(article_id, name):
    """Callback: jump to the Articles view with this article pinned and selected, where its 'why' is shown."""
    st.session_state["view"] = st.session_state["last_view"] = "Articles"
    st.session_state["selected"] = st.session_state["focus"] = article_id
    st.session_state["focus_from"] = name


def show(scope, scope_all_dates, entities, entity_types, end):
    """scope: articles after the sidebar filters; scope_all_dates: the same without the date range."""
    in_scope = entities[entities["article_id"].isin(scope["article_id"])]
    counts = in_scope["entity"].value_counts()
    options = counts[counts >= MIN_ARTICLES].index.tolist()
    if not options:
        st.info("No name has enough articles with the current filters.")
        return
    if st.session_state.get("entity") not in options:
        st.session_state["entity"] = options[0]

    pick_col, info_col = st.columns([1.2, 2], vertical_alignment="bottom")
    pick_col.selectbox("Person, place or organisation", options, key="entity",
                       format_func=lambda e: f"{e} ({entity_types[e]}, {counts[e]} articles)",
                       help="Type to search. Uses the sidebar filters for outlets, topic, 'only news' and dates.")
    name = st.session_state["entity"]

    ids = in_scope.loc[in_scope["entity"] == name, "article_id"]
    mine = scope[scope["article_id"].isin(ids)]
    analysed = mine[mine["roberta_label"] != "not analysed"]
    everyone = scope[scope["roberta_label"] != "not analysed"]
    info_col.markdown(f'<p class="gna-h" style="margin:0">{esc(name)}</p><p class="muted">{entity_types[name]} · '
                      f'{len(mine)} articles in {mine["outlet"].nunique()} outlets · '
                      f'{mine["date"].min():%d %b} – {mine["date"].max():%d %b %Y}</p>', unsafe_allow_html=True)

    # ---- key numbers ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Articles", f"{len(mine):,}", help="Articles whose headline or summary mentions the name.")
    neg, neg_all = (analysed["roberta_label"] == "negative").mean(), (everyone["roberta_label"] == "negative").mean()
    k2.metric("Negative headlines", f"{neg:.0%}", f"{(neg - neg_all) * 100:+.0f} pts vs all articles",
              delta_color="inverse", help=f"All articles with the same filters: {neg_all:.0%} negative.")
    score, score_all = analysed["roberta_score"].mean(), everyone["roberta_score"].mean()
    k3.metric("Average tone", f"{score:+.2f}", f"{score - score_all:+.2f} vs all articles",
              help="RoBERTa's score (positive − negative), from −1 to +1, averaged over the headlines.")
    all_ids = entities.loc[entities["entity"] == name, "article_id"]    # also outside the date range
    dated = scope_all_dates[scope_all_dates["article_id"].isin(all_ids)]
    now = ((dated["date"] > end - timedelta(days=7)) & (dated["date"] <= end)).sum()
    before = ((dated["date"] > end - timedelta(days=14)) & (dated["date"] <= end - timedelta(days=7))).sum()
    k4.metric("Last 7 days", f"{now}", "new" if before == 0 and now else f"{now - before:+d} vs the 7 days before",
              help=f"Articles in the 7 days to {end:%d %b}.")

    # ---- coverage over time | outlets ----
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        heading("Coverage over time", "Articles per week, coloured by RoBERTa's label")
        weekly = analysed.assign(week=pd.to_datetime(analysed["date"]).dt.to_period("W").dt.start_time) \
            .groupby(["week", "roberta_label"]).size().reset_index(name="articles")
        order = ["positive", "neutral", "negative"]
        st.altair_chart(alt.Chart(weekly).mark_bar().encode(
            x=alt.X("week:T", title=None),
            y=alt.Y("articles:Q", title=None),
            color=alt.Color("roberta_label:N", title=None, sort=order,
                            scale=alt.Scale(domain=order, range=[SENTIMENT_COLOURS[s] for s in order]),
                            legend=alt.Legend(orient="bottom")),
            order=alt.Order("order:Q"),
            tooltip=[alt.Tooltip("week:T", title="week of"), "roberta_label:N", "articles:Q"],
        ).transform_calculate(order=f"indexof({order}, datum.roberta_label)").properties(height=230), width="stretch")

    with right:
        heading("Who covers it, and how", "Share = how many of the outlet's articles mention the name")
        per_outlet = scope.groupby("outlet").size().rename("all")
        table = analysed.groupby("outlet").agg(articles=("article_id", "size"),
                                               negative=("roberta_label", lambda s: (s == "negative").mean()),
                                               tone=("roberta_score", "mean"))
        table["share"] = table["articles"] / per_outlet.reindex(table.index)
        # with few articles one mention looks like a big share (1 of 31 = 3%): leave it empty
        table.loc[per_outlet.reindex(table.index) < MIN_OUTLET_ARTICLES, "share"] = None
        table["negative"] = table["negative"] * 100
        table = table.sort_values("articles", ascending=False)[["articles", "share", "negative", "tone"]]
        st.dataframe(table, column_config={
            "articles": "Articles",
            "share": st.column_config.ProgressColumn("Share of outlet", format="percent", min_value=0,
                                                     max_value=float(max(table["share"].max(skipna=True) or 0, 0.01))),
            "negative": st.column_config.NumberColumn("Negative", format="%.0f%%"),
            "tone": st.column_config.NumberColumn("Tone", format="%+.2f"),
        }, width="stretch")
        st.caption(f"Share is left empty for outlets with fewer than {MIN_OUTLET_ARTICLES} articles in the filters "
                   "(one mention would look big). Outlets also cover different topics.")

    # ---- together with | headlines ----
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        heading("Mentioned together with", "In the same articles · click a name to open its profile")
        together = in_scope[in_scope["article_id"].isin(ids) & (in_scope["entity"] != name)]["entity"].value_counts()
        together = together[together.index.isin(options)].head(12)
        if together.empty:
            st.caption("No other name appears often enough.")
        else:
            st.pills("together", together.index.tolist(), key="together_pick", label_visibility="collapsed",
                     format_func=lambda e: f"{e} · {together[e]}",
                     on_change=open_together)

    with right:
        heading("Most negative and most positive headlines", "Click “Why?” to see the explanation in Articles")
        for title, part in [("Most negative", analysed.nsmallest(TOP_HEADLINES, "roberta_score")),
                            ("Most positive", analysed.nlargest(TOP_HEADLINES, "roberta_score"))]:
            st.markdown(f'<p class="muted" style="margin-top:.4rem">{title.upper()}</p>', unsafe_allow_html=True)
            for a in part.itertuples():
                text_col, button_col = st.columns([6, 1], vertical_alignment="center")
                text_col.markdown(f'{sentiment_tag(a.roberta_label, a.roberta_score)}'
                                  f'<span class="muted">{esc(a.outlet)} · {a.date:%d %b}</span><br>'
                                  f'{highlight(a.headline, a.roberta_words)}', unsafe_allow_html=True)
                button_col.button("Why?", key=f"why_{title}_{a.article_id}", type="tertiary",
                                  on_click=open_article, args=(a.article_id, name))
        st.caption("⚠ Positive labels are often wrong: about 4 in 10 are really good news.")
