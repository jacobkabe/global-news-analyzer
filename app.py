
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent


SOURCE_NAMES = {
    # The New York Times & Guardian
    "nyt": "The New York Times",
    "guardian": "The Guardian",

    # RSS feeds
    "bbc": "BBC News",
    "aljazeera": "Al Jazeera",
    "foxnews": "Fox News",
    "npr": "NPR",

    # The News API - final selection
    "abcnews.go.com": "ABC News",
    "cbsnews.com": "CBS News",
    "nbcnews.com": "NBC News",
    "cnbc.com": "CNBC",
    "time.com": "TIME",
    "dw.com": "Deutsche Welle",
    "washingtontimes.com": "The Washington Times",
    "theconversation.com": "The Conversation",

    # Older articles (if still present)
    "cnn.com": "CNN",
    "cbc.ca": "CBC News",
    "thenewsapi": "The News API (unclassified)",
}


st.set_page_config(
    page_title="Global News Analyzer",
    page_icon="🌍",
    layout="wide"
)

@st.cache_data
def load_data():
    path = ROOT / "data" /"news.csv"
    df = pd.read_csv(path)
    df["source_name"] = (
        df["source"]
        .map(SOURCE_NAMES)
        .fillna(df["source"])
    )

    df["published_at"] = pd.to_datetime(
        df["published_at"], errors="coerce", utc=True
    )

    # Add sentiment if a compatible result file exists
    sentiment_path = ROOT / "data" / "outputs" / "roberta_results.csv"

    if sentiment_path.exists():
        results = pd.read_csv(sentiment_path)

        if {"url", "roberta_sentiment"}.issubset(results.columns):
            sentiment = results[
                ["url", "roberta_sentiment"]
            ].drop_duplicates("url")

            df = df.merge(sentiment, on="url", how="left")

    if "roberta_sentiment" not in df.columns:
        df["roberta_sentiment"] = "Not analyzed"

    return df


st.title("🌍 Global News Analyzer")
st.caption("Explore global news, sources and sentiment")

try:
    df = load_data()
except (FileNotFoundError, KeyError) as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

# SIDEBAR
with st.sidebar:
    st.header("Filters")

    sources = sorted(df["source_name"].dropna().unique())

    selected_sources = st.multiselect(
        "News sources",
        sources,
        default=sources
    )

    keyword = st.text_input("Search headlines")

    sentiments = sorted(
        df["roberta_sentiment"].dropna().unique()
    )

    selected_sentiments = st.multiselect(
        "Sentiment",
        sentiments,
        default=sentiments
    )

    valid_dates = df["published_at"].dropna()

    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        date_range = st.date_input(
            "Publication date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        date_range = ()

# APPLY FILTERS
filtered = df[
    df["source_name"].isin(selected_sources)
    & df["roberta_sentiment"].isin(selected_sentiments)
].copy()

if keyword:
    filtered = filtered[
        filtered["title"].fillna("").str.contains(
            keyword, case=False, regex=False
        )
    ]

if len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        filtered["published_at"].dt.date.between(start, end)
    ]

filtered = filtered.sort_values(
    "published_at", ascending=False
)

# METRICS
m1, m2, m3 = st.columns(3)

m1.metric("Articles", len(filtered))
m2.metric("Sources", filtered["source"].nunique())
m3.metric(
    "Analyzed",
    filtered["roberta_sentiment"].ne("Not analyzed").sum()
)

st.divider()

# MAIN LAYOUT
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Latest articles")

    if filtered.empty:
        st.info("No articles match your filters.")

    for _, article in filtered.head(40).iterrows():
        title = str(article["title"])
        url = article.get("url")

        if pd.notna(url):
            st.markdown(f"**[{title}]({url})**")
        else:
            st.markdown(f"**{title}**")

        st.caption(
            f"{article['source_name']} · "
            f"{article['published_at'].strftime('%d %b %Y') if pd.notna(article['published_at']) else 'Unknown date'}"
        )

        st.write(
            f"Sentiment: {article['roberta_sentiment']}"
        )

        if pd.notna(article.get("summary")):
            with st.expander("Read summary"):
                st.write(article["summary"])

        st.divider()

with right:
    st.subheader("Analytics")

    st.write("Sentiment distribution")
    analyzed = filtered[
        filtered["roberta_sentiment"].ne("Not analyzed")
        & filtered["roberta_sentiment"].notna()
    ]

    if not analyzed.empty:
        counts = analyzed[
            "roberta_sentiment"
        ].value_counts()
        st.bar_chart(counts)
    else:
        st.info("Sentiment results not available yet.")

    st.write("Articles by source")

    if not filtered.empty:
        source_counts = filtered["source"].value_counts()
        st.bar_chart(source_counts)

    st.write("Articles over time")

    if not filtered.empty:
        daily = (
            filtered.dropna(subset=["published_at"])
            .set_index("published_at")
            .resample("D")
            .size()
        )
        st.line_chart(daily)
