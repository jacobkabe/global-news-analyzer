"""
clean.py - step 1 of the pipeline: one clean table of articles that every NLP step uses.

Reads the articles table from data/raw/news.db and writes data/raw/outputs/articles_clean.csv
(one row per article; ignored by Git because it contains article text).

    python pipeline\clean.py                       read data/raw/news.db
    python pipeline\clean.py --input news.csv      read a CSV export instead (e.g. from OneDrive)

What it does
  headline       title without HTML, invisible characters, curly quotes, the Guardian " | Author" ending
                 and series labels such as "Country diary:" or "LIVE:"   <- what the models read
  headline_raw   the title exactly as collected (for the dashboard)
  headline_sentcase  NYT-style Title Case turned into sentence case (for the NER casing experiment)
  content_type   news, opinion, briefing, column, letter, live, media or callout.
                 Only "news" rows are a fair input for sentiment; the others are kept but flagged.
  topic          the outlets' sections merged into one list (e.g. "U.S." and "US news" -> "US")
  outlet         a readable outlet name for the dashboard (e.g. "The Guardian")

Nothing is dropped except rows without a title. Duplicate URLs are already removed when collecting;
the same title twice is usually a weekly series ("Who's Winning the Race for Congress?"), so it stays.
"""
import argparse
import html
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "raw" / "news.db"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "outputs" / "articles_clean.csv"

OUTLETS = {
    "guardian": "The Guardian",
    "nyt": "The New York Times",
    "bbc": "BBC News",
    "aljazeera": "Al Jazeera",
    "npr": "NPR",
    "foxnews": "Fox News",
    # The News API (src/collect/thenewsapi.py): the source is the outlet's domain
    "abcnews.go.com": "ABC News",
    "cbsnews.com": "CBS News",
    "nbcnews.com": "NBC News",
    "cnbc.com": "CNBC",
    "time.com": "TIME",
    "dw.com": "Deutsche Welle",
    "washingtontimes.com": "The Washington Times",
    "theconversation.com": "The Conversation",
}

TOPICS = {
    "u.s.": "US", "us news": "US",
    "world": "World", "world news": "World",
    "uk news": "UK",
    "politics": "Politics",
    "business": "Business",
    "technology": "Technology",
    "science": "Science",
    "health": "Health",
    "environment": "Environment & climate", "climate": "Environment & climate",
    # The News API's categories
    "tech": "Technology", "sports": "Sport", "entertainment": "Culture",
}

# Series labels at the start of a headline -> content type. The label is removed from "headline".
PREFIXES = [
    (r"(Ukraine war|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) briefing:", "briefing"),
    (r"First Thing:", "briefing"),
    (r"(Country diary|Starwatch|Birdwatch|Weather tracker|Digested week):", "column"),
    (r"Letters?:", "letter"),
    (r"Tell us:", "callout"),
    (r"LIVE:", "live"),
    (r"(Photos|Watch):", "media"),
]
MEDIA_SUFFIX = re.compile(r"\s+[–-]\s+(in pictures|in charts|visual guide|video|podcast)\s*$", re.IGNORECASE)
LETTERS_SUFFIX = re.compile(r"\|\s*(brief )?letters\s*$", re.IGNORECASE)
UNINFORMATIVE = {"here's the latest."}   # NYT live-blog placeholder title


# ---- text helpers (clean, is_title_case, sentence_case: from Klaudi's nlp/ner/ner.py) ---------------
_INVISIBLE = re.compile("[​-‏‪-‮⁠﻿]")
_AUTHOR_SUFFIX = re.compile(r"\s+\|\s+[^|]+$")                 # Guardian: "Title | Author"
_TAGS = re.compile(r"<[^>]+>")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def clean(s):
    """Unescape HTML entities (e.g. &#039;), drop HTML tags and invisible characters, collapse whitespace."""
    if not s or pd.isna(s):
        return ""
    s = html.unescape(str(s))
    s = _TAGS.sub(" ", s)
    s = _INVISIBLE.sub("", s).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def straight_quotes(s):
    # VADER strips only straight punctuation, so "‘thrilled’" would not match its word "thrilled"
    return s.translate(_QUOTES)


def is_title_case(s):
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'’-]*", s) if len(w) >= 4]
    return len(words) >= 3 and sum(w[0].isupper() for w in words) / len(words) >= 0.8


def sentence_case(s):
    """Only touches headlines that look Title Case. Keeps ALLCAPS acronyms and the first word.
    Note: this also lowercases names ('Donald Trump' -> 'donald trump'), so it can hurt NER."""
    if not is_title_case(s):
        return s
    return " ".join(w if (i == 0 or w.isupper() or w == "I") else w.lower()
                    for i, w in enumerate(s.split(" ")))


# ---- per-article rules ------------------------------------------------------------------------------
def content_type_and_headline(title, url):
    """Return (content_type, headline). Opinion pieces are recognised by the Guardian URL."""
    kind = "news"
    if "/commentisfree/" in (url or ""):
        kind = "opinion"
    if LETTERS_SUFFIX.search(title):
        kind = "letter"

    if title.endswith("| First Thing"):   # the Guardian's US morning briefing, as a suffix
        kind = "briefing"

    headline = _AUTHOR_SUFFIX.sub("", title)
    for pattern, label in PREFIXES:
        match = re.match(pattern + r"\s*", headline, re.IGNORECASE)
        if match:
            kind = label
            headline = headline[match.end():]
            break
    if MEDIA_SUFFIX.search(headline) or "/video" in (url or ""):    # NBC, CBS: video pages
        kind = "media"
        headline = MEDIA_SUFFIX.sub("", headline)
    headline = straight_quotes(headline.strip())
    if headline.lower() in UNINFORMATIVE:
        kind = "live"
    return kind, headline


def topic(section):
    """Map an outlet's section to our topic list. The News API gives a list such as 'general, politics':
    use the first category that says something ('general' only if there is nothing else)."""
    if not section:
        return "Other"
    parts = [x.strip() for x in section.split(",") if x.strip()]
    specific = [x for x in parts if x.lower() != "general"]
    first = (specific or parts)[0]
    return TOPICS.get(first.lower(), first.capitalize())


# ---- main -------------------------------------------------------------------------------------------
def load(input_path):
    if input_path:
        return pd.read_csv(input_path, dtype=str, keep_default_na=False)
    if not DEFAULT_DB.exists():
        sys.exit(f"{DEFAULT_DB} not found. Use --input path\\to\\news.csv")
    with sqlite3.connect(DEFAULT_DB) as conn:
        return pd.read_sql("SELECT * FROM articles", conn).fillna("")


def build(df):
    rows = []
    for a in df.to_dict("records"):
        title = clean(a["title"])
        if not title:
            continue
        kind, headline = content_type_and_headline(title, a["url"])
        text = clean(a["text"])
        rows.append({
            "article_id": int(a["id"]),
            "source": a["source"],
            "outlet": OUTLETS.get(a["source"], a["source"]),
            "published_at": a["published_at"],
            "date": str(a["published_at"])[:10],
            "topic": topic(clean(a["section"])),
            "content_type": kind,
            "headline": headline,
            "headline_sentcase": sentence_case(headline),
            "is_title_case": is_title_case(headline),
            "headline_raw": title,
            "lead": straight_quotes(clean(a["summary"])),
            "n_text_chars": len(text),
            "url": a["url"],
        })
    return pd.DataFrame(rows).sort_values("article_id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make data/raw/outputs/articles_clean.csv")
    parser.add_argument("--input", help="a news CSV instead of data/raw/news.db")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    raw = load(args.input)
    out = build(raw)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")

    print(f"{len(raw)} articles read, {len(out)} written to {args.output}")
    print("\ncontent types:\n" + out["content_type"].value_counts().to_string())
    print("\ntopics:\n" + out["topic"].value_counts().to_string())
    print(f"\nheadlines changed by cleaning: {(out['headline'] != out['headline_raw']).sum()}")
    print(f"Title Case headlines: {out['is_title_case'].sum()}")
