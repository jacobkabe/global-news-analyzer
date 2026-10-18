import argparse
import os
import time
from datetime import date, timedelta

import requests
from dotenv import load_dotenv

from db import save_articles

load_dotenv()

ARCHIVE_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
SEARCH_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

# Keep only real news: type_of_material "News" in these sections (no Opinion, Arts, Style, ...)
SECTIONS = ["World", "U.S.", "Politics", "Business", "Science", "Climate", "Health", "Technology"]


def get_json(url, params):
    """Call the NYT API and return the JSON answer. Waits and retries if we hit the rate limit."""
    params = dict(params, **{"api-key": os.getenv("NYT_API_KEY")})
    for attempt in range(3):
        # Error messages of requests contain the full URL including our API key
        try:
            response = requests.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            raise RuntimeError(f"NYT API request failed: {type(e).__name__}") from None
        if response.status_code == 429:
            print("  rate limit reached, waiting 60 seconds")
            time.sleep(60)
            continue
        # Do not use raise_for_status(): its error message would print the URL with our API key
        if response.status_code != 200:
            raise RuntimeError(f"NYT API returned HTTP {response.status_code}")
        return response.json()
    raise RuntimeError("NYT API: still rate limited (HTTP 429) after 3 tries")


def is_news(doc):
    """True for real news articles in SECTIONS."""
    return doc.get("type_of_material") == "News" and doc.get("section_name") in SECTIONS


def to_article(doc):
    """Turn one NYT document into a dict with the same keys as the other collectors."""
    # summary = abstract + lead paragraph (skip empty parts and exact repeats)
    # Article Search has no lead_paragraph field; its "snippet" is used instead.
    parts = []
    for part in (doc.get("abstract"), doc.get("lead_paragraph") or doc.get("snippet")):
        if part and part not in parts:
            parts.append(part)
    return {
        "source": "nyt",
        "url": doc["web_url"],
        "published_at": doc.get("pub_date"),
        "title": (doc.get("headline") or {}).get("main"),
        "summary": "\n\n".join(parts) or None,
        "text": None,  # the NYT APIs have no article text
        "section": doc.get("section_name"),
    }


def fetch_archive(year, month):
    """Download the NYT Archive for one month (one request) and keep only the news."""
    data = get_json(ARCHIVE_URL.format(year=year, month=month), {})
    docs = data["response"]["docs"] or []
    articles = [to_article(d) for d in docs if d.get("web_url") and is_news(d)]
    print(f"archive {year}-{month:02d}: {len(docs)} documents, {len(articles)} are news")
    return articles


def fetch_search(days_back=1, last_day=None):
    """Download the news of the days_back days before last_day, and of last_day itself
    (last_day is today if not given), with the Article Search API.
    One search per day, 10 results per page, 12 seconds between requests (limit: ~5 per minute).
    Each day is saved to the database as soon as it is downloaded, so if the run is stopped
    only the day in progress is lost. Returns (articles downloaded, new articles saved)."""
    downloaded = 0
    added = 0
    last_day = last_day or date.today()
    for i in range(days_back, -1, -1):
        day = last_day - timedelta(days=i)
        day_articles = []
        page = 0
        while True:
            # The API's own filter (fq) returns nothing for us, so we filter with is_news() below
            data = get_json(SEARCH_URL, {
                "begin_date": day.strftime("%Y%m%d"),
                "end_date": day.strftime("%Y%m%d"),
                "sort": "oldest",
                "page": page,
            })
            docs = data["response"]["docs"] or []
            hits = data["response"]["metadata"]["hits"]
            news = [to_article(d) for d in docs if d.get("web_url") and is_news(d)]
            day_articles += news
            print(f"{day} page {page + 1}: {len(docs)} articles, {len(news)} are news ({hits} in total that day)")
            page += 1
            time.sleep(12)  # be polite to the API
            if not docs or page * 10 >= hits or page > 100:  # the API allows pages 0-100
                break

        day_added = save_articles(day_articles)
        print(f"{day} saved: {len(day_articles)} news articles, {day_added} new")
        downloaded += len(day_articles)
        added += day_added
    return downloaded, added


if __name__ == "__main__":
    # python src\collect\nyt.py            last day via Article Search
    # python src\collect\nyt.py 14         last 14 days via Article Search
    # python src\collect\nyt.py --archive 2026 9   a whole month via Archive
    parser = argparse.ArgumentParser()
    parser.add_argument("days", nargs="?", type=int, default=1, help="days back (default 1)")
    parser.add_argument("--archive", nargs=2, type=int, metavar=("YEAR", "MONTH"))
    args = parser.parse_args()

    if args.archive:
        articles = fetch_archive(*args.archive)
        downloaded, added = len(articles), save_articles(articles)
    else:
        downloaded, added = fetch_search(args.days)  # saves day by day
    print(f"Downloaded {downloaded} news articles, {added} new saved to the database.")
