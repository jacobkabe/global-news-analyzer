
import os
import sys
import time
from collections import Counter
from datetime import date, timedelta

import requests
from dotenv import load_dotenv

from db import save_articles

load_dotenv()

API_URL = "https://api.thenewsapi.com/v1/news/all"


SOURCES = [
    "abcnews.go.com",
    "cbsnews.com",
    "theconversation.com",
    "nbcnews.com",
    "cnbc.com",
    "washingtontimes.com",
    "time.com",
    "dw.com",
]


# Maximum requests used by this script per run.
MAX_REQUESTS = 80

# Maximum pages requested for one source.
MAX_PAGES_PER_SOURCE = 15


def fetch_thenewsapi(from_date, to_date, max_requests=MAX_REQUESTS):
    token = os.getenv("THENEWSAPI_API_KEY")

    if not token:
        raise RuntimeError("Missing THENEWSAPI_API_KEY in .env")

    articles = []
    requests_used = 0

    for source in SOURCES:
        if requests_used >= max_requests:
            break

        print(f"\n--- {source} ---")

        for page in range(1, MAX_PAGES_PER_SOURCE + 1):
            if requests_used >= max_requests:
                break

            params = {
                "api_token": token,
                "language": "en",
                "domains": source,
                "published_after": from_date,
                "published_before": to_date,
                "limit": 3,
                "page": page,
            }

            try:
                response = requests.get(
                    API_URL,
                    params=params,
                    timeout=30,
                )
            except requests.RequestException as exc:
                print(f"{source}: request failed ({type(exc).__name__})")
                break

            requests_used += 1

            if response.status_code == 429:
                print("API rate limit reached. Stopping collection.")
                print(f"Requests used in this run: {requests_used}")
                return articles

            if response.status_code != 200:
                print(f"{source}: HTTP {response.status_code}")
                break

            result = response.json()
            items = result.get("data", [])
            meta = result.get("meta", {})

            if page == 1:
                print(f"Available articles: {meta.get('found', 'unknown')}")

            for item in items:
                if not item.get("url"):
                    continue

                categories = item.get("categories") or []

                articles.append({
                    "source": item.get("source") or source,
                    "url": item["url"],
                    "published_at": item.get("published_at"),
                    "title": item.get("title"),
                    "summary": (
                        item.get("description")
                        or item.get("snippet")
                    ),
                    "text": None,
                    "section": ", ".join(categories) or None,
                })

            print(f"Page {page}: {len(items)} articles")

            if not items:
                break

            if meta.get("returned", len(items)) < 3:
                break

            if page * 3 >= meta.get("found", float("inf")):
                break

            time.sleep(1)

    print(f"\nRequests used in this run: {requests_used}/{max_requests}")
    print(f"Total downloaded: {len(articles)}")

    counts = Counter(article["source"] for article in articles)
    for source, count in counts.most_common():
        print(f"  {source}: {count}")

    return articles


if __name__ == "__main__":
    days_back = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    today = date.today()
    start = today - timedelta(days=days_back)

    articles = fetch_thenewsapi(
        start.isoformat(),
        (today + timedelta(days=1)).isoformat(),
    )

    added = save_articles(articles)

    print(
        f"\nDownloaded {len(articles)} articles, "
        f"{added} new saved to the database."
    )