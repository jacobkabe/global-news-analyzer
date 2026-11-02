
import os
import sys
import time
from datetime import date, timedelta

import requests
from dotenv import load_dotenv

from db import save_articles

load_dotenv()

API_URL = "https://api.thenewsapi.com/v1/news/all"


def fetch_thenewsapi(from_date, to_date, max_pages=30):
    """Fetch English news and return articles in our database format."""
    token = os.getenv("THENEWSAPI_API_KEY")
    if not token:
        raise RuntimeError("Missing THENEWSAPI_API_KEY in .env")

    articles = []

    for page in range(1, max_pages + 1):
        params = {
            "api_token": token,
            "language": "en",
            "published_after": from_date,
            "published_before": to_date,
            "limit": 3,  # Free plan limit
            "page": page,
        }

        try:
            response = requests.get(
                API_URL, params=params, timeout=30
            )
        except requests.RequestException as e:
            raise RuntimeError(
                f"The News API request failed: {type(e).__name__}"
            ) from None

        if response.status_code != 200:
            raise RuntimeError(
                f"The News API returned HTTP {response.status_code}"
            )

        result = response.json()
        items = result.get("data", [])
        meta = result.get("meta", {})

        for item in items:
            if not item.get("url"):
                continue

            categories = item.get("categories") or []

            articles.append({
                "source": "thenewsapi",
                "url": item["url"],
                "published_at": item.get("published_at"),
                "title": item.get("title"),
                "summary": item.get("description")
                           or item.get("snippet"),
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

    return articles


if __name__ == "__main__":
    days_back = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    today = date.today()
    start = today - timedelta(days=days_back)

    articles = fetch_thenewsapi(
        start.isoformat(),
        (today + timedelta(days=1)).isoformat(),
    )

    added = save_articles(articles)

    print(
        f"Downloaded {len(articles)} articles, "
        f"{added} new saved to the database."
    )
