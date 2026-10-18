import os
import sys
import time
from datetime import date, timedelta

import requests
from dotenv import load_dotenv

from db import save_articles

load_dotenv()

API_URL = "https://content.guardianapis.com/search"
SECTIONS = "world|uk-news|us-news|politics|business|environment|science|technology"


def fetch_guardian(from_date, to_date):
    """Download all Guardian articles in SECTIONS between two dates (YYYY-MM-DD)."""
    articles = []
    page = 1
    while True:
        # Error messages of requests contain the full URL including our API key,
        # so we raise our own short messages instead (never print the key).
        try:
            response = requests.get(API_URL, params={
                "api-key": os.getenv("GUARDIAN_API_KEY"),
                "section": SECTIONS,
                "type": "article",
                "from-date": from_date,
                "to-date": to_date,
                "show-fields": "trailText,bodyText",
                "order-by": "oldest",
                "page-size": 50,
                "page": page,
            }, timeout=30)
        except requests.RequestException as e:
            raise RuntimeError(f"Guardian API request failed: {type(e).__name__}") from None
        if response.status_code != 200:
            raise RuntimeError(f"Guardian API returned HTTP {response.status_code}")
        data = response.json()["response"]

        for a in data["results"]:
            fields = a.get("fields", {})
            articles.append({
                "source": "guardian",
                "url": a["webUrl"],
                "published_at": a["webPublicationDate"],
                "title": a["webTitle"],
                "summary": fields.get("trailText"),
                "text": fields.get("bodyText"),
                "section": a.get("sectionName"),
            })

        print(f"page {page}/{data['pages']}: {len(data['results'])} articles")
        if page >= data["pages"]:
            break
        page += 1
        time.sleep(1)  # be polite to the API

    return articles


if __name__ == "__main__":
    # How many days back to collect: python src\collect\guardian.py 14  (default: 1)
    days_back = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    to_date = date.today()
    from_date = to_date - timedelta(days=days_back)

    articles = fetch_guardian(from_date.isoformat(), to_date.isoformat())
    added = save_articles(articles)
    print(f"Downloaded {len(articles)} articles, {added} new saved to the database.")
