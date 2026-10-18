import time
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import feedparser
import requests
import trafilatura

from db import get_connection, save_articles

FEEDS = {
    "bbc": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "npr": "https://feeds.npr.org/1004/rss.xml",
    "foxnews": "https://moxie.foxnews.com/google-publisher/world.xml",
}

# What each outlet's terms of use say about scraping (details and quotes: docs/data-sources.md)
TERMS = {
    "bbc": "not allowed",
    "aljazeera": "not allowed",
    "npr": "unclear",
    "foxnews": "not allowed",
}

# Keep False until the teacher approves downloading full article text.
# False = save only URL, title, summary and date (no article pages are downloaded).
FULL_TEXT = False


def known_urls():
    """Return the set of article URLs that are already in the database."""
    conn = get_connection()
    urls = {row[0] for row in conn.execute("SELECT url FROM articles")}
    conn.close()
    return urls


robots_cache = {}  # one parsed robots.txt per website


def allowed_by_robots(url):
    """Check the website's robots.txt: may we download this URL? (rules for all bots, "*")"""
    parts = urlparse(url)
    site = f"{parts.scheme}://{parts.netloc}"

    if site not in robots_cache:
        parser = RobotFileParser()
        try:
            response = requests.get(site + "/robots.txt", timeout=30)
            if response.status_code in (401, 403):
                parser.disallow_all = True  # site refuses us: do not download anything
            elif response.status_code >= 400:
                parser.allow_all = True  # no robots.txt: everything is allowed
            else:
                # Drop blank lines: Python's parser ends a rule group at a blank line,
                # which made it ignore the BBC rules that follow its Sitemap lines.
                lines = [line for line in response.text.splitlines() if line.strip()]
                parser.parse(lines)
        except requests.RequestException as e:
            print(f"  could not read robots.txt of {site}: {e}")
            parser.disallow_all = True  # unsure, so be careful
        robots_cache[site] = parser
        time.sleep(1)  # be polite to the website

    return robots_cache[site].can_fetch("*", url)


def fetch_feed(source, feed_url, known):
    """Read one RSS feed and save every article not yet in the database.
    The full text is downloaded only if FULL_TEXT is True, the outlet's terms do not say
    "not allowed" and robots.txt allows the URL. Otherwise text stays None."""
    articles = []
    feed = feedparser.parse(feed_url)

    for entry in feed.entries:
        url = entry.get("link")
        if not url or url in known:
            continue

        text = None
        if FULL_TEXT and TERMS[source] != "not allowed":
            if not allowed_by_robots(url):
                print(f"  skipped (robots.txt): {url}")
                continue

            # Full text from the article page (None if the download or extraction fails)
            try:
                html = trafilatura.fetch_url(url)
                text = trafilatura.extract(html) if html else None
            except Exception as e:
                print(f"  could not download {url}: {e}")
            time.sleep(1)  # be polite to the website

        if entry.get("published_parsed"):
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        else:
            published = entry.get("published")

        articles.append({
            "source": source,
            "url": url,
            "published_at": published,
            "title": entry.get("title"),
            "summary": entry.get("summary"),
            "text": text,
            "section": "world",
        })

    return articles


if __name__ == "__main__":
    known = known_urls()
    total = 0
    for source, feed_url in FEEDS.items():
        print(f"{source}: reading {feed_url}")
        articles = fetch_feed(source, feed_url, known)
        added = save_articles(articles)
        with_text = sum(1 for a in articles if a["text"])
        print(f"{source}: {added} new articles saved ({with_text} with full text)")
        total += added
    print(f"Done. {total} new articles saved to the database.")
