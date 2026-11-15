# Data sources: what the outlets allow

Checked on 2026-10-17 by reading each outlet's `robots.txt` and terms of use.
We are not lawyers: "unclear" means the teacher should decide.

**Result: no outlet clearly allows scraping.** `robots.txt` alone is not enough: it allowed
most of these sites, but their terms of use forbid scraping or archiving.

Current setting in `src/collect/rss.py`: `FULL_TEXT = False`. For every RSS outlet we save
only URL, title, summary and date. Full text is downloaded only if the teacher approves
(then set `FULL_TEXT = True`; outlets marked "not allowed" in `TERMS` are still skipped).

## RSS outlets

| Outlet | RSS feed | Status | Quote and where it is |
|---|---|---|---|
| BBC (ruled out) | `https://feeds.bbci.co.uk/news/world/rss.xml` | **Not allowed** | `robots.txt` header: "No scraping, crawling, or systematic extraction of content" and "No creating datasets from BBC content". |
| Al Jazeera (current) | `https://www.aljazeera.com/xml/rss/all.xml` | **Not allowed** | `robots.txt`: "Any other uses are not permitted, including but not limited to: (1) the development of any software, machine learning, artificial intelligence (AI), and/or large language models (LLMs); (2) text and data mining activities; (3) creating or providing archived or cached data sets containing our content to others". Terms: "You shall not authorise, encourage, permit or assist any unauthorised copying, text or data mining, or web scraping". |
| Fox News (current) | `https://moxie.foxnews.com/google-publisher/world.xml` | **Not allowed** | Terms of Use: "you may not copy, download, stream, scrape, capture, reproduce, duplicate, archive ... any Content". Also: "you may not and shall not copy, data mine, scrape or in any way extract or use the Content ... for the purpose of training any artificial intelligence algorithm ... or any large language or machine learning model". |
| Deutsche Welle (DW) | `https://rss.dw.com/xml/rss-en-world` | **Not allowed** | `robots.txt`: "Use of any device, tool, or process designed to data mine or scrape the content using automated means is prohibited without prior written permission from Deutsche Welle." No separate terms page found, so this is its only statement. |
| France 24 | `https://www.france24.com/en/rss` | **Not allowed** | Legal notice of its parent France Medias Monde (`francemm.com/legal-notice`): "It is also strictly forbidden to collect, store, use, extract, reproduce ... all or part of the content without authorization, by any means and for any purpose, automatically or manually". It also opts out of text and data mining. |
| CBS News | `https://www.cbsnews.com/latest/rss/world` | **Not allowed** (needs permission) | Terms of Use (linked from the cbsnews.com footer): "Engage in unauthorized spidering, "scraping," data mining or harvesting of Content, or use any other unauthorized automated means to gather data from or about the Services." |
| NPR (current) | `https://feeds.npr.org/1004/rss.xml` | **Unclear** | Terms: "You may not reproduce, distribute, republish ... scrape for the purposes of replicating or republishing ... without the prior written consent of NPR". Also: "You may not use ... any content ... to build or train large language models, machine learning tools, or artificial intelligence (AI) systems without the prior written consent of NPR." The scraping ban covers republishing, not analysis. Content is for personal, non-commercial use only. |
| The Independent | `https://www.independent.co.uk/news/world/rss` | **Unclear** | Terms of Use: "The Website and the Content may only be used for your personal, non-commercial use." Also: "you agree not to download, copy, reproduce, modify, store, archive ... any part of the Content without the prior written permission". It never mentions scraping or data mining, but storing articles in a database arguably counts as archiving. |
| Sky News | `https://feeds.skynews.com/feeds/rss/world.xml` | **Unclear** | `robots.txt` allows normal bots. It blocks AI-training crawlers and the scrapers `Scrapy` and `news-please`, and says "Disallow news aggregators except on RSS." We could not read Sky's website terms (`news.sky.com` returns 403 to scripts). |

## API sources

The Guardian (Open Platform) and the NYT (Article Search API) are accessed through their official APIs
with our own API keys, so they do not need scraping. Check their API terms before publishing anything.

### The News API (`thenewsapi.com`)

Checked on 2026-11-07, updated 2026-11-12. Collector: `src/collect/thenewsapi.py` (key `THENEWSAPI_API_KEY` in `.env`).

- **What we get:** title, description, a ~60-character snippet, URL, date, categories, and the outlet as a
  domain (`source`, e.g. `cbsnews.com`). No full text. We save the domain in our `source` column.
- **Which outlets:** the collector asks for 8 outlets one by one with the `domains=` filter:
  `abcnews.go.com`, `cbsnews.com`, `nbcnews.com`, `cnbc.com`, `time.com`, `dw.com`, `washingtontimes.com`,
  `theconversation.com`. Older rows may still have other domains (`cnn.com`, `cbc.ca`) or `thenewsapi`.
- **Free plan:** 100 requests a day, 3 articles per request. The collector uses at most 80 requests per run
  (15 pages per outlet, so up to 45 articles per outlet) and stops when the API answers "limit reached" (HTTP 429).
  Outlets are asked in a fixed order, so on a day with a second run the last outlets in the list may get nothing.
- **Terms:** the Terms of Service (`thenewsapi.com/tos`, last updated 2021-07-08) are written for the website,
  not the API: "granted a limited license to access and use the Site ... solely for your personal,
  non-commercial use". They say nothing about storing API results, research use or attribution.
  **Unclear**, so ask the teacher. The articles still belong to the original outlets, so their terms
  in the table above apply too (CBS News and Deutsche Welle forbid scraping their websites; we only get
  title and description through the API, not their pages).
