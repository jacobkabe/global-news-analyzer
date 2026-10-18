# Global News Analyzer

University text-analytics team project (5 students, beginners with Python and Git).
Goal and design: read `docs/solution-idea.md` first.

**In short:** collect news from several outlets, group articles about the same event into
"stories", and show on a dashboard how the outlets differ (sentiment vs. the story average,
entities, headlines, coverage). Every label must be explainable ("why this label").

## Environment
- Windows + PowerShell, VS Code, Python virtual environment in `.venv` (activate with `.venv\Scripts\Activate.ps1`).
- Dependencies in `requirements.txt`; add new packages there.
- API keys live in `.env` (`GUARDIAN_API_KEY`, `NYT_API_KEY`), loaded with `python-dotenv`. Never print or commit them.

## Data collection (`src/collect/`)
- `db.py`: SQLite database at `data/raw/news.db`, table `articles`
  (`source, url UNIQUE, published_at, title, summary, text, section, collected_at`).
  Use `save_articles(list_of_dicts)`; duplicates by URL are skipped.
- `guardian.py`: Guardian Open Platform API, full text. `python src\collect\guardian.py 14` = last 14 days.
- Every new collector returns dicts with the same keys and saves via `save_articles()`.
- `rss.py`: BBC, Al Jazeera, NPR, Fox News via RSS (`feedparser`). `python src\collect\rss.py`.
  With `FULL_TEXT = False` (current setting) it saves only URL, title, summary and date, so `text` is NULL.
  Do not set `FULL_TEXT = True` before the teacher approves; `TERMS` still skips outlets marked "not allowed".
- `nyt.py`: NYT Article Search (`python src\collect\nyt.py 14` = last 14 days, saved day by day) or Archive
  (`--archive 2026 9` = a whole month). Keeps only type "News" in World, U.S., Politics, Business, Science,
  Climate, Health, Technology. Headline + abstract + lead paragraph only; waits 12 s between requests (rate limit).
- `collect.py`: runs all sources (`python src\collect\collect.py`, or only some: `... guardian nyt`).
  Guardian and NYT fetch everything since the newest article we already have (at least yesterday + today),
  so a missed day or week is filled in on the next run. Long gaps are filled in steps (NYT 14 days, Guardian 90 days per run).
  Each source has its own try/except and a log line in `data/raw/collect.log`; exit code 1 if a source failed.
  Run it twice a day with Windows Task Scheduler on one team machine (RSS has no history; missed days are lost).
- `save_articles()` stores dates as UTC (`2026-10-17T14:28:07Z`) and URLs without tracking parameters.
  `python src\collect\db.py` applies this to old rows.
- Error messages must never contain the API key: raise short messages (see `guardian.py`, `nyt.py`), not `raise_for_status()`.
- Sources: Guardian (API, full text) is the only source with full text. NYT is headline + abstract only.
- No outlet clearly allows scraping (BBC, Al Jazeera, Fox News: not allowed; NPR: unclear).
  Quotes and RSS URLs for 9 outlets: `docs/data-sources.md`. Check an outlet's terms of use before adding it.
- Be polite to websites: wait ~1 second between requests, respect robots.txt (full-text downloads only).
- Collection runs once a day (RSS has no history).

## Course rules (must follow)
- Never commit raw scraped data: `data/raw/`, `.env` and `.venv/` are in `.gitignore`.
  Only a small anonymised sample goes in `data/sample/` (no author names).
- For each NLP task compare at least 2 approaches (baseline vs. pre-trained) with metrics on a hand-labelled sample,
  and keep concrete failure examples.
- Anyone in the team may be asked to explain any part of the code at the defence.

## How to help us
- We are beginners: keep code simple and readable, explain changes in plain words.
- Small steps: one script or feature at a time, then we run it and commit.
- Ask before deleting files or changing the database schema.
