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
- Planned sources: Guardian (API), NYT (API, headline + abstract only), BBC, Al Jazeera, NPR, Fox News (RSS + `trafilatura` for full text).
- Be polite to websites: wait ~1 second between requests, respect robots.txt.
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
