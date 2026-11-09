# Global News Analyzer
Collects news articles from several sources, analyses them with NLP
(e.g. entities, sentiment, topics) and shows the insights in a dashboard.
Team: Adriana Talavera, Jakub Kabele, Klaudie Polednová, Matylda Smolová, Sára Košelíková
Course: Text Analytics 2, VŠE Prague
## Where to find what
| Folder | Contents |
|---|---|
| `src/` | Python code: collection script, NLP pipeline, dashboard |
| `data/sample/` | Small anonymised data sample, schema and example outputs |
| `results/` | Metrics table comparing approaches, plots |
| `failures/` | Analysed failure cases |
| `screenshots/` | Dashboard screenshots or demo link |
## Setup
Windows PowerShell, from the project folder:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
Copy-Item .env.example .env
```
Then open `.env` and fill in your API keys (`GUARDIAN_API_KEY`, `NYT_API_KEY`, `THENEWSAPI_API_KEY`).
Never commit `.env`.
## Data
Raw scraped articles are not published (personal content, terms of service).
We provide the collection script, the schema and a small anonymised sample instead.
## Status
Work in progress. The `prefinal` tag marks the version handed in for peer review.
