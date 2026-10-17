# Global News Analyzer – Solution Idea

_Draft, 17 Oct 2026. Captures the agreed direction; details will change as we build._

## In one sentence

A dashboard that takes one news story, collects how several outlets report it, gives a short neutral summary, and shows **how the outlets differ**: sentiment, framing, who and what they mention, and what they leave out.

## Why it is useful (user benefits)

- **Save time:** one short summary per story instead of reading five articles.
- **See different angles:** the same event as told by different outlets, side by side.
- **Spot framing and bias:** which outlet is more negative or positive than the others, which people and countries it focuses on, how the headlines differ.
- **See coverage gaps:** which outlets covered a story at all, who was first, and what an outlet ignores.
- **Transparency (our "no black box" rule):** every label shows why it was given (driving words, model scores).

Comparable product to cite: Ground News (cross-outlet comparison with bias ratings).

## Main screens

### 1. Story page (main screen)
- Story title and a short summary (from a full-text article in the cluster)
- List of outlets that covered it, with their headlines side by side
- **Sentiment per outlet, shown as the gap from the story average.** A flood story is negative everywhere, so the raw score says little; "BBC is more negative than the others" is the interesting part.
- Entities per outlet: people, organisations, countries mentioned
- Coverage: who published first, article length, which outlets are missing
- "Why these labels?" panel

### 2. Outlet profile
The same stats summed across all stories: topic mix, share of negative coverage, most-mentioned countries and people, its political leaning rating.

### 3. Overview / trends
Top stories, trending entities, sentiment over time, filters (topic, sentiment, outlet, country, person, date).

## Sources

| Outlet | How we get it | Full text? | Leaning (external rating) |
|---|---|---|---|
| The Guardian | Open Platform API (free developer key) | Yes, from API (`show-fields=body`) | to look up |
| BBC News | RSS + text extraction (`trafilatura`) | Yes | to look up |
| The New York Times | Article Search API | No: headline, abstract, lead paragraph (paywall) | to look up |
| Al Jazeera | RSS + text extraction | Yes | to look up |
| NPR | RSS + text extraction | Yes | to look up |
| Fox News (proposed) | RSS + text extraction | Yes | to look up |

- The current five outlets probably all lean centre-left (to be confirmed with the ratings), so **at least one right-leaning outlet (e.g. Fox News)** is needed for a left/right comparison to mean anything.
- **Political leaning is taken from published outlet ratings (AllSides, Ad Fontes Media), not predicted by our model.** We have no labelled data for leaning and it would be hard to defend. We cite the rating source.
- Not used: NewsAPI.org (free plan is dev-only, 24h delay, text cut to ~200 characters), GDELT (gives only links from thousands of sites; does not match our fixed outlet list).

## Pipeline

1. **Collect:** Guardian API + NYT API + RSS feeds, then full text with `trafilatura`. Run **once a day from now on**, because RSS feeds only show the latest items (no history).
2. **Store:** SQLite, one common schema for all sources: `id, source, url, published_at, title, text, section`.
3. **Clean:** strip boilerplate, deduplicate.
4. **NLP:**
   - NER (spaCy vs. BERT-based)
   - Sentiment (VADER vs. RoBERTa)
   - Topic classification (TF-IDF + logistic regression vs. transformer)
   - Story clustering: group articles about the same event (TF-IDF + cosine vs. sentence embeddings)
   - Summary per story (TextRank or similar, from a full-text article)
5. **Dashboard:** story page, outlet profile, trends.

Most tasks (NER, sentiment, topic, clustering) work on headline + first paragraph, so the NYT can take part. Only the summary needs full text. An extra comparison we can run: **headline only vs. full text** for sentiment or clustering.

## Evaluation (what the course grades)

- At least 2 approaches per task, baseline vs. pre-trained, compared in one metrics table.
- A small hand-labelled sample: sentiment, entities, topic, and which articles belong to the same story.
- Concrete failure cases and why they fail (e.g. VADER on neutral news language, clustering merging two different stories).
- Failure journal kept from the start.

## Data rules

- Do not publish raw scraped data. The repo gets the collection scripts, the schema, and a **small anonymised sample** (`data/sample/`).
- `data/raw/` and `.env` (API keys) go in `.gitignore`.
- Scraping: check `robots.txt`, keep requests slow (about 1 per second), use the text only for our analysis. EU text-and-data-mining rules generally allow analysis for research but not republishing. (Our reading, not legal advice.)

## Risks and limits

- **Clustering quality:** wrong clusters mean every comparison on the story page is wrong.
- **Sentiment ≠ bias:** a more negative tone can come from the facts, not the outlet's leaning. Comparing against the story average helps but doesn't remove this.
- **Outlet-level leaning** is a coarse label; a single article can differ from its outlet.
- **English-only, mostly Western outlets:** "global" is limited to what these outlets cover.
- **Paywalls and terms of use** limit which outlets we can add.

## Next steps

1. Get Guardian (and NYT) API keys; keys go in `.env`.
2. Write the collector for Guardian + one RSS feed and start daily collection.
3. Look up the leaning ratings for each outlet and confirm the final outlet list.
4. Agree on the shared schema and set up the SQLite database.
