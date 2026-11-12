# Global News Analyzer – Solution Idea

_Draft 17 Oct 2026, updated 18 Oct 2026 after the data-collection phase. One decision is still open: how we use the RSS outlets (teacher's answer expected 19 Oct)._

## In one sentence

A dashboard that takes one news story, collects how several outlets report it, gives a short neutral summary, and shows **how the outlets differ**: sentiment, framing, who and what they mention, and what they leave out.

## Why it is useful (user benefits)

- **Save time:** one short summary per story instead of reading five articles.
- **See different angles:** the same event as told by different outlets, side by side.
- **Spot framing and bias:** which outlet is more negative or positive than the others, which people and countries it focuses on, how the headlines differ.
- **See coverage gaps:** which outlets covered a story at all, who was first, and what an outlet ignores.
- **Transparency (our "no black box" rule):** every label shows why it was given (driving words, model scores).

Comparable product to cite: Ground News (cross-outlet comparison with bias ratings).

## Status of the data phase (18 Oct 2026)

| Part | Status |
|---|---|
| Guardian collector (API, full text) | Built. 3,312 articles, 31 Aug – 18 Oct. |
| NYT collector (API, headline + abstract + lead paragraph) | Built. 2,688 news articles, 1 Sep – 18 Oct. |
| RSS collector (BBC, Al Jazeera, NPR, Fox) | Built, but saves **only title, summary and date** until the teacher decides. About 140 articles, 11 – 18 Oct. |
| Database (SQLite, one schema, UTC dates, clean URLs) | Built. |
| `collect.py` (all sources, one log line per source, fills in missed days for Guardian and NYT) | Built. |
| Daily run with Windows Task Scheduler | **Not set up yet.** Planned after the RSS decision. |
| Cleaning (boilerplate, deduplication) | Not started. |
| NLP and dashboard | Not started. |

Guardian and NYT overlap for about 48 days (about 70 articles a day each). That is our main corpus for story clustering. The RSS outlets only start on 11 – 15 Oct and add a few articles per day.

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

Terms of use were checked on 17 Oct 2026: **no outlet clearly allows scraping.** Quotes and RSS addresses for nine outlets are in `docs/data-sources.md`.

| Outlet | How we get it | What we store now | Terms of use | Leaning (external rating) |
|---|---|---|---|---|
| The Guardian | Open Platform API (free developer key) | Full text, from the API | API terms still to read | to look up |
| The New York Times | Article Search + Archive API | Headline, abstract, lead paragraph (no article text) | API terms still to read | to look up |
| BBC News | RSS | Title, summary, date | Not allowed (ruled out for full text) | to look up |
| Al Jazeera | RSS | Title, summary, date | Not allowed | to look up |
| NPR | RSS | Title, summary, date | Unclear | to look up |
| Fox News (proposed) | RSS | Title, summary, date | Not allowed | to look up |

- **Only the Guardian gives us full text.** The NYT gives a headline, abstract and lead paragraph. Full text from the RSS outlets is switched off (`FULL_TEXT = False` in `src/collect/rss.py`) until the teacher approves it.
- The Guardian and NYT both probably lean centre-left (to be confirmed with the ratings), so **at least one right-leaning outlet** is needed for a left/right comparison to mean anything. Fox is the only candidate so far, and it is the one whose terms say "not allowed". This is an open question for the teacher.
- **Political leaning is taken from published outlet ratings (AllSides, Ad Fontes Media), not predicted by our model.** We have no labelled data for leaning and it would be hard to defend. We cite the rating source.
- Not used: NewsAPI.org (free plan is dev-only, 24h delay, text cut to ~200 characters), GDELT (gives only links from thousands of sites; does not match our fixed outlet list).

## Pipeline

1. **Collect:** `src/collect/collect.py` runs the Guardian API, the NYT API and the RSS feeds. Guardian and NYT are fetched from the newest article we already have, so a missed day is filled in on the next run. **RSS feeds only show the latest items (no history), so a missed day is lost for good.** The plan is to run it **twice a day** with Windows Task Scheduler on one team machine (not set up yet).
2. **Store:** SQLite, one common schema for all sources: `id, source, url, published_at, title, summary, text, section, collected_at`. Dates are UTC (`2026-10-17T14:28:07Z`), URLs have no tracking parameters. `text` is empty for every source except the Guardian.
3. **Clean:** strip boilerplate, deduplicate (not built yet).
4. **NLP:**
   - NER (spaCy vs. BERT-based)
   - Sentiment (VADER vs. RoBERTa)
   - Topic classification (TF-IDF + logistic regression vs. transformer)
   - Story clustering: group articles about the same event (TF-IDF + cosine vs. sentence embeddings)
   - Summary per story (TextRank or similar, from a full-text article, which means a Guardian article)
5. **Dashboard:** story page, outlet profile, trends.

Most tasks (NER, sentiment, topic, clustering) work on headline + summary or first paragraph, so the NYT and the RSS outlets can take part even without full text. Only the summary needs full text. An extra comparison we can run: **headline only vs. full text** for sentiment or clustering (the Guardian has both).

## Open decision: RSS outlets (teacher, expected 19 Oct)

| Answer | What changes |
|---|---|
| Title and summary are fine | Nothing in the code. We keep `FULL_TEXT = False`. |
| Full text is approved for some outlets | We switch `FULL_TEXT` on and mark those outlets as approved in `TERMS`. A small script fetches the missing text for the rows we already have. |
| No RSS at all | We remove the RSS feeds from the collection and delete the stored RSS rows. The corpus is then Guardian + NYT only, and we need another idea for a right-leaning outlet. |

Questions to ask: (1) May we store title and summary from these outlets? (2) May we download full text for our own analysis (not republishing)? Which outlets? (3) Is there an acceptable right-leaning outlet instead of Fox? (4) Are the Guardian and NYT API terms fine for what we store?

## Evaluation (what the course grades)

- At least 2 approaches per task, baseline vs. pre-trained, compared in one metrics table.
- A small hand-labelled sample: sentiment, entities, topic, and which articles belong to the same story.
- Concrete failure cases and why they fail (e.g. VADER on neutral news language, clustering merging two different stories).
- Failure journal kept from the start.

## Data rules

- Do not publish raw scraped data. The repo gets the collection scripts, the schema, and a **small anonymised sample** (`data/sample/`, still to be created; what it may contain depends on the terms of the sources).
- `data/raw/` and `.env` (API keys) go in `.gitignore`. API keys must never be printed or logged.
- Scraping: **no outlet clearly allows it** (see `docs/data-sources.md`), so we do not download article pages from the RSS outlets until the teacher approves. When we do, we check `robots.txt` and the terms of use, keep requests slow (about 1 per second), and use the text only for our analysis. EU text-and-data-mining rules may allow analysis for research but not republishing, and some outlets opt out of this. (Our reading, not legal advice.)

## Risks and limits

- **Clustering quality:** wrong clusters mean every comparison on the story page is wrong.
- **Sentiment ≠ bias:** a more negative tone can come from the facts, not the outlet's leaning. Comparing against the story average helps but doesn't remove this.
- **Outlet-level leaning** is a coarse label; a single article can differ from its outlet.
- **Few outlets with enough data:** today only the Guardian and NYT have more than a few weeks of data, and both are Western and probably centre-left. The RSS outlets add only a few articles a day, so most stories will not have a match from them.
- **English-only, mostly Western outlets:** "global" is limited to what these outlets cover.
- **Paywalls and terms of use** limit which outlets we can add and what we may store.
- **The database lives on one machine** (it is not in Git), so the team needs a way to share a copy.

## Next steps

1. **Get the teacher's answer on RSS** (see above) and change the code and this document accordingly.
2. **Set up the daily run** (Task Scheduler, twice a day) once the RSS decision is made, and check the first runs in `data/raw/collect.log`.
3. **Read the Guardian and NYT API terms** and write what we found in `docs/data-sources.md`.
4. **Look up the leaning ratings** (AllSides, Ad Fontes) and confirm the final outlet list, including a right-leaning one.
5. **Decide how the team shares the database** and create the small sample in `data/sample/`.
6. **Start the NLP work:** hand-label about 100 article pairs (same story or not) and build the first clustering baseline on Guardian + NYT.
