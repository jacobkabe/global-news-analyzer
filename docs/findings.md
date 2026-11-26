# Findings and limitations

One sentence per finding, with where the evidence is. Keep this file up to date: it is the raw material for the
"Evaluation", "Failure analysis" and "Limitations" parts of the report. Newest findings at the end of each section.

## Data
- **Most of our data comes from two outlets:** 94% of the 10,259 articles (to 15 Nov) are from the Guardian and the NYT; the RSS
  outlets (BBC, Al Jazeera, Fox News, NPR) only started in mid-October with a few articles a day, so their numbers
  rest on very few articles. *(dashboard, "How the labels work")*
- **Only the Guardian gives full text**, so every model reads the headline (sentiment) or headline + summary
  (entities), which keeps the outlets comparable. *(docs/data-sources.md)*
- **A CSV opened and saved in Excel is damaged** (Czech settings): Tala's export of The News API (14 Nov) had ';;;'
  added to each line, 444 of 517 lines packed into one quoted field, and 3 lines split where a summary had a line
  break. merge_db.py now repairs such files; all 517 articles were recovered. Rule: never open our CSV exports
  in Excel; share news.db or use merge_db.py. *(src/collect/merge_db.py)*
- **The News API returns many video pages:** 119 of 172 CBS News and 37 of 54 NBC News articles (10-14 Nov) are
  videos (NBC's 'Nightly News' covers many events at once). URLs with '/video' now count as media, so 'Only news'
  hides them; this also moved 25 older RSS/NYT/Guardian video pages from news to media. Exception, decided by the
  team: CBS News videos stay news, because each clip reports one news event. *(pipeline/clean.py)*
- **The News API's categories are vague:** 175 of 517 articles are only 'general'; we take the first specific
  category ('general, politics' -> Politics). *(pipeline/clean.py topic)*
- **No outlet clearly allows scraping**, so we use APIs and RSS only (headline and summary). *(docs/data-sources.md)*

## Hand labels
- **Labelling without the codebook at hand is subjective:** a second pass against the codebook agreed with the first
  labels on 75% of 230 headlines (Cohen's kappa 0.52, "moderate") and corrected 30 clear mistakes, mostly war,
  violence and threats labelled neutral. *(data/sample/sentiment_gold.csv: label_before_review, note)*
- **The second pass was not fully blind** (some model outputs had been seen in the audit), but RoBERTa's accuracy went
  down after it (73% → 71%), so the corrections did not favour the models. *(results/sentiment_metrics.csv)*
- **The "random 150" are random within each outlet** (20 per small RSS outlet, 38/37 Guardian/NYT), not
  proportional to our data. *(data/sample/README.md)*
- **Good news is rare:** 11 of the 150 random headlines are positive. *(sentiment_gold.csv)*
- **The 80 "positive candidates" were picked by the models**, so they are not random and are not used for the main
  numbers (16 of them are really positive). *(data/sample/README.md)*
- **Team labelling (Nov 2026):** 4 people each label 30 shared headlines (agreement), 25 new random headlines and 15
  new positive candidates, plus the entity sheets. 2 of 4 files are back (Adriana, Klaudie, 15 Nov).
  *(pipeline/labels.py, docs/labelling-guide.md)*
- **Two teammates agree on 70% of the shared headlines** (Cohen's kappa 0.50, moderate): the same level as the first
  and second pass of our own labels (kappa 0.52). Each agrees with our reviewed labels more often (80% and 77%,
  kappa 0.68 and 0.62). People differ in how strict they are: of the same 30 headlines Adriana called 14 negative,
  Klaudie 10. *(results/sentiment_agreement.csv)*
- **The returned files had small slips:** a typo ("neagtive"), the types of missed names in an extra column, the guide
  pasted next to the table, and 2 pre-filled names left unchanged ("Trump" in "Trump-Xi State Dinner", "Zack
  Polanski's"). labels.py collect fixes the first three; the last two are corrected with a reason in the `note`
  column of data/sample/ner_gold_same.csv. *(pipeline/labels.py)*

## Sentiment models
- **No model is significantly better than the others:** RoBERTa 71%, VADER-tuned 71%, BART-tuned 70% accuracy on the
  random 150; paired bootstrap differences between them include zero. *(results/sentiment_metrics.csv)*
- **Accuracy alone is misleading:** always answering "neutral" is already right 53% of the time (macro-F1 0.23), so
  we report macro-F1 too. *(pipeline/README.md)*
- **RoBERTa gives the dashboard label for practical reasons,** not for a better score: a real neutral class, fast,
  and it can explain itself. *(pipeline/README.md)*
- **BART-MNLI calls almost everything negative:** its own labels reach 45% accuracy and almost never say neutral;
  with our tuned cut-off (negative only at −0.90 or lower) it reaches 70%. *(results/sentiment_cutoffs.csv)*
- **DistilBERT SST-2 is a failure example:** trained on movie reviews, it has no neutral class (42% accuracy).
  *(results/sentiment_metrics.csv)*
- **"Positive" labels are unreliable:** only 13 of RoBERTa's 32 positive labels on the 230 headlines are really good
  news, because positive words are not good news ("Who's Winning the Race for Congress?").
  *(failures/sentiment_gold_vs_models.csv)*
- **Good news told with bad words is missed** ("Rescuers in Nepal find two more people alive after flash flood
  disaster" is called negative). *(failures/sentiment_gold_vs_models.csv)*
- **RoBERTa's accuracy differs by outlet** (BBC 90%, NYT 84%, Guardian 58%, Fox News 55%; 15-38 headlines each),
  so part of an outlet's "more negative" gap on the dashboard can be model error. *(results/sentiment_by_outlet.csv)*
- **RoBERTa does not lean negative after the label review:** before it, it seemed to call many neutral headlines
  negative; with the corrected labels its errors go both ways (19 neutral→negative, 14 negative→neutral of 150).
  *(results/sentiment_confusion.png)*
- **NYT Title Case does not hurt RoBERTa:** on 68 Title Case headlines it is more accurate on the original (79%)
  than on our sentence-case version (74%), so it reads the original. *(audit, 12 Nov 2026)*
- **A finer cut-off grid is not worth it:** it improves macro-F1 by at most 0.03, even measured on the headlines it
  was tuned on. *(audit, 12 Nov 2026)*
- **On the teammates' labels the models score lower (held-out test):** 50 new random headlines (in proportion
  to our data: 29 Guardian, 21 NYT), labelled by teammates and never used for tuning: RoBERTa 58% (tuned cut-offs
  62%), VADER-tuned 56%, BART-tuned 54%, always "neutral" 52%; RoBERTa macro-F1 0.41 (0.32-0.50), and only 3 of the
  50 are positive. So the 71-73% on our own labels is partly fitted to one person's way of labelling.
  *(results/sentiment_metrics.csv, eval_set team_random)*
- **RoBERTa agrees with a person about as often as two people agree:** on the 30 shared headlines it agrees with
  Adriana on 70% and with Klaudie on 73%; Adriana and Klaudie agree on 70%. *(results/sentiment_agreement.csv)*
- **Its main miss is bad news told in calm words:** the team calls it negative, RoBERTa neutral (10 of the 21
  negative held-out headlines with tuned cut-offs), e.g. "Houthis seize key Yemeni port of Mocha in drive to take control of Red Sea
  coast", "UK households have paid extra £13,961 on bills since 2021 amid soaring inflation". *(sentiment_team.csv)*

## Explanations ("why this label")
- **Deleting a word to measure its effect can break the sentence:** "US workers facing deadly heat borrow…" without
  "heat" reads "facing deadly borrow…", so "heat" looked positive (+0.33). *(failure example, 12 Nov 2026)*
- **Masking the word instead works better** (two approaches compared on the 230 labelled headlines: hide the top-2
  words of each method and measure how much the score moves): masking 0.315 vs deleting 0.299; masking finds the
  more important words in 44% of headlines, deleting in 25% (tie 31%); the top word is a "nonsense" word
  (effect < 0.05) in 3.5% vs 11% of headlines. Caveat: the test hides words with a mask, which favours masking a little.
  *(results/sentiment_explain_comparison.csv: `python pipeline\sentiment.py compare-explain`)*
- **scikit-learn's stop-word list hid important words** ("not", "no", "never", "without", "against", "fire"), which
  appear in 7% of our headlines; we now use our own short filler list. *(sentiment.py FILLER)*
- **The new filler list shows the words that decide the meaning:** about 5% of headlines now show "not", "no",
  "never", "without" etc. among their driving words ("Starlink access still not fully restored…" → *not* −0.28),
  none did before. Side effect: "No" in a name ("No Kings", "No 10") is also shown, as a weak negative word.
  *(data/raw/outputs/sentiment_explain.csv)*
- **Punctuation was shown as a driving word:** in 67 headlines a dash ("–") was among the top words, because hiding
  it changes RoBERTa's score a little; punctuation is now never shown. *(sentiment.py as_text, app/data.py)*
- **The driving words explain the score, not the label:** the label is the most likely of three probabilities, so the
  dashboard now shows these probabilities above the words. *(app/streamlit_app.py)*

## Entities
- **The small spaCy model depends on capital letters:** NYT Title Case headlines give mistakes such as "Trump Says"
  as an organisation; the transformer model reads the whole sentence. *(pipeline/README.md)*
- **Name normalisation matters:** "Donald Trump" is found in 1,537 articles, but the exact text "Donald Trump" in only 76.
- **The 80% rule keeps unclear surnames apart, sometimes too strictly:** "Biden" stayed a separate entity because the
  full names were split between "Joe Biden" (5), "Joseph R. Biden Jr." (7, the NYT's style) and "Hunter Biden" (2).
  After adding the NYT spelling to our alias list, Joe Biden has 12 of 14 full names and "Biden" becomes Joe Biden
  (28 articles); Hunter Biden stays separate. Found by searching the dashboard (15 Nov 2026).
  *(pipeline/entity_aliases.csv)*
  *(results/ner_top_entities.csv)*
- **The transformer model finds names much better:** on 85 names in 50 headlines labelled by teammates, F1 (name and
  type right) 0.84 for en_core_web_trf vs 0.68 for en_core_web_sm; name only 0.92 vs 0.79; 26 vs 52 mistakes.
  *(results/ner_metrics.csv, failures/ner_errors.csv)*
- **Many of the transformer's remaining "errors" are type questions people disagree on:** "Kyiv books in 1,000
  Patriots" (Kyiv = the government, an organisation, or a place?), "EU" (place or organisation), "Australian" labelled
  as a place although the guide says group. So the type F1 is a lower bound. *(failures/ner_errors.csv)*
- **The name-rule check is too small to show much:** on 76 labelled mentions, pairs of mentions of the same entity
  score F1 0.994 before our rules and 1.000 after; the sample has few name variants (mostly 13 × "Trump"), so the
  better evidence for the rules is the count above. *(results/ner_metrics.csv)*

## Dashboard and comparison of outlets
- **Each outlet was collected over a different period:** Guardian 31 Aug – 14 Nov, NYT 1 Sep – 15 Nov (a gap of
  4 – 15 Nov filled in later because of the daily API limit), BBC, Al Jazeera, Fox News and NPR only about 11 – 18 Oct (the RSS collection was not
  kept running), The News API outlets only 10 – 14 Nov, Deutsche Welle 1 article. Comparing outlets over "all
  dates" would mix different news weeks. *(data/raw/news.db; dashboard header, Sources)*
- **So the dashboard compares each outlet with the Guardian on the same days** (the Guardian is the only outlet we have
  for every day), with a 95% interval, and only outlets with at least 30 articles. The Guardian has the largest share of
  negative headlines: on the same days, the NYT has 11 points fewer negative headlines (30% vs 41%, 76 days), ABC News
  20 and CBS News 15 points fewer; Fox News (+13), Al Jazeera and The Conversation show no clear difference (the gap
  may be chance with so few articles).
  *(dashboard, Outlets compared)*
- **The same day is not the same story, and outlets cover different topics** (CNBC is mostly business), so a difference
  can come from the topic mix and not from the tone. RoBERTa is also more accurate on some outlets than others.
- **For the NYT the gap is not the topic mix:** on the same days and within the same topic, the NYT has fewer
  negative headlines in 5 of 6 topics with 50+ articles each (US 29% vs 42%, World 38% vs 48%, Business 22% vs 38%,
  Technology 21% vs 35%, Environment 34% vs 38%; Science equal at 21%). Weighted to the NYT's topic mix, the Guardian
  is still 42% negative vs the NYT's 30%. Part of it can still be model error: RoBERTa is more accurate on NYT headlines
  (84%) than on Guardian ones (58%) in our hand labels. *(15 Nov 2026, articles_clean.csv + sentiment.csv)*

## Not built: story clustering (future work)
- **Planned:** group the articles about the same event into "stories" and compare each outlet with the story average
  (`archive/solution-idea-oct-2026.md`). This would be the fairest outlet comparison.
- **Why we did not build it:** (1) our data has little overlap between outlets: on only 10 of 77 days do three or more
  outlets have at least 5 news articles, and all outlets except the Guardian and the NYT together are 6% of the news
  articles, so most "stories" would contain one or two outlets; (2) a clustering needs its own hand labels (pairs of
  articles: same event or not) and two approaches to compare, which we could not also do well in the time; (3) we
  chose to spend the time on evaluating and explaining the labels we show (hand labels, label review, team check,
  driving words), because every label on the dashboard must be explainable. *(news.db; this file)*
- **A first try did not work well:** sentence embeddings + DBSCAN on 6,140 articles left 69% of them in no cluster,
  and 16 of the 20 biggest clusters contain only Guardian or NYT articles. *(failures/clustering.ipynb)*
- **What it would need:** collect the same outlets over the same weeks, then TF-IDF vs sentence embeddings with about
  100 labelled article pairs.

## Reproducibility
- **The model versions are pinned** (Hugging Face commit ids in sentiment.py, package versions in requirements.txt),
  and results are redone when a headline changes. *(pipeline/sentiment.py)*
