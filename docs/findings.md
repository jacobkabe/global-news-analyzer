# Findings and limitations

One sentence per finding, with where the evidence is. Keep this file up to date: it is the raw material for the
"Evaluation", "Failure analysis" and "Limitations" parts of the report. Newest findings at the end of each section.

## Data
- **Most of our data comes from two outlets:** 98% of the 6,140 articles are from the Guardian and the NYT; the RSS
  outlets (BBC, Al Jazeera, Fox News, NPR) only started in mid-October with a few articles a day, so their numbers
  rest on very few articles. *(dashboard, "How the labels work")*
- **Only the Guardian gives full text**, so every model reads the headline (sentiment) or headline + summary
  (entities), which keeps the outlets comparable. *(docs/data-sources.md)*
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
- **Team labelling (Nov 2026):** 4 people label 30 shared headlines (agreement), 100 new random headlines and 60 new
  positive candidates, plus the entity sheets. *(pipeline/labels.py, docs/labelling-guide.md; results to come)*

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

## Explanations ("why this label")
- **Deleting a word to measure its effect can break the sentence:** "US workers facing deadly heat borrow…" without
  "heat" reads "facing deadly borrow…", so "heat" looked positive (+0.33). *(failure example, 12 Nov 2026)*
- **Masking the word instead works better** (two approaches compared on the 230 labelled headlines: hide the top-2
  words of each method and measure how much the score moves): masking 0.316 vs deleting 0.299; masking finds the
  more important words in 45% of headlines, deleting in 25% (tie 30%); the top word is a "nonsense" word
  (effect < 0.05) in 3.5% vs 11% of headlines. Caveat: the test hides words with a mask, which favours masking a little.
  *(results/sentiment_explain_comparison.csv: `python pipeline\sentiment.py compare-explain`)*
- **scikit-learn's stop-word list hid important words** ("not", "no", "never", "without", "against", "fire"), which
  appear in 7% of our headlines; we now use our own short filler list. *(sentiment.py FILLER)*
- **The new filler list shows the words that decide the meaning:** about 5% of headlines now show "not", "no",
  "never", "without" etc. among their driving words ("Starlink access still not fully restored…" → *not* −0.28),
  none did before. Side effect: "No" in a name ("No Kings", "No 10") is also shown, as a weak negative word.
  *(data/raw/outputs/sentiment_explain.csv)*
- **The driving words explain the score, not the label:** the label is the most likely of three probabilities, so the
  dashboard now shows these probabilities above the words. *(app/streamlit_app.py)*

## Entities
- **The small spaCy model depends on capital letters:** NYT Title Case headlines give mistakes such as "Trump Says"
  as an organisation; the transformer model reads the whole sentence. *(pipeline/README.md)*
- **Name normalisation matters:** "Donald Trump" is found in 957 articles, but the exact text "Donald Trump" in only 48.
  *(results/ner_top_entities.csv)*
- **The entity evaluation is still to come** (team labels). *(pipeline/labels.py)*

## Dashboard and comparison of outlets
- **Outlet differences are not fair yet:** outlets cover different topics, so a gap in average sentiment can come from
  the topic mix; the fair comparison (the same story in every outlet) needs story clustering. *(dashboard, Outlets tab)*

## Reproducibility
- **The model versions are pinned** (Hugging Face commit ids in sentiment.py, package versions in requirements.txt),
  and results are redone when a headline changes. *(pipeline/sentiment.py)*
