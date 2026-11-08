Small anonymised sample of collected articles and outputs.

## Files
- `sentiment_sample.csv`: 50 random headlines with the labels of the 5 sentiment models (made by `nlp/sentiment/comparison.py`).
- `sentiment_gold.csv`: 150 headlines labelled by hand (`my_label`): 95 neutral, 47 negative, 8 positive.
  Picked by `nlp/sentiment/make_label_sheet.py`, spread across outlets: Guardian 38, NYT 37, Al Jazeera 20, BBC 20,
  Fox News 20, NPR 15. The outlet and the model labels were hidden while labelling.
- `sentiment_to_label_extra.xlsx` (not in Git until labelled): 80 more headlines that a model thinks are positive
  (made by `nlp/sentiment/make_label_sheet_extra.py`), because 8 positives are too few to evaluate. Not a random sample: say so in the report.

## How to label (our codebook)
Ask one question: "Reading only this headline, is it reporting something bad, something good, or neither?"
Label what the headline says, since that is all the models see. Write `negative`, `neutral` or `positive` in `my_label`;
use `note` for anything unclear (unclear cases become our failure examples).

- **Headline only.** Don't open the article and don't use what you know about the outlet.
- **Negative:** death, injury, illness, crime, violence, disaster, war, scandal, crisis, failure, loss.
  Also threats and warnings about things that haven't happened yet ("risks", "warns", "on alert").
- **Positive:** a win, breakthrough, recovery, rescue, improvement, record achievement, or a light feel-good story.
- **Neutral:** process news with no clear harm or benefit: elections, debates, campaigns, policy, reports,
  explainers, announcements.
- **Stance test:** if supporters and opponents of a side would label it differently, it's neutral.
  Protests and rallies are neutral unless the headline itself mentions violence or damage (riots, looting, clashes);
  then they're negative.

Do not look at the model labels while labelling.
