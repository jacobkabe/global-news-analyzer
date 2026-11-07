Small anonymised sample of collected articles and outputs.

## Files
- `sentiment_sample.csv`: 50 random headlines with the labels of the 5 sentiment models (made by `nlp/sentiment/comparison.py`).
- `sentiment_to_label.csv`: 150 headlines to label by hand (made by `nlp/sentiment/make_label_sheet.py`).
  Spread across outlets: Guardian 38, NYT 37, Al Jazeera 20, BBC 20, Fox News 20, NPR 15.
  The outlet and the model labels are hidden on purpose, so they do not influence the labeller.

## How to label `sentiment_to_label.csv`
Write one of `negative`, `neutral`, `positive` in `my_label`. Use `note` for anything unclear
(e.g. "sarcasm", "two events in one headline"); unclear cases become our failure examples.

Rule: label the headline as a typical reader would feel about it: the event it reports *and* its wording.
- `negative`: bad news or negative wording (deaths, attacks, crisis, loss, criticism, "slams", "warns").
- `positive`: good news or positive wording (recovery, success, rescue, agreement, record high).
- `neutral`: a plain fact with no clear good or bad side (announcements, schedules, explainers, "What we know about ...").

If you are unsure between two labels, pick `neutral` and write why in `note`.
Do not look at the model labels while labelling.
