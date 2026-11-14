# Labelling guide

Open **your own** file `labelling_<Name>.xlsx` in this folder, fill **only the yellow columns** (use the drop-downs),
save it under the same name. Don't sort or delete rows. **Work alone and don't discuss headlines** until everyone is
done: we measure how much we agree. Not sure? Choose the best answer and write `x` in *unsure*. About 60 minutes.

## 1 Sentiment: *does the headline report something bad, something good, or neither?*
Read **only the headline** (don't open the article, ignore the outlet). Judge the **event**, not the style.

| my_label | When | Example |
|---|---|---|
| **negative** | death, injury, illness, crime, violence, war, attacks, disaster, scandal, crisis, failure, loss; threats and warnings; protests **with** violence (clashes, riots, looting) | "Two people killed in light aircraft crash" · "Millions on alert as hurricane set to bring storm surge" |
| **positive** | win, breakthrough, recovery, rescue, improvement, record achievement, light feel-good story, sports win | "Brazil sees record fall in Amazon deforestation" · "Bendy elm crowned UK tree of the year" |
| **neutral** | elections, debates, policy, reports, explainers, announcements, business; protests **without** violence | "Sweden's parties launch coalition talks" · "'No Kings' protests spread across the US" |

- **Stance test:** if supporters and opponents of a side would label it differently, it is **neutral** (an election win, a court ruling for one side).
- War is negative even if one side is winning. A death is negative, also in an obituary.
- Positive *words* are not good *news*: "Who's Winning the Race for Congress?" is neutral.
- Mixed headline: choose what it is **mainly** about ("Rescuers find two more people alive after flash flood" = positive).

## 2 Entity types: *what kind of name is this?*
One row per name a model found in the headline (rows with an empty headline belong to the headline above).

| my_type | Examples |
|---|---|
| **person** (titles and 's are fine) | Donald Trump, "Pope Leo's" |
| **organisation**: company, agency, party, team, newspaper | OpenAI, NATO, Labour, UN |
| **place**: country, city, region, building | France, Gaza, Florida |
| **group**: nationality, religion, political group as a word | Russian, Saudi (in "Saudi pipeline"), Republicans |
| **event** | Fat Bear Week, Oktoberfest |
| **other**: a real name of another kind | a law, a product (iPhone), a book, a ship |
| **no**: not a name, or the wrong piece of text | "Trump Says", "Saudi Aramco HQ" (the name is "Saudi Aramco") |

**missed** (first row of each headline): names in the headline that no row lists, as `Keir Starmer (person); Gaza (place)`.

## 3 Same entity: *who or what is it really?*
**real_entity** is already filled with the name as it was found, so you only type when it needs changing (most rows are
already right). Change it to the **full English Wikipedia name**, so the same thing always has the same text:
Trump → `Donald Trump` · US / U.S. → `United States` · the EU → `European Union` · Starmer → `Keir Starmer`.
Use the lead to see who is meant (you may search the web). Not a real name at all? Write `no`.
