# Labelling guide

For the team, November 2026. Every person has one Excel file, `labelling_<Name>.xlsx`, in the shared OneDrive
folder **TA2 / Semestral project - TA / Labelling**.
It has a short version of this guide on the first sheet (**READ ME**) and three sheets to fill.
In total about 60-90 minutes.

## Before you start
- **Work alone.** Don't discuss headlines with the others until everyone has sent their file back. We measure how
  much we agree, so we need independent answers.
- **Fill only the yellow columns.** Don't sort, delete or add rows, and don't change the other columns.
- Use the drop-down lists where there is one.
- When you are not sure, choose the best answer anyway and write `x` in **unsure** (sheet 1) or a short note.
- Open **your own** file in the OneDrive folder (Excel or Excel online), fill it, save it there with the same name,
  and tell Jakub when you are done. Don't open the other people's files before you have finished yours.

Why we do this: the course asks us to compare our models with hand labels. With your labels we can
(1) show how much people agree (if we don't agree, a model can't be "right" either), (2) get more examples of
positive news, and (3) measure the entity models, which have no labels yet.

---

## Sheet 1: Sentiment (70 headlines)

Column **my_label**: `negative`, `neutral` or `positive`.

**Ask one question:** *Reading only this headline, does it report something bad, something good, or neither?*

- **Headline only.** Don't open the article, don't search for it, don't think about which outlet wrote it.
- **Judge the event, not the style.** A calm headline about a bombing is still negative; an excited headline
  about an election poll is still neutral.

| Label | When | Examples |
|---|---|---|
| **negative** | death, injury, illness, crime, violence, war and attacks, disaster, scandal, crisis, failure, loss | "Two people killed in light aircraft crash near Loch Lomond" |
| | threats and warnings about things that have not happened yet ("warns", "risks", "on alert") | "Millions on alert as Hurricane Isaias set to bring dangerous storm surge" |
| | protests **with** violence or damage (clashes, riots, looting, fires) | "Protesters clash with police during nationwide demonstrations in France" |
| **positive** | a win, breakthrough, recovery, rescue, improvement, record achievement | "Brazil sees record fall in Amazon deforestation" |
| | a light feel-good story | "Bendy elm still standing against all odds crowned UK tree of the year" |
| | sports wins | "Haaland's record scoring spree helps Man City to seventh Premier League win" |
| **neutral** | process news with no clear harm or benefit: elections, debates, campaigns, policy, reports, explainers, announcements, business and market news | "Sweden's political parties launch coalition talks" |
| | protests and rallies **without** violence | "'No Kings' protests spread across the US ahead of midterm elections" |

**Stance test:** if supporters and opponents of a side would label the headline differently, it is **neutral**.
For example, an election win ("Trump-backed John Sununu wins New Hampshire Republican Senate primary"), a court
decision that helps one side, or a party's plan.

**Tricky cases. Decide like this:**

- **War news is negative, even if one side is winning** ("St Petersburg under attack from Ukrainian drone swarm").
- **A death is negative, also in an obituary** ("Margaret Hamilton … dies aged 90").
- **Crime and court news is negative** when the headline is about the crime ("Three men jailed for murdering
  surfers"). A process step without a crime in the headline can be neutral.
- **Mixed headline** (good and bad): choose what the headline is *mainly* about and write `x` in **unsure**.
  "Rescuers in Nepal find two more people alive after flash flood disaster": the news is the rescue, so **positive**.
- **Positive words don't make good news.** "Who's Winning the Race for Congress?" is election news, so **neutral**.
- **Questions and explainers** ("What's Next for Paramount") are neutral, unless the question is about something
  bad ("Is there a different way to fix the small boats crisis?" is about a crisis, so negative).

---

## Sheet 2: Entity types

One row for every **name** that a model found in a headline. The headline is written on the first row only;
the rows below it (with an empty headline) belong to the same headline.

Column **my_type**: choose one.

| my_type | What | Examples |
|---|---|---|
| `person` | a person (titles and 's are fine: "Pope Leo's" is a person) | Donald Trump, Leo, Keir Starmer |
| `organisation` | company, government agency, party, team, newspaper, international body | OpenAI, NATO, Labour, Man City, UN |
| `place` | country, city, region, building, sea | France, Gaza, Florida, Strait of Hormuz |
| `group` | nationality, religion or political group used as a word | Russian, Saudi (in "Saudi pipeline"), Republicans, Muslim |
| `event` | a named event | Fat Bear Week, Premier League (as a competition), Oktoberfest |
| `other` | a real name of a type we don't use | a law, a product (iPhone), a book or TV series, a ship |
| `no` | not a name, or **the wrong piece of text** | "Trump Says", "Saudi Aramco HQ" (the name is "Saudi Aramco"), "Photos" |

Column **missed** (on the **first row of each headline**): names in the headline that **no row lists**, also the
correct version of a row you marked `no`. Write each as `Name (type)` and separate them with `;`:

```
Keir Starmer (person); Gaza (place)
```

Only names in the **headline** count (not in the article).

---

## Sheet 3: Same entity

One row for each name found somewhere in our articles, with the headline and the summary (lead) to show the context.
Column **real_entity** is already filled with the name as found.

**Change it to the full real-world name when needed**, so the same thing always has exactly the same text:

| mention | real_entity |
|---|---|
| Trump | Donald Trump |
| US, U.S., America | United States |
| the EU | European Union |
| Starmer | Keir Starmer |
| Labour | Labour Party (UK) |

- Use the **English Wikipedia name** of the person, place or organisation. If two have the same name, add what
  makes it clear, as Wikipedia does: `Labour Party (UK)`.
- Read the headline and the lead to see who is meant. You may search the web for this sheet.
- If it is already the full name, leave it.
- If it is not a real name at all, write `no`.

---

## What happens next
Jakub runs `python pipeline\labels.py collect --folder <the OneDrive Labelling folder>`.
This checks every file, computes how much we agree (Cohen's kappa), and writes the labels that `sentiment.py` and
`ner.py` use for their evaluation.
