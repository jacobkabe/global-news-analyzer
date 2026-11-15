"""
ui.py - colours and small helpers that build HTML, shared by every view of the dashboard.
"""
import html

import streamlit as st

# Colours of the mockup: yellow accent, blue = positive, grey = neutral, orange = negative
YELLOW = "#FFDE59"
SENTIMENT_COLOURS = {"positive": "#6B9BFF", "neutral": "#8A8080", "negative": "#FF8A3D"}

esc = html.escape   # every text from the outlets goes through this before it is put into HTML


def sentiment_tag(label, score):
    if label == "not analysed":
        return '<span class="tag tag-not">Not analysed</span>'
    return f'<span class="tag tag-{label}">{label.capitalize()} · {score:+.2f}</span>'


def entity_chips(article_entities, limit=6):
    chips = [f'<span class="chip"><b>{row.tag}</b>{esc(row.entity)}</span>'
             for row in article_entities.head(limit).itertuples()]
    return "".join(chips)


def highlight(headline, words):
    """Colour the driving words in the headline: orange pulls towards negative, blue towards positive."""
    effects = dict(words)
    out = []
    for token in headline.split():
        word = token.strip(".,:;!?'\"()[]")   # the same rule as words_of() in pipeline/sentiment.py
        if word in effects:
            effect = effects.pop(word)          # highlight only the first time the word appears
            colour = SENTIMENT_COLOURS["negative"] if effect < 0 else SENTIMENT_COLOURS["positive"]
            strength = min(0.25 + abs(effect), 0.9)
            out.append(f'<span class="word" style="background:{colour}{int(strength * 255):02x}">{esc(token)}</span>')
        else:
            out.append(esc(token))
    return " ".join(out)


def word_chips(words):
    chips = []
    for word, effect in words:
        label = "negative" if effect < 0 else "positive"
        chips.append(f'<span class="tag tag-{label}">{esc(word)} {effect:+.2f}</span>')
    return " ".join(chips) or '<span class="muted">no word changes the score much</span>'


def probability_bar(probs, label):
    """One bar split into negative / neutral / positive, as wide as RoBERTa's probability for each."""
    parts = []
    for name, p in probs.items():
        text = f"{p:.0%}" if p >= 0.10 else ""          # only the number; too narrow parts stay empty
        weight = "700" if name == label else "400"      # the label's part in bold, nothing else stands out
        parts.append(f'<div title="{name} {p:.0%}" style="width:{p * 100:.1f}%;background:{SENTIMENT_COLOURS[name]}40;'
                     f'color:#EDE6E3;font-size:.7rem;font-weight:{weight};display:flex;align-items:center;'
                     f'justify-content:center;white-space:nowrap;overflow:hidden">{text}</div>')
    legend = " · ".join(f"<b>{n} {p:.0%}</b>" if n == label else f"{n} {p:.0%}" for n, p in probs.items())
    return (f'<div style="display:flex;height:16px;border-radius:4px;overflow:hidden;margin:.2rem 0">'
            f'{"".join(parts)}</div><p class="muted">{legend}</p>')


def heading(text, note=""):
    st.markdown(f'<p class="gna-h">{text}</p>' + (f'<p class="muted">{note}</p>' if note else ""),
                unsafe_allow_html=True)
