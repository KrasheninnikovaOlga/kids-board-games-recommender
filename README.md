# 🎲 Kids Board Game Explorer & Recommender

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_URL)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KrasheninnikovaOlga/kids-board-games-recommender/blob/main/kids_board_games_recommender.ipynb)

**Which board games are actually worth it for kids, and what should we play next?**

A Python notebook that explores ~20,000 board games from BoardGameGeek and builds a content-based
recommender that suggests games for a child's age, the number of players, and the time available.

## What's inside

- **Exploration**: how kids' games differ from grown-up games (mechanics, themes, ratings, cooperative games over time)
- **Best-rated games by age group**, ranked by BGG's Bayesian "geek rating"
- **Clustering**: TF-IDF + TruncatedSVD + K-Means groups kids' games into "game types", with a t-SNE map
- **Recommender #1**: *"We loved Dixit and Codenames, what's similar?"* (cosine similarity on mechanics, categories, description text and numeric features)
- **Recommender #2**: *"My kid is 7, we're 3 players, we have 30 minutes"*, with hard filters plus a preference score, and every pick explains why it was chosen
- **Evaluation against expert picks**: winners and nominees of the *Kinderspiel des Jahres* (Children's Game of the Year, 1989–2021) are used to test the recommender. Given 3 award games a family liked, how high does it rank a hidden award game, compared with random, most-popular and highest-rated baselines? (hit@10, hit@50, median rank)
- **Interactive widgets** to try your own combinations

## Live app

**[Try the app](YOUR_STREAMLIT_APP_URL)**: pick your kid's age, the number of players and the time you have, or list the games your family loves, and get recommendations with explanations, award badges and BoardGameGeek links.

The app (`app.py`, built with Streamlit) and the notebook share the same logic (`recsys.py`).

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notebook

Click **Open in Colab** above, then **Runtime → Run all**. No setup or API keys are needed: the data loads directly from GitHub.

## Tech

Python · Streamlit · pandas · scikit-learn · SciPy · matplotlib · ipywidgets · Google Colab

## Data

[TidyTuesday 2022-01-25: Board Games](https://github.com/rfordatascience/tidytuesday/tree/main/data/2022/2022-01-25),
which is BoardGameGeek data collected via Kaggle (snapshot from January 2022).
Award winners and nominees: [Wikipedia — Spiel des Jahres](https://en.wikipedia.org/wiki/Spiel_des_Jahres).

## Limitations & next steps

- The data is a January 2022 snapshot, and BGG raters are mostly adult hobbyists.
- `minage` is the publisher's recommendation.
- The recommender is content-based. A next step is collaborative filtering on user-level ratings,
  evaluated offline with hit-rate@10.
