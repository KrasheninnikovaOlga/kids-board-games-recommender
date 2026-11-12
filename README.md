# 🎲 Kids Board Game Explorer & Recommender

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
- **Interactive widgets** to try your own combinations

## How to run

Click **Open in Colab** above, then **Runtime → Run all**. No setup or API keys are needed: the data loads directly from GitHub.

## Tech

Python · pandas · scikit-learn · SciPy · matplotlib · ipywidgets · Google Colab

## Data

[TidyTuesday 2022-01-25: Board Games](https://github.com/rfordatascience/tidytuesday/tree/main/data/2022/2022-01-25),
which is BoardGameGeek data collected via Kaggle (snapshot from January 2022).

## Limitations & next steps

- The data is a January 2022 snapshot, and BGG raters are mostly adult hobbyists.
- `minage` is the publisher's recommendation.
- The recommender is content-based. A next step is collaborative filtering on user-level ratings,
  evaluated offline with hit-rate@10.
