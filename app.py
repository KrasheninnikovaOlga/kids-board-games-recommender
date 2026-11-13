"""Kids Board Game Recommender: Streamlit app.

Run locally:  streamlit run app.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import recsys

REPO_URL = "https://github.com/KrasheninnikovaOlga/kids-board-games-recommender"
ACCENT, GREY, ACCENT2 = "#4C6EF5", "#ADB5BD", "#F59F00"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.titleweight": "bold"})

st.set_page_config(page_title="Kids Board Game Recommender", page_icon="🎲", layout="wide")


# ---------------------------------------------------------------- data (cached)
@st.cache_data(show_spinner="Loading ~20,000 board games from BoardGameGeek…")
def get_data():
    base = os.environ.get("BGG_DATA_BASE", recsys.DATA_URL)  # override for local testing
    return recsys.prepare(recsys.load_raw(base))


@st.cache_resource(show_spinner="Building game features (mechanics, themes, descriptions)…")
def get_features():
    games, _ = get_data()
    return recsys.build_features(games)


@st.cache_resource(show_spinner="Evaluating the recommender against award winners…")
def get_evaluation():
    games, _ = get_data()
    return recsys.evaluate(games, get_features())


games, kdj = get_data()
X = get_features()


def show_games(df, extra=()):
    """Render a results table with thumbnails and BGG links."""
    if df.empty:
        st.info("No games match these filters. Try more time or a different number of players.")
        return
    out = pd.DataFrame({
        "thumbnail": df["thumbnail"],
        "Game": df["name"],
        "Year": df["yearpublished"].astype("Int64"),
        "Age": df["minage"].astype(int).astype(str) + "+",
        "Players": df["minplayers"].astype(int).astype(str) + "–" + df["maxplayers"].astype(int).astype(str),
        "Minutes": df["playingtime"].astype(int),
        "BGG rating": df["bayes_average"].round(2),
        **{label: df[col] for label, col in extra},
        "Award": df["award"],
        "BGG": df["bgg_url"],
    })
    st.dataframe(
        out, hide_index=True, width="stretch",
        column_config={
            "thumbnail": st.column_config.ImageColumn("", width="small"),
            "BGG": st.column_config.LinkColumn("BGG", display_text="open ↗"),
            "BGG rating": st.column_config.NumberColumn(format="%.2f",
                                                        help="BGG 'geek rating' (Bayesian average)"),
        },
    )


# ---------------------------------------------------------------- header & sidebar
st.title("🎲 Kids Board Game Recommender")
st.caption(f"{len(games):,} board games from BoardGameGeek · content-based ML recommendations · "
           "checked against *Kinderspiel des Jahres* (Children's Game of the Year) picks")

with st.sidebar:
    st.header("About")
    st.markdown(
        "Find board games for kids by age, number of players and time, or get games similar to "
        "the ones your family already loves.\n\n"
        "**How it works:** each game is a vector of its mechanics, categories, description "
        "(TF-IDF) and basic stats; recommendations use cosine similarity blended with the BGG rating. "
        "Versions of the same game (*Catan Junior*, *Ticket to Ride: Europe*) are filtered out.\n\n"
        "**Data:** BoardGameGeek via "
        "[TidyTuesday](https://github.com/rfordatascience/tidytuesday/tree/main/data/2022/2022-01-25) "
        "(snapshot Jan 2022, games with 200+ ratings) · award list from "
        "[Wikipedia](https://en.wikipedia.org/wiki/Spiel_des_Jahres).\n\n"
        f"**Code & notebook:** [GitHub]({REPO_URL})"
    )

tab_find, tab_similar, tab_explore, tab_eval = st.tabs(
    ["🎯 Find a game", "🔁 More like this", "📊 Explore", "✅ How good is it?"])

# ---------------------------------------------------------------- tab 1: find a game
with tab_find:
    st.subheader("My kid is… we are… we have…")
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Kid's age", 3, 14, 7)
    players = c2.slider("Players", 1, 8, 3)
    max_time = c3.slider("Max minutes", 10, 120, 30, step=5)
    kid_tags = games[games["minage"] <= 10]["tags"].explode().value_counts().head(40).index.tolist()
    default_likes = [t for t in ["Cooperative Game", "Animals"] if t in kid_tags]
    likes = st.multiselect("We like… (mechanics & themes)", kid_tags, default=default_likes)
    n = st.slider("How many suggestions", 5, 25, 10, key="n_find")
    res = recsys.recommend(games, age, players, max_time, likes=likes, n=n)
    show_games(res, extra=[("Why", "why")] if not res.empty else ())

# ---------------------------------------------------------------- tab 2: more like this
with tab_similar:
    st.subheader("We loved these games. What's similar?")
    order = games.sort_values("users_rated", ascending=False).index.tolist()
    label = lambda i: f"{games.at[i, 'name']} ({int(games.at[i, 'yearpublished'])})"
    defaults = [i for i in order if games.at[i, "name"] in ("Ticket to Ride", "Dixit")][:2]
    liked = st.multiselect("Games your family loves", order, default=defaults, format_func=label,
                           placeholder="Start typing a game name…")
    c1, c2, c3 = st.columns([2, 2, 3])
    max_age = c1.slider("Suitable from age", 3, 14, 8)
    n2 = c2.slider("How many suggestions", 5, 25, 10, key="n_similar")
    allow_series = c3.checkbox("Include versions of the same game", value=False,
                               help="e.g. Ticket to Ride: Europe for Ticket to Ride")
    if liked:
        res2 = recsys.similar_games(games, X, liked, max_age=max_age, n=n2, allow_same_series=allow_series)
        show_games(res2, extra=[("Similarity", "similarity")])
    else:
        st.info("Pick at least one game you like.")

# ---------------------------------------------------------------- tab 3: explore
with tab_explore:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Games and median rating by recommended minimum age**")
        by_age = games.groupby("age_group", observed=True).agg(
            n_games=("id", "size"), median_rating=("average", "median"))
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(by_age.index.astype(str), by_age["n_games"], color=ACCENT, alpha=0.85)
        ax.set_ylabel("Number of games")
        ax2 = ax.twinx()
        ax2.plot(by_age.index.astype(str), by_age["median_rating"], color=ACCENT2, marker="o", lw=2.5)
        ax2.set_ylabel("Median user rating", color=ACCENT2)
        ax2.grid(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with c2:
        st.markdown("**Share of cooperative games: kids vs older audiences**")
        coop = recsys.coop_trend(games)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for col, color in zip(coop.columns, [ACCENT, GREY]):
            ax.plot(coop.index, coop[col] * 100, label=col, color=color, lw=2.5)
        ax.set_ylabel("% cooperative (3-yr rolling)")
        ax.legend(frameon=False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("**What makes a kids' game?** Share of kids' games (age ≤ 8) minus share of grown-up games (12+)")
    top = recsys.kids_vs_grownups(games)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(top.index, top.values * 100, color=[ACCENT if v > 0 else GREY for v in top.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("percentage points")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("**Top-rated games by age group**")
    group = st.segmented_control("Minimum age", ["2–5", "6–7", "8–9", "10–11", "12+"], default="6–7")
    if group:
        show_games(games[games["age_group"] == group].nlargest(10, "bayes_average"))

# ---------------------------------------------------------------- tab 4: evaluation
with tab_eval:
    st.subheader("Does the model agree with the experts?")
    st.markdown(
        "The jury of the German *Kinderspiel des Jahres* award plays games with children and picks the best "
        "kids' game every year. We use its winners and nominees (1989–2021) as a test:\n\n"
        "1. Give the model **3 random award games** as a family's favourites.\n"
        "2. **Hide another award game** and see how high it ranks among all kids' games (age ≤ 8).\n"
        "3. Repeat for every award game and compare with baselines that ignore what the family liked."
    )
    results, pool, award_pos = get_evaluation()
    found = int(kdj["id"].notna().sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Award games 1989–2021", len(kdj))
    m2.metric("Found on BGG by title", found)
    m3.metric("In the test pool (age ≤ 8, 200+ ratings)", len(award_pos))

    if results is None:
        st.warning("Not enough award games in the data to run the evaluation.")
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            order = results.index.tolist()[::-1]
            ax.barh(order, results.loc[order, "hit_at_50"] * 100,
                    color=[ACCENT if m == "our model" else GREY for m in order])
            ax.set_xlabel(f"hit@50, %: hidden award game in the top 50 of {len(pool):,} kids' games")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        with c2:
            st.dataframe(
                (results * 100).rename(columns={"hit_at_10": "hit@10", "hit_at_50": "hit@50",
                                                "median_rank_pct": "median rank"}),
                column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                               for c in ["hit@10", "hit@50", "median rank"]},
                width="stretch")
            st.caption("hit@k: share of tests where the hidden game is in the top k. "
                       "Median rank: position as % of the list (lower is better).")

    with st.expander("Award games in the data"):
        show_games(games.loc[award_pos].sort_values("bayes_average", ascending=False))
    missing = kdj.loc[kdj["id"].isna(), "title"].tolist()
    if missing:
        with st.expander(f"Not found on BGG by title ({len(missing)})"):
            st.write(", ".join(missing))
