"""Kids Board Game Recommender: Streamlit app.

Run locally:  streamlit run app.py
Uses precomputed ./artifacts/ when present (built by build_artifacts.py / GitHub Actions),
otherwise downloads the data and computes everything on startup.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import recsys

REPO_URL = "https://github.com/KrasheninnikovaOlga/kids-board-games-recommender"
ARTIFACTS = Path(__file__).parent / "artifacts"
ACCENT, GREY, ACCENT2 = "#4C6EF5", "#ADB5BD", "#F59F00"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.titleweight": "bold"})

st.set_page_config(page_title="Kids Board Game Recommender", page_icon="🎲", layout="wide")


# ---------------------------------------------------------------- data (cached)
@st.cache_resource(show_spinner="Loading ~20,000 board games…")
def get_model():
    """Returns (games, kdj, X, feature_names, feature_blocks, evaluation)."""
    loaded = recsys.load_artifacts(ARTIFACTS)
    if loaded is not None:
        return loaded
    base = os.environ.get("BGG_DATA_BASE", recsys.DATA_URL)  # override for local testing
    games, kdj = recsys.prepare(recsys.load_raw(base))
    X, names, blocks = recsys.build_features(games)
    return games, kdj, X, names, blocks, recsys.evaluate(games, X)


games, kdj, X, feat_names, feat_blocks, evaluation = get_model()
ids_to_idx = pd.Series(games.index, index=games["id"])

PAGES = {"find": "🎯 Find a game", "similar": "🔁 More like this",
         "explore": "📊 Explore", "eval": "✅ How good is it?"}
KID_TAGS = games[games["minage"] <= 10]["tags"].explode().value_counts().head(40).index.tolist()


# ---------------------------------------------------------------- state <-> URL (shareable links)
def _int(qp, key, default, lo, hi):
    try:
        return min(max(int(qp.get(key, default)), lo), hi)
    except (TypeError, ValueError):
        return default


def _list(qp, key):
    v = qp.get(key, "")
    return [x for x in str(v).split(",") if x] if v else []


ss = st.session_state
if "initialized" not in ss:                      # first visit: read settings from the URL
    qp = st.query_params
    ss.page = PAGES.get(qp.get("page", "find"), PAGES["find"])
    ss.age = _int(qp, "age", 7, 3, 14)
    ss.players = _int(qp, "players", 3, 1, 8)
    ss.max_time = _int(qp, "time", 30, 10, 120) // 5 * 5
    likes = [t for t in _list(qp, "likes") if t in KID_TAGS]
    ss.likes = likes if "likes" in qp else [t for t in ["Cooperative Game", "Animals"] if t in KID_TAGS]
    liked_ids = [int(x) for x in _list(qp, "games") if x.isdigit() and int(x) in ids_to_idx.index]
    if liked_ids:
        ss.liked = [int(ids_to_idx[i]) for i in liked_ids]
    else:
        popular = games.sort_values("users_rated", ascending=False)
        ss.liked = [int(i) for i in popular.index[popular["name"].isin(["Ticket to Ride", "Dixit"])][:2]]
    ss.max_age = _int(qp, "max_age", 8, 3, 14)
    ss.versions = qp.get("versions", "0") == "1"
    ss.n = _int(qp, "n", 12, 4, 24) // 4 * 4
    ss.view = "Table" if qp.get("view") == "table" else "Cards"
    ss.initialized = True

WIDGET_KEYS = ["page", "age", "players", "max_time", "likes", "liked", "max_age", "versions", "n", "view"]
for k in WIDGET_KEYS:          # keep values when a widget is not rendered on the current page
    ss[k] = ss[k]


def sync_url(page_key):
    qp = {"page": page_key}
    if page_key == "find":
        qp.update(age=ss.age, players=ss.players, time=ss.max_time, likes=",".join(ss.likes))
    elif page_key == "similar":
        qp.update(games=",".join(str(int(games.at[i, "id"])) for i in ss.liked),
                  max_age=ss.max_age, versions=int(ss.versions))
    if page_key in ("find", "similar"):
        qp.update(n=ss.n, view=ss.view.lower())
    st.query_params.from_dict({k: str(v) for k, v in qp.items()})


# ---------------------------------------------------------------- result rendering
def _meta_line(r):
    return (f"👶 {int(r.minage)}+ · 👥 {int(r.minplayers)}–{int(r.maxplayers)} · "
            f"⏱ {int(r.playingtime)} min · ⭐ {r.bayes_average:.2f}")


def show_cards(df, why_label="Why"):
    per_row = 4
    rows = list(df.itertuples())
    for start in range(0, len(rows), per_row):
        cols = st.columns(per_row)
        for col, r in zip(cols, rows[start:start + per_row]):
            with col.container(border=True):
                # thumbnails in the data are small: show them at native size, never upscaled
                c_img, c_txt = st.columns([1, 3])
                if isinstance(r.thumbnail, str) and r.thumbnail.startswith("http"):
                    c_img.image(r.thumbnail, width="content")
                year = f" ({int(r.yearpublished)})" if pd.notna(r.yearpublished) else ""
                c_txt.markdown(f"**{r.name}**{year}")
                c_txt.caption(_meta_line(r))
                if r.award:
                    st.markdown(f"**{r.award}**")
                why = getattr(r, "why", "")
                if why:
                    st.caption(f"{why_label}: {why}")
                st.link_button("BoardGameGeek ↗", r.bgg_url)


def show_table(df, why_label="Why"):
    out = pd.DataFrame({
        "thumbnail": df["thumbnail"],
        "Game": df["name"],
        "Year": df["yearpublished"].astype("Int64"),
        "Age": df["minage"].astype(int).astype(str) + "+",
        "Players": df["minplayers"].astype(int).astype(str) + "–" + df["maxplayers"].astype(int).astype(str),
        "Minutes": df["playingtime"].astype(int),
        "BGG rating": df["bayes_average"].round(2),
        "Award": df["award"],
        "BGG": df["bgg_url"],
    })
    if "why" in df:
        out.insert(7, why_label, df["why"].to_numpy())
    st.dataframe(
        out, hide_index=True, width="stretch",
        column_config={
            "thumbnail": st.column_config.ImageColumn("", width="small"),
            "BGG": st.column_config.LinkColumn("BGG", display_text="open ↗"),
            "BGG rating": st.column_config.NumberColumn(format="%.2f",
                                                        help="BGG 'geek rating' (Bayesian average)"),
        },
    )


def show_results(df, why_label="Why", force_table=False):
    if df.empty:
        st.info("No games match these settings. Try more time or a different number of players.")
        return
    (show_table if force_table or ss.view == "Table" else show_cards)(df, why_label)


def result_controls():
    c1, c2, c3 = st.columns([2, 2, 3])
    c1.slider("How many suggestions", 4, 24, step=4, key="n")
    c2.radio("View", ["Cards", "Table"], horizontal=True, key="view")
    c3.caption("🔗 Your settings are saved in the address bar. Copy the link to share these results.")


# ---------------------------------------------------------------- header, sidebar, navigation
st.title("🎲 Kids Board Game Recommender")
st.caption(f"{len(games):,} board games from BoardGameGeek · content-based ML recommendations · "
           "checked against *Kinderspiel des Jahres* (Children's Game of the Year) picks")

with st.sidebar:
    st.header("About")
    st.markdown(
        "Find board games for kids by age, number of players and time, or get games similar to "
        "the ones your family already loves.\n\n"
        "**How it works:** each game is a vector of its mechanics, categories, description "
        "(TF-IDF) and basic stats. Recommendations use cosine similarity blended with the BGG rating, "
        "and each pick shows the features that made it similar. Versions of the same game "
        "(*Catan Junior*, *Ticket to Ride: Europe*) are filtered out.\n\n"
        "**Data:** BoardGameGeek via "
        "[TidyTuesday](https://github.com/rfordatascience/tidytuesday/tree/main/data/2022/2022-01-25) "
        "(snapshot Jan 2022, games with 200+ ratings) · award list from "
        "[Wikipedia](https://en.wikipedia.org/wiki/Spiel_des_Jahres).\n\n"
        f"**Code & notebook:** [GitHub]({REPO_URL})"
    )

page_label = st.radio("Section", list(PAGES.values()), horizontal=True, key="page",
                      label_visibility="collapsed")
page = next(k for k, v in PAGES.items() if v == page_label)

# ---------------------------------------------------------------- find a game
if page == "find":
    st.subheader("My kid is… we are… we have…")
    c1, c2, c3 = st.columns(3)
    c1.slider("Kid's age", 3, 14, key="age")
    c2.slider("Players", 1, 8, key="players")
    c3.slider("Max minutes", 10, 120, step=5, key="max_time")
    st.multiselect("We like… (mechanics & themes)", KID_TAGS, key="likes")
    result_controls()
    res = recsys.recommend(games, ss.age, ss.players, ss.max_time, likes=ss.likes, n=ss.n)
    show_results(res, why_label="Matches")

# ---------------------------------------------------------------- more like this
elif page == "similar":
    st.subheader("We loved these games. What's similar?")
    order = games.sort_values("users_rated", ascending=False).index.tolist()
    st.multiselect("Games your family loves", order, key="liked",
                   format_func=lambda i: games.at[i, "name"] + (
                       f" ({int(games.at[i, 'yearpublished'])})" if games.at[i, "yearpublished"] > 0 else ""),
                   placeholder="Start typing a game name…")
    c1, c2 = st.columns([2, 3])
    c1.slider("Suitable from age", 3, 14, key="max_age")
    c2.checkbox("Include versions of the same game", key="versions",
                help="e.g. Ticket to Ride: Europe for Ticket to Ride")
    result_controls()
    if ss.liked:
        res = recsys.similar_games(games, X, ss.liked, max_age=ss.max_age, n=ss.n,
                                   allow_same_series=ss.versions, names=feat_names, blocks=feat_blocks)
        show_results(res, why_label="In common")
    else:
        st.info("Pick at least one game you like.")

# ---------------------------------------------------------------- explore
elif page == "explore":
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
        show_results(games[games["age_group"] == group].nlargest(8, "bayes_average"))

# ---------------------------------------------------------------- evaluation
else:
    st.subheader("Does the model agree with the experts?")
    st.markdown(
        "The jury of the German *Kinderspiel des Jahres* award plays games with children and picks the best "
        "kids' game every year. We use its winners and nominees (1989–2021) as a test:\n\n"
        "1. Give the model **3 random award games** as a family's favourites.\n"
        "2. **Hide another award game** and see how high it ranks among all kids' games (age ≤ 8).\n"
        "3. Repeat for every award game and compare with baselines that ignore what the family liked."
    )
    results, pool, award_pos = evaluation
    m1, m2, m3 = st.columns(3)
    m1.metric("Award games 1989–2021", len(kdj))
    m2.metric("Found on BGG by title", int(kdj["id"].notna().sum()))
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
        show_results(games.loc[list(award_pos)].sort_values("bayes_average", ascending=False),
                     force_table=True)
    missing = kdj.loc[kdj["id"].isna(), "title"].tolist()
    if missing:
        with st.expander(f"Not found on BGG by title ({len(missing)})"):
            st.write(", ".join(missing))

sync_url(page)
