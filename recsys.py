"""Data prep, features, recommenders and evaluation for the Kids Board Game Recommender.

Shared by the Streamlit app (app.py). Same logic as the Colab notebook.
"""
import ast
import html
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize

DATA_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2022/2022-01-25/"
MIN_RATINGS = 200
FEATURE_WEIGHTS = {"mechanics": 1.0, "categories": 1.0, "text": 0.7, "numeric": 0.5}

# Kinderspiel des Jahres (Children's Game of the Year), 1989–2021.
# Source: https://en.wikipedia.org/wiki/Spiel_des_Jahres  (BGG snapshot is from Jan 2022)
KDJ = [
    (1989, "winner", ["Gute Freunde", "Good Friends"]),
    (1990, "winner", ["Das Geisterschloss", "My Haunted Castle"]),
    (1991, "winner", ["Corsaro", "Corsaro: Irrfahrt im Piratenmeer"]),
    (1992, "winner", ["Schweinsgalopp", "Galloping Pigs"]),
    (1993, "winner", ["Ringel Rangel"]),
    (1994, "winner", ["Loopin' Louie"]),
    (1995, "winner", ["Karambolage"]),
    (1996, "winner", ["Vier zu Mir", "Noah's Park"]),
    (1997, "winner", ["Leinen Los!", "Leinen Los"]),
    (1998, "winner", ["Zicke Zacke Hühnerkacke", "Chicken Cha Cha Cha"]),
    (1999, "winner", ["Kayanak"]),
    (2000, "winner", ["Arbos"]),
    (2001, "winner", ["Klondike"]),
    (2001, "nominee", ["Rüsselbande", "The Piggyback Brigade"]),
    (2001, "nominee", ["Im Märchenwald"]),
    (2002, "winner", ["Maskenball der Käfer", "The Ladybug's Costume Party"]),
    (2002, "nominee", ["Höchst verdächtig", "Highly Suspect"]),
    (2002, "nominee", ["Bärenstark"]),
    (2003, "winner", ["Viva Topo!", "Viva Topo"]),
    (2003, "nominee", ["Robbys Rutschpartie"]),
    (2003, "nominee", ["Lauras Sternenspiel"]),
    (2004, "winner", ["Geistertreppe", "Spooky Stairs"]),
    (2004, "nominee", ["Schnelle Welle", "Treasure Hunt"]),
    (2004, "nominee", ["Schatz der Drachen"]),
    (2004, "nominee", ["Mare Polare"]),
    (2004, "nominee", ["Macius: Achtung, fertig los!"]),
    (2005, "winner", ["Das kleine Gespenst"]),
    (2005, "nominee", ["Schildkrötenrennen", "Ribbit"]),
    (2005, "nominee", ["Schildi Schildkröte"]),
    (2005, "nominee", ["Mago Magino"]),
    (2005, "nominee", ["Daddy Cool"]),
    (2006, "winner", ["Der schwarze Pirat"]),
    (2006, "nominee", ["Piratissimo"]),
    (2006, "nominee", ["Nacht der Magier"]),
    (2006, "nominee", ["Los Mampfos"]),
    (2006, "nominee", ["Giro Galoppo"]),
    (2007, "winner", ["Beppo der Bock"]),
    (2007, "nominee", ["Rettet den Märchenschatz!", "Save the Treasure of Fairy Tales"]),
    (2007, "nominee", ["Hüpf hüpf, Hurra!", "Hop Hop Hooray!"]),
    (2007, "nominee", ["Gesagt – getan!"]),
    (2007, "nominee", ["Burg-Ritter", "Castle Knights"]),
    (2008, "winner", ["Wer war's?", "Whoowasit?"]),
    (2008, "nominee", ["Geisterjäger", "Ghost Hunters"]),
    (2008, "nominee", ["Fluss der Drachen"]),
    (2008, "nominee", ["Didi Dotter"]),
    (2008, "nominee", ["Capt'n Sharky"]),
    (2009, "winner", ["Das magische Labyrinth", "The Magic Labyrinth"]),
    (2009, "nominee", ["Curli Kuller"]),
    (2009, "nominee", ["Zoowaboo"]),
    (2009, "nominee", ["Nicht zu fassen", "Hide the Kids!"]),
    (2009, "nominee", ["Land in Sicht", "Tonga Island"]),
    (2010, "winner", ["Diego Drachenzahn"]),
    (2010, "nominee", ["Panic Tower!"]),
    (2010, "nominee", ["Vampire der Nacht", "Vampires of the Night"]),
    (2010, "nominee", ["Turi-Tour"]),
    (2010, "nominee", ["Kraken-Alarm"]),
    (2011, "winner", ["Da ist der Wurm drin"]),
    (2011, "nominee", ["Die kleinen Zauberlehrlinge", "Magician's Kitchen"]),
    (2011, "nominee", ["Monster-Falle", "Monster Trap"]),
    (2012, "winner", ["Schnappt Hubi!"]),
    (2012, "nominee", ["Spinnengift und Krötenschleim"]),
    (2012, "nominee", ["Die kleinen Drachenritter"]),
    (2013, "winner", ["Der verzauberte Turm", "The Enchanted Tower"]),
    (2013, "nominee", ["Gold am Orinoko"]),
    (2013, "nominee", ["Mucca Pazza"]),
    (2014, "winner", ["Geister, Geister, Schatzsuchmeister!", "Ghost Fightin' Treasure Hunters"]),
    (2014, "nominee", ["Flizz & Miez"]),
    (2014, "nominee", ["Richard Ritterschlag", "Brandon the Brave"]),
    (2015, "winner", ["Spinderella"]),
    (2015, "nominee", ["Push a Monster"]),
    (2015, "nominee", ["Schatz-Rabatz"]),
    (2016, "winner", ["Stone Age Junior", "My First Stone Age"]),
    (2016, "nominee", ["Mmm!"]),
    (2016, "nominee", ["Leo muss zum Friseur", "Leo Goes to the Barber"]),
    (2017, "winner", ["ICECOOL", "Ice Cool"]),
    (2017, "nominee", ["Captain Silver"]),
    (2017, "nominee", ["Der Mysteriöse Wald", "The Mysterious Forest"]),
    (2018, "winner", ["Funkelschatz", "Dragon's Breath"]),
    (2018, "nominee", ["Emojito!"]),
    (2018, "nominee", ["Panic Mansion"]),
    (2019, "winner", ["Tal der Wikinger", "Valley of the Vikings"]),
    (2019, "nominee", ["Fabulantica"]),
    (2019, "nominee", ["Go Gecko Go!"]),
    (2020, "winner", ["Speedy Roll", "Hedgehog Roll"]),
    (2020, "nominee", ["Foto Fish"]),
    (2020, "nominee", ["Wir sind die Roboter", "Robots"]),
    (2021, "winner", ["Dragomino"]),
    (2021, "nominee", ["Fabelwelten", "Storytailors"]),
    (2021, "nominee", ["Mia London"]),
]


# ---------------------------------------------------------------- loading & cleaning
def parse_list(x):
    if not isinstance(x, str):
        return []
    try:
        v = ast.literal_eval(x)
        return [str(i) for i in v] if isinstance(v, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = html.unescape(html.unescape(s))
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def base_title(name):
    # "Ticket to Ride: Europe" -> "ticket to ride";  "Catan (Junior Edition)" -> "catan"
    return re.split(r"\s*(?:[:(–—]| - )", str(name))[0].strip().lower()


def norm_title(t):
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def strip_title(desc, name, series):
    for t in {str(name), series}:
        if len(t) > 2:
            desc = re.sub(re.escape(t), " ", desc, flags=re.IGNORECASE)
    return desc


def load_raw(base=DATA_URL):
    ratings = pd.read_csv(base + "ratings.csv")
    details = pd.read_csv(base + "details.csv")
    return ratings.merge(details.drop(columns=["num"]), on="id", how="inner")


def prepare(raw, min_ratings=MIN_RATINGS):
    """Clean the raw BGG data and add award labels. Returns (games, kdj_table)."""
    raw = raw.copy()
    raw["categories"] = raw["boardgamecategory"].map(parse_list)
    raw["mechanics"] = raw["boardgamemechanic"].map(parse_list)
    raw["series"] = raw["name"].map(base_title)
    raw["implementations"] = raw["boardgameimplementation"].map(parse_list)
    desc = raw["description"].map(clean_text)
    raw["description_clean"] = [strip_title(d, n, s) for d, n, s
                                in zip(desc, raw["name"], raw["series"])]

    mask = (
        (raw["users_rated"] >= min_ratings)
        & (raw["bayes_average"] > 0)
        & raw["minage"].between(2, 18)
        & raw["playingtime"].between(5, 360)
        & (raw["minplayers"] >= 1)
        & (raw["maxplayers"] >= raw["minplayers"])
        & ~raw["categories"].map(lambda c: "Expansion for Base-game" in c)
    )
    games = raw.loc[mask].copy().reset_index(drop=True)
    games["maxplayers"] = games["maxplayers"].clip(upper=12)
    games["age_group"] = pd.cut(games["minage"], bins=[0, 5, 7, 9, 11, 99],
                                labels=["2–5", "6–7", "8–9", "10–11", "12+"])
    games["tags"] = (games["mechanics"] + games["categories"]).map(lambda t: sorted(set(t)))
    r = games["bayes_average"]
    games["rating_norm"] = (r - r.min()) / (r.max() - r.min())
    games["bgg_url"] = "https://boardgamegeek.com" + games["url"].astype(str)

    # award labels (exact match on normalized title, searched in the full data)
    raw_norm = raw.assign(n=raw["name"].map(norm_title))
    rows = []
    for year, status, titles in KDJ:
        keys = {norm_title(t) for t in titles}
        hit = raw_norm[raw_norm["n"].isin(keys)]
        bgg_id = hit.loc[hit["users_rated"].idxmax(), "id"] if len(hit) else None
        rows.append({"year": year, "status": status, "title": titles[0], "id": bgg_id})
    kdj = pd.DataFrame(rows)
    found = kdj.dropna(subset=["id"]).astype({"id": int}).drop_duplicates("id")
    label = {r.id: f"🏆 KdJ {r.year}" if r.status == "winner" else f"KdJ nominee {r.year}"
             for r in found.itertuples()}
    games["award"] = games["id"].map(label).fillna("")
    kdj["in_app"] = kdj["id"].isin(games["id"])
    return games, kdj


# ---------------------------------------------------------------- features
def build_features(games, w=FEATURE_WEIGHTS):
    X_mech = MultiLabelBinarizer(sparse_output=True).fit_transform(games["mechanics"]).astype(float)
    X_cat = MultiLabelBinarizer(sparse_output=True).fit_transform(games["categories"]).astype(float)
    tfidf = TfidfVectorizer(max_features=6000, stop_words="english", min_df=5,
                            ngram_range=(1, 2), sublinear_tf=True)
    X_text = tfidf.fit_transform(games["description_clean"])
    num = games[["minage", "minplayers", "maxplayers"]].assign(log_time=np.log1p(games["playingtime"]))
    X_num = sparse.csr_matrix(StandardScaler().fit_transform(num) / np.sqrt(num.shape[1]))
    X = sparse.hstack([
        normalize(X_mech) * w["mechanics"],
        normalize(X_cat) * w["categories"],
        normalize(X_text) * w["text"],
        X_num * w["numeric"],
    ]).tocsr()
    return normalize(X)


# ---------------------------------------------------------------- recommenders
def same_series(a, b):
    if a == b:
        return True
    short, long_ = sorted([a, b], key=len)
    return len(short) >= 4 and long_.startswith(short + " ")


def diversify(ranked, n, exclude_series=()):
    """Walk down the ranking and keep at most one game per series."""
    picked, seen = [], list(exclude_series)
    for i, s in zip(ranked.index, ranked["series"]):
        if any(same_series(s, x) for x in seen):
            continue
        picked.append(i)
        seen.append(s)
        if len(picked) == n:
            break
    return ranked.loc[picked]


def similar_games(games, X, liked_idx, max_age=None, n=10, rating_weight=0.3, allow_same_series=False):
    """'We loved these games, what's similar?' liked_idx: row indices in `games`."""
    idx = list(liked_idx)
    profile = normalize(sparse.csr_matrix(X[idx].mean(axis=0)))
    sim = (X @ profile.T).toarray().ravel()
    cand = games.assign(similarity=sim).drop(index=idx)
    if max_age is not None:
        cand = cand[cand["minage"] <= max_age]
    cand = cand.assign(score=(1 - rating_weight) * cand["similarity"] + rating_weight * cand["rating_norm"])
    ranked = cand.sort_values("score", ascending=False)
    if allow_same_series:
        return ranked.head(n)
    liked_names = set(games.loc[idx, "name"])
    liked_impl = set().union(*games.loc[idx, "implementations"])
    related = ranked["name"].isin(liked_impl) | ranked["implementations"].map(
        lambda imp: bool(liked_names & set(imp)))
    return diversify(ranked[~related], n, exclude_series=games.loc[idx, "series"])


def recommend(games, age, players, max_time, likes=(), n=10, rating_weight=0.4):
    """'My kid is 7, we're 3 players, we have 30 minutes.'"""
    likes = set(likes)
    cand = games[(games["minage"] <= age)
                 & (games["minplayers"] <= players) & (games["maxplayers"] >= players)
                 & (games["playingtime"] <= max_time)].copy()
    if cand.empty:
        return cand.assign(why=pd.Series(dtype=str), score=pd.Series(dtype=float))
    if likes:
        cand["matched"] = cand["tags"].map(lambda t: sorted(likes & set(t)))
        pref = cand["matched"].map(len) / len(likes)
        cand["score"] = (1 - rating_weight) * pref + rating_weight * cand["rating_norm"]
    else:
        cand["matched"] = [[] for _ in range(len(cand))]
        cand["score"] = cand["rating_norm"]
    cand["why"] = cand["matched"].map(lambda m: ", ".join(m) if m else "highly rated")
    return diversify(cand.sort_values("score", ascending=False), n)


# ---------------------------------------------------------------- exploration helpers
def tag_share(df):
    return df["tags"].explode().value_counts() / len(df)


def kids_vs_grownups(games, k=8):
    diff = tag_share(games[games["minage"] <= 8]).sub(tag_share(games[games["minage"] >= 12]), fill_value=0)
    return pd.concat([diff.nlargest(k), diff.nsmallest(k)]).sort_values()


def coop_trend(games):
    return (games[games["yearpublished"].between(2000, 2021)]
            .assign(coop=lambda d: d["mechanics"].map(lambda m: "Cooperative Game" in m),
                    segment=lambda d: np.where(d["minage"] <= 8, "Kids (age ≤ 8)", "Older (9+)"))
            .groupby(["yearpublished", "segment"])["coop"].mean().unstack()
            .rolling(3, min_periods=1).mean())


# ---------------------------------------------------------------- evaluation
def evaluate(games, X, n_liked=3, n_draws=5, rating_weight=0.3, seed=42):
    """Leave-one-out on award games: give the model 3 random award games as 'liked',
    hide another one and see how high it ranks among all kids' games (age <= 8)."""
    rng = np.random.default_rng(seed)
    pool = games.index[games["minage"] <= 8]
    award_pos = [i for i in pool if games.at[i, "award"]]
    if len(award_pos) <= n_liked:
        return None, pool, award_pos
    X_pool = X[pool]
    pos = {g: k for k, g in enumerate(pool)}
    rating = games.loc[pool, "rating_norm"].to_numpy()
    popularity = games.loc[pool, "users_rated"].to_numpy(dtype=float)
    records = []
    for held in award_pos:
        others = [a for a in award_pos if a != held]
        for _ in range(n_draws):
            liked = list(rng.choice(others, size=n_liked, replace=False))
            excl, t = [pos[l] for l in liked], pos[held]
            n_cand = len(pool) - len(excl)
            profile = normalize(sparse.csr_matrix(X[liked].mean(axis=0)))
            sim = (X_pool @ profile.T).toarray().ravel()
            methods = {"our model": (1 - rating_weight) * sim + rating_weight * rating,
                       "random": rng.random(len(pool)),
                       "most popular": popularity,
                       "highest rated": rating}
            for name, sc in methods.items():
                s = sc.copy()
                s[excl] = -np.inf
                r = int((s > s[t]).sum())
                records.append({"method": name, "rank": r, "pct": r / n_cand})
    ev = pd.DataFrame(records)
    results = (ev.groupby("method")
               .agg(hit_at_10=("rank", lambda r: (r < 10).mean()),
                    hit_at_50=("rank", lambda r: (r < 50).mean()),
                    median_rank_pct=("pct", "median"))
               .sort_values("median_rank_pct"))
    return results, pool, award_pos
