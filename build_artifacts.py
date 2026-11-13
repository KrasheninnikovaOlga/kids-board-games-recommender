"""Precompute data & features for the Streamlit app.

Run:  python build_artifacts.py   (writes ./artifacts/)
Runs automatically in GitHub Actions on every change to recsys.py (see .github/workflows/).
"""
import time

import recsys

t = time.time()
games, kdj = recsys.prepare(recsys.load_raw())
X, names, blocks = recsys.build_features(games)
evaluation = recsys.evaluate(games, X)
recsys.save_artifacts("artifacts", games, kdj, X, names, blocks, evaluation)
print(f"Built artifacts for {len(games):,} games, {X.shape[1]:,} features in {time.time() - t:.0f}s")
