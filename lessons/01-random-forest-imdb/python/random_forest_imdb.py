# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "scikit-learn", "matplotlib"]
# ///
"""Lesson 01 in code: the same Random Forest workflow we built in Orange.

Run from the lab root:
    uv run lessons/01-random-forest-imdb/python/random_forest_imdb.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline

LAB = Path(__file__).resolve().parents[3]
DATA = LAB / "data"
OUT = Path(__file__).resolve().parent / "output"

# Select Columns: features, target, meta
FEATURES = ["duration", "budget", "gross", "num_voted_users", "num_user_for_reviews"]
TARGET = "imdb_score"
META = "movie_title"

train = pd.read_csv(DATA / "movies_train_demo.csv")
new = pd.read_csv(DATA / "new_movies_2019_2024.csv")
X, y = train[FEATURES], train[TARGET]

# Random Forest widget. Orange imputes missing values for us; here we do it explicitly.
forest = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestRegressor(n_estimators=100, min_samples_split=5, random_state=0),
)
baseline = make_pipeline(SimpleImputer(strategy="median"), DummyRegressor())

# Test and Score: 5-fold cross-validation (5 throwaway models, then averaged scores)
cv = KFold(n_splits=5, shuffle=True, random_state=0)
print("Test and Score (5-fold cross-validation)")
for name, model in [("Constant", baseline), ("Random Forest", forest)]:
    s = cross_validate(model, X, y, cv=cv, scoring=["neg_mean_absolute_error", "r2"])
    print(f"  {name:14s} MAE {-s['test_neg_mean_absolute_error'].mean():.3f}   R2 {s['test_r2'].mean():.3f}")

# Train the real model on all the data, then Predictions on movies it has never seen
forest.fit(X, y)
new["predicted"] = forest.predict(new[FEATURES]).round(2)
new["error"] = (new["predicted"] - new[TARGET]).round(2)

print("\nPredictions on new movies (error = predicted - real)")
print(new[[META, TARGET, "predicted", "error"]].to_string(index=False))
print(f"\nMAE on new movies: {new['error'].abs().mean():.2f}")

# Which features the forest relied on most
importance = pd.Series(forest[-1].feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importance")
print(importance.round(3).to_string())

# Scatter Plot + Bar Plot
OUT.mkdir(exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.scatter(new[TARGET], new["predicted"])
ax1.plot([2, 9], [2, 9], linestyle="--", color="grey")
for _, r in new.iterrows():
    ax1.annotate(r[META], (r[TARGET], r["predicted"]), fontsize=7)
ax1.set(xlabel="Real IMDb rating", ylabel="Predicted", title="Actual vs predicted")
ordered = new.sort_values("error")
ax2.barh(ordered[META], ordered["error"])
ax2.set(xlabel="Error (predicted - real)", title="Error per movie")
ax2.tick_params(axis="y", labelsize=7)
fig.tight_layout()
fig.savefig(OUT / "random_forest_results.png", dpi=120)
print(f"\nChart saved to {OUT / 'random_forest_results.png'}")
