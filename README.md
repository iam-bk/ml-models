# ML Models Training Lab

We're learning machine learning together, hands-on: build each model visually in **Orange Data Mining**, then rebuild it in code with Python ML libraries (pandas, scikit-learn, and later PyTorch) to see what Orange does behind the scenes.

Each lesson covers one model on a real dataset, explaining every click, every concept and the results we got.

## Lessons

| # | Lesson | Model | Dataset | Date |
|---|---|---|---|---|
| 01 | [Predicting IMDb ratings with Random Forest](lessons/01-random-forest-imdb/README.md) | Random Forest (regression) | IMDb 5000 + 30 recent movies | 2026-10-28 |

## Layout

```
ML-Models-Training-Lab/
├── data/                      shared datasets used by the lessons
│   ├── movie_metadata.csv         IMDb 5000 (Kaggle), 5,043 movies up to 2016, all columns
│   ├── movies_train_demo.csv      the same movies, trimmed to title + 5 features + rating
│   ├── new_movies_2019_2024.csv   30 movies released 2017–2024, for testing
│   └── imdb_top_250_movies.csv    IMDb Top 250, used to show a sampling-bias pitfall
└── lessons/
    └── 01-random-forest-imdb/
        ├── README.md              the lesson
        └── python/                the same workflow in scikit-learn
```

## Data sources

- `movie_metadata.csv`: the "IMDB 5000 Movie Dataset" from Kaggle (scraped from IMDb, ~2016).
- `imdb_top_250_movies.csv`: an IMDb Top 250 export.
- `new_movies_2019_2024.csv`: compiled for this lab from IMDb (ratings, votes, reviews, runtime) and Box Office Mojo (domestic gross, budgets), fetched 2026-10-28. Budgets marked `reported` weren't checked against a source.

IMDb data is for personal and non-commercial use. These files are included for learning only.

## Setup

- **Orange Data Mining:** download it from orangedatamining.com.
- **Python:** the lesson scripts declare their own dependencies, so `uv` installs them automatically:
  ```sh
  uv run lessons/01-random-forest-imdb/python/random_forest_imdb.py
  ```

## How we work

1. Start with a question we can answer with data, such as "can we predict a movie's rating?"
2. Build it in Orange, with every widget and link explained.
3. Test it on data the model has never seen.
4. Rebuild it in Python and check that the numbers agree.
5. Write down what surprised us. That's usually the real lesson.
