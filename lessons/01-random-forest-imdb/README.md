# Lesson 01: Predicting IMDb Ratings with Random Forest

> I am going to predict a movie's IMDb rating from its basic facts using a Random Forest, a model that combines the guesses of hundreds of decision trees. I train it on about 5,000 older movies and check how accurate it is on movies it hasn't seen during training. I then use it to predict the ratings of 30 recent movies and compare each prediction with the real rating. Through this, I see how well the model ranks good and bad films, and why a model trained on older data can over-rate newer ones.

**Tools:** Orange Data Mining; Python with pandas and scikit-learn
**Data:** `data/movies_train_demo.csv` (training) and `data/new_movies_2019_2024.csv` (testing)
**Time:** about 30 minutes to build in Orange

---

## Part 1: Core ideas

### 1.1 What we're doing: regression
We predict a **number**, the IMDb rating, so this is **regression**. If we predicted a category such as "hit" or "flop", it would be **classification**.

Every supervised model works the same way:
1. **Train:** show it many examples where both the inputs **and** the answer are known.
2. **Predict:** give it a new example with only the inputs, and it returns an answer.

### 1.2 The three column roles
| Role | Meaning | In this lesson |
|---|---|---|
| **Features** | Inputs the model learns from | `duration`, `budget`, `gross`, `num_voted_users`, `num_user_for_reviews` |
| **Target (label)** | What we predict | `imdb_score` |
| **Meta** | Shown for reference, never learned from | `movie_title` |

Text columns such as URLs, IDs and descriptions can't be used directly by these models, so we leave them out.

### 1.3 How a decision tree is built
A tree is a flowchart of yes/no questions:
```
Did more than 90,000 people vote?
├── Yes → Is it longer than 2 hours?
│         ├── Yes → 8.0
│         └── No  → 7.2
└── No  → Did it earn less than $10M?
          ├── Yes → 5.4
          └── No  → 6.3
```

**Choosing a question (a split):** for a candidate feature, sort the movies by it and try a cut between every pair of neighbouring values (`votes ≤ 1,000?`, `votes ≤ 1,050?` …). Score each cut by how **similar the ratings are inside each resulting group** (low variance is good). Keep the best cut. Then repeat inside each group.

Example: for six movies rated 4.0, 4.5, 5.0, 7.5, 8.0, 8.5, the cut "votes ≤ 90K?" separates {4.0, 4.5, 5.0} from {7.5, 8.0, 8.5}, so both groups are tight and the question is a **good** one.

**When a branch stops (base conditions):** it becomes a **leaf** when
- the group has fewer than 5 movies (Orange's *Do not split subsets smaller than*),
- it hits a maximum depth, if one is set (off by default),
- all the ratings in the group are equal, or
- no question improves the split.

**A leaf's prediction is the average rating of the training movies in it.**

### 1.4 From one tree to a forest
One deep tree **memorises** its training data and is unstable. A Random Forest builds many trees (the number is a setting, such as 10, 50 or 100) and makes each one **different** using two sources of randomness:

| Randomness | When |
|---|---|
| A random sample of movies, drawn with replacement (bootstrap) | Once per tree |
| A random subset of features to choose from | At every split |

The trees aren't "one tree per feature". Every tree can use all the features across its many splits, often the same feature several times. It just isn't allowed to see them all at any single question. Without that rule, every tree would start with `num_voted_users` and the trees would all be the same.

### 1.5 How the forest predicts
1. The new movie goes down **every** tree, answering each question with its own values.
2. Each tree returns the average rating of the leaf it lands in.
3. The forest **averages** all the trees' answers. For classification, the trees **vote**.

The trees make **different** mistakes, so averaging cancels many of them out.

Two consequences:
- **It can't predict outside the training range.** Every answer is an average of real ratings.
- **It pulls toward the middle.** Extreme movies (very good or very bad) get softened.

### 1.6 Why Random Forest for this lesson
- **No data preparation needed.** Trees only ask "above or below?", so scale, outliers and huge numbers don't matter.
- **Best results** of the models we tried (Part 4).
- **Easy to explain:** many trees, each asking questions, averaged.

A Neural Network needed a log transform plus normalization and still did worse. Linear Regression was thrown off by outliers (see Part 4).

### 1.7 How the other models relate (for later lessons)
| Family | Models | What training adjusts |
|---|---|---|
| Weighted sum (a single neuron) | Linear Regression, Logistic Regression, SGD, linear SVM, PLS | Weights and bias |
| Layers of neurons | Neural Network | Weights and biases, via backpropagation and gradient descent |
| Trees | Tree, Random Forest, Gradient Boosting, AdaBoost | Split questions and thresholds |
| Memory-based | kNN | Nothing; it stores the examples |
| Probability counting | Naive Bayes | Probability tables |
| Rules and points | CN2 Rule Induction, Scoring Sheet | Rules and points |

**Linear Regression is a single neuron** with no activation, trained on MSE loss. Add a sigmoid activation and you get Logistic Regression. Add hidden layers with non-linear activations and you get a Neural Network.

### 1.8 Test and Score: grading, not training
Test and Score runs **5-fold cross-validation**. It splits the data into 5 parts, trains 5 **throwaway** models (each tested on the part it didn't see), and averages the scores. It tells you **how well this kind of model will do on unseen data**, which is how you compare models and tune settings without needing new labelled data.

The model you actually use is trained by the **Random Forest widget** on **all** the data and sent to Predictions through its *Model* output.

### 1.9 Reading the metrics
| Metric | Full name | Meaning |
|---|---|---|
| **MAE** | Mean Absolute Error | Average miss, in rating points. **The easiest to explain.** |
| MSE | Mean Squared Error | Average squared miss; big misses count much more |
| RMSE | Root Mean Squared Error | √MSE, back in rating points |
| MAPE | Mean Absolute Percentage Error | Average miss as a percentage of the real value |
| sMAPE | Symmetric MAPE | Same idea, but over- and under-prediction are scored evenly |
| **R2** | R-squared (coefficient of determination) | Share of variation explained: 0 = no better than the average, 1 = perfect |

Worked example (predicting 8.3 for every movie):

| Real | Error | Error² |
|---|---|---|
| 9.3 | 1.0 | 1.00 |
| 9.1 | 0.8 | 0.64 |
| 8.2 | 0.1 | 0.01 |
| 8.1 | 0.2 | 0.04 |

MAE = 2.1 / 4 = **0.525** · MSE = 1.69 / 4 = **0.4225** · RMSE = √0.4225 = **0.65**

RMSE comes out above MAE because a few large misses dominate it.

### 1.10 Normalization: needed by some models, not by Random Forest
Orange's default is **Standardize to μ=0, σ²=1**: `(value − mean) / standard deviation`.

Worked example, durations 90, 100, 110, 120, 130:
- Mean μ = 550 / 5 = **110**
- Deviations −20, −10, 0, +10, +20 → squared 400, 100, 0, 100, 400 → sum **1000**
- Variance σ² = 1000 / 5 = **200** → standard deviation σ = √200 ≈ **14.14**
- Standardized: −1.41, −0.71, 0, +0.71, +1.41

**Neural Networks, kNN and SVM need this. Trees don't.** Connect Preprocess to a model's **Preprocessor** input so the scaling is learned only from training data and reapplied to new data.

---

## Part 2: Build it in Orange

### The final canvas
```
CSV File Import ─→ Select Columns ──Data──────────────→ Test and Score
                         │                                  ↑ Learner
                         └──Data──→ Random Forest ──────────┘
                                         │ Model
                                         ↓
CSV File Import (1) ─────Data──────→ Predictions ──→ Scatter Plot
                                         └──→ Formula ──→ Bar Plot
```

**Rule for every link:** data flows **left to right**. Drag from a widget's **right** edge (its output) to the next widget's **left** edge (its input). A **dashed** line means the link isn't carrying anything yet.

### Step 1: Load the training data
1. **Data** panel → drag **CSV File Import** onto the canvas.
2. Double-click it and open `data/movies_train_demo.csv` (4,996 movies).
3. Check that the five features and `imdb_score` are **Numeric**.

### Step 2: Choose the columns with Select Columns
CSV File Import can't mark a target column, so we need this widget.

1. **Transform** → **Select Columns**. Connect **CSV File Import → Select Columns**.
2. Drag columns into the boxes:
   - **Features:** `duration`, `budget`, `gross`, `num_voted_users`, `num_user_for_reviews`
   - **Target:** `imdb_score`
   - **Metas:** `movie_title`
3. Check the status bar: the output should read **4996 | 5**.

**Alternative:** the plain **File** widget has a *Role* column, so you can set the target there and skip Select Columns.

### Step 3: Add Random Forest
1. **Model** → **Random Forest**.
2. Connect **Select Columns → Random Forest** (Data). Without this link, it has nothing to train on.
3. Double-click it and set **Number of trees** to **100** (the default is 10).

### Step 4: Grade it with Test and Score
1. **Evaluate** → **Test and Score**.
2. Connect **Select Columns → Test and Score** (Data) and **Random Forest → Test and Score** (Learner).
3. Choose **Cross validation, 5 folds**, and untick **Stratified** (it only applies to classification).
4. Optional: add **Constant** as a baseline. Any useful model must beat it.

### Step 5: Predict the new movies
1. Add a second **CSV File Import** with `data/new_movies_2019_2024.csv`.
2. **Evaluate** → **Predictions**.
3. Connect **Random Forest → Predictions** (*Model → Predictors*) and **CSV File Import (1) → Predictions** (*Data*).
4. Predictions has two inputs and nothing going out (until the charts in Step 6).

Orange matches columns **by name**, so the new file must use the same column names as the training data.

### Step 6: Chart the errors
**Actual vs predicted:** **Visualize** → **Scatter Plot**. Connect **Predictions → Scatter Plot**. Set x = `imdb_score`, y = `Random Forest`, and Label = `movie_title`. Dots above the diagonal were over-predicted.

**Error per movie:**
1. **Transform** → **Formula** (it used to be called *Feature Constructor*). Connect **Predictions → Formula**.
2. **New → Numeric**, Name `error`, Expression `Random_Forest - imdb_score`. Insert the column through the *Select Feature* dropdown so its name is exact.
3. **Visualize** → **Bar Plot**. Connect **Formula → Bar Plot**. Set Values = `error` and Annotations = `movie_title`.

### Step 7: Look inside the forest (optional)
- **Random Forest → Pythagorean Forest** (Visualize) shows every tree. Each square is a split, its size is the number of movies, and each tree has a different shape.
- Select a tree, then connect **Pythagorean Forest → Tree Viewer** and set its depth to 3 to read the actual questions.

### Mistakes we hit, and the fixes
| Symptom | Cause | Fix |
|---|---|---|
| Red **!**: "Data has no target variable" | CSV File Import can't set a target | Add Select Columns, or use the File widget |
| Link label reads *Predictions → Data* or *Selected Predictions → …* | The link was dragged backwards | Delete it (right-click → Remove) and drag from the **right** edge of the source |
| Dashed *Model → Predictors* line | The model widget has no training data | Connect Select Columns → model (Data) |
| Test and Score table is empty | No learner is reaching it | Check the model widget is applied and linked through *Learner* |
| Create Instance has no feature fields | The wanted columns were in **Ignored** and the junk in **Features** | Fix the boxes in Select Columns |
| Can't find Feature Constructor | Renamed in newer Orange | Use **Formula** |
| White **!** on Test and Score | "Stratification is ignored for regression" | Harmless; untick Stratified |

---

## Part 3: Things we learned along the way

### The training data limits what the model can predict
We first trained on the **IMDb Top 250**, where every movie is rated 8.0–9.3. The model **never predicted below about 8.0**, even for made-up "bad" movies, because it had never seen one. And within that list, **fewer votes meant a higher rating** (a little-known film only makes the Top 250 if its rating is very high), which is the opposite of the general pattern.

**Lesson:** a model only knows the world in its training data. Switching to the 5,000-movie dataset (ratings 1.6–9.5) fixed this.

### Outliers and mixed units break some models
The IMDb 5000 `budget` column mixes currencies. *The Host* is listed at **12,215,500,000** (Korean won), and *Princess Mononoke* at 2.4 billion (yen).
- **Linear Regression** multiplies a weight by every value, so a few huge values distort it.
- **Random Forest** only asks "is the budget above $50M?", so a 12-billion value just falls on the "yes" side.
- A **log transform** (`log(budget)`) compresses huge values and helps weighted-sum models.

### Transform new data the same way as training data
Any log or normalization step must be applied to new data **using the training data's parameters**. Orange does this automatically when the transform is built into the workflow (Formula columns remember their formula; Preprocess is connected as a *Preprocessor*). In Python, a scikit-learn `Pipeline` does the same.

---

## Part 4: Results

### Cross-validation on the training data (Python, 100 trees)
| Model | MAE | R2 |
|---|---|---|
| Constant (always predicts the average) | 0.87 | 0.00 |
| **Random Forest** | **0.63** | **0.42** |

### First 6 test movies: three models compared (errors in Orange, predicted − real)
| Movie | Real | Random Forest | Linear Regression | Neural Network |
|---|---|---|---|---|
| Joker | 8.3 | +0.69 | +0.53 | +0.58 |
| Oppenheimer | 8.2 | +0.38 | +0.97 | −0.43 |
| Barbie | 6.8 | +1.08 | +0.42 | +0.98 |
| Morbius | 5.1 | +0.54 | +0.96 | +1.53 |
| Madame Web | 4.1 | +1.04 | +2.13 | +2.69 |
| Cats | 2.8 | +1.81 | +2.89 | +3.38 |
| **MAE** | | **0.92** | 1.32 | 1.60 |

The Neural Network column comes from a run **before** the log transform, while it was still trained on the raw columns.

### All 30 test movies with Random Forest (from the Python script)
- **MAE: 0.70 rating points.**
- **23 of 30 over-predicted.** The 7 under-predictions were small (at most 0.35).
- Biggest misses: Fifty Shades Freed (+1.51), Madame Web (+1.43), Cats (+1.41), The Emoji Movie (+1.19), Venom (+1.18).
- Closest: Top Gun: Maverick (−0.04), Spider-Verse (−0.13), Dune: Part Two (−0.15), Borderlands (−0.16).
- **Feature importance:** `num_voted_users` 0.38, `duration` 0.20, `budget` 0.16, `num_user_for_reviews` 0.13, `gross` 0.13.

### What the results mean
1. **It separates good films from bad ones well.** Acclaimed films are predicted high and flops low.
2. **It mostly over-predicts: this is data drift.** The training data ends in 2016. Online voting has grown since then, so a recent movie with lots of votes looks better to the model than it really is.
3. **It softens extremes.** Averaging pulls the worst films toward the middle.
4. **Votes dominate.** The model leans on popularity, which is partly an effect of quality rather than a cause of it.

Data sources for the test movies: ratings, votes, review counts and runtimes are from IMDb (fetched 2026-10-28); domestic gross is from Box Office Mojo. Budgets are from Box Office Mojo where listed; the rest are commonly reported figures that weren't checked against a source, marked `reported` in the `budget_source` column. Box Office Mojo's $465M budget for *Jurassic World: Dominion* looks like an error, so we used the widely reported $265M.

---

## Part 5: The same workflow in Python
`python/random_forest_imdb.py` repeats every Orange step in scikit-learn:

| Orange widget | Python |
|---|---|
| CSV File Import | `pd.read_csv` |
| Select Columns | `X = train[FEATURES]`, `y = train[TARGET]` |
| Random Forest | `RandomForestRegressor(n_estimators=100, min_samples_split=5)` |
| (built-in missing-value handling) | `SimpleImputer(strategy="median")` in a `Pipeline` |
| Test and Score | `cross_validate(..., cv=KFold(5))` |
| Predictions | `forest.fit(X, y)` then `forest.predict(new[FEATURES])` |
| Scatter Plot / Bar Plot | matplotlib, saved to `python/output/random_forest_results.png` |

Run it from the lab root:
```sh
uv run lessons/01-random-forest-imdb/python/random_forest_imdb.py
```

---

## Part 6: Exercises for next time
1. **Drop `num_voted_users`** in Select Columns. How much worse does MAE get, and does the over-prediction shrink?
2. **Try 10, 100 and 500 trees.** When does adding trees stop helping?
3. **Add Gradient Boosting** next to Random Forest in Test and Score. Which wins?
4. **Switch to classification:** use Formula to create `hit = imdb_score >= 7` as a categorical target, then try Logistic Regression and Random Forest and compare their AUC.
5. **Retrain with recent movies included** and see whether the data drift goes away.

## Glossary
| Term | Meaning |
|---|---|
| Feature | An input column the model learns from |
| Target / label | The value being predicted |
| Regression / classification | Predicting a number / predicting a category |
| Split / node / leaf | A yes/no question / a point in the tree / a final group that gives the prediction |
| Bootstrap | Sampling with replacement to give each tree different data |
| Cross-validation | Repeated train/test splits to estimate performance on unseen data |
| Overfitting | Memorising the training data instead of learning the general pattern |
| Data drift | The world changing after the training data was collected |
| Standardize | (value − mean) / standard deviation |
