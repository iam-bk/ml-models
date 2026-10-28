# ML Models Training Lab

A hands-on learning lab: each lesson teaches one ML model on a real dataset, first built visually in **Orange Data Mining**, then rebuilt in **Python** to show what Orange does behind the scenes. The audience is learning, so lessons explain concepts plainly and show every click.

## Layout
- `data/`: datasets shared by all lessons. Lessons reference them by relative path; don't duplicate data inside lesson folders.
- `lessons/NN-<model>-<dataset>/`: one folder per lesson, numbered in order.
  - `README.md`: the lesson.
  - `python/`: the code version of the Orange workflow. Generated charts go in `python/output/`.
- `README.md`: the lab overview and the **lesson index table**. Add a row for every new lesson.

## Lesson structure
Follow `lessons/01-random-forest-imdb/README.md`:
1. A short first-person task description at the top ("I am going to predict…").
2. **Core ideas:** how the model works, with small worked numeric examples.
3. **Build it in Orange:** every widget, setting and link in order, plus a canvas diagram and a table of mistakes and fixes.
4. **Results:** real numbers from an actual run, never estimates.
5. **The same workflow in Python:** a table mapping each Orange widget to its code equivalent.
6. **Exercises** and a **glossary**.

## Python conventions
- Scripts are standalone `uv` scripts with PEP 723 inline dependencies (`# /// script`). Run them from the repo root with `uv run lessons/<lesson>/python/<script>.py`.
- Use scikit-learn `Pipeline`s, so preprocessing is learned from training data only and reapplied to new data.
- Set `random_state=0` everywhere so results are reproducible.
- Mirror Orange's defaults where it matters (for example `min_samples_split=5` for trees), and comment which Orange widget each block corresponds to.

## Accuracy rules
- Run the code before quoting any number in a lesson. If Orange and Python disagree, say so; don't hide it.
- Cite where data came from and when it was fetched. Mark any value that wasn't checked against a source.
- When explaining Orange links, always state the direction: from the source widget's **right** edge to the target's **left** edge, with the expected link label.

## Commits
Concise subject line, `[lesson-NN] Sentence case description` or `[lab] …` for shared changes. Add a body only when the reason isn't obvious from the diff.
