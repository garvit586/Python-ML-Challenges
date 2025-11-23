# Rapid Fit — Ingredient Line Classifier (≈90 minutes)

Build a small classifier that tags short **ingredient lines** into one of these classes:
- `ingredient_only` — e.g., "Tomato"
- `ingredient_with_qty` — e.g., "Milk 200 ml"
- `instruction_like` — e.g., "Chop the onions"
- `non_food` — e.g., "Plastic wrap"

### Data
- `train.csv` and `test.csv` in `data/` with columns: `text`, `label` (label only in train).

### Requirements
- Provide a Python script or notebook that trains a simple model and predicts labels for `test.csv`.
- Output a `predictions.csv` with columns: `text`, `pred`.
- Include **at least 3 unit tests** (e.g., expected behavior on edge cases).
- Aim to finish in ≈90 minutes; don't over-engineer.

### Scoring
- Macro F1 on the test set (we'll hold out some extra lines).
- Code quality, structure, and clarity of `DECISIONS.md`.
