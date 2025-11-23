# Mini Project — Supplier Item -> Canonical Ingredient Matching (≤4 hours)

Given a noisy supplier item list, map each row to a **canonical ingredient** from our master list.

### Files
- `data/ingredients_master.csv`: canonical list with `ingredient_id`, `name`.
- `data/supplier_items.csv`: noisy items with `item_id`, `raw_name`.
- You must produce `matches.csv` with `item_id`, `ingredient_id`, `confidence` in [0,1].

### Functional Requirements
1. Design a text‑matching pipeline (normalization + tokenization + embeddings or fuzzy).
2. Return **exactly one** best match per supplier item with a confidence score.
3. Build a tiny **FastAPI** service with one route:
   - `POST /match` that accepts a JSON body with `raw_name` and returns a JSON response containing keys `ingredient_id` and `confidence`.
4. Include a small evaluation script reporting precision@1 and coverage.

### Non‑Functional
- Unit tests (`pytest`).
- A `Dockerfile` to run the FastAPI app.
- `DECISIONS.md` describing the matching approach, thresholds, and failure modes.

### Bonus (optional)
- Handle stop‑words, misspellings, and size/pack info (e.g., "1kg", "500 ml").
- Add simple blocking to speed up matching (prefix/simhash/token‑set).

We'll run: `docker build .` then `docker run -p 8000:8000 app` and call `/match`.
