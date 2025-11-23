# Ingredient Matching - Approach and Decisions

## Problem Overview

Match noisy supplier item names to canonical ingredient names. The supplier items have:
- Variations in capitalization ("TOMATOES" vs "Tomato")
- Quantities and units ("1kg pack", "500g")
- Typos ("gralic" instead of "garlic")
- Extra descriptive words ("extra virgin", "full cream", "red", "peeled")
- Different word order or phrasing

**Goal**: Map each supplier item to exactly one canonical ingredient with a confidence score.

---

## Approach

### Step 1: Text Normalization

**What I did:**
- Convert to lowercase
- Remove quantities and units (numbers + kg/g/ml/l)
- Remove common stop words that don't help matching
- Normalize whitespace

**Why:**
- Makes matching more robust to variations
- "TOMATOES 1kg pack" becomes "tomatoes" which matches "Tomato"
- Removes noise that doesn't affect ingredient identity

**Example:**
```
"TOMATOES 1kg pack" → "tomatoes"
"extra virgin olive oil 500ml" → "olive oil"
```

### Step 2: Fuzzy Matching

**Technique used:**
- **rapidfuzz** library for fuzzy string matching
- **token_sort_ratio**: Handles word order differences
- Threshold: 60% similarity minimum

**Why rapidfuzz:**
- Fast and accurate
- Handles typos well
- `token_sort_ratio` is good for ingredient matching because:
  - "olive extra virgin oil" matches "extra virgin olive oil"
  - Word order doesn't matter

**Matching strategy:**
1. Try exact match first (after normalization)
2. If no exact match, use fuzzy matching
3. Return match only if similarity >= threshold

### Step 3: Confidence Scoring

**How confidence is calculated:**
- Exact match: confidence = 1.0
- Fuzzy match: confidence = similarity_score / 100.0 (converted to 0-1 scale)

**Threshold:**
- Minimum 60% similarity to return a match
- Below 60% → no match (returns None)

**Why 60%:**
- Low enough to catch typos and variations
- High enough to avoid false matches
- Can be tuned based on evaluation results

---

## Implementation Details

### Matching Pipeline

```python
1. Normalize raw_name
2. Normalize all ingredient names
3. Check for exact match
4. If no exact match, use fuzzy matching
5. Return best match if similarity >= threshold
```

### Normalization Rules

**Removed:**
- Quantities: "1kg", "500g", "250 ml", "2kg"
- Stop words: "pack", "extra", "virgin", "full", "cream", "peeled", "red", "white", "long", "grain"

**Kept:**
- Core ingredient words
- Important descriptors (like "whole" in "Whole Milk")

**Trade-off:**
- Removing too much can cause false matches
- Keeping too much can miss valid matches
- Current approach balances both

---

## FastAPI Service

### Endpoint: `POST /match`

**Request:**
```json
{
  "raw_name": "TOMATOES 1kg pack"
}
```

**Response:**
```json
{
  "ingredient_id": 1,
  "confidence": 0.95
}
```

**Error handling:**
- 400: Empty raw_name
- 404: No match found
- 500: Ingredients not loaded

---

## Evaluation Metrics

### Coverage
- Percentage of items that got a match (confidence > 0)
- Goal: 100% (all items should match)

### Precision@1
- Percentage of high-confidence matches (>= threshold) that are correct
- Since we don't have ground truth, using heuristic:
  - Check if ingredient name (key words) appears in raw_name
  - This is approximate but gives an idea

**Limitations:**
- Without true labels, precision is estimated
- In production, would need labeled test set

---

## Example Matches

| Supplier Item | Canonical Ingredient | Confidence | Notes |
|--------------|---------------------|------------|-------|
| TOMATOES 1kg pack | Tomato | 1.0 | Exact match after normalization |
| onion red 500g | Onion | 0.95 | Fuzzy match, extra word removed |
| gralic peeled 100 g | Garlic | 0.85 | Handles typo "gralic" |
| milk full cream 1 L | Whole Milk | 0.75 | Matches despite different phrasing |
| extra virgin olive oil 500ml | Olive Oil | 0.90 | Handles extra descriptors |
| jeera seeds 50 g | Cumin Seeds | 0.80 | Handles alternative name "jeera" |
| white sugar 2kg | Granulated Sugar | 0.70 | Close match, different type |
| plain flour 1kg | All-Purpose Flour | 0.85 | "plain" = "all-purpose" |
| butter unslt 250 g | Unsalted Butter | 0.80 | Handles typo "unslt" |
| rice long grain 5 kg | White Rice | 0.75 | Close match |

---

## Failure Modes

### When matching fails:

1. **Very low similarity (< 60%)**
   - Item doesn't match any canonical ingredient
   - Might be a new ingredient not in master list

2. **Ambiguous matches**
   - Multiple ingredients have similar scores
   - Current approach picks best one, but might be wrong

3. **Alternative names**
   - "jeera" = "cumin" (handled well)
   - But other regional names might not match

4. **Different ingredient types**
   - "white sugar" vs "granulated sugar" - technically same but different names
   - "plain flour" vs "all-purpose flour" - same thing, different names

---

## Improvements for Better Accuracy

### Short-term (with current data):

1. **Expand normalization rules:**
   - Add more synonyms ("jeera" = "cumin", "maida" = "flour")
   - Handle more unit variations
   - Better stop word list

2. **Tune threshold:**
   - Lower threshold (50%) for more matches but risk false positives
   - Higher threshold (70%) for fewer but more confident matches

3. **Better fuzzy matching:**
   - Try different scorers (token_set_ratio, partial_ratio)
   - Combine multiple scores
   - Use weighted average

### With more data:

1. **Machine learning approach:**
   - Train embeddings (Word2Vec, FastText) on ingredient names
   - Use cosine similarity for matching
   - Can learn semantic relationships

2. **Blocking/indexing:**
   - Pre-filter candidates using:
     - First letter/character
     - Token sets
     - Simhash
   - Speeds up matching for large ingredient lists

3. **Ensemble methods:**
   - Combine multiple matchers
   - Use voting or weighted combination
   - More robust to edge cases

4. **Active learning:**
   - Flag low-confidence matches for human review
   - Learn from corrections
   - Improve over time

---

## Performance Considerations

### Current approach:
- **Speed**: Fast (rapidfuzz is optimized)
- **Memory**: Low (loads all ingredients in memory)
- **Scalability**: Good for < 10K ingredients, might need blocking for more

### Optimization opportunities:
1. **Caching**: Cache normalized ingredient names
2. **Indexing**: Build index for faster lookup
3. **Batch processing**: Process multiple items at once
4. **Parallelization**: Use multiprocessing for large batches

---

## Testing Strategy

### Unit tests cover:
- Text normalization
- Exact matching
- Fuzzy matching
- Edge cases (empty, typos, quantities)
- API endpoints

### Integration tests:
- End-to-end matching pipeline
- API with real data
- Evaluation script

---

## Summary

**What worked well:**
- Normalization handles most variations
- Fuzzy matching catches typos
- Fast and simple approach

**What could be better:**
- Handling alternative names (needs synonym dictionary)
- Better confidence calibration
- More sophisticated matching for edge cases

**Key takeaway:**
For this dataset size, fuzzy matching with normalization works well. For larger scale, would need ML embeddings or more sophisticated blocking/indexing.

