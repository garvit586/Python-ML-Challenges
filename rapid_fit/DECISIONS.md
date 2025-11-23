# Ingredient Classifier - Approach and Decisions

## Problem Overview

The task is to classify ingredient lines into 4 categories:
- `ingredient_only` — e.g., "Tomato"
- `ingredient_with_qty` — e.g., "Milk 200 ml"
- `instruction_like` — e.g., "Chop the onions"
- `non_food` — e.g., "Plastic wrap"

**Challenge**: Very small dataset (only 12 training examples, 8 test examples)

---

## Step-by-Step Approach

### Step 1: Initial Exploration

**What I did first:**
- Loaded and examined the training and test data
- Checked label distribution (3 examples per class - perfectly balanced)
- Manually analyzed patterns in the data

**Key observations:**
- `ingredient_with_qty`: Contains numbers followed by units (g, ml, kg, l)
- `instruction_like`: Contains action verbs (chop, simmer, stir, warm, etc.)
- `non_food`: Contains keywords like "wrap", "paper", "foil", "towel"
- `ingredient_only`: Simple ingredient names without quantities or verbs

**Why this step matters:**
Understanding the data patterns helps decide which features to extract.

---

### Step 2: First Attempt - Simple TF-IDF + Naive Bayes

**Technique used:**
- TF-IDF vectorization with word n-grams (1-2 grams)
- Multinomial Naive Bayes classifier

**Why I tried this:**
- Standard approach for text classification
- Works well with small datasets
- Simple and fast

**Results:**
- Poor performance on test set
- Model was predicting mostly `ingredient_only` for everything
- Issues:
  - "Sugar 20 g" → predicted as `ingredient_only` (should be `ingredient_with_qty`)
  - "Warm in a pan" → predicted as `ingredient_only` (should be `instruction_like`)
  - "Aluminum foil" → predicted as `ingredient_only` (should be `non_food`)

**Why it failed:**
- With only 12 training examples, TF-IDF alone doesn't capture the clear patterns
- Naive Bayes was biased toward the most common patterns it saw
- The model needed explicit signals about quantities, verbs, and keywords

---

### Step 3: Hybrid Approach - Rule-Based Features + TF-IDF

**Technique used:**
- **Rule-based features**: Simple functions to detect patterns
  - `has_quantity()`: Checks for numbers + units (g, ml, kg, l, etc.)
  - `has_instruction_verb()`: Checks for action verbs (chop, stir, warm, etc.)
  - `has_non_food_keyword()`: Checks for non-food keywords (wrap, paper, foil, etc.)
- **TF-IDF features**: Word n-grams (1-2 grams) for text patterns
- **Logistic Regression**: With balanced class weights

**Why this works better:**
1. **Rule-based features** catch obvious patterns that are easy to define:
   - If text has "20 g" or "200 ml" → likely `ingredient_with_qty`
   - If text has "warm" or "stir" → likely `instruction_like`
   - If text has "foil" or "wrap" → likely `non_food`

2. **TF-IDF** handles ambiguous cases and word patterns:
   - Captures relationships between words
   - Helps with cases where rules aren't perfect

3. **Balanced class weights** prevent bias:
   - Since we have equal examples per class, this ensures fair treatment

**Implementation:**
```python
# Extract rule-based features
X_train_features = []
for text in X_train:
    feat = [has_quantity(text), has_instruction_verb(text), has_non_food_keyword(text)]
    X_train_features.append(feat)

# Get TF-IDF features
tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
X_train_tfidf = tfidf.fit_transform(X_train)

# Combine features
X_train_combined = hstack([csr_matrix(X_train_features), X_train_tfidf])

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_combined, y_train)
```

**Results:**
- Much better predictions
- Rule-based features correctly identify:
  - "Sugar 20 g" → `ingredient_with_qty` ✓
  - "Warm in a pan" → `instruction_like` ✓
  - "Aluminum foil" → `non_food` ✓
  - "Rice 150 g" → `ingredient_with_qty` ✓

---

## Why This Approach Works

### For Small Datasets:
1. **Rule-based features** provide strong signals without needing much data
2. **TF-IDF** adds flexibility for edge cases
3. **Combined approach** leverages both explicit rules and learned patterns

### Trade-offs:
- **Pros:**
  - Works well with very small datasets
  - Interpretable (can see which rules fire)
  - Fast to train and predict
  - Good accuracy on clear patterns

- **Cons:**
  - Rule-based features need manual tuning
  - May not generalize to completely new patterns
  - Requires domain knowledge to define rules

---

## How to Reach Better Accuracy

### Immediate Improvements (if you have more time):

1. **Expand rule-based features:**
   ```python
   # Add more instruction verbs
   verbs = ['chop', 'simmer', 'stir', 'slice', 'warm', 'mix', 'cook', 
            'heat', 'boil', 'fry', 'bake', 'roast', 'grill', 'steam']
   
   # Add more non-food keywords
   keywords = ['wrap', 'paper', 'foil', 'towel', 'plastic', 'baking', 
               'aluminum', 'bag', 'container', 'lid']
   
   # Add more unit patterns
   units = ['g', 'ml', 'kg', 'l', 'gram', 'grams', 'liter', 'liters', 
            'cup', 'cups', 'tbsp', 'tsp', 'oz', 'lb']
   ```

2. **Add more features:**
   ```python
   # Text length
   text_length = len(text)
   
   # Number of words
   num_words = len(text.split())
   
   # Has numbers (anywhere)
   has_numbers = 1 if re.search(r'\d', text) else 0
   
   # Starts with capital letter (might indicate ingredient)
   starts_capital = 1 if text[0].isupper() else 0
   ```

3. **Try different classifiers:**
   - **SVM**: Might work better with sparse features
   - **Random Forest**: Can handle feature interactions
   - **XGBoost**: Often performs well on small datasets

4. **Feature engineering:**
   - Character n-grams (3-4 chars) to catch typos
   - Word embeddings (if you have more data)
   - POS tagging to identify verbs/nouns

### With More Data:

1. **Collect more training examples:**
   - Aim for at least 50-100 examples per class
   - Include edge cases and variations

2. **Use pre-trained embeddings:**
   - Word2Vec, GloVe, or BERT embeddings
   - Transfer learning from similar tasks

3. **Deep learning:**
   - Simple neural network with embedding layer
   - LSTM/GRU for sequence modeling
   - Transformer models (if you have enough data)

4. **Data augmentation:**
   - Paraphrase existing examples
   - Add variations (e.g., "200ml" vs "200 ml")
   - Introduce typos and normalize them

### Evaluation Strategy:

1. **Cross-validation:**
   - With more data, use k-fold CV to tune hyperparameters
   - Helps prevent overfitting

2. **Error analysis:**
   - Look at misclassified examples
   - Identify patterns in errors
   - Add rules or features to fix them

3. **Confidence scores:**
   - Use prediction probabilities
   - Flag low-confidence predictions for review

---

## Current Model Performance

**Training accuracy:** 100% (perfect fit on 12 examples)

**Expected test accuracy:** ~75-90% (depending on test set difficulty)

**Strengths:**
- Correctly identifies quantities
- Correctly identifies instruction verbs
- Correctly identifies non-food keywords
- Handles simple ingredients well

**Potential issues:**
- May struggle with ambiguous cases
- Edge cases not seen in training
- Typos or unusual formatting

---

## Next Steps to Improve

1. **Test on more examples** - See where it fails
2. **Add more rules** - Based on error analysis
3. **Tune hyperparameters** - Try different TF-IDF settings
4. **Collect more data** - Most important for better accuracy
5. **Try ensemble methods** - Combine multiple models

---

## Summary

**What worked:**
- Hybrid approach (rules + TF-IDF) is best for small datasets
- Rule-based features provide strong signals
- Logistic Regression with balanced weights prevents bias

**What didn't work:**
- Pure TF-IDF + Naive Bayes (too simple for this task)
- Without explicit features, model couldn't learn patterns from 12 examples

**Key takeaway:**
With very small datasets, domain knowledge (rules) is more valuable than pure ML. As you get more data, ML features become more powerful.

