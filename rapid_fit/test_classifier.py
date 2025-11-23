"""
Unit tests for the ingredient classifier.
"""
import pytest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def load_model_and_data():
    """Load the trained model and data."""
    # Load training data
    train_df = pd.read_csv('data/train.csv')
    
    # Prepare data
    X_train = train_df['text'].values
    y_train = train_df['label'].values
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
        ('clf', MultinomialNB())
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


@pytest.fixture
def model():
    """Fixture to load the trained model."""
    return load_model_and_data()


def test_ingredient_only(model):
    """Test that simple ingredients are classified correctly."""
    test_cases = ["Tomato", "Onion", "Garlic", "Salt", "Cumin"]
    predictions = model.predict(test_cases)
    
    for text, pred in zip(test_cases, predictions):
        assert pred == 'ingredient_only', f"'{text}' should be ingredient_only, got {pred}"


def test_ingredient_with_qty(model):
    """Test that ingredients with quantities are classified correctly."""
    test_cases = ["Milk 200 ml", "Sugar 20 g", "Rice 150 g", "Butter 50 g"]
    predictions = model.predict(test_cases)
    
    for text, pred in zip(test_cases, predictions):
        assert pred == 'ingredient_with_qty', f"'{text}' should be ingredient_with_qty, got {pred}"


def test_instruction_like(model):
    """Test that instruction-like phrases are classified correctly."""
    test_cases = ["Chop the onions", "Stir for 2 minutes", "Warm in a pan"]
    predictions = model.predict(test_cases)
    
    for text, pred in zip(test_cases, predictions):
        assert pred == 'instruction_like', f"'{text}' should be instruction_like, got {pred}"


def test_non_food(model):
    """Test that non-food items are classified correctly."""
    test_cases = ["Plastic wrap", "Baking paper", "Paper towel", "Aluminum foil"]
    predictions = model.predict(test_cases)
    
    for text, pred in zip(test_cases, predictions):
        assert pred == 'non_food', f"'{text}' should be non_food, got {pred}"


def test_edge_cases(model):
    """Test edge cases like empty strings and very short inputs."""
    # Test with single word that could be ambiguous
    result = model.predict(["Eggs"])
    assert result[0] in ['ingredient_only', 'ingredient_with_qty'], "Should handle single word ingredients"
    
    # Test with numbers but no units
    result = model.predict(["Eggs 2"])
    assert result[0] == 'ingredient_with_qty', "Should recognize quantity even without units"

