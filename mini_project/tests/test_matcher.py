"""
Unit tests for ingredient matcher.
"""
import pytest
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.matcher import normalize_text, find_best_match, load_ingredients


@pytest.fixture
def sample_ingredients():
    """Create sample ingredients dataframe."""
    data = {
        'ingredient_id': [1, 2, 3, 4, 5],
        'name': ['Tomato', 'Onion', 'Garlic', 'Whole Milk', 'Olive Oil']
    }
    return pd.DataFrame(data)


def test_normalize_text():
    """Test text normalization."""
    assert normalize_text("TOMATOES 1kg pack") == "tomatoes"
    assert normalize_text("onion red 500g") == "onion"
    assert normalize_text("extra virgin olive oil") == "olive oil"
    assert normalize_text("") == ""
    assert normalize_text("  white  sugar  ") == "sugar"


def test_find_best_match_exact(sample_ingredients):
    """Test exact matching."""
    ingredient_id, confidence = find_best_match("Tomato", sample_ingredients)
    assert ingredient_id == 1
    assert confidence == 1.0


def test_find_best_match_fuzzy(sample_ingredients):
    """Test fuzzy matching."""
    # Should match "Tomato" even with variations
    ingredient_id, confidence = find_best_match("TOMATOES", sample_ingredients)
    assert ingredient_id == 1
    assert confidence > 0.7
    
    # Should match "Olive Oil" even with extra words
    ingredient_id, confidence = find_best_match("extra virgin olive oil", sample_ingredients)
    assert ingredient_id == 5
    assert confidence > 0.6


def test_find_best_match_with_quantity(sample_ingredients):
    """Test matching with quantities."""
    ingredient_id, confidence = find_best_match("Tomato 1kg pack", sample_ingredients)
    assert ingredient_id == 1
    assert confidence > 0.7


def test_find_best_match_typo(sample_ingredients):
    """Test matching with typos."""
    # "gralic" should match "Garlic"
    ingredient_id, confidence = find_best_match("gralic", sample_ingredients)
    assert ingredient_id == 3
    assert confidence > 0.6


def test_find_best_match_no_match(sample_ingredients):
    """Test when no match is found."""
    ingredient_id, confidence = find_best_match("Random Item XYZ", sample_ingredients, threshold=80)
    assert ingredient_id is None
    assert confidence == 0.0


def test_find_best_match_empty(sample_ingredients):
    """Test with empty input."""
    ingredient_id, confidence = find_best_match("", sample_ingredients)
    assert ingredient_id is None
    assert confidence == 0.0


def test_load_ingredients():
    """Test loading ingredients from file."""
    df = load_ingredients('data/ingredients_master.csv')
    assert len(df) > 0
    assert 'ingredient_id' in df.columns
    assert 'name' in df.columns

