"""
Fuzzy matching module for supplier items to canonical ingredients.
"""
import pandas as pd
import re
from rapidfuzz import fuzz, process


def normalize_text(text):
    """Normalize text for better matching."""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common units and quantities (for matching purposes)
    # But keep them for context
    text = re.sub(r'\d+\s*(kg|g|ml|l|gram|grams|liter|liters|pack|packs)', '', text)
    text = re.sub(r'\d+', '', text)  # Remove remaining numbers
    
    # Remove common words that don't help matching
    stop_words = ['pack', 'packs', 'extra', 'virgin', 'full', 'cream', 'peeled', 
                  'long', 'grain', 'red', 'white', 'unslt', 'unsalted']
    words = text.split()
    words = [w for w in words if w not in stop_words]
    text = ' '.join(words)
    
    # Clean up again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_ingredients(filepath='data/ingredients_master.csv'):
    """Load canonical ingredients."""
    df = pd.read_csv(filepath)
    return df


def find_best_match(raw_name, ingredients_df, threshold=60):
    """
    Find best matching ingredient for a raw name.
    
    Returns:
        tuple: (ingredient_id, confidence_score)
    """
    if not raw_name or pd.isna(raw_name):
        return None, 0.0
    
    # Normalize the raw name
    normalized_raw = normalize_text(raw_name)
    
    # Get all ingredient names
    ingredient_names = ingredients_df['name'].tolist()
    
    # Try exact match first (after normalization)
    normalized_ingredients = [normalize_text(name) for name in ingredient_names]
    
    # Check for exact match
    if normalized_raw in normalized_ingredients:
        idx = normalized_ingredients.index(normalized_raw)
        return ingredients_df.iloc[idx]['ingredient_id'], 1.0
    
    # Use fuzzy matching
    # Try token_sort_ratio (handles word order differences)
    best_match = process.extractOne(
        normalized_raw,
        normalized_ingredients,
        scorer=fuzz.token_sort_ratio
    )
    
    if best_match and best_match[1] >= threshold:
        # Find the original ingredient
        matched_name = best_match[0]
        idx = normalized_ingredients.index(matched_name)
        ingredient_id = ingredients_df.iloc[idx]['ingredient_id']
        confidence = best_match[1] / 100.0  # Convert to 0-1 scale
        return ingredient_id, confidence
    
    # If no good match found, return None
    return None, 0.0


def match_all_items(supplier_file='data/supplier_items.csv', 
                   ingredients_file='data/ingredients_master.csv',
                   output_file='matches.csv',
                   threshold=60):
    """
    Match all supplier items to canonical ingredients.
    
    Args:
        supplier_file: Path to supplier items CSV
        ingredients_file: Path to ingredients master CSV
        output_file: Path to output matches CSV
        threshold: Minimum similarity threshold (0-100)
    """
    # Load data
    supplier_df = pd.read_csv(supplier_file)
    ingredients_df = pd.read_csv(ingredients_file)
    
    print(f"Loaded {len(supplier_df)} supplier items")
    print(f"Loaded {len(ingredients_df)} canonical ingredients")
    
    # Match each item
    matches = []
    for _, row in supplier_df.iterrows():
        item_id = row['item_id']
        raw_name = row['raw_name']
        
        ingredient_id, confidence = find_best_match(raw_name, ingredients_df, threshold)
        
        matches.append({
            'item_id': item_id,
            'ingredient_id': ingredient_id if ingredient_id else None,
            'confidence': confidence
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(matches)
    results_df.to_csv(output_file, index=False)
    
    print(f"\nMatched {len(results_df)} items")
    print(f"Saved results to {output_file}")
    
    return results_df

