"""
Evaluation script for matching accuracy.
Calculates precision@1 and coverage.
"""
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_matches(matches_file='matches.csv', 
                    supplier_file='data/supplier_items.csv',
                    ingredients_file='data/ingredients_master.csv'):
    """
    Evaluate matching performance.
    
    For this evaluation, we'll assume:
    - Precision@1: % of matches with confidence > threshold that are correct
    - Coverage: % of items that got a match (confidence > 0)
    
    Since we don't have ground truth labels, we'll use heuristics:
    - Check if matched ingredient name appears in raw_name (normalized)
    """
    matches_df = pd.read_csv(matches_file)
    supplier_df = pd.read_csv(supplier_file)
    ingredients_df = pd.read_csv(ingredients_file)
    
    # Merge to get ingredient names
    results = matches_df.merge(
        ingredients_df, 
        on='ingredient_id', 
        how='left'
    )
    results = results.merge(
        supplier_df,
        on='item_id',
        how='left'
    )
    
    # Coverage: % of items with a match (confidence > 0)
    total_items = len(results)
    matched_items = len(results[results['confidence'] > 0])
    coverage = (matched_items / total_items) * 100 if total_items > 0 else 0
    
    print(f"Total items: {total_items}")
    print(f"Matched items: {matched_items}")
    print(f"Coverage: {coverage:.2f}%")
    
    # For precision, check if match makes sense
    # Simple heuristic: check if ingredient name (normalized) appears in raw_name
    def check_match_quality(row):
        if pd.isna(row['name']) or pd.isna(row['raw_name']):
            return False
        
        raw_lower = str(row['raw_name']).lower()
        name_lower = str(row['name']).lower()
        
        # Extract key words from ingredient name
        name_words = name_lower.split()
        # Remove common words
        name_words = [w for w in name_words if w not in ['whole', 'all-purpose', 'unsalted', 'granulated']]
        
        # Check if any key word appears in raw_name
        for word in name_words:
            if word in raw_lower:
                return True
        
        return False
    
    results['is_correct'] = results.apply(check_match_quality, axis=1)
    
    # Precision@1: % of matches (confidence > threshold) that are correct
    threshold = 0.6
    high_confidence = results[results['confidence'] >= threshold]
    
    if len(high_confidence) > 0:
        correct_matches = len(high_confidence[high_confidence['is_correct'] == True])
        precision = (correct_matches / len(high_confidence)) * 100
        print(f"\nHigh confidence matches (>= {threshold}): {len(high_confidence)}")
        print(f"Correct matches: {correct_matches}")
        print(f"Precision@1: {precision:.2f}%")
    else:
        print(f"\nNo matches with confidence >= {threshold}")
        print("Precision@1: N/A")
    
    # Show all matches
    print("\nAll matches:")
    print(results[['item_id', 'raw_name', 'ingredient_id', 'name', 'confidence', 'is_correct']].to_string())
    
    return {
        'coverage': coverage,
        'precision': precision if len(high_confidence) > 0 else 0,
        'total_items': total_items,
        'matched_items': matched_items
    }


if __name__ == "__main__":
    evaluate_matches()

