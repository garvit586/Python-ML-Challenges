"""
Main script to generate matches.csv
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.matcher import match_all_items

if __name__ == "__main__":
    print("Starting ingredient matching...")
    results = match_all_items(
        supplier_file='data/supplier_items.csv',
        ingredients_file='data/ingredients_master.csv',
        output_file='matches.csv',
        threshold=60
    )
    
    print("\nMatching complete!")
    print("\nSample results:")
    print(results.head(10))

