"""
FastAPI service for ingredient matching.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.matcher import load_ingredients, find_best_match

app = FastAPI(title="Ingredient Matcher API")

# Load ingredients on startup
ingredients_df = None

@app.on_event("startup")
async def startup_event():
    """Load ingredients when app starts."""
    global ingredients_df
    try:
        ingredients_df = load_ingredients('data/ingredients_master.csv')
        print(f"Loaded {len(ingredients_df)} ingredients")
    except Exception as e:
        print(f"Error loading ingredients: {e}")
        raise


class MatchRequest(BaseModel):
    raw_name: str


class MatchResponse(BaseModel):
    ingredient_id: int
    confidence: float


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Ingredient Matcher API"}


@app.post("/match", response_model=MatchResponse)
def match_ingredient(request: MatchRequest):
    """
    Match a raw ingredient name to canonical ingredient.
    
    Args:
        request: JSON with 'raw_name' field
        
    Returns:
        JSON with 'ingredient_id' and 'confidence' (0-1)
    """
    if ingredients_df is None:
        raise HTTPException(status_code=500, detail="Ingredients not loaded")
    
    if not request.raw_name:
        raise HTTPException(status_code=400, detail="raw_name cannot be empty")
    
    # Find best match
    ingredient_id, confidence = find_best_match(
        request.raw_name, 
        ingredients_df, 
        threshold=60
    )
    
    if ingredient_id is None:
        raise HTTPException(
            status_code=404, 
            detail=f"No match found for '{request.raw_name}'"
        )
    
    return MatchResponse(
        ingredient_id=int(ingredient_id),
        confidence=round(confidence, 3)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

