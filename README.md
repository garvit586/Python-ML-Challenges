# Python ML Challenge - MenuWise

This repository contains two ML projects completed as part of a Python ML Developer Challenge:

## Projects

### 1. Rapid Fit - Ingredient Line Classifier
A small NLP classification task that tags ingredient lines into 4 categories:
- `ingredient_only` — e.g., "Tomato"
- `ingredient_with_qty` — e.g., "Milk 200 ml"
- `instruction_like` — e.g., "Chop the onions"
- `non_food` — e.g., "Plastic wrap"

**Location:** `rapid_fit/`

**Features:**
- Hybrid approach combining rule-based features with TF-IDF
- Logistic Regression classifier
- Jupyter notebook with full analysis
- Unit tests with pytest
- See `rapid_fit/DECISIONS.md` for detailed approach

### 2. Mini Project - Supplier Item Matching
Fuzzy matching system to map noisy supplier items to canonical ingredients with a FastAPI service.

**Location:** `mini_project/`

**Features:**
- Text normalization and fuzzy matching using rapidfuzz
- FastAPI service with `/match` endpoint
- Docker containerization
- Evaluation script for precision@1 and coverage
- Unit tests with pytest
- See `mini_project/DECISIONS.md` for detailed approach

## Quick Start

### Rapid Fit
```bash
cd rapid_fit
pip install -r requirements.txt
# Open ingredient_classifier.ipynb in Jupyter
```

### Mini Project
```bash
cd mini_project
pip install -r requirements.txt

# Generate matches
python -m src.main

# Run FastAPI service
uvicorn src.app:app --host 0.0.0.0 --port 8000

# Or using Docker
docker build -t ingredient-matcher .
docker run -p 8000:8000 ingredient-matcher
```

## Project Structure

```
.
├── rapid_fit/              # Ingredient line classifier
│   ├── data/
│   ├── ingredient_classifier.ipynb
│   ├── test_classifier.py
│   ├── requirements.txt
│   └── DECISIONS.md
│
├── mini_project/           # Supplier item matcher
│   ├── src/                # Source code
│   ├── tests/              # Unit tests
│   ├── data/               # Data files
│   ├── Dockerfile
│   ├── requirements.txt
│   └── DECISIONS.md
│
└── requirements.txt         # Root dependencies
```

## Technologies Used

- **Python 3.9+**
- **scikit-learn** - Machine learning
- **rapidfuzz** - Fuzzy string matching
- **FastAPI** - Web framework
- **pandas** - Data manipulation
- **pytest** - Testing
- **Jupyter** - Notebooks

## License

This project was completed as part of a coding challenge.
"# PYTHON-ML-CHALLANGES" 
