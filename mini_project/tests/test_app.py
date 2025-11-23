"""
Unit tests for FastAPI app.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.app import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_match_success():
    """Test successful matching."""
    response = client.post("/match", json={"raw_name": "Tomato"})
    assert response.status_code == 200
    data = response.json()
    assert "ingredient_id" in data
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1


def test_match_with_quantity():
    """Test matching with quantity."""
    response = client.post("/match", json={"raw_name": "TOMATOES 1kg pack"})
    assert response.status_code == 200
    data = response.json()
    assert data["ingredient_id"] == 1
    assert data["confidence"] > 0.7


def test_match_fuzzy():
    """Test fuzzy matching."""
    response = client.post("/match", json={"raw_name": "gralic"})
    assert response.status_code == 200
    data = response.json()
    assert data["ingredient_id"] == 3  # Garlic
    assert data["confidence"] > 0.6


def test_match_empty():
    """Test with empty raw_name."""
    response = client.post("/match", json={"raw_name": ""})
    assert response.status_code == 400


def test_match_invalid():
    """Test with invalid input."""
    response = client.post("/match", json={})
    assert response.status_code == 422  # Validation error

