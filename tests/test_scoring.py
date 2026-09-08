"""Tests for Scoring Engine and Explainability."""
import pytest
from app.services.scoring_engine import compute_comprehensive_analysis

def test_scoring_bounds_and_structure():
    resume = "Senior Python developer with FastAPI, SQL, PostgreSQL, and Docker experience. BS in Computer Science."
    jd = "Seeking Python engineer with FastAPI, PostgreSQL, Docker, and AWS skills. Degree required."
    
    result = compute_comprehensive_analysis(resume, jd, "Test Resume", "Test Job")
    
    assert 0.0 <= result["overall_score"] <= 100.0
    assert 0.0 <= result["skill_match_score"] <= 100.0
    assert 0.0 <= result["semantic_similarity_score"] <= 100.0
    assert 0.0 <= result["experience_match_score"] <= 100.0
    assert 0.0 <= result["education_match_score"] <= 100.0
    
    assert "positive_signals" in result
    assert "negative_signals" in result
    assert "recommendations" in result
    assert len(result["recommendations"]) > 0

def test_score_monotonicity():
    good_resume = "Machine Learning Engineer with Python, PyTorch, Scikit-learn, NLP, Transformers, and FastAPI. BS in Computer Science."
    bad_resume = "Classical pastry chef with expertise in baking sourdough bread, chocolate eclairs, and French culinary techniques."
    ml_jd = "Senior ML Engineer required with Python, PyTorch, Scikit-learn, NLP, and FastAPI. Computer Science degree."
    
    good_res = compute_comprehensive_analysis(good_resume, ml_jd)
    bad_res = compute_comprehensive_analysis(bad_resume, ml_jd)
    
    assert good_res["overall_score"] > bad_res["overall_score"]
    assert good_res["skill_match_score"] > bad_res["skill_match_score"]

def test_empty_input_validation():
    with pytest.raises(ValueError):
        compute_comprehensive_analysis("", "Valid JD Text")
    with pytest.raises(ValueError):
        compute_comprehensive_analysis("Valid Resume", "")
