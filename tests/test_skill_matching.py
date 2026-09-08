"""Tests for Skill Extraction, Levenshtein Fuzzy Matching, and Taxonomy."""
import pytest
from app.services.skill_extractor import extract_skills_from_text, determine_skill_requirement_type
from app.services.fuzzy_matcher import compute_levenshtein_distance, calculate_fuzzy_similarity, is_fuzzy_match

def test_levenshtein_distance():
    assert compute_levenshtein_distance("python", "python") == 0
    assert compute_levenshtein_distance("pyhton", "python") == 2
    assert compute_levenshtein_distance("cat", "hat") == 1

def test_fuzzy_matching_with_typos():
    sim_exact = calculate_fuzzy_similarity("python", "python")
    sim_typo = calculate_fuzzy_similarity("python", "pyhton")
    sim_unrelated = calculate_fuzzy_similarity("python", "docker")
    
    assert sim_exact == 1.0
    assert sim_typo >= 0.65
    assert sim_unrelated < 0.40
    
    is_match, score = is_fuzzy_match("postgresql", "posgresql", threshold=0.80)
    assert is_match is True
    assert score >= 0.80

def test_multi_word_skill_extraction():
    text = "Experience with Machine Learning, Natural Language Processing, REST API design, and Python."
    extracted = extract_skills_from_text(text)
    
    all_s = extracted["all_skills"]
    assert "machine learning" in all_s
    assert "natural language processing" in all_s
    assert "rest api" in all_s
    assert "python" in all_s

def test_requirement_type_detection():
    jd_text = "Required: Python, FastAPI, and SQL. Nice to have: AWS and Kubernetes."
    
    assert determine_skill_requirement_type("python", jd_text) == "Required"
    assert determine_skill_requirement_type("fastapi", jd_text) == "Required"
    assert determine_skill_requirement_type("kubernetes", jd_text) == "Preferred"
