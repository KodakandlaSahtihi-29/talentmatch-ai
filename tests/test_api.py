"""Tests for FastAPI REST endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import init_db

# Ensure test DB tables are initialized
init_db()

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "nlp_engine" in data

def test_analyze_endpoint():
    payload = {
        "resume_name": "Test Candidate",
        "job_title": "Test Job",
        "resume_text": "Experienced Python and FastAPI developer with PostgreSQL, Docker, and Git. BS in Computer Science.",
        "job_text": "Looking for a Python developer with FastAPI, PostgreSQL, Docker, and AWS. Bachelor's in CS required."
    }
    response = client.post("/api/analysis/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "overall_score" in data
    assert "skill_match_score" in data
    assert "matched_skills" in data
    assert "recommendations" in data
    assert len(data["matched_skills"]) > 0

def test_history_endpoint():
    response = client.get("/api/analysis/history")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_academic_demo_endpoints():
    r_morph = client.get("/api/analysis/academic/morphology?text=Testing+FastAPI+services")
    assert r_morph.status_code == 200
    assert isinstance(r_morph.json(), list)
    
    r_edit = client.get("/api/analysis/academic/edit-distance?word1=pyhton&word2=python")
    assert r_edit.status_code == 200
    assert r_edit.json()["distance"] == 2
