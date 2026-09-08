"""Schemas for candidate-job analysis and matching."""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict

class AnalysisRequest(BaseModel):
    resume_id: Optional[int] = None
    job_id: Optional[int] = None
    resume_text: Optional[str] = None
    job_text: Optional[str] = None
    resume_name: Optional[str] = "Candidate Resume"
    job_title: Optional[str] = "Target Job Description"
    weight_tfidf: Optional[float] = None
    weight_w2v: Optional[float] = None

class SkillMatchItem(BaseModel):
    skill: str
    resume_evidence: str
    requirement_type: str  # "Required" or "Preferred"
    match_type: str        # "Exact", "Fuzzy", "Semantic", "Missing"
    similarity: float      # 0.0 - 1.0
    category: str          # e.g., "Programming Languages", "Machine Learning"

class SimilarityBreakdown(BaseModel):
    tfidf_similarity: float
    word2vec_similarity: float
    hybrid_similarity: float
    tfidf_weight: float
    word2vec_weight: float
    explanation: str

class POSDistribution(BaseModel):
    nouns_pct: float
    verbs_pct: float
    adjectives_pct: float
    adverbs_pct: float
    others_pct: float
    top_pos_tags: Dict[str, int]

class NgramInsights(BaseModel):
    resume_top_bigrams: List[Dict[str, Any]]
    jd_top_bigrams: List[Dict[str, Any]]
    resume_top_trigrams: List[Dict[str, Any]]
    jd_top_trigrams: List[Dict[str, Any]]

class ExperienceMatchItem(BaseModel):
    requirement: str
    evidence: str
    match_strength: str  # "Strong", "Moderate", "Weak / Missing"
    relevance_score: float

class RecommendationItem(BaseModel):
    category: str
    skill_or_area: str
    priority: str  # "High", "Medium", "Low"
    message: str
    context: str

class AnalysisResponse(BaseModel):
    id: Optional[int] = None
    resume_name: str
    job_title: str
    
    # Overall and Component Scores (0 - 100)
    overall_score: float
    skill_match_score: float
    semantic_similarity_score: float
    experience_match_score: float
    education_match_score: float
    
    # Component Weights Used
    scoring_weights: Dict[str, float]
    
    # Similarity Metrics
    similarity: SimilarityBreakdown
    
    # Skill Gap Classification
    matched_skills: List[SkillMatchItem]
    partial_skills: List[SkillMatchItem]
    missing_skills: List[SkillMatchItem]
    all_skills_comparison: List[SkillMatchItem]
    
    # Experience Matching Breakdown
    experience_matches: List[ExperienceMatchItem]
    
    # NLP Linguistics & Syntax
    resume_pos: POSDistribution
    jd_pos: POSDistribution
    ngram_insights: NgramInsights
    
    # Explainability & Recommendations
    positive_signals: List[str]
    negative_signals: List[str]
    recommendations: List[RecommendationItem]
    
    created_at: Optional[datetime] = None

class AnalysisHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    resume_name: str
    job_title: str
    overall_score: float
    skill_score: float
    semantic_score: float
    experience_score: float
    education_score: float
    hybrid_similarity: float
    created_at: datetime
