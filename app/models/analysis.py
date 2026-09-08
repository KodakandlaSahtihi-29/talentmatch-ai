"""SQLAlchemy model for talent matching analysis history."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from app.core.database import Base

class AnalysisModel(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    resume_name = Column(String(255), nullable=False)
    job_title = Column(String(255), nullable=False)
    
    # Overall and Component Scores (0 - 100)
    overall_score = Column(Float, nullable=False)
    skill_score = Column(Float, nullable=False)
    semantic_score = Column(Float, nullable=False)
    experience_score = Column(Float, nullable=False)
    education_score = Column(Float, nullable=False)
    
    # Similarity Metrics
    tfidf_similarity = Column(Float, nullable=False)
    word2vec_similarity = Column(Float, nullable=False)
    hybrid_similarity = Column(Float, nullable=False)
    
    # Detailed JSON payload fields
    matched_skills = Column(Text, nullable=True)     # JSON string
    partial_skills = Column(Text, nullable=True)     # JSON string
    missing_skills = Column(Text, nullable=True)     # JSON string
    skill_comparison_matrix = Column(Text, nullable=True) # JSON string
    pos_distribution = Column(Text, nullable=True)   # JSON string
    ngram_insights = Column(Text, nullable=True)     # JSON string
    recommendations = Column(Text, nullable=True)    # JSON string
    why_explanation = Column(Text, nullable=True)    # JSON string
    
    created_at = Column(DateTime, default=datetime.utcnow)
