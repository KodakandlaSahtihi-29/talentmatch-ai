"""Analysis and Compatibility Matching API routes."""
import json
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.resume import ResumeModel
from app.models.job import JobModel
from app.models.analysis import AnalysisModel
from app.schemas.analysis import AnalysisRequest, AnalysisResponse, AnalysisHistoryItem
from app.services.scoring_engine import compute_comprehensive_analysis
from app.services.academic_nlp import (
    analyze_morphology_comparison,
    compute_edit_distance_demo,
    analyze_pos_tags_detailed,
    parse_syntax_cky_demo,
    analyze_vector_semantics_demo
)

router = APIRouter(prefix="/analysis", tags=["Analysis & Matching"])
logger = logging.getLogger(__name__)

@router.post("/run", response_model=AnalysisResponse)
def run_analysis(
    payload: AnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Executes the full NLP matching pipeline between a Candidate Resume and Job Description.
    Stores the structured results in the database and returns explainable scores.
    """
    resume_text = payload.resume_text
    job_text = payload.job_text
    resume_name = payload.resume_name or "Candidate Resume"
    job_title = payload.job_title or "Target Job Description"
    
    # 1. Fetch from database if IDs provided
    if payload.resume_id:
        resume_record = db.query(ResumeModel).filter(ResumeModel.id == payload.resume_id).first()
        if not resume_record:
            raise HTTPException(status_code=404, detail=f"Resume with ID {payload.resume_id} not found.")
        resume_text = resume_record.raw_text
        resume_name = resume_record.filename
        
    if payload.job_id:
        job_record = db.query(JobModel).filter(JobModel.id == payload.job_id).first()
        if not job_record:
            raise HTTPException(status_code=404, detail=f"Job with ID {payload.job_id} not found.")
        job_text = job_record.raw_text
        job_title = job_record.title
        
    if not resume_text or not resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text is required (provide resume_id or resume_text).")
    if not job_text or not job_text.strip():
        raise HTTPException(status_code=400, detail="Job Description text is required (provide job_id or job_text).")
        
    # 2. Run Comprehensive NLP Analysis Pipeline
    try:
        results = compute_comprehensive_analysis(
            resume_text=resume_text,
            jd_text=job_text,
            resume_name=resume_name,
            job_title=job_title,
            weight_tfidf=payload.weight_tfidf,
            weight_w2v=payload.weight_w2v
        )
    except Exception as e:
        logger.error(f"Analysis pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis pipeline encountered an error: {str(e)}")
        
    # 3. Store in Database
    analysis_record = AnalysisModel(
        resume_name=resume_name,
        job_title=job_title,
        overall_score=results["overall_score"],
        skill_score=results["skill_match_score"],
        semantic_score=results["semantic_similarity_score"],
        experience_score=results["experience_match_score"],
        education_score=results["education_match_score"],
        tfidf_similarity=results["similarity"]["tfidf_similarity"],
        word2vec_similarity=results["similarity"]["word2vec_similarity"],
        hybrid_similarity=results["similarity"]["hybrid_similarity"],
        matched_skills=json.dumps(results["matched_skills"]),
        partial_skills=json.dumps(results["partial_skills"]),
        missing_skills=json.dumps(results["missing_skills"]),
        skill_comparison_matrix=json.dumps(results["all_skills_comparison"]),
        pos_distribution=json.dumps({"resume": results["resume_pos"], "jd": results["jd_pos"]}),
        ngram_insights=json.dumps(results["ngram_insights"]),
        recommendations=json.dumps(results["recommendations"]),
        why_explanation=json.dumps({
            "positive": results["positive_signals"],
            "negative": results["negative_signals"]
        })
    )
    
    db.add(analysis_record)
    db.commit()
    db.refresh(analysis_record)
    
    results["id"] = analysis_record.id
    results["created_at"] = analysis_record.created_at
    
    return results

@router.get("/history", response_model=List[AnalysisHistoryItem])
def get_analysis_history(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Fetches past matching analysis history from the database."""
    analyses = db.query(AnalysisModel).order_by(AnalysisModel.created_at.desc()).limit(limit).all()
    return analyses

@router.get("/{analysis_id}", response_model=AnalysisResponse)
def get_analysis_by_id(analysis_id: int, db: Session = Depends(get_db)):
    """Retrieves full analysis record by ID."""
    record = db.query(AnalysisModel).filter(AnalysisModel.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Analysis with ID {analysis_id} not found.")
        
    matched = json.loads(record.matched_skills) if record.matched_skills else []
    partial = json.loads(record.partial_skills) if record.partial_skills else []
    missing = json.loads(record.missing_skills) if record.missing_skills else []
    comparison = json.loads(record.skill_comparison_matrix) if record.skill_comparison_matrix else []
    pos_data = json.loads(record.pos_distribution) if record.pos_distribution else {}
    ngrams = json.loads(record.ngram_insights) if record.ngram_insights else {
        "resume_top_bigrams": [], "jd_top_bigrams": [], "resume_top_trigrams": [], "jd_top_trigrams": []
    }
    recs = json.loads(record.recommendations) if record.recommendations else []
    why = json.loads(record.why_explanation) if record.why_explanation else {"positive": [], "negative": []}
    
    return AnalysisResponse(
        id=record.id,
        resume_name=record.resume_name,
        job_title=record.job_title,
        overall_score=record.overall_score,
        skill_match_score=record.skill_score,
        semantic_similarity_score=record.semantic_score,
        experience_match_score=record.experience_score,
        education_match_score=record.education_score,
        scoring_weights={
            "skill_match": 0.45,
            "semantic_similarity": 0.30,
            "experience_relevance": 0.15,
            "education_match": 0.10
        },
        similarity={
            "tfidf_similarity": record.tfidf_similarity,
            "word2vec_similarity": record.word2vec_similarity,
            "hybrid_similarity": record.hybrid_similarity,
            "tfidf_weight": 0.45,
            "word2vec_weight": 0.55,
            "explanation": "Stored historical similarity evaluation."
        },
        matched_skills=matched,
        partial_skills=partial,
        missing_skills=missing,
        all_skills_comparison=comparison,
        experience_matches=[],
        resume_pos=pos_data.get("resume", {
            "nouns_pct": 0, "verbs_pct": 0, "adjectives_pct": 0, "adverbs_pct": 0, "others_pct": 0, "top_pos_tags": {}
        }),
        jd_pos=pos_data.get("jd", {
            "nouns_pct": 0, "verbs_pct": 0, "adjectives_pct": 0, "adverbs_pct": 0, "others_pct": 0, "top_pos_tags": {}
        }),
        ngram_insights=ngrams,
        positive_signals=why.get("positive", []),
        negative_signals=why.get("negative", []),
        recommendations=recs,
        created_at=record.created_at
    )

@router.get("/diagnostics/morphology", tags=["NLP Diagnostics"])
@router.get("/academic/morphology", include_in_schema=False)
def get_academic_morphology(text: str = Query("Developing scalable Python applications and APIs")):
    """Compares WordNet lemmatization vs. Porter stemming."""
    return analyze_morphology_comparison(text)

@router.get("/diagnostics/edit-distance", tags=["NLP Diagnostics"])
@router.get("/academic/edit-distance", include_in_schema=False)
def get_academic_edit_distance(word1: str = Query("pyhton"), word2: str = Query("python")):
    """Computes Levenshtein edit distance matrix and typo similarity."""
    return compute_edit_distance_demo(word1, word2)

@router.get("/diagnostics/pos-breakdown", tags=["NLP Diagnostics"])
@router.get("/academic/pos-breakdown", include_in_schema=False)
def get_academic_pos(text: str = Query("Candidate built high performance machine learning systems using PyTorch.")):
    """Provides detailed Penn Treebank POS tag classification."""
    return analyze_pos_tags_detailed(text)

@router.get("/diagnostics/syntax-structure", tags=["NLP Diagnostics"])
@router.get("/academic/syntax-cky", include_in_schema=False)
def get_academic_syntax(sentence: str = Query("The senior developer built reliable web services.")):
    """Demonstrates constituent phrase structure (Noun Phrase / Verb Phrase chunking)."""
    return parse_syntax_cky_demo(sentence)

@router.get("/diagnostics/vector-neighbors", tags=["NLP Diagnostics"])
@router.get("/academic/vector-neighbors", include_in_schema=False)
def get_academic_vectors(word: str = Query("python")):
    """Retrieves continuous vector space nearest neighbors using Word2Vec."""
    return analyze_vector_semantics_demo(word)

