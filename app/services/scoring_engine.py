"""Explainable Scoring Engine."""
import logging
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.services.preprocessing import preprocess_text
from app.services.skill_extractor import (
    extract_skills_from_text,
    determine_skill_requirement_type,
    SKILL_TO_CATEGORY
)
from app.services.fuzzy_matcher import calculate_fuzzy_similarity
from app.services.similarity_engine import compute_all_similarities
from app.services.experience_matcher import (
    extract_requirement_sentences,
    match_experience_sentences,
    evaluate_education_match
)
from app.services.recommendation_engine import generate_recommendations

logger = logging.getLogger(__name__)

def compute_comprehensive_analysis(
    resume_text: str,
    jd_text: str,
    resume_name: str = "Candidate Resume",
    job_title: str = "Target Job Description",
    weight_tfidf: Optional[float] = None,
    weight_w2v: Optional[float] = None
) -> Dict[str, Any]:
    """
    Executes the full NLP matching pipeline and calculates explainable compatibility metrics.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty. Please provide a valid resume.")
    if not jd_text or not jd_text.strip():
        raise ValueError("Job Description text is empty. Please provide a valid job description.")
        
    # 1. NLP Preprocessing
    resume_doc = preprocess_text(resume_text)
    jd_doc = preprocess_text(jd_text)
    
    # 2. Skill Extraction from both documents
    resume_skills_data = extract_skills_from_text(resume_text)
    jd_skills_data = extract_skills_from_text(jd_text)
    
    resume_skills_set = set(resume_skills_data["all_skills"])
    jd_skills_list = jd_skills_data["all_skills"]
    
    # 3. Classify each JD Skill into Exact, Fuzzy, Semantic, or Missing
    matched_skills: List[Dict[str, Any]] = []
    partial_skills: List[Dict[str, Any]] = []
    missing_skills: List[Dict[str, Any]] = []
    all_comparison: List[Dict[str, Any]] = []
    
    # Score accumulator for skills
    total_skill_weight = 0.0
    earned_skill_weight = 0.0
    
    for skill in jd_skills_list:
        req_type = determine_skill_requirement_type(skill, jd_text)
        cat = SKILL_TO_CATEGORY.get(skill, "Technical Skills")
        
        importance_weight = 1.0 if req_type == "Required" else 0.6
        total_skill_weight += importance_weight
        
        # Check Exact Match
        if skill in resume_skills_set or skill in resume_skills_data["exact_skills"]:
            item = {
                "skill": skill,
                "resume_evidence": "Found in resume",
                "requirement_type": req_type,
                "match_type": "Exact",
                "similarity": 1.0,
                "category": cat
            }
            matched_skills.append(item)
            earned_skill_weight += (1.0 * importance_weight)
            all_comparison.append(item)
            continue
            
        # Check Fuzzy Match against candidate tokens
        fuzzy_found = False
        for f_item in resume_skills_data["fuzzy_skills"]:
            if f_item["skill"] == skill:
                item = {
                    "skill": skill,
                    "resume_evidence": f"Fuzzy matched token: '{f_item['matched_token']}'",
                    "requirement_type": req_type,
                    "match_type": "Fuzzy",
                    "similarity": f_item["similarity"],
                    "category": cat
                }
                partial_skills.append(item)
                earned_skill_weight += (0.85 * importance_weight)
                all_comparison.append(item)
                fuzzy_found = True
                break
                
        if fuzzy_found:
            continue
            
        # Check Semantic Match (via WordNet or category overlap in resume)
        # If resume has other skills in the exact same category, partial semantic credit is recognized
        cat_skills_in_resume = [s for s in resume_skills_set if SKILL_TO_CATEGORY.get(s) == cat]
        if cat_skills_in_resume:
            sim_score = 0.70
            item = {
                "skill": skill,
                "resume_evidence": f"Related domain skills found: {', '.join(cat_skills_in_resume[:2])}",
                "requirement_type": req_type,
                "match_type": "Semantic",
                "similarity": sim_score,
                "category": cat
            }
            partial_skills.append(item)
            earned_skill_weight += (0.70 * importance_weight)
            all_comparison.append(item)
        else:
            item = {
                "skill": skill,
                "resume_evidence": "Not found in resume",
                "requirement_type": req_type,
                "match_type": "Missing",
                "similarity": 0.0,
                "category": cat
            }
            missing_skills.append(item)
            all_comparison.append(item)
            
    # Calculate Skill Match Score percentage
    if total_skill_weight > 0:
        skill_match_score = round((earned_skill_weight / total_skill_weight) * 100.0, 1)
    else:
        # Fallback if no skills were in taxonomy
        skill_match_score = 75.0
        
    # 4. Lexical, Semantic, and Hybrid Similarity
    similarity_data = compute_all_similarities(
        doc1_clean_text=resume_doc.clean_text,
        doc2_clean_text=jd_doc.clean_text,
        tokens1=resume_doc.filtered_tokens,
        tokens2=jd_doc.filtered_tokens,
        weight_tfidf=weight_tfidf,
        weight_w2v=weight_w2v
    )
    
    # 5. Experience and Education Matching
    req_sentences = extract_requirement_sentences(jd_text)
    experience_matches = match_experience_sentences(req_sentences, resume_doc.sentences)
    
    if experience_matches:
        avg_exp_score = sum(e["relevance_score"] for e in experience_matches) / len(experience_matches)
        # Rescale into typical experience score range [40, 100] based on evidence
        strong_count = sum(1 for e in experience_matches if e["match_strength"] == "Strong")
        mod_count = sum(1 for e in experience_matches if e["match_strength"] == "Moderate")
        exp_score = min(100.0, round(40.0 + (strong_count * 15.0) + (mod_count * 8.0) + (avg_exp_score * 0.2), 1))
    else:
        exp_score = 70.0
        
    education_eval = evaluate_education_match(resume_text, jd_text)
    education_score = education_eval["score"]
    
    # 6. Overall Transparent Scoring Formula
    # Overall = 45% Skill + 30% Semantic + 15% Experience + 10% Education
    w_skill = settings.WEIGHT_SKILL_MATCH
    w_sem = settings.WEIGHT_SEMANTIC_SIM
    w_exp = settings.WEIGHT_EXPERIENCE_MATCH
    w_edu = settings.WEIGHT_EDUCATION_MATCH
    
    semantic_sim_pct = similarity_data["word2vec_similarity"]
    
    overall_score = (
        (w_skill * skill_match_score) +
        (w_sem * semantic_sim_pct) +
        (w_exp * exp_score) +
        (w_edu * education_score)
    )
    overall_score = max(0.0, min(100.0, round(overall_score, 1)))
    
    # 7. Positive & Negative Explainability Diagnostics
    positive_signals = []
    negative_signals = []
    
    if matched_skills:
        exact_names = [s["skill"].title() for s in matched_skills[:4]]
        positive_signals.append(f"Strong direct alignment on core skills: {', '.join(exact_names)}")
    if similarity_data["word2vec_similarity"] >= 70.0:
        positive_signals.append(f"High conceptual/semantic alignment ({similarity_data['word2vec_similarity']}%) across domain vocabulary")
    if education_score >= 90.0:
        positive_signals.append(education_eval["reason"])
    if any(e["match_strength"] == "Strong" for e in experience_matches):
        positive_signals.append("Demonstrated direct project evidence for major JD responsibility areas")
        
    if missing_skills:
        missing_req_names = [s["skill"].title() for s in missing_skills if s["requirement_type"] == "Required"][:4]
        if missing_req_names:
            negative_signals.append(f"Missing required technical competencies: {', '.join(missing_req_names)}")
        missing_pref_names = [s["skill"].title() for s in missing_skills if s["requirement_type"] == "Preferred"][:3]
        if missing_pref_names:
            negative_signals.append(f"Preferred skills not explicitly evidenced: {', '.join(missing_pref_names)}")
    if similarity_data["tfidf_similarity"] < 40.0:
        negative_signals.append("Low direct lexical keyword overlap with Job Description phrasing")
    if education_score < 80.0:
        negative_signals.append(education_eval["reason"])
        
    if not positive_signals:
        positive_signals.append("General technical vocabulary detected")
    if not negative_signals:
        negative_signals.append("No significant skill or qualification gaps detected")
        
    # 8. Actionable Ethical Recommendations
    recommendations = generate_recommendations(
        matched_skills=matched_skills,
        partial_skills=partial_skills,
        missing_skills=missing_skills,
        overall_score=overall_score,
        experience_matches=experience_matches
    )
    
    return {
        "resume_name": resume_name,
        "job_title": job_title,
        "overall_score": overall_score,
        "skill_match_score": skill_match_score,
        "semantic_similarity_score": semantic_sim_pct,
        "experience_match_score": exp_score,
        "education_match_score": education_score,
        "scoring_weights": {
            "skill_match": w_skill,
            "semantic_similarity": w_sem,
            "experience_relevance": w_exp,
            "education_match": w_edu
        },
        "similarity": similarity_data,
        "matched_skills": matched_skills,
        "partial_skills": partial_skills,
        "missing_skills": missing_skills,
        "all_skills_comparison": all_comparison,
        "experience_matches": experience_matches,
        "resume_pos": resume_doc.pos_distribution,
        "jd_pos": jd_doc.pos_distribution,
        "ngram_insights": {
            "resume_top_bigrams": resume_doc.top_bigrams,
            "jd_top_bigrams": jd_doc.top_bigrams,
            "resume_top_trigrams": resume_doc.top_trigrams,
            "jd_top_trigrams": jd_doc.top_trigrams
        },
        "positive_signals": positive_signals,
        "negative_signals": negative_signals,
        "recommendations": recommendations,
        "education_evaluation": education_eval
    }
