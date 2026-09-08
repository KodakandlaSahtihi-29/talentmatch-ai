"""Experience and Requirement Matching Engine."""
from typing import List, Dict, Any, Tuple
from app.services.tfidf_engine import calculate_tfidf_similarity
from app.utils.text_utils import detect_education_level
from app.services.tokenizer import tokenize_sentences

def extract_requirement_sentences(jd_text: str) -> List[str]:
    """Extracts requirement sentences from Job Description."""
    sentences = tokenize_sentences(jd_text)
    reqs = []
    
    req_indicators = [
        'experience', 'responsible', 'develop', 'build', 'design', 'manage',
        'proficiency', 'working with', 'knowledge of', 'seeking', 'requirements',
        'must have', 'years of', 'hands-on', 'architect', 'lead', 'maintain'
    ]
    
    for s in sentences:
        s_clean = s.strip()
        lower = s_clean.lower()
        if len(s_clean) > 20 and any(ind in lower for ind in req_indicators):
            reqs.append(s_clean)
            
    # If no specific sentences matched, take top 4 substantive sentences
    if not reqs and sentences:
        reqs = [s.strip() for s in sentences if len(s.strip()) > 20][:4]
        
    return reqs[:6]

def match_experience_sentences(
    jd_sentences: List[str],
    resume_sentences: List[str]
) -> List[Dict[str, Any]]:
    """
    Finds the most semantically relevant resume evidence for each JD requirement sentence.
    """
    matches = []
    
    for req in jd_sentences:
        best_sent = "No direct evidence found in resume"
        best_score = 0.0
        
        for res_sent in resume_sentences:
            if len(res_sent.strip()) < 15:
                continue
            sim_res = calculate_tfidf_similarity(req, res_sent)
            score = sim_res["similarity"]
            if score > best_score:
                best_score = score
                best_sent = res_sent.strip()
                
        if best_score >= 0.35:
            strength = "Strong"
        elif best_score >= 0.15:
            strength = "Moderate"
        else:
            strength = "Weak / Missing"
            if best_score < 0.10:
                best_sent = "No direct evidence found in resume"
                
        matches.append({
            "requirement": req,
            "evidence": best_sent,
            "match_strength": strength,
            "relevance_score": round(best_score * 100.0, 1)
        })
        
    return matches

def evaluate_education_match(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """
    Compares degree levels and field of study between candidate resume and JD.
    """
    res_edu = detect_education_level(resume_text)
    jd_edu = detect_education_level(jd_text)
    
    degree_hierarchy = {"PhD": 4, "Masters": 3, "Bachelors": 2, "Associate/Diploma": 1, "None Detected": 0}
    
    res_level = degree_hierarchy.get(res_edu["highest_degree"], 0)
    jd_level = degree_hierarchy.get(jd_edu["highest_degree"], 0)
    
    # If JD doesn't mention degree, candidate matches by default
    if jd_level == 0:
        score = 90.0 if res_level > 0 else 80.0
        reason = "No strict degree requirement specified in Job Description."
    elif res_level >= jd_level:
        score = 100.0 if res_edu["is_cs_related"] else 90.0
        reason = f"Candidate degree ({res_edu['highest_degree']}) meets or exceeds required degree ({jd_edu['highest_degree']})."
    elif res_level > 0:
        score = 70.0
        reason = f"Candidate degree ({res_edu['highest_degree']}) is lower than target ({jd_edu['highest_degree']}), but demonstrates formal education."
    else:
        score = 50.0
        reason = f"Job requires {jd_edu['highest_degree']}, but no clear degree was detected on resume."
        
    return {
        "score": score,
        "resume_degree": res_edu["highest_degree"],
        "jd_degree": jd_edu["highest_degree"],
        "cs_discipline": res_edu["is_cs_related"],
        "reason": reason
    }
