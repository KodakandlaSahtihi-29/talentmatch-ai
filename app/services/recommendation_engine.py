"""Ethical and Actionable Recommendation Engine."""
from typing import List, Dict, Any

def generate_recommendations(
    matched_skills: List[Dict[str, Any]],
    partial_skills: List[Dict[str, Any]],
    missing_skills: List[Dict[str, Any]],
    overall_score: float,
    experience_matches: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generates actionable, ethical recommendations based on gaps detected in the Job Description.
    
    IMPORTANT: The system enforces zero experience fabrication. It advises candidates
    how to better surface legitimate project experience or acquire target skills.
    """
    recommendations = []
    
    # 1. High Priority: Missing Required Skills
    missing_required = [s for s in missing_skills if s.get("requirement_type") == "Required"]
    for item in missing_required[:4]:
        skill_name = item["skill"].title()
        cat = item.get("category", "Technical Skills")
        recommendations.append({
            "category": cat,
            "skill_or_area": skill_name,
            "priority": "High",
            "message": (
                f"The Job Description lists {skill_name} as an essential requirement. "
                f"If you have verified academic or industry experience with {skill_name}, "
                f"explicitly highlight it in your resume's technical skills and project bullets."
            ),
            "context": f"Missing core requirement in {cat}"
        })
        
    # 2. Medium Priority: Fuzzy / Partial Term Normalization
    for item in partial_skills[:3]:
        skill_name = item["skill"].title()
        found_tok = item.get("resume_evidence", "")
        recommendations.append({
            "category": item.get("category", "Technical Skills"),
            "skill_or_area": skill_name,
            "priority": "Medium",
            "message": (
                f"Your resume mentions '{found_tok}', which was fuzzy-matched to '{skill_name}'. "
                f"Standardize your resume phrasing to '{skill_name}' to ensure ATS parsers recognize it with 100% precision."
            ),
            "context": "Keyword phrasing optimization"
        })
        
    # 3. Medium Priority: Missing Preferred Skills
    missing_pref = [s for s in missing_skills if s.get("requirement_type") == "Preferred"]
    for item in missing_pref[:2]:
        skill_name = item["skill"].title()
        recommendations.append({
            "category": item.get("category", "Technical Skills"),
            "skill_or_area": skill_name,
            "priority": "Medium",
            "message": (
                f"{skill_name} is listed as a preferred/nice-to-have skill. "
                f"Building a demo or including coursework related to {skill_name} will increase candidate competitiveness."
            ),
            "context": f"Preferred bonus skill in {item.get('category', 'Technical Skills')}"
        })
        
    # 4. Experience Impact Recommendations
    weak_exp = [e for e in experience_matches if e.get("match_strength") == "Weak / Missing"]
    if weak_exp and overall_score < 80.0:
        recommendations.append({
            "category": "Impact Phrasing",
            "skill_or_area": "Action Verbs & Quantifiable Metrics",
            "priority": "Low",
            "message": (
                "Strengthen your bullet points using the Google XYZ formula: "
                "'Accomplished [X], as measured by [Y], by doing [Z]' with strong action verbs (e.g., Architected, Optimized, Implemented)."
            ),
            "context": "Resume impact formatting"
        })
        
    if not recommendations:
        recommendations.append({
            "category": "Profile Strength",
            "skill_or_area": "Strong Candidate Profile",
            "priority": "Low",
            "message": "Your resume demonstrates excellent alignment across technical skills and experience requirements.",
            "context": "Strong alignment"
        })
        
    return recommendations
