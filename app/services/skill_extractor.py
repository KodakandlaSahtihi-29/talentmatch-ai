"""Skill extraction and classification engine."""
import json
import re
from typing import List, Dict, Set, Tuple, Any, Optional
from pathlib import Path
from app.core.config import settings, SKILLS_FILE
from app.services.fuzzy_matcher import calculate_fuzzy_similarity
from app.services.tokenizer import tokenize_words

def load_skills_taxonomy(filepath: Optional[Path] = None) -> Dict[str, List[str]]:
    """Loads taxonomy of categorized skills from JSON."""
    path = filepath or SKILLS_FILE
    if not path.exists():
        return {
            "programming_languages": ["python", "java", "c++", "c#", "javascript", "sql"],
            "machine_learning_and_ai": ["machine learning", "nlp", "deep learning", "pytorch", "scikit-learn"],
            "backend_and_apis": ["fastapi", "flask", "django", "rest api", "postgresql"],
            "cloud_and_infrastructure": ["aws", "azure", "docker", "kubernetes"]
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

SKILLS_TAXONOMY = load_skills_taxonomy()

def get_all_skills_flat() -> Dict[str, str]:
    """Returns a mapping of skill -> category name."""
    skill_to_cat = {}
    for category, skills in SKILLS_TAXONOMY.items():
        cat_title = category.replace("_", " ").title()
        for skill in skills:
            skill_to_cat[skill.lower().strip()] = cat_title
    return skill_to_cat

SKILL_TO_CATEGORY = get_all_skills_flat()

# Cues for detecting requirement type (Required vs Preferred)
REQUIRED_CUES = [
    r'\brequired\b', r'\bmust have\b', r'\bessential\b', r'\bminimum\b',
    r'\bproven experience with\b', r'\bproficiency in\b', r'\bstrong background in\b'
]
PREFERRED_CUES = [
    r'\bpreferred\b', r'\bnice to have\b', r'\bplus\b', r'\bbonus\b',
    r'\bdesirable\b', r'\bfamiliarity with\b', r'\boptional\b', r'\bgood to have\b'
]

def determine_skill_requirement_type(skill: str, full_jd_text: str) -> str:
    """
    Analyzes sentence context in JD to identify whether a skill is Required or Preferred.
    """
    lower_jd = full_jd_text.lower()
    sentences = re.split(r'[\n\.\?!;]', lower_jd)
    
    for sent in sentences:
        if skill.lower() in sent:
            for pref in PREFERRED_CUES:
                if re.search(pref, sent):
                    return "Preferred"
            for req in REQUIRED_CUES:
                if re.search(req, sent):
                    return "Required"
                    
    # Default assumption if not explicitly marked preferred
    return "Required"

def extract_skills_from_text(text: str, custom_taxonomy: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """
    Extracts all recognized skills from a given text using exact phrase matching,
    normalized token matching, and fuzzy matching for typos.
    
    Returns:
        Dict:
            - "exact_skills": List[str]
            - "fuzzy_skills": List[Dict[str, Any]] (skill, found_token, similarity)
            - "all_skills": List[str]
            - "by_category": Dict[str, List[str]]
    """
    if not text:
        return {"exact_skills": [], "fuzzy_skills": [], "all_skills": [], "by_category": {}}
        
    taxonomy = custom_taxonomy or SKILLS_TAXONOMY
    skill_map = {}
    for cat, skills in taxonomy.items():
        cat_name = cat.replace("_", " ").title()
        for s in skills:
            skill_map[s.lower().strip()] = cat_name
            
    lower_text = " " + text.lower() + " "
    raw_tokens = tokenize_words(text)
    
    found_exact: Set[str] = set()
    found_fuzzy: List[Dict[str, Any]] = []
    
    # 1. Multi-word phrase matching & single word exact matching
    for skill in skill_map.keys():
        # Match as whole word/phrase
        pattern = r'(?<![a-zA-Z0-9_\#\+])' + re.escape(skill) + r'(?![a-zA-Z0-9_\#\+])'
        if re.search(pattern, lower_text):
            found_exact.add(skill)
            
    # 2. Fuzzy matching on remaining unmatched single-word skills (e.g. typos like 'pyhton' or 'posgres')
    for skill in skill_map.keys():
        if " " in skill or skill in found_exact or len(skill) <= 3:
            continue
            
        for tok in raw_tokens:
            if len(tok) <= 3 or tok in found_exact:
                continue
            sim = calculate_fuzzy_similarity(skill, tok)
            if sim >= settings.FUZZY_MATCH_THRESHOLD and sim < 1.0:
                found_fuzzy.append({
                    "skill": skill,
                    "matched_token": tok,
                    "similarity": round(sim, 3),
                    "category": skill_map[skill]
                })
                break
                
    all_skills = sorted(list(found_exact.union(set(item["skill"] for item in found_fuzzy))))
    
    by_cat: Dict[str, List[str]] = {}
    for s in all_skills:
        cat = skill_map.get(s, "Other Skills")
        by_cat.setdefault(cat, []).append(s)
        
    return {
        "exact_skills": sorted(list(found_exact)),
        "fuzzy_skills": found_fuzzy,
        "all_skills": all_skills,
        "by_category": by_cat
    }
