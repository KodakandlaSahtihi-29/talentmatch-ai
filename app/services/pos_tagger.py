"""POS Tagging service and distribution analysis."""
from typing import List, Tuple, Dict, Any
from collections import Counter
import nltk
from app.services.tokenizer import ensure_nltk_resources

ensure_nltk_resources()

def tag_pos(tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Performs Part-of-Speech tagging on tokens using NLTK Averaged Perceptron Tagger.
    """
    if not tokens:
        return []
    try:
        return nltk.pos_tag(tokens)
    except Exception:
        return [(t, "NN") for t in tokens]

def calculate_pos_distribution(pos_tags: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Computes syntactic tag distribution percentages for analytics dashboards.
    """
    if not pos_tags:
        return {
            "nouns_pct": 0.0,
            "verbs_pct": 0.0,
            "adjectives_pct": 0.0,
            "adverbs_pct": 0.0,
            "others_pct": 0.0,
            "top_pos_tags": {}
        }
        
    total = len(pos_tags)
    nouns = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
    verbs = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    adjectives = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    adverbs = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
    others = total - (nouns + verbs + adjectives + adverbs)
    
    tag_counts = Counter(tag for _, tag in pos_tags)
    
    return {
        "nouns_pct": round((nouns / total) * 100.0, 1),
        "verbs_pct": round((verbs / total) * 100.0, 1),
        "adjectives_pct": round((adjectives / total) * 100.0, 1),
        "adverbs_pct": round((adverbs / total) * 100.0, 1),
        "others_pct": round((others / total) * 100.0, 1),
        "top_pos_tags": dict(tag_counts.most_common(6))
    }

def extract_action_verbs(pos_tags: List[Tuple[str, str]]) -> List[str]:
    """
    Extracts action verbs (e.g., developed, implemented, architected, optimized).
    """
    return list(set(word.lower() for word, tag in pos_tags if tag.startswith('VB') and len(word) > 2))
