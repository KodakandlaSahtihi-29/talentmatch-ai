"""Text processing utilities and technical term preservation helpers."""
import re
from typing import List, Tuple, Dict, Any
from collections import Counter

# Regex patterns for technical symbols that must NOT be stripped or mangled
TECH_TERMS_PRESERVE = [
    (r'(?<!\w)c\+\+(?!\w)', 'c_plus_plus'),
    (r'(?<!\w)c\#(?!\w)', 'c_sharp'),
    (r'(?<!\w)\.net(?!\w)', 'dot_net'),
    (r'(?<!\w)node\.js(?!\w)', 'nodejs'),
    (r'(?<!\w)vue\.js(?!\w)', 'vuejs'),
    (r'(?<!\w)next\.js(?!\w)', 'nextjs'),
    (r'(?<!\w)ci/cd(?!\w)', 'ci_cd'),
    (r'(?<!\w)pl/sql(?!\w)', 'pl_sql'),
    (r'(?<!\w)tcp/ip(?!\w)', 'tcp_ip'),
    (r'(?<!\w)restful(?!\w)', 'rest_api'),
    (r'(?<!\w)rest api(?!\w)', 'rest_api'),
    (r'(?<!\w)power bi(?!\w)', 'power_bi'),
    (r'(?<!\w)amazon web services(?!\w)', 'aws'),
    (r'(?<!\w)google cloud platform(?!\w)', 'gcp'),
]

TECH_TERMS_RESTORE = {
    'c_plus_plus': 'c++',
    'c_sharp': 'c#',
    'dot_net': '.net',
    'nodejs': 'node.js',
    'vuejs': 'vue.js',
    'nextjs': 'next.js',
    'ci_cd': 'ci/cd',
    'pl_sql': 'pl/sql',
    'tcp_ip': 'tcp/ip',
    'rest_api': 'rest api',
    'power_bi': 'power bi',
    'aws': 'aws',
    'gcp': 'gcp',
}

def protect_tech_terms(text: str) -> str:
    """Replaces tech punctuation with safe alpha tokens before tokenization."""
    lower = text.lower()
    for pattern, replacement in TECH_TERMS_PRESERVE:
        lower = re.sub(pattern, f" {replacement} ", lower, flags=re.IGNORECASE)
    return lower

def restore_tech_terms(tokens: List[str]) -> List[str]:
    """Restores protected alpha tokens back to their standard representation."""
    return [TECH_TERMS_RESTORE.get(t, t) for t in tokens]

def clean_text_for_nlp(text: str) -> str:
    """
    Cleans raw text while preserving alphanumeric tokens and technical markers.
    Removes emails, URLs, excess punctuation, and normalizes whitespace.
    """
    if not text:
        return ""
    
    # Remove email addresses and URLs
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Protect technical terms
    text = protect_tech_terms(text)
    
    # Remove unwanted punctuation but keep hyphens and alphanumeric characters
    text = re.sub(r'[^\w\s\-_]', ' ', text)
    
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_ngrams(tokens: List[str], n: int) -> List[str]:
    """Generates contiguous n-grams from a token sequence."""
    if len(tokens) < n or n < 1:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def get_top_ngrams(tokens: List[str], n: int, top_k: int = 8) -> List[Dict[str, int]]:
    """Returns the top_k most frequent n-grams with counts."""
    ngrams = generate_ngrams(tokens, n)
    if not ngrams:
        return []
    counts = Counter(ngrams)
    return [{"ngram": ngram, "count": count} for ngram, count in counts.most_common(top_k)]

def detect_education_level(text: str) -> Dict[str, Any]:
    """
    Detects educational degrees and relevant CS/Engineering disciplines in text.
    """
    lower = text.lower()
    degrees = []
    
    degree_patterns = [
        (r'\b(ph\.?d|doctor of philosophy|doctorate)\b', 'PhD'),
        (r'\b(master(\'s)?|m\.?s\.?|m\.?tech|m\.?sc|mba)\b', 'Masters'),
        (r'\b(bachelor(\'s)?|b\.?s\.?|b\.?tech|b\.?e\.?|b\.?sc)\b', 'Bachelors'),
        (r'\b(associate(\'s)?|diploma)\b', 'Associate/Diploma')
    ]
    
    for pattern, name in degree_patterns:
        if re.search(pattern, lower):
            degrees.append(name)
            
    is_cs_related = bool(re.search(
        r'\b(computer science|software engineering|data science|information technology|electrical engineering|computer engineering|informatics)\b',
        lower
    ))
    
    return {
        "degrees_found": degrees,
        "highest_degree": degrees[0] if degrees else "None Detected",
        "is_cs_related": is_cs_related
    }
