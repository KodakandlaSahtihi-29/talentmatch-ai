"""Levenshtein Edit Distance and Fuzzy Matching service."""
from typing import Tuple, Optional

def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes classic dynamic programming Levenshtein distance matrix.
    Academic demonstration for NLP Unit 1 edit distance.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )
    return dp[m][n]

def get_levenshtein_matrix(s1: str, s2: str):
    """
    Returns full DP matrix with row and column headers for educational visualization.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
            
    return {
        "s1": s1,
        "s2": s2,
        "matrix": dp,
        "distance": dp[m][n],
        "similarity": 1.0 - (dp[m][n] / max(m, n, 1))
    }

def calculate_fuzzy_similarity(str1: str, str2: str) -> float:
    """
    Computes normalized string similarity in range [0.0, 1.0].
    Applies safeguards for short acronyms to avoid spurious fuzzy matches.
    """
    s1, s2 = str1.strip().lower(), str2.strip().lower()
    
    if s1 == s2:
        return 1.0
        
    # Safeguard: Short tokens (<= 3 chars) like 'go', 'r', 'c', 'sql', 'aws' must match closely or exactly
    if len(s1) <= 3 or len(s2) <= 3:
        if abs(len(s1) - len(s2)) > 1:
            return 0.0
        dist = compute_levenshtein_distance(s1, s2)
        if dist > 1:
            return 0.0
        return max(0.0, 1.0 - (dist / max(len(s1), len(s2))))
        
    dist = compute_levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return max(0.0, 1.0 - (dist / max_len))

def is_fuzzy_match(target_skill: str, candidate_text_token: str, threshold: float = 0.82) -> Tuple[bool, float]:
    """
    Checks if candidate_text_token is a fuzzy match for target_skill above threshold.
    """
    sim = calculate_fuzzy_similarity(target_skill, candidate_text_token)
    return (sim >= threshold, sim)
