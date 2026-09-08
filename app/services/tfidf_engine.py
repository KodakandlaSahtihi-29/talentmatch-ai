"""TF-IDF Vectorization and Lexical Cosine Similarity Engine."""
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import settings

def calculate_tfidf_similarity(
    doc1_text: str,
    doc2_text: str,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Dict[str, Any]:
    """
    Computes TF-IDF lexical cosine similarity between two texts and extracts top aligned terms.
    
    Returns:
        Dict:
            - "similarity": float (0.0 - 1.0)
            - "similarity_pct": float (0.0 - 100.0)
            - "top_shared_terms": List[Dict[str, float]]
    """
    if not doc1_text.strip() or not doc2_text.strip():
        return {
            "similarity": 0.0,
            "similarity_pct": 0.0,
            "top_shared_terms": []
        }
        
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=settings.TFIDF_MIN_DF,
            max_df=settings.TFIDF_MAX_DF,
            stop_words='english',
            sublinear_tf=True
        )
        
        tfidf_matrix = vectorizer.fit_transform([doc1_text, doc2_text])
        cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        sim_val = max(0.0, min(1.0, float(cos_sim)))
        
        # Calculate shared term weights for explainability
        feature_names = np.array(vectorizer.get_feature_names_out())
        v1 = tfidf_matrix[0].toarray()[0]
        v2 = tfidf_matrix[1].toarray()[0]
        shared_weights = v1 * v2
        
        top_indices = np.argsort(shared_weights)[::-1]
        top_shared = []
        for idx in top_indices[:10]:
            if shared_weights[idx] > 0.0:
                top_shared.append({
                    "term": str(feature_names[idx]),
                    "weight": round(float(shared_weights[idx]), 4)
                })
                
        return {
            "similarity": sim_val,
            "similarity_pct": round(sim_val * 100.0, 1),
            "top_shared_terms": top_shared
        }
    except Exception as e:
        return {
            "similarity": 0.0,
            "similarity_pct": 0.0,
            "top_shared_terms": [],
            "error": str(e)
        }
