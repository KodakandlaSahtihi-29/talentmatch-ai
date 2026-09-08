"""Similarity Orchestration Engine: TF-IDF, Word2Vec, and Hybrid."""
from typing import Dict, Any, Optional
from app.core.config import settings
from app.services.tfidf_engine import calculate_tfidf_similarity
from app.services.word2vec_engine import compute_w2v_similarity

def compute_all_similarities(
    doc1_clean_text: str,
    doc2_clean_text: str,
    tokens1: list,
    tokens2: list,
    weight_tfidf: Optional[float] = None,
    weight_w2v: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes TF-IDF lexical similarity, Word2Vec semantic similarity,
    and the configurable weighted Hybrid similarity.
    """
    w_tfidf = weight_tfidf if weight_tfidf is not None else settings.WEIGHT_TFIDF
    w_w2v = weight_w2v if weight_w2v is not None else settings.WEIGHT_WORD2VEC
    
    # Normalize weights so they sum to 1.0
    total_w = w_tfidf + w_w2v
    if total_w > 0:
        w_tfidf = w_tfidf / total_w
        w_w2v = w_w2v / total_w
    else:
        w_tfidf, w_w2v = 0.45, 0.55
        
    tfidf_res = calculate_tfidf_similarity(doc1_clean_text, doc2_clean_text)
    w2v_res = compute_w2v_similarity(tokens1, tokens2)
    
    sim_tfidf = tfidf_res["similarity"]
    sim_w2v = w2v_res["similarity"]
    
    sim_hybrid = (w_tfidf * sim_tfidf) + (w_w2v * sim_w2v)
    sim_hybrid = max(0.0, min(1.0, sim_hybrid))
    
    explanation = (
        f"Hybrid similarity combines {round(w_tfidf*100)}% Lexical TF-IDF (keyword overlap) "
        f"and {round(w_w2v*100)}% Word2Vec Semantic Embeddings (conceptual alignment)."
    )
    
    return {
        "tfidf_similarity": round(sim_tfidf * 100.0, 1),
        "word2vec_similarity": round(sim_w2v * 100.0, 1),
        "hybrid_similarity": round(sim_hybrid * 100.0, 1),
        "tfidf_weight": round(w_tfidf, 2),
        "word2vec_weight": round(w_w2v, 2),
        "top_shared_terms": tfidf_res.get("top_shared_terms", []),
        "explanation": explanation
    }
