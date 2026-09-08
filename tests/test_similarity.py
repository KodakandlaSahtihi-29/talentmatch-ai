"""Tests for TF-IDF, Word2Vec, and Hybrid similarity engines."""
import pytest
from app.services.tfidf_engine import calculate_tfidf_similarity
from app.services.word2vec_engine import compute_document_vector, compute_w2v_similarity
from app.services.similarity_engine import compute_all_similarities

def test_tfidf_cosine_similarity():
    doc1 = "Python machine learning natural language processing scikit-learn"
    doc2 = "Python machine learning deep learning neural networks"
    doc_unrelated = "French cooking pastry baking gourmet desserts"
    
    sim_related = calculate_tfidf_similarity(doc1, doc2)
    sim_unrelated = calculate_tfidf_similarity(doc1, doc_unrelated)
    
    assert 0.0 <= sim_related["similarity"] <= 1.0
    assert 0.0 <= sim_unrelated["similarity"] <= 1.0
    assert sim_related["similarity"] > sim_unrelated["similarity"]

def test_word2vec_document_vector_and_oov():
    tokens_known = ["python", "fastapi", "machine", "learning"]
    tokens_oov = ["randomwordxyz123", "nonexistenttoken456"]
    
    vec_known = compute_document_vector(tokens_known)
    vec_oov = compute_document_vector(tokens_oov)
    
    assert len(vec_known) == 100
    assert len(vec_oov) == 100
    # OOV should return zero vector safely without error
    assert (vec_oov == 0).all()

def test_hybrid_similarity_bounds():
    tokens1 = ["python", "sql", "fastapi", "docker"]
    tokens2 = ["python", "sql", "postgresql", "docker"]
    
    sim_results = compute_all_similarities(
        doc1_clean_text="python sql fastapi docker",
        doc2_clean_text="python sql postgresql docker",
        tokens1=tokens1,
        tokens2=tokens2,
        weight_tfidf=0.45,
        weight_w2v=0.55
    )
    
    assert 0.0 <= sim_results["tfidf_similarity"] <= 100.0
    assert 0.0 <= sim_results["word2vec_similarity"] <= 100.0
    assert 0.0 <= sim_results["hybrid_similarity"] <= 100.0
    assert sim_results["tfidf_weight"] == 0.45
    assert sim_results["word2vec_weight"] == 0.55
