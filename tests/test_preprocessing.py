"""Tests for NLP preprocessing, tokenization, lemmatization, and POS tagging."""
import pytest
from app.utils.text_utils import protect_tech_terms, restore_tech_terms, clean_text_for_nlp, get_top_ngrams
from app.services.tokenizer import tokenize_words, lemmatize_tokens, stem_tokens
from app.services.pos_tagger import tag_pos, calculate_pos_distribution
from app.services.preprocessing import preprocess_text

def test_tech_term_preservation():
    raw_text = "Proficient in C++, C#, .NET, and Node.js backend development."
    cleaned = clean_text_for_nlp(raw_text)
    tokens = tokenize_words(cleaned)
    
    assert "c++" in tokens
    assert "c#" in tokens
    assert ".net" in tokens
    assert "node.js" in tokens

def test_tokenization_and_stopwords():
    text = "Developed machine learning models using Python and PyTorch."
    doc = preprocess_text(text)
    
    assert "python" in doc.filtered_tokens
    assert "pytorch" in doc.filtered_tokens
    assert "and" not in doc.filtered_tokens  # Standard stopword filtered

def test_lemmatization_vs_stemming():
    tokens = ["developed", "services", "technologies", "running"]
    lemmas = lemmatize_tokens(tokens)
    stems = stem_tokens(tokens)
    
    assert len(lemmas) == len(tokens)
    assert len(stems) == len(tokens)
    assert "service" in lemmas or "services" in lemmas
    assert "servic" in stems or "develop" in stems

def test_pos_tagging_distribution():
    tokens = ["senior", "engineer", "built", "fastapi", "services", "quickly"]
    pos_tags = tag_pos(tokens)
    dist = calculate_pos_distribution(pos_tags)
    
    assert len(pos_tags) == len(tokens)
    assert "nouns_pct" in dist
    assert "verbs_pct" in dist
    assert dist["nouns_pct"] + dist["verbs_pct"] + dist["adjectives_pct"] + dist["adverbs_pct"] + dist["others_pct"] == pytest.approx(100.0, abs=0.5)

def test_ngram_generation():
    tokens = ["machine", "learning", "model", "deployment"]
    bigrams = get_top_ngrams(tokens, 2, top_k=5)
    trigrams = get_top_ngrams(tokens, 3, top_k=5)
    
    assert len(bigrams) == 3
    assert bigrams[0]["ngram"] == "machine learning"
    assert len(trigrams) == 2
    assert trigrams[0]["ngram"] == "machine learning model"
