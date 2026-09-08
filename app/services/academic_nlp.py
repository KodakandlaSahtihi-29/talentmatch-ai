"""NLP Diagnostics & Algorithmic Playground Utilities.

Provides interactive inspection tools for morphology, edit distance matrices,
POS syntactic distributions, constituent structures, and Word2Vec vector neighborhoods.
"""
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from app.services.tokenizer import lemmatize_tokens, stem_tokens, tokenize_words
from app.services.pos_tagger import tag_pos
from app.services.fuzzy_matcher import get_levenshtein_matrix
from app.services.word2vec_engine import get_word_similarities

# Penn Treebank POS tag glossary for human-readable inspection
PENN_TAG_GLOSSARY = {
    "NN": "Noun, singular or mass",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund/present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "RB": "Adverb",
    "IN": "Preposition or subordinating conjunction",
    "CD": "Cardinal number",
    "DT": "Determiner"
}

def analyze_morphology_comparison(text: str) -> List[Dict[str, str]]:
    """Compares WordNet Lemmatization vs Porter Stemming on candidate text.
    
    Why this matters: Stemming chops word endings aggressively (e.g. 'organization' -> 'organ'),
    whereas WordNet lemmatization uses POS tags to map words back to real dictionary roots.
    """
    tokens = tokenize_words(text)[:15]
    pos = tag_pos(tokens)
    lemmas = lemmatize_tokens(tokens, pos)
    stems = stem_tokens(tokens)
    
    results = []
    for t, p, l, s in zip(tokens, pos, lemmas, stems):
        results.append({
            "token": t,
            "pos_tag": p[1],
            "lemma": l,
            "stem": s,
            "notes": "Preserves valid dictionary root" if l != s else "Identical reduction"
        })
    return results

def compute_edit_distance_demo(word1: str, word2: str) -> Dict[str, Any]:
    """Generates the full Levenshtein dynamic programming matrix and similarity score."""
    return get_levenshtein_matrix(word1, word2)

def analyze_pos_tags_detailed(text: str) -> List[Dict[str, str]]:
    """Provides a detailed POS tag breakdown with Penn Treebank explanations."""
    tokens = tokenize_words(text)[:25]
    pos = tag_pos(tokens)
    
    breakdown = []
    for tok, tag in pos:
        breakdown.append({
            "token": tok,
            "tag": tag,
            "description": PENN_TAG_GLOSSARY.get(tag, "Syntactic element")
        })
    return breakdown

def parse_syntax_cky_demo(sentence: str) -> Dict[str, Any]:
    """Demonstrates constituent syntax structure (Noun Phrase vs Verb Phrase breakdown).
    
    Explains how phrase chunking is used in modern ATS systems rather than full parse trees.
    """
    words = [w.lower() for w in re.findall(r'\b\w+\b', sentence)]
    if not words:
        return {"tokens": [], "parse_tree": "Empty sentence", "constituents": []}
        
    pos = tag_pos(words)
    
    # Practical syntactic mapping
    simplified_tags = []
    for w, t in pos:
        if t.startswith('N'):
            simplified_tags.append((w, 'Noun'))
        elif t.startswith('V'):
            simplified_tags.append((w, 'Verb'))
        elif t.startswith('J') or t == 'DT':
            simplified_tags.append((w, 'Modifier'))
        else:
            simplified_tags.append((w, 'Other'))
            
    # Simple constituent representation
    tree_repr = f"(S\n  (NP {' '.join([w for w, tag in simplified_tags if tag in ['Noun', 'Modifier']])})\n  (VP {' '.join([w for w, tag in simplified_tags if tag == 'Verb'])})\n)"
    
    return {
        "tokens": words,
        "pos_tags": pos,
        "simplified_tags": simplified_tags,
        "parse_tree": tree_repr,
        "discussion": (
            "Modern ATS systems analyze phrases (like 'scalable backend services' or 'deployed microservices') "
            "as functional chunks (Noun Phrases / Verb Phrases). While deep syntactic parse trees are sensitive to "
            "informal resume formatting, combining shallow phrase extraction with vector semantics yields highly robust results."
        )
    }

def analyze_vector_semantics_demo(word: str) -> Dict[str, Any]:
    """Retrieves top vector space nearest neighbors for a technical term via Word2Vec."""
    sims = get_word_similarities(word, top_n=6)
    return {
        "query_word": word,
        "nearest_neighbors": sims,
        "concept": "Distributional semantics: terms occurring in similar technical contexts (e.g. FastAPI and Flask) cluster together in the embedding space."
    }
