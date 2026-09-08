"""Academic NLP Educational Module (Units 1 to 5 Reference Engine)."""
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from app.services.tokenizer import lemmatize_tokens, stem_tokens, tokenize_words
from app.services.pos_tagger import tag_pos
from app.services.fuzzy_matcher import get_levenshtein_matrix
from app.services.word2vec_engine import get_word_similarities

# Penn Treebank POS tag reference glossary
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
    """Unit 1: Demonstrates WordNet Lemmatization vs. Porter Stemming."""
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
            "notes": "WordNet dictionary root" if l != s else "Identical root"
        })
    return results

def compute_edit_distance_demo(word1: str, word2: str) -> Dict[str, Any]:
    """Unit 1: Levenshtein dynamic programming matrix visualization."""
    return get_levenshtein_matrix(word1, word2)

def analyze_pos_tags_detailed(text: str) -> List[Dict[str, str]]:
    """Unit 2: Detailed POS tag breakdown with Penn Treebank explanations."""
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
    """
    Unit 3: Lightweight educational demo of Context-Free Grammar (CFG) / CKY Parsing concepts.
    Demonstrates syntactic constituent reduction: S -> NP VP, NP -> Det N | N, VP -> V NP | V.
    """
    words = [w.lower() for w in re.findall(r'\b\w+\b', sentence)]
    if not words:
        return {"tokens": [], "parse_tree": "Empty sentence", "constituents": []}
        
    pos = tag_pos(words)
    
    # Educational simplified grammar rules
    # N = Nouns, V = Verbs, D = Determiners/Adjectives
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
            
    # Mock constituency tree representation
    tree_repr = f"(S\n  (NP {' '.join([w for w, tag in simplified_tags if tag in ['Noun', 'Modifier']])})\n  (VP {' '.join([w for w, tag in simplified_tags if tag == 'Verb'])})\n)"
    
    return {
        "tokens": words,
        "pos_tags": pos,
        "simplified_tags": simplified_tags,
        "parse_tree": tree_repr,
        "discussion": (
            "In NLP Unit 3, Context-Free Grammars (CFGs) and probabilistic parsers (PCFG / CKY) "
            "decompose sentence structure into hierarchical constituent phrase markers (Noun Phrase NP, Verb Phrase VP). "
            "While deep syntactic trees handle structural ambiguity (e.g. PP attachment), talent matching predominantly "
            "utilizes shallow phrase chunking and vector semantics for semantic alignment."
        )
    }

def analyze_vector_semantics_demo(word: str) -> Dict[str, Any]:
    """Unit 4: Vector semantics and Word2Vec neighboring words."""
    sims = get_word_similarities(word, top_n=6)
    return {
        "query_word": word,
        "nearest_neighbors": sims,
        "concept": "Distributional semantics: words appearing in similar contexts acquire proximate vector representations."
    }
