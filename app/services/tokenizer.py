"""Tokenization, Lemmatization, and Stemming service using NLTK."""
import logging
from typing import List, Tuple, Dict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
from app.utils.text_utils import protect_tech_terms, restore_tech_terms

logger = logging.getLogger(__name__)

# Ensure NLTK models are downloaded safely
def ensure_nltk_resources():
    packages = ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "omw-1.4"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            logger.warning(f"NLTK download notice for {pkg}: {e}")

ensure_nltk_resources()

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def get_wordnet_pos(treebank_tag: str) -> str:
    """Map Penn Treebank POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def tokenize_sentences(text: str) -> List[str]:
    """Splits text into individual sentences."""
    if not text:
        return []
    try:
        return sent_tokenize(text)
    except Exception:
        # Fallback if sentence tokenizer fails
        return [s.strip() for s in text.split("\n") if s.strip()]

def tokenize_words(text: str) -> List[str]:
    """
    Tokenizes text while preserving technical terms (e.g., C++, .NET, Node.js).
    """
    if not text:
        return []
    protected = protect_tech_terms(text)
    try:
        raw_tokens = word_tokenize(protected)
    except Exception:
        raw_tokens = protected.split()
    
    # Filter out empty or pure whitespace, allowing alphanumeric and internal underscores/hyphens
    valid_tokens = [t.lower() for t in raw_tokens if t.isalnum() or '_' in t or '-' in t]
    return restore_tech_terms(valid_tokens)

def lemmatize_tokens(tokens: List[str], pos_tags: List[Tuple[str, str]] = None) -> List[str]:
    """
    Lemmatizes tokens using WordNet.
    If pos_tags are provided, uses POS-guided lemmatization for accurate verbal and adjectival reduction.
    """
    if not tokens:
        return []
    
    if pos_tags and len(pos_tags) == len(tokens):
        lemmas = []
        for (tok, tag) in pos_tags:
            wn_pos = get_wordnet_pos(tag)
            lemmas.append(lemmatizer.lemmatize(tok, pos=wn_pos))
        return lemmas
    else:
        return [lemmatizer.lemmatize(t) for t in tokens]

def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Stems tokens using the Porter Stemmer (isolated comparison mode).
    """
    return [stemmer.stem(t) for t in tokens]

def compare_stemming_vs_lemmatization(sample_words: List[str]) -> List[Dict[str, str]]:
    """
    Produces a comparative mapping for academic NLP demonstration.
    """
    comparison = []
    for word in sample_words:
        comparison.append({
            "original": word,
            "lemmatized": lemmatizer.lemmatize(word.lower()),
            "stemmed": stemmer.stem(word.lower())
        })
    return comparison
