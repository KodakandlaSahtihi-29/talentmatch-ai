"""Comprehensive NLP Preprocessing Pipeline."""
from typing import List, Dict, Any, Optional
from nltk.corpus import stopwords
from app.utils.text_utils import clean_text_for_nlp, get_top_ngrams
from app.services.tokenizer import tokenize_words, tokenize_sentences, lemmatize_tokens, stem_tokens, ensure_nltk_resources
from app.services.pos_tagger import tag_pos, calculate_pos_distribution, extract_action_verbs

ensure_nltk_resources()

STOP_WORDS = set(stopwords.words('english'))

# Words to never filter out as stopwords in technical resumes
TECH_EXEMPT_WORDS = {'c', 'r', 'go', 'ai', 'ml', 'it', 'db', 'os'}
EFFECTIVE_STOPWORDS = STOP_WORDS - TECH_EXEMPT_WORDS

class PreprocessedDocument:
    def __init__(
        self,
        raw_text: str,
        clean_text: str,
        sentences: List[str],
        tokens: List[str],
        filtered_tokens: List[str],
        lemmas: List[str],
        pos_tags: List[Any],
        pos_distribution: Dict[str, Any],
        action_verbs: List[str],
        top_unigrams: List[Dict[str, int]],
        top_bigrams: List[Dict[str, int]],
        top_trigrams: List[Dict[str, int]],
    ):
        self.raw_text = raw_text
        self.clean_text = clean_text
        self.sentences = sentences
        self.tokens = tokens
        self.filtered_tokens = filtered_tokens
        self.lemmas = lemmas
        self.pos_tags = pos_tags
        self.pos_distribution = pos_distribution
        self.action_verbs = action_verbs
        self.top_unigrams = top_unigrams
        self.top_bigrams = top_bigrams
        self.top_trigrams = top_trigrams

def preprocess_text(text: str, remove_stopwords: bool = True) -> PreprocessedDocument:
    """
    Executes the full NLP preprocessing pipeline on raw text.
    """
    if not text:
        text = ""
        
    sentences = tokenize_sentences(text)
    clean_str = clean_text_for_nlp(text)
    raw_tokens = tokenize_words(clean_str)
    
    if remove_stopwords:
        filtered_tokens = [t for t in raw_tokens if t not in EFFECTIVE_STOPWORDS and len(t) > 1 or t in TECH_EXEMPT_WORDS]
    else:
        filtered_tokens = raw_tokens
        
    pos_tags = tag_pos(filtered_tokens)
    lemmas = lemmatize_tokens(filtered_tokens, pos_tags)
    pos_dist = calculate_pos_distribution(pos_tags)
    action_verbs = extract_action_verbs(pos_tags)
    
    top_unigrams = get_top_ngrams(lemmas, 1, top_k=10)
    top_bigrams = get_top_ngrams(lemmas, 2, top_k=10)
    top_trigrams = get_top_ngrams(lemmas, 3, top_k=8)
    
    return PreprocessedDocument(
        raw_text=text,
        clean_text=" ".join(lemmas),
        sentences=sentences,
        tokens=raw_tokens,
        filtered_tokens=filtered_tokens,
        lemmas=lemmas,
        pos_tags=pos_tags,
        pos_distribution=pos_dist,
        action_verbs=action_verbs,
        top_unigrams=top_unigrams,
        top_bigrams=top_bigrams,
        top_trigrams=top_trigrams
    )
