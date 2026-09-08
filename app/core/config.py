"""Application configuration settings."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SKILLS_FILE = DATA_DIR / "skills.json"
CORPUS_FILE = DATA_DIR / "sample_corpus.txt"
MODEL_DIR = BASE_DIR / "models_cache"
WORD2VEC_MODEL_PATH = MODEL_DIR / "word2vec_tech.model"
EVAL_FILE = DATA_DIR / "evaluation" / "eval_dataset.json"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "TalentMatch AI — NLP-Based Talent Intelligence & Job Matching Platform"
    PROJECT_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Database URL: Defaults to local SQLite, easily overridden with PostgreSQL URI
    # Example PostgreSQL: postgresql://postgres:postgres@localhost:5432/talentmatch
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'talentmatch.db'}")
    
    # NLP Preprocessing & Modeling parameters
    NGRAM_RANGE_MIN: int = 1
    NGRAM_RANGE_MAX: int = 2
    TFIDF_MIN_DF: int = 1
    TFIDF_MAX_DF: float = 1.0
    
    # Word2Vec dimensions and parameters
    W2V_VECTOR_SIZE: int = 100
    W2V_WINDOW: int = 5
    W2V_MIN_COUNT: int = 1
    W2V_EPOCHS: int = 30
    
    # Hybrid Similarity Weights (Configurable Heuristic)
    WEIGHT_TFIDF: float = 0.45
    WEIGHT_WORD2VEC: float = 0.55
    
    # Transparent Overall Scoring Weights
    WEIGHT_SKILL_MATCH: float = 0.45
    WEIGHT_SEMANTIC_SIM: float = 0.30
    WEIGHT_EXPERIENCE_MATCH: float = 0.15
    WEIGHT_EDUCATION_MATCH: float = 0.10
    
    # Fuzzy Matching Threshold (Levenshtein normalized similarity)
    FUZZY_MATCH_THRESHOLD: float = 0.82
    SEMANTIC_SKILL_THRESHOLD: float = 0.70

settings = Settings()
