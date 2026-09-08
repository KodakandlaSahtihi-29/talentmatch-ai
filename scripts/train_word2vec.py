"""Script to train and evaluate domain Word2Vec model."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.word2vec_engine import get_or_train_word2vec, get_word_similarities
from app.core.config import WORD2VEC_MODEL_PATH

def main():
    print("=" * 60)
    print("TalentMatch AI — Word2Vec Domain Training Script")
    print("=" * 60)
    
    # If model exists, remove it to force fresh training
    if WORD2VEC_MODEL_PATH.exists():
        print(f"Removing existing model cache at {WORD2VEC_MODEL_PATH} for retraining...")
        try:
            WORD2VEC_MODEL_PATH.unlink()
        except Exception as e:
            print(f"Notice: {e}")
            
    print("Training Word2Vec embeddings on technical domain corpus...")
    model = get_or_train_word2vec()
    
    if model is None:
        print("[ERROR] Failed to train Word2Vec model.")
        return 1
        
    vocab_size = len(model.wv)
    print(f"[SUCCESS] Model trained successfully! Vocabulary size: {vocab_size} terms.")
    print(f"Model saved to: {WORD2VEC_MODEL_PATH}")
    print("\n--- Semantic Neighbor Tests ---")
    
    test_terms = ["python", "fastapi", "machine", "cloud", "sql"]
    for term in test_terms:
        neighbors = get_word_similarities(term, top_n=4)
        print(f"\nNearest semantic neighbors for '{term}':")
        for item in neighbors:
            print(f"  -> {item['word']}: {item['similarity']:.4f}")
            
    print("\nWord2Vec training and verification complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
