"""Model Evaluation and Benchmark Experiment Script.

Compares:
1. Baseline: TF-IDF + Cosine Similarity
2. Semantic Model: Word2Vec + Cosine Similarity
3. Hybrid Model: 0.45 * TF-IDF + 0.55 * Word2Vec

Evaluates classification accuracy on the ground-truth labeled evaluation dataset.
"""
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import EVAL_FILE
from app.services.preprocessing import preprocess_text
from app.services.tfidf_engine import calculate_tfidf_similarity
from app.services.word2vec_engine import compute_w2v_similarity
from app.services.similarity_engine import compute_all_similarities
from app.services.scoring_engine import compute_comprehensive_analysis

def evaluate_category(score: float) -> str:
    if score >= 70.0:
        return "High"
    elif score >= 50.0:
        return "Medium"
    else:
        return "Low"

def run_evaluation_benchmark():
    print("=" * 80)
    print("TalentMatch AI — NLP Matching Engine Benchmark & Evaluation")
    print("=" * 80)
    
    if not EVAL_FILE.exists():
        print(f"[ERROR] Evaluation file not found at {EVAL_FILE}")
        return
        
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        pairs = json.load(f)
        
    print(f"Loaded {len(pairs)} ground-truth labeled Candidate-Job evaluation pairs.\n")
    
    tfidf_correct = 0
    w2v_correct = 0
    hybrid_correct = 0
    full_pipeline_correct = 0
    
    results_table = []
    
    for item in pairs:
        pair_id = item["id"]
        pair_name = item["pair_name"]
        expected_cat = item["expected_category"]
        resume_text = item["resume_text"]
        jd_text = item["jd_text"]
        
        # Preprocessing
        res_doc = preprocess_text(resume_text)
        jd_doc = preprocess_text(jd_text)
        
        # 1. TF-IDF Lexical Only
        tfidf_res = calculate_tfidf_similarity(res_doc.clean_text, jd_doc.clean_text)
        tfidf_score = tfidf_res["similarity_pct"]
        tfidf_pred = evaluate_category(tfidf_score)
        if tfidf_pred == expected_cat:
            tfidf_correct += 1
            
        # 2. Word2Vec Semantic Only
        w2v_res = compute_w2v_similarity(res_doc.filtered_tokens, jd_doc.filtered_tokens)
        w2v_score = w2v_res["similarity_pct"]
        w2v_pred = evaluate_category(w2v_score)
        if w2v_pred == expected_cat:
            w2v_correct += 1
            
        # 3. Hybrid Similarity Only (0.45 TF-IDF + 0.55 W2V)
        hybrid_score = round(0.45 * tfidf_score + 0.55 * w2v_score, 1)
        hybrid_pred = evaluate_category(hybrid_score)
        if hybrid_pred == expected_cat:
            hybrid_correct += 1
            
        # 4. Full TalentMatch Multi-Stage Pipeline (Skill + Semantic + Exp + Edu)
        full_res = compute_comprehensive_analysis(resume_text, jd_text, resume_name=pair_id, job_title=pair_name)
        full_score = full_res["overall_score"]
        full_pred = evaluate_category(full_score)
        if full_pred == expected_cat:
            full_pipeline_correct += 1
            
        results_table.append({
            "pair": pair_name,
            "expected": expected_cat,
            "tfidf_score": tfidf_score,
            "tfidf_pred": tfidf_pred,
            "w2v_score": w2v_score,
            "w2v_pred": w2v_pred,
            "hybrid_score": hybrid_score,
            "hybrid_pred": hybrid_pred,
            "full_score": full_score,
            "full_pred": full_pred
        })
        
    total = len(pairs)
    acc_tfidf = round((tfidf_correct / total) * 100.0, 1)
    acc_w2v = round((w2v_correct / total) * 100.0, 1)
    acc_hybrid = round((hybrid_correct / total) * 100.0, 1)
    acc_full = round((full_pipeline_correct / total) * 100.0, 1)
    
    print(f"{'Pair Name':<45} | {'Target':<6} | {'TF-IDF':<8} | {'W2V':<8} | {'Hybrid':<8} | {'Full Match'}")
    print("-" * 95)
    for r in results_table:
        print(f"{r['pair'][:44]:<45} | {r['expected']:<6} | {r['tfidf_score']:>5.1f}%   | {r['w2v_score']:>5.1f}%   | {r['hybrid_score']:>5.1f}%   | {r['full_score']:>5.1f}% ({r['full_pred']})")
        
    print("\n" + "=" * 80)
    print("BENCHMARK EXPERIMENT SUMMARY RESULTS")
    print("=" * 80)
    print(f"{'Approach / Architecture':<35} | {'Classification Accuracy':<25} | {'Notes'}")
    print("-" * 80)
    print(f"{'1. Baseline (TF-IDF Lexical Only)':<35} | {acc_tfidf:>6.1f}%                   | Exact keyword dependent")
    print(f"{'2. Semantic Model (Word2Vec Only)':<35} | {acc_w2v:>6.1f}%                   | Vector space semantic proximity")
    print(f"{'3. Hybrid (TF-IDF + Word2Vec)':<35} | {acc_hybrid:>6.1f}%                   | Weighted 0.45/0.55 balance")
    print(f"{'4. Full TalentMatch AI Pipeline':<35} | {acc_full:>6.1f}%                   | Skill + Semantic + Exp + Edu")
    print("=" * 80)
    
    return {
        "accuracy_tfidf": acc_tfidf,
        "accuracy_word2vec": acc_w2v,
        "accuracy_hybrid": acc_hybrid,
        "accuracy_full_pipeline": acc_full,
        "details": results_table
    }

if __name__ == "__main__":
    run_evaluation_benchmark()
