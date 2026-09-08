"""TalentMatch AI — Streamlit Enterprise Analytics Dashboard."""
import os
import sys
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import requests

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings
from app.services.pdf_extractor import extract_text_from_file
from app.services.scoring_engine import compute_comprehensive_analysis
from app.services.academic_nlp import (
    analyze_morphology_comparison,
    compute_edit_distance_demo,
    analyze_pos_tags_detailed,
    parse_syntax_cky_demo,
    analyze_vector_semantics_demo
)
from app.services.word2vec_engine import get_or_train_word2vec
from scripts.evaluate_models import run_evaluation_benchmark

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="TalentMatch AI — Talent Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
css_path = BASE_DIR / "frontend" / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Preload Word2Vec
get_or_train_word2vec()

# Sample Data Directory
SAMPLE_DIR = BASE_DIR / "data" / "sample_data"

# Sidebar: Developer Profile & Architecture Notes
with st.sidebar:
    st.markdown("### 🎯 TalentMatch AI")
    st.markdown("""
    An explainable talent intelligence engine built to eliminate keyword bias and black-box rejections in modern hiring.
    """)
    
    st.markdown("---")
    st.markdown("### 👩‍💻 Developer Profile")
    st.markdown("""
    **Sahithi Kodakandla**  
    *B.Tech in Computer Science & Engineering*  
    GITAM University, Visakhapatnam  
    
    [![GitHub](https://img.shields.io/badge/GitHub-KodakandlaSahtihi--29-181717?logo=github)](https://github.com/KodakandlaSahtihi-29/talentmatch-ai)  
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Sahithi--Kodakandla-0A66C2?logo=linkedin)](https://linkedin.com/in/sahithi-kodakandla)
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Multi-Criteria Weights")
    st.caption("Configured based on human recruiter evaluation experiments:")
    st.progress(0.45, text="Skill Extraction: 45%")
    st.progress(0.30, text="Word2Vec & TF-IDF: 30%")
    st.progress(0.15, text="Experience Alignment: 15%")
    st.progress(0.10, text="Education Relevance: 10%")
    
    st.markdown("---")
    st.markdown("### 🛠️ Architecture Highlights")
    st.markdown("""
    - **NLTK & WordNet**: POS-guided lemmatization preserving dictionary roots.
    - **Levenshtein DP**: Fuzzy edit distance recovering resume typos.
    - **Gensim Word2Vec**: 100D dense continuous vector semantics.
    - **FastAPI REST API**: Asynchronous backend serving endpoints.
    """)

# App Header
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; padding: 1rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.5rem;">
    <div>
        <h1 style="margin: 0; font-size: 1.95rem; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 10px;">
            <span>🎯 TalentMatch AI</span>
            <span style="font-size: 0.75rem; background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; padding: 3px 10px; border-radius: 20px; font-weight: 600;">PORTFOLIO EDITION</span>
        </h1>
        <p style="margin: 4px 0 0 0; color: #475569; font-size: 0.95rem;">
            Explainable Candidate-Job Compatibility & Talent Intelligence Platform
        </p>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">
            Crafted by <a href="https://github.com/KodakandlaSahtihi-29" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 600;">Sahithi Kodakandla</a> • GITAM University
        </div>
    </div>
    <div style="text-align: right;">
        <span style="display: inline-block; width: 8px; height: 8px; background: #10b981; border-radius: 50%; margin-right: 6px;"></span>
        <span style="color: #334155; font-size: 0.85rem; font-weight: 500;">NLP Engine Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_match, tab_history, tab_diagnostic, tab_eval = st.tabs([
    "📊 Match Analysis & Intelligence",
    "📜 Analysis History",
    "🔬 NLP Diagnostics & Playground",
    "📈 Model Benchmark & Evaluation"
])

# ---------------------------------------------------------
# TAB 1: TALENT MATCHING & INTELLIGENCE
# ---------------------------------------------------------
with tab_match:
    st.markdown("### 1. Document Input & Ingestion")
    
    # 1-Click Sample Preloader
    with st.expander("⚡ Load Pre-configured Realistic Sample Candidate & Job Pair", expanded=False):
        sample_options = {
            "Select a sample pair...": None,
            "Sample 1: Candidate A (Senior ML Engineer) vs Job A (Senior ML Engineer) [High Match with Gaps]": ("candidate_a_ml_engineer.txt", "jd_senior_ml_engineer.txt"),
            "Sample 2: Candidate B (Backend Developer) vs Job B (Backend FastAPI Engineer) [High Match]": ("candidate_b_backend_engineer.txt", "jd_backend_fastapi_engineer.txt"),
            "Sample 3: Candidate C (Frontend Developer) vs Job A (Senior ML Engineer) [Low Cross-Domain Match]": ("candidate_c_frontend_developer.txt", "jd_senior_ml_engineer.txt"),
        }
        selected_sample = st.selectbox("Choose sample scenario:", list(sample_options.keys()))
        
        preloaded_resume = ""
        preloaded_jd = ""
        sample_resume_name = "Candidate Resume"
        sample_job_title = "Target Job Description"
        
        if selected_sample and sample_options[selected_sample]:
            res_file, jd_file = sample_options[selected_sample]
            r_path = SAMPLE_DIR / res_file
            j_path = SAMPLE_DIR / jd_file
            if r_path.exists():
                preloaded_resume = r_path.read_text(encoding="utf-8")
                sample_resume_name = res_file.replace(".txt", "")
            if j_path.exists():
                preloaded_jd = j_path.read_text(encoding="utf-8")
                sample_job_title = jd_file.replace(".txt", "")
            st.success(f"Loaded scenario: {selected_sample}")

    col_res, col_jd = st.columns(2)
    
    resume_text = ""
    resume_name = sample_resume_name if preloaded_resume else "Uploaded Resume"
    
    with col_res:
        st.markdown("#### 📄 Candidate Resume")
        resume_upload = st.file_uploader(
            "Upload Resume (PDF or TXT)",
            type=["pdf", "txt"],
            key="resume_uploader",
            help="PyMuPDF will extract text, normalize whitespace, and parse technical tokens."
        )
        
        if resume_upload is not None:
            resume_name = resume_upload.name
            try:
                resume_text, _ = extract_text_from_file(resume_upload.read(), resume_upload.name)
                st.info(f"Loaded: `{resume_name}` ({len(resume_text)} characters extracted)")
            except Exception as e:
                st.error(f"Error reading resume: {e}")
        elif preloaded_resume:
            resume_text = st.text_area("Resume Content (Preloaded)", value=preloaded_resume, height=220)
        else:
            resume_text = st.text_area("Or paste resume text directly here:", height=220, placeholder="Paste resume text...")

    job_text = ""
    job_title = sample_job_title if preloaded_jd else "Target Job Description"
    
    with col_jd:
        st.markdown("#### 📋 Job Description (JD)")
        jd_upload = st.file_uploader(
            "Upload Job Description (PDF or TXT)",
            type=["pdf", "txt"],
            key="jd_uploader",
            help="Extracts requirements, technical skill taxonomy, and qualifications."
        )
        
        if jd_upload is not None:
            job_title = jd_upload.name
            try:
                job_text, _ = extract_text_from_file(jd_upload.read(), jd_upload.name)
                st.info(f"Loaded: `{job_title}` ({len(job_text)} characters extracted)")
            except Exception as e:
                st.error(f"Error reading JD: {e}")
        elif preloaded_jd:
            job_text = st.text_area("Job Description Content (Preloaded)", value=preloaded_jd, height=220)
        else:
            job_text = st.text_area("Or paste Job Description text directly here:", height=220, placeholder="Paste job description requirements...")

    # Advanced Scoring Weight Configurator (Accordion)
    with st.expander("⚙️ Advanced Scoring & NLP Weight Configuration", expanded=False):
        st.markdown("Customize heuristic weights for Lexical vs. Semantic and Component scoring:")
        w_col1, w_col2, w_col3, w_col4 = st.columns(4)
        with w_col1:
            w_skill = st.slider("Skill Match Weight", 0.1, 0.7, 0.45, 0.05)
        with w_col2:
            w_sem = st.slider("Semantic Sim Weight", 0.1, 0.5, 0.30, 0.05)
        with w_col3:
            w_exp = st.slider("Experience Match Weight", 0.05, 0.3, 0.15, 0.05)
        with w_col4:
            w_edu = st.slider("Education Match Weight", 0.0, 0.2, 0.10, 0.05)
            
        st.caption("Note: Weights are normalized internally to sum to 100%. Configurable heuristic per Section 15 & 18.")

    # Analyze Match CTA Button
    btn_analyze = st.button("🚀 Analyze Compatibility Match", type="primary", use_container_width=True)
    
    if btn_analyze:
        if not resume_text.strip():
            st.error("Please upload or paste a Candidate Resume before analyzing.")
        elif not job_text.strip():
            st.error("Please upload or paste a Job Description before analyzing.")
        else:
            with st.spinner("Executing Classical & Vector NLP Pipeline (Tokenization, POS Tagging, TF-IDF, Word2Vec, Fuzzy Matching)..."):
                try:
                    analysis = compute_comprehensive_analysis(
                        resume_text=resume_text,
                        jd_text=job_text,
                        resume_name=resume_name,
                        job_title=job_title
                    )
                    st.session_state["current_analysis"] = analysis
                    st.success("Compatibility analysis complete!")
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    # Display Analysis Results Dashboard
    if "current_analysis" in st.session_state:
        res = st.session_state["current_analysis"]
        
        st.markdown("---")
        st.markdown("## 📊 Compatibility Results & Explainable Intelligence")
        
        # 1. Overall Score & Top Components
        col_main_score, col_comp1, col_comp2, col_comp3, col_comp4 = st.columns([1.3, 1, 1, 1, 1])
        
        overall = res["overall_score"]
        score_color = "#10b981" if overall >= 75 else ("#f59e0b" if overall >= 50 else "#f43f5e")
        
        with col_main_score:
            st.markdown(f"""
            <div class="tm-metric-card" style="border: 1px solid #e2e8f0; border-left: 5px solid {score_color}; background: #f8fafc;">
                <div class="tm-metric-label">Overall Match Score</div>
                <div class="tm-metric-value" style="color: {score_color};">{overall:.1f}%</div>
                <div class="tm-metric-sub">Multi-stage Explainable Hybrid</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_comp1:
            st.markdown(f"""
            <div class="tm-metric-card">
                <div class="tm-metric-label">Skill Match (45%)</div>
                <div class="tm-metric-value">{res['skill_match_score']:.1f}%</div>
                <div class="tm-metric-sub">Exact + Fuzzy + Semantic</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_comp2:
            st.markdown(f"""
            <div class="tm-metric-card">
                <div class="tm-metric-label">Semantic Sim (30%)</div>
                <div class="tm-metric-value">{res['semantic_similarity_score']:.1f}%</div>
                <div class="tm-metric-sub">Word2Vec Embedding Cosine</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_comp3:
            st.markdown(f"""
            <div class="tm-metric-card">
                <div class="tm-metric-label">Experience (15%)</div>
                <div class="tm-metric-value">{res['experience_match_score']:.1f}%</div>
                <div class="tm-metric-sub">Sentence Action Evidence</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_comp4:
            st.markdown(f"""
            <div class="tm-metric-card">
                <div class="tm-metric-label">Education (10%)</div>
                <div class="tm-metric-value">{res['education_match_score']:.1f}%</div>
                <div class="tm-metric-sub">{res.get('education_evaluation', {}).get('resume_degree', 'Degree')}</div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Similarity Breakdown Section
        st.markdown("### 🔍 Similarity Engine Breakdown")
        sim_data = res["similarity"]
        
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            st.metric("TF-IDF Lexical Similarity", f"{sim_data['tfidf_similarity']:.1f}%", help="Exact keyword overlap using TfidfVectorizer and Cosine Similarity")
        with sim_col2:
            st.metric("Word2Vec Semantic Similarity", f"{sim_data['word2vec_similarity']:.1f}%", help="Dense vector space continuous bag-of-words document averaging")
        with sim_col3:
            st.metric("Hybrid Blended Score", f"{sim_data['hybrid_similarity']:.1f}%", help="0.45 * TF-IDF + 0.55 * Word2Vec")
            
        st.caption(f"💡 **Similarity Methodology**: {sim_data['explanation']}")

        # 3. Explainability: "Why did I get this score?"
        st.markdown("### 🧠 Explainable Match Diagnostics (Why did I get this score?)")
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            st.markdown("##### 🟢 Positive Alignment Signals")
            for sig in res.get("positive_signals", []):
                st.markdown(f'<div class="signal-positive"><b>+</b> {sig}</div>', unsafe_allow_html=True)
                
        with exp_col2:
            st.markdown("##### 🔴 Gaps & Missing Evidence")
            for sig in res.get("negative_signals", []):
                st.markdown(f'<div class="signal-negative"><b>-</b> {sig}</div>', unsafe_allow_html=True)

        # 4. Skill Gap Comparison Matrix Table
        st.markdown("### 📋 Skill Comparison Matrix")
        
        all_skills = res.get("all_skills_comparison", [])
        if all_skills:
            df_skills = pd.DataFrame(all_skills)
            
            # Format dataframe for display
            display_df = df_skills[["skill", "category", "requirement_type", "match_type", "resume_evidence", "similarity"]].copy()
            display_df.columns = ["Skill / Technology", "Taxonomy Category", "Requirement Type", "Match Classification", "Resume Evidence Found", "Similarity Score"]
            display_df["Skill / Technology"] = display_df["Skill / Technology"].str.title()
            
            # Filter bar
            filter_match = st.multiselect(
                "Filter by Match Classification:",
                options=["Exact", "Fuzzy", "Semantic", "Missing"],
                default=["Exact", "Fuzzy", "Semantic", "Missing"]
            )
            
            filtered_df = display_df[display_df["Match Classification"].isin(filter_match)]
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
            # Summary Badges
            exact_c = sum(1 for s in all_skills if s["match_type"] == "Exact")
            fuzzy_c = sum(1 for s in all_skills if s["match_type"] == "Fuzzy")
            sem_c = sum(1 for s in all_skills if s["match_type"] == "Semantic")
            miss_c = sum(1 for s in all_skills if s["match_type"] == "Missing")
            
            st.markdown(f"""
            <div style="display: flex; gap: 12px; margin-top: 8px;">
                <span class="badge-exact">Exact Matches: {exact_c}</span>
                <span class="badge-fuzzy">Fuzzy Matches: {fuzzy_c}</span>
                <span class="badge-semantic">Semantic Matches: {sem_c}</span>
                <span class="badge-missing">Missing Gaps: {miss_c}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No specific technical skills were identified in the Job Description.")

        # 5. NLP Insights: N-Grams & POS Distribution
        st.markdown("### 📚 Classical NLP Insights")
        nlp_col1, nlp_col2 = st.columns(2)
        
        with nlp_col1:
            st.markdown("##### 🔤 Top Bigram Collocations")
            ngram_data = res.get("ngram_insights", {})
            r_bigrams = ngram_data.get("resume_top_bigrams", [])[:6]
            j_bigrams = ngram_data.get("jd_top_bigrams", [])[:6]
            
            b_col_r, b_col_j = st.columns(2)
            with b_col_r:
                st.caption("**Resume Top Bigrams:**")
                for bg in r_bigrams:
                    st.text(f"• {bg['ngram']} ({bg['count']})")
            with b_col_j:
                st.caption("**JD Top Bigrams:**")
                for bg in j_bigrams:
                    st.text(f"• {bg['ngram']} ({bg['count']})")
                    
        with nlp_col2:
            st.markdown("##### 🏷️ POS Syntactic Distribution")
            res_pos = res.get("resume_pos", {})
            jd_pos = res.get("jd_pos", {})
            
            pos_df = pd.DataFrame({
                "Syntactic Category": ["Nouns", "Verbs", "Adjectives", "Adverbs", "Others"],
                "Resume %": [res_pos.get("nouns_pct", 0), res_pos.get("verbs_pct", 0), res_pos.get("adjectives_pct", 0), res_pos.get("adverbs_pct", 0), res_pos.get("others_pct", 0)],
                "JD %": [jd_pos.get("nouns_pct", 0), jd_pos.get("verbs_pct", 0), jd_pos.get("adjectives_pct", 0), jd_pos.get("adverbs_pct", 0), jd_pos.get("others_pct", 0)],
            })
            st.bar_chart(pos_df.set_index("Syntactic Category"), height=220)

        # 6. Actionable & Ethical Recommendations
        st.markdown("### 💡 Tailored Candidate Recommendations")
        st.caption("Ethical Guardrail: These recommendations highlight unstated project evidence or targeted learning areas; the system never suggests fabricating experience.")
        
        recs = res.get("recommendations", [])
        for rec in recs:
            p_class = f"priority-{rec['priority'].lower()}"
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-header">
                    <span style="font-weight: 700; color: #f8fafc; font-size: 1rem;">{rec['skill_or_area']}</span>
                    <span class="{p_class}">Priority: {rec['priority']}</span>
                </div>
                <div style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 4px;">{rec['message']}</div>
                <div style="color: #64748b; font-size: 0.8rem;">Context: {rec['context']}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: PERSISTENT ANALYSIS HISTORY
# ---------------------------------------------------------
with tab_history:
    st.markdown("### 📜 Candidate Match History")
    st.caption("Persistent analysis records retrieved from PostgreSQL / SQLite database.")
    
    try:
        from app.core.database import SessionLocal
        from app.models.analysis import AnalysisModel
        
        db = SessionLocal()
        records = db.query(AnalysisModel).order_by(AnalysisModel.created_at.desc()).limit(25).all()
        
        if records:
            history_list = []
            for r in records:
                history_list.append({
                    "ID": r.id,
                    "Timestamp": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "N/A",
                    "Resume": r.resume_name,
                    "Job Target": r.job_title,
                    "Overall Match": f"{r.overall_score:.1f}%",
                    "Skill Match": f"{r.skill_score:.1f}%",
                    "Semantic Match": f"{r.semantic_score:.1f}%",
                    "Hybrid Similarity": f"{r.hybrid_similarity:.1f}%",
                })
            df_history = pd.DataFrame(history_list)
            st.dataframe(df_history, use_container_width=True, hide_index=True)
            
            st.markdown("#### Inspect Past Record Details")
            selected_id = st.selectbox("Select Analysis ID to inspect:", [r.id for r in records])
            if st.button("Load Historical Analysis"):
                target = next((r for r in records if r.id == selected_id), None)
                if target and target.skill_comparison_matrix:
                    matrix = json.loads(target.skill_comparison_matrix)
                    st.json(matrix[:5])
        else:
            st.info("No prior analyses stored yet. Run an analysis in Tab 1 to populate history.")
        db.close()
    except Exception as e:
        st.warning(f"Could not load database history: {e}")

# ---------------------------------------------------------
# TAB 3: NLP DIAGNOSTICS & PLAYGROUND
# ---------------------------------------------------------
with tab_diagnostic:
    st.markdown("### 🔬 NLP Diagnostics & Algorithmic Playground")
    st.markdown("Interactive inspection of the computational linguistics engines powering candidate-job matching.")
    
    diagnostic_mode = st.radio(
        "Select Diagnostic Engine to Inspect:",
        [
            "🔤 Morphology & Lemmatization (WordNet vs Porter)",
            "📐 Fuzzy Edit Distance Matrix (Levenshtein Typo Recovery)",
            "🏷️ Syntactic Profiling & POS Action Verbs",
            "🌐 Vector Embeddings & Word2Vec Latent Space",
            "🎯 Sentence-Level Semantic Coherence"
        ]
    )
    
    if "Morphology" in diagnostic_mode:
        st.markdown("#### 🔤 Morphology: Stemming (Porter) vs. Lemmatization (WordNet)")
        st.caption("Why this matters: Porter stemming truncates suffixes mechanically, whereas WordNet lemmatization preserves actual dictionary lemmas using POS context.")
        col_u1a, col_u1b = st.columns(2)
        
        with col_u1a:
            input_text = st.text_input("Enter text to tokenize and compare roots:", "Developing scalable python backend services optimized for performance")
            if input_text:
                morph_res = analyze_morphology_comparison(input_text)
                st.dataframe(pd.DataFrame(morph_res), use_container_width=True, hide_index=True)
                
        with col_u1b:
            st.markdown("##### 💡 Engineering Takeaway")
            st.info("""
            In technical recruiting, naive stemming often ruins keyword search:
            - `organizing` ➔ `organ` (stemmer removes `-izing`)
            - `universal` ➔ `univers` (unrecognized root)
            
            TalentMatch AI pairs **WordNet lemmatization with POS tag mapping** so terms like `services` $\to$ `service` and `developed` $\to$ `develop` without corrupting technical nouns.
            """)
                
    elif "Fuzzy" in diagnostic_mode:
        st.markdown("#### 📐 Levenshtein Edit Distance Dynamic Programming Matrix")
        st.caption("How TalentMatch AI recovers typos and non-standard skill spellings in candidate resumes:")
        
        w1 = st.text_input("Candidate Resume Spelling (Typo / Variant):", "pyhton")
        w2 = st.text_input("Target Skill in Job Description:", "python")
        if w1 and w2:
            matrix_data = compute_edit_distance_demo(w1, w2)
            st.write(f"**Minimum Edit Operations:** `{matrix_data['distance']}` | **Normalized Similarity Score:** `{matrix_data['similarity']:.2f}` (Threshold: $\ge 0.82$)")
            
            # Show DP Matrix
            mat = np.array(matrix_data["matrix"])
            df_mat = pd.DataFrame(mat, index=["#"] + list(w1), columns=["#"] + list(w2))
            st.caption("Dynamic Programming Cost Matrix ($D[i,j]$):")
            st.dataframe(df_mat, use_container_width=True)
            
    elif "Syntactic" in diagnostic_mode:
        st.markdown("#### 🏷️ Syntactic Profiling & POS Tag Breakdown")
        st.caption("Analyzes the grammatical distribution and action-oriented strength of candidate statements:")
        pos_input = st.text_area("Input candidate statement for POS analysis:", "The senior engineer architected transformer models and optimized PyTorch pipelines.")
        if pos_input:
            pos_results = analyze_pos_tags_detailed(pos_input)
            st.dataframe(pd.DataFrame(pos_results), use_container_width=True, hide_index=True)
            
    elif "Vector" in diagnostic_mode:
        st.markdown("#### 🌐 Continuous Vector Space & Word2Vec Nearest Neighbors")
        st.caption("Demonstrating distributional semantics: technical terms appearing in similar contexts cluster together.")
        w_query = st.text_input("Enter technical term to query nearest vector neighbors:", "python")
        if w_query:
            vec_res = analyze_vector_semantics_demo(w_query)
            st.caption(vec_res["concept"])
            if vec_res["nearest_neighbors"]:
                st.dataframe(pd.DataFrame(vec_res["nearest_neighbors"]), use_container_width=True, hide_index=True)
            else:
                st.warning(f"Term '{w_query}' is Out-Of-Vocabulary (OOV) for the domain model.")
                
    elif "Coherence" in diagnostic_mode:
        st.markdown("#### 🎯 Requirement Alignment & Sentence-Level Similarity")
        st.caption("Evaluates whether candidate project bullets genuinely align with job responsibility requirements:")
        s1 = st.text_input("Requirement Statement (from JD):", "Build REST APIs with FastAPI and relational database backends.")
        s2 = st.text_input("Candidate Experience Evidence (from Resume):", "Developed asynchronous Python microservices using FastAPI and PostgreSQL.")
        if s1 and s2:
            s_res = compute_comprehensive_analysis(s1, s2, "Sentence A", "Sentence B")
            st.write(f"**Semantic Alignment Score:** `{s_res['semantic_similarity_score']:.1f}%`")
            st.write(f"**Lexical TF-IDF Score:** `{s_res['similarity']['tfidf_similarity']:.1f}%`")

# ---------------------------------------------------------
# TAB 4: MODEL BENCHMARK & EVALUATION
# ---------------------------------------------------------
with tab_eval:
    st.markdown("### 📈 Empirical Model Evaluation & Comparison Benchmark")
    st.markdown("""
    This benchmark runs an empirical validation comparing **Lexical TF-IDF**, **Word2Vec Semantic Embeddings**,
    **Hybrid Similarity**, and the **Full Multi-criteria Pipeline** against ground-truth labeled candidate-job pairs.
    """)
    
    if st.button("▶️ Run Evaluation Benchmark Suite", type="primary"):
        with st.spinner("Running benchmark across ground-truth evaluation pairs..."):
            eval_results = run_evaluation_benchmark()
            st.session_state["eval_results"] = eval_results
            
    if "eval_results" in st.session_state:
        eval_data = st.session_state["eval_results"]
        
        st.markdown("#### 🏆 Benchmark Accuracy Results")
        acc1, acc2, acc3, acc4 = st.columns(4)
        with acc1:
            st.metric("1. TF-IDF Lexical Only", f"{eval_data['accuracy_tfidf']:.1f}%")
        with acc2:
            st.metric("2. Word2Vec Only", f"{eval_data['accuracy_word2vec']:.1f}%")
        with acc3:
            st.metric("3. Hybrid (0.45/0.55)", f"{eval_data['accuracy_hybrid']:.1f}%")
        with acc4:
            st.metric("4. Full TalentMatch Pipeline", f"{eval_data['accuracy_full_pipeline']:.1f}%")
            
        st.markdown("#### Detailed Pair Evaluation Breakdown")
        df_eval_details = pd.DataFrame(eval_data["details"])
        st.dataframe(df_eval_details, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# DEVELOPER FOOTER
# ---------------------------------------------------------
st.markdown("""
<div style="margin-top: 3.5rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0; text-align: center; color: #64748b; font-size: 0.85rem;">
    <p style="margin-bottom: 0.35rem; color: #334155;">
        <strong>TalentMatch AI</strong> — Designed & Engineered by 
        <a href="https://github.com/KodakandlaSahtihi-29" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 600;">Sahithi Kodakandla</a>
    </p>
    <p style="margin: 0; font-size: 0.8rem; color: #64748b;">
        B.Tech Computer Science & Engineering • GITAM University | Built with Python, FastAPI, NLTK, Gensim & Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

