"""Script to generate realistic sample resumes and job descriptions (PDF & TXT)."""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_data"

SAMPLE_FILES = {
    "candidate_a_ml_engineer.txt": """ALEX RIVERA
Email: alex.rivera@example.com | GitHub: github.com/alexrivera-ml | Location: San Francisco, CA

PROFESSIONAL SUMMARY
Senior Machine Learning Engineer with 5+ years of experience designing and deploying scalable natural language processing (NLP) and deep learning models in production. Demonstrated expertise in Python, PyTorch, Scikit-learn, and FastAPI backend services with PostgreSQL data persistence.

CORE SKILLS & TECHNOLOGIES
• Programming & Frameworks: Python, SQL, C++, PyTorch, Scikit-learn, TensorFlow, HuggingFace
• NLP & Machine Learning: Natural Language Processing, BERT, Transformers, Word2Vec, TF-IDF, Feature Engineering
• Backend & APIs: FastAPI, Flask, REST API, Microservices, Asynchronous Tasks
• Databases & Storage: PostgreSQL, Redis, Vector Databases, SQLite
• DevOps & Practices: Docker, Git, CI/CD, PyTest, Agile, Unit Testing

PROFESSIONAL EXPERIENCE
Senior Machine Learning Engineer | NeuralScale Technologies (2022 – Present)
• Architected enterprise semantic document retrieval engine using PyTorch, BERT transformers, and FastAPI, improving search precision by 34%.
• Built distributed feature engineering pipelines using Python and Scikit-learn, processing 2M+ daily text documents.
• Designed RESTful microservices wrapped in Docker containers with PostgreSQL and Redis caching.
• Mentored 4 junior engineers on NLP preprocessing, tokenization, and model evaluation metrics.

Machine Learning Developer | DataFlow Systems (2019 – 2022)
• Developed text classification and entity extraction models using NLTK and Scikit-learn.
• Automated model deployment pipelines using Git and Docker, reducing deployment turnaround by 45%.
• Executed SQL queries and database schema migrations on PostgreSQL databases.

EDUCATION & CERTIFICATIONS
• Bachelor of Science in Computer Science — University of California, Berkeley (2019)
""",

    "candidate_b_backend_engineer.txt": """JORDAN PATEL
Email: jordan.patel@example.com | GitHub: github.com/jordanpatel-dev | Location: Austin, TX

PROFESSIONAL SUMMARY
High-impact Backend Software Engineer with 4+ years of experience building resilient microservices, high-throughput REST APIs, and event-driven architectures. Specialist in Python, FastAPI, Django, PostgreSQL, and Redis caching.

CORE TECHNICAL SKILLS
• Languages & Frameworks: Python, SQL, JavaScript, FastAPI, Django, Flask, Node.js
• Databases & Caching: PostgreSQL, Redis, MySQL, MongoDB
• Cloud & DevOps: Docker, GitHub Actions, AWS, Linux, CI/CD, Terraform
• Engineering Methodologies: PyTest, TDD, Unit Testing, System Design, RESTful Architecture, Agile Scrum

WORK EXPERIENCE
Backend Software Engineer | CloudVantage Labs (2021 – Present)
• Developed core asynchronous REST APIs using FastAPI and Python handling 10,000+ requests/minute.
• Optimized PostgreSQL query performance, reducing database latency by 42% through query indexing and Redis caching.
• Automated continuous integration and deployment CI/CD workflows using Docker and GitHub Actions.
• Implemented unit testing and integration test suites using PyTest, achieving 92% code coverage.

Software Developer | NextGen Soft (2019 – 2021)
• Built Django web applications and integrated REST APIs with relational database backends.
• Designed database schemas and managed migrations in PostgreSQL.
• Participated in agile sprints, daily standups, and bi-weekly code reviews.

EDUCATION
• Bachelor of Engineering in Computer Science — University of Texas at Austin (2019)
""",

    "candidate_c_frontend_developer.txt": """MORGAN LEE
Email: morgan.lee@example.com | Portfolio: morganlee.dev | Location: New York, NY

PROFESSIONAL SUMMARY
Creative Frontend Web Developer with 4 years of experience crafting responsive, accessible, and high-performance user interfaces. Expert in React, Next.js, TypeScript, JavaScript, HTML5, CSS3, and Tailwind CSS.

CORE SKILLS
• Frontend Technologies: React, Next.js, TypeScript, JavaScript, HTML5, CSS3, Tailwind CSS, Redux, Svelte
• UI/UX & Design: Responsive Web Design, Figma, Accessibility (a11y), Cross-Browser Compatibility
• Tooling & Testing: Git, GitHub, Webpack, Vite, Jest, NPM
• Backend Basics: Node.js, Express, REST API consumption

PROFESSIONAL EXPERIENCE
Frontend Developer | PixelCraft Interactive (2022 – Present)
• Built dynamic web applications using React, Next.js, and TypeScript, delivering 60fps responsive interfaces.
• Implemented state management architectures using Redux Toolkit and React Context API.
• Designed pixel-perfect component systems with Tailwind CSS and CSS3 modules.

Junior Web Developer | Apex Digital (2020 – 2022)
• Developed responsive landing pages and interactive dashboards using JavaScript, HTML5, and CSS3.
• Integrated client-side applications with RESTful APIs developed by the backend team.

EDUCATION
• Bachelor of Science in Information Systems — New York University (2020)
""",

    "jd_senior_ml_engineer.txt": """SENIOR MACHINE LEARNING ENGINEER
Company: QuantumAI Solutions | Location: San Francisco, CA / Remote

ROLE OVERVIEW
We are seeking an experienced Senior Machine Learning Engineer to join our core AI Intelligence team. You will lead the research, development, and production deployment of natural language processing (NLP), deep learning models, and big data architectures.

CORE RESPONSIBILITIES
• Design, train, and deploy state-of-the-art machine learning and deep learning models for text processing and semantic matching.
• Build scalable REST APIs and microservices using Python and FastAPI or Flask.
• Architect distributed data pipelines using PySpark and Apache Spark for large-scale document analytics.
• Deploy and maintain containerized ML services in the cloud using AWS (EC2, S3) and Docker.
• Collaborate with cross-functional product and engineering teams in an agile environment.

MANDATORY REQUIREMENTS
• Strong proficiency in Python and classical machine learning libraries (scikit-learn, PyTorch, or TensorFlow).
• Proven track record in Natural Language Processing (NLP) techniques: tokenization, lemmatization, TF-IDF, embeddings, and transformers.
• Hands-on experience building production REST APIs using FastAPI or Flask.
• Proficiency in SQL and relational database systems (PostgreSQL).
• Hands-on experience with big data processing using PySpark.
• Cloud computing experience with AWS.
• Bachelor of Science in Computer Science or related engineering field.

PREFERRED QUALIFICATIONS (PLUS)
• Familiarity with Docker containerization and Kubernetes orchestration.
• Experience with LLMs, BERT, or vector databases.
""",

    "jd_backend_fastapi_engineer.txt": """SENIOR BACKEND SOFTWARE ENGINEER
Company: NovaCloud Systems | Location: Austin, TX / Remote

ABOUT THE POSITION
We are looking for a Senior Backend Software Engineer to design, scale, and maintain our enterprise microservices platform.

REQUIREMENTS & QUALIFICATIONS
• 4+ years of professional backend engineering experience with Python.
• Strong experience with FastAPI or Django building asynchronous RESTful APIs.
• Deep understanding of PostgreSQL database design, indexing, and query optimization.
• Hands-on experience with in-memory caching systems such as Redis.
• Proficiency with Docker containerization and CI/CD automation using GitHub Actions.
• Experience deploying services on AWS cloud infrastructure.
• Commitment to unit testing, test-driven development (TDD), and PyTest.
• Bachelor's degree in Computer Science or equivalent practical experience.
"""
}

def create_pdf_from_text(text: str, output_path: Path):
    """Creates a formatted PDF document from plain text using PyMuPDF."""
    if fitz is None:
        return
    doc = fitz.open()
    page = doc.new_page(width=595, height=842) # A4 size
    
    rect = fitz.Rect(50, 50, 545, 792)
    # Insert text into page rectangle
    page.insert_textbox(rect, text, fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1), align=0)
    doc.save(str(output_path))
    doc.close()

def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating realistic sample dataset in {SAMPLE_DIR}...")
    
    for filename, content in SAMPLE_FILES.items():
        txt_path = SAMPLE_DIR / filename
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"  [TXT] Created: {txt_path.name}")
        
        # Generate corresponding PDF for resumes and JDs
        if fitz is not None:
            pdf_name = filename.replace(".txt", ".pdf")
            pdf_path = SAMPLE_DIR / pdf_name
            create_pdf_from_text(content.strip(), pdf_path)
            print(f"  [PDF] Created: {pdf_path.name}")
            
    print("[SUCCESS] Sample dataset generation complete.")

if __name__ == "__main__":
    main()
