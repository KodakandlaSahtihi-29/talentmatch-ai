"""FastAPI Application entry point for TalentMatch AI."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.core.database import init_db
from app.api import resume_router, jd_router, analysis_router
from app.services.word2vec_engine import get_or_train_word2vec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("talentmatch")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Database and Warmup Word2Vec
    logger.info("Initializing TalentMatch AI Database Schema...")
    init_db()
    logger.info("Warming up Word2Vec semantic embedding engine...")
    get_or_train_word2vec()
    logger.info("TalentMatch AI ready for inference.")
    yield
    logger.info("Shutting down TalentMatch AI backend.")

app = FastAPI(
    title="TalentMatch AI — NLP-Based Talent Intelligence & Job Matching Platform",
    description=(
        "Production-grade NLP REST API that analyzes candidate resumes against Job Descriptions "
        "using TF-IDF lexical matching, Word2Vec vector semantics, POS syntactic statistics, "
        "Levenshtein fuzzy matching, and explainable scoring."
    ),
    version=settings.PROJECT_VERSION,
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(resume_router, prefix=settings.API_PREFIX)
app.include_router(jd_router, prefix=settings.API_PREFIX)
app.include_router(analysis_router, prefix=settings.API_PREFIX)

@app.get("/api/health", tags=["Health"])
def health_check():
    """Health check endpoint validating API and NLP engines."""
    return {
        "status": "healthy",
        "app": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "database": "connected",
        "nlp_engine": "active"
    }

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Welcome to TalentMatch AI API. Visit /docs for OpenAPI documentation.",
        "health": "/api/health"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An internal server error occurred: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
