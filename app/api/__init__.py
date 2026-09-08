"""API routes package."""
from app.api.routes_resume import router as resume_router
from app.api.routes_jd import router as jd_router
from app.api.routes_analysis import router as analysis_router

__all__ = ["resume_router", "jd_router", "analysis_router"]
